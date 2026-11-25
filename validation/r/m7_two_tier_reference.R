# M7 reference values (VALIDATION.md V33: two-tier models with dimension reduction) from R mirt.
# Run from the repository root:  Rscript validation/r/m7_two_tier_reference.R
# Writes tests/reference/v33_two_tier.json. CI does not run R; the JSON is committed.
#
# Settings match the other references: bfactor(data, model, model2 = ) (Cai's 2010 two-tier
# dimension reduction), accelerate = "none", TOL = 1e-10, SE.type = "Oakes", theta_lim = c(-6, 6)
# inside technical = list(...), and quadpts = 15 (mirt's default for two primary factors,
# select_quadpts(nfact - nspec + 1) = select_quadpts(3): one 15^3 grid per specific factor).
# The M-step optimizer is BFGS with reltol = 1e-13, as in the 3D case of V14: mirt's NR uses
# numerical Hessians for GroupPars and is very slow on 3,375 nodes. Convergence and grid_check
# are asserted for every case.
#
# Parameter mapping:
#   - Item slopes: a1..aP (primary factors in the order of model2), a(P+1)..aD (specific factors
#     in order; unloaded slopes fixed at 0).
#   - Item intercepts: d (2PL) or d1..d{K-1} (GRM, categories 0..K-1).
#   - Latent parameters: MEAN_1..MEAN_D -> mean_1..mean_D (fixed at 0), COV_11..COV_DD ->
#     var_1..var_D (fixed at 1), COV_21 -> cor_12 (free; the variances are 1, so the covariance
#     is the correlation). Specific factors are orthogonal to everything.
#
# EAP: mirt's fscores() integrates over the full D-dimensional product grid, which equals the
# factorized posterior on the same 1-D grid (the weights are the product of the primary
# multivariate-normal weights and each specific factor's weights). EAPs are computed with
# eap_quadpts points per dimension at mirt's estimates (9^5 = 59,049 nodes for D = 5,
# 15^4 = 50,625 for D = 4); the Python test scores at the same grid.
suppressMessages({
  library(mirt)
  library(jsonlite)
})

out_dir <- file.path("tests", "reference")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

extract_item <- function(x, itemtype, n_factors) {
  nm <- colnames(x)
  slopes <- paste0("a", seq_len(n_factors))
  intercepts <- if (itemtype == "graded") nm[grepl("^d[1-9][0-9]*$", nm)] else "d"
  keep <- c(slopes, intercepts)
  par <- x["par", keep]
  se <- x["SE", keep]
  names(se) <- paste0(keep, "_se")
  c(as.list(par), as.list(se))
}

extract_latent <- function(gp, n_factors) {
  out <- list()
  for (i in seq_len(n_factors)) out[[sprintf("mean_%d", i)]] <- unname(gp["par", sprintf("MEAN_%d", i)])
  for (i in seq_len(n_factors)) out[[sprintf("var_%d", i)]] <- unname(gp["par", sprintf("COV_%d%d", i, i)])
  out$cor_12 <- unname(gp["par", "COV_21"])
  out$cor_12_se <- unname(gp["SE", "COV_21"])
  out
}

pack_patterns <- function(dat) {
  key <- apply(dat, 1, paste, collapse = ",")
  first <- !duplicated(key)
  list(
    patterns = unname(as.matrix(dat[first, , drop = FALSE])),
    counts = as.integer(table(key)[key[first]])
  )
}

# "F1 = 1,2,5\nF2 = 3,4\nCOV = F1*F2" from named item lists.
primary_syntax <- function(primary, items) {
  lines <- vapply(seq_along(primary), function(i) {
    sprintf("F%d = %s", i, paste(match(primary[[i]], items), collapse = ","))
  }, character(1))
  paste(c(lines, "COV = F1*F2"), collapse = "\n")
}

run_case <- function(case, description, dat, primary, specific, itemtype, n_cat, truth,
                     quadpts = 15L, eap_quadpts = 9L, theta_lim = c(-6, 6)) {
  items <- colnames(dat)
  model <- rep(NA_integer_, ncol(dat))
  for (s in seq_along(specific)) model[match(specific[[s]], items)] <- s
  n_primary <- length(primary)
  stopifnot(n_primary == 2L)
  n_factors <- n_primary + length(specific)
  t0 <- proc.time()[3L]
  m <- bfactor(
    dat, model,
    model2 = mirt.model(primary_syntax(primary, items)),
    itemtype = itemtype,
    SE = TRUE,
    SE.type = "Oakes",
    quadpts = quadpts,
    TOL = 1e-10,
    accelerate = "none",
    optimizer = "BFGS",
    control = list(reltol = 1e-13, maxit = 500L),
    technical = list(NCYCLES = 20000L, theta_lim = theta_lim),
    verbose = FALSE
  )
  elapsed <- proc.time()[3L] - t0

  conv <- isTRUE(extract.mirt(m, "converged"))
  grid_check <- range(m@Model$Theta)
  stopifnot(conv, all(grid_check == theta_lim), nrow(m@Model$Theta) == quadpts^(n_primary + 1L))

  cf <- coef(m, printSE = TRUE)
  params <- lapply(items, function(i) extract_item(cf[[i]], itemtype, n_factors))
  names(params) <- items
  latent <- extract_latent(cf$GroupPars, n_factors)
  # Every other covariance is fixed at 0.
  cov_names <- colnames(cf$GroupPars)[grepl("^COV_", colnames(cf$GroupPars))]
  off <- setdiff(cov_names, c(sprintf("COV_%d%d", seq_len(n_factors), seq_len(n_factors)), "COV_21"))
  stopifnot(all(cf$GroupPars["par", off] == 0), all(is.na(cf$GroupPars["SE", off])))

  packed <- pack_patterns(dat)
  n_pat <- nrow(packed$patterns)
  eap_idx <- if (n_pat <= 200L) seq_len(n_pat) else seq(1L, n_pat, length.out = 200L)
  eap_idx <- unique(as.integer(round(eap_idx)))
  sc <- fscores(
    m,
    method = "EAP",
    response.pattern = packed$patterns[eap_idx, , drop = FALSE],
    quadpts = eap_quadpts,
    theta_lim = theta_lim
  )
  fnames <- c(names(primary), names(specific))
  score_cols <- colnames(sc)[seq_len(n_factors) + (ncol(sc) - 2L * n_factors)]
  se_cols <- paste0("SE_", score_cols)
  stopifnot(all(se_cols %in% colnames(sc)))
  eap <- list(
    pattern_indices = eap_idx - 1L,
    quadpts = eap_quadpts,
    theta = setNames(lapply(score_cols, function(f) unname(sc[, f])), fnames),
    se = setNames(lapply(se_cols, function(f) unname(sc[, f])), fnames)
  )

  cat(sprintf(
    "%-16s loglik=%.6f iter=%d n_pat=%d cor_12=%.6f time=%.1fs score_cols=%s\n",
    case, extract.mirt(m, "logLik"), extract.mirt(m, "iterations"), n_pat, latent$cor_12, elapsed,
    paste(score_cols, collapse = ",")
  ))

  list(
    meta = list(
      case = case,
      description = description,
      itemtype = itemtype,
      n_factors = n_factors,
      n_primary = n_primary,
      n_cat = n_cat,
      quadpts = quadpts,
      theta_lim = theta_lim,
      grid_check = grid_check,
      n_nodes = nrow(m@Model$Theta),
      TOL = 1e-10,
      accelerate = "none",
      optimizer = "BFGS",
      converged = conv,
      iterations = extract.mirt(m, "iterations")
    ),
    items = items,
    primary = primary,
    specific = specific,
    patterns = packed$patterns,
    counts = packed$counts,
    loglik = extract.mirt(m, "logLik"),
    estimates = list(params = params, latent = latent),
    eap = eap,
    truth = truth
  )
}

simulate_two_tier <- function(a, d, n, itemtype, rho) {
  sigma <- diag(ncol(a))
  sigma[1, 2] <- sigma[2, 1] <- rho
  as.data.frame(simdata(a, d, n, itemtype = itemtype, sigma = sigma))
}

cases <- list()

# ---- V33a: two-tier 2PL, J = 16, two primary factors (rho = 0.5), three specific factors -------
# s1 lies within F1, s2 within F2, s3 spans both primary factors; q07, q08, q15, q16 have no
# specific factor.
set.seed(20261127)
J <- 16L; N <- 2000L
a <- matrix(0, nrow = J, ncol = 5L)
a[1:8, 1] <- c(1.4, 1.2, 1.6, 1.0, 1.3, 1.5, 1.1, 1.4)
a[9:16, 2] <- c(1.2, 1.7, 1.0, 1.3, 1.5, 1.1, 1.2, 1.4)
a[1:4, 3] <- c(0.9, 1.1, 0.7, 1.0)
a[9:12, 4] <- c(1.2, 0.8, 1.0, 0.9)
a[c(5, 6, 13, 14), 5] <- c(0.8, 1.0, 1.1, 0.7)
d <- matrix(seq(-1.2, 1.2, length.out = J))
dat <- simulate_two_tier(a, d, N, "2PL", rho = 0.5)
colnames(dat) <- sprintf("q%02d", seq_len(J))
q <- function(i) sprintf("q%02d", i)
cases$two_tier_2pl <- run_case(
  case = "two_tier_2pl",
  description = paste(
    "V33a: two-tier 2PL (F1 = q01-q08, F2 = q09-q16, rho = 0.5; s1 = q01-q04, s2 = q09-q12,",
    "s3 = q05, q06, q13, q14; q07, q08, q15, q16 primary only)"
  ),
  dat = dat,
  primary = list(F1 = q(1:8), F2 = q(9:16)),
  specific = list(s1 = q(1:4), s2 = q(9:12), s3 = q(c(5, 6, 13, 14))),
  itemtype = "2PL",
  n_cat = 2L,
  truth = list(a = a, d = as.numeric(d), rho = 0.5),
  eap_quadpts = 9L
)

# ---- V33b: two-tier GRM (K = 4), J = 12, cross-loading item, two testlets ----------------------
set.seed(20261128)
J <- 12L; N <- 2000L; K <- 4L
a <- matrix(0, nrow = J, ncol = 4L)
a[1:6, 1] <- c(1.3, 1.1, 1.5, 1.2, 1.4, 1.0)
a[7:12, 2] <- c(1.6, 1.2, 1.1, 1.3, 1.5, 1.2)
a[6, 2] <- 0.6
a[1:4, 3] <- c(0.8, 1.0, 0.9, 1.1)
a[7:10, 4] <- c(1.0, 0.7, 1.2, 0.9)
d <- t(sapply(seq_len(J), function(j) c(1.5, 0.2, -1.1) + (j - 6.5) * 0.08))
dat <- simulate_two_tier(a, d, N, "graded", rho = 0.4)
colnames(dat) <- sprintf("g%02d", seq_len(J))
g <- function(i) sprintf("g%02d", i)
stopifnot(all(apply(dat, 2, function(x) length(unique(x)) == K)))
cases$two_tier_grm <- run_case(
  case = "two_tier_grm",
  description = paste(
    "V33b: two-tier GRM (K = 4; F1 = g01-g06, F2 = g06-g12 with g06 cross-loading, rho = 0.4;",
    "s1 = g01-g04, s2 = g07-g10)"
  ),
  dat = dat,
  primary = list(F1 = g(1:6), F2 = g(6:12)),
  specific = list(s1 = g(1:4), s2 = g(7:10)),
  itemtype = "graded",
  n_cat = K,
  truth = list(a = a, d = d, rho = 0.4),
  eap_quadpts = 15L
)

bundle <- list(
  meta = list(
    case = "v33_two_tier",
    description = "V33: two-tier models (2PL with three specific factors, GRM with a cross-loading) by bfactor(model2 = )",
    r_version = R.version.string,
    mirt_version = as.character(packageVersion("mirt")),
    TOL = 1e-10,
    accelerate = "none",
    optimizer = "BFGS",
    theta_lim = c(-6, 6),
    grid_check = c(-6, 6),
    converged = all(vapply(cases, function(c) isTRUE(c$meta$converged), logical(1)))
  ),
  cases = cases
)
stopifnot(isTRUE(bundle$meta$converged))

path <- file.path(out_dir, "v33_two_tier.json")
write_json(bundle, path, auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null")
cat("wrote", path, "\n")
