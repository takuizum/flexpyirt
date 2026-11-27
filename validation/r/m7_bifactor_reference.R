# M7 reference values (VALIDATION.md V32: bifactor models with dimension reduction) from R mirt.
# Run from the repository root:  Rscript validation/r/m7_bifactor_reference.R
# Writes tests/reference/v32_bifactor.json. CI does not run R; the JSON is committed.
#
# Settings match the other references: bfactor() (Gibbons-Hedeker dimension reduction), optimizer =
# "NR", accelerate = "none", TOL = 1e-10, SE.type = "Oakes", theta_lim = c(-6, 6) inside
# technical = list(...), and quadpts = 31 (mirt's default for bfactor, select_quadpts(2): one
# 31 x 31 grid per specific factor). Convergence and grid_check are asserted for every case.
#
# Parameter mapping:
#   - Item slopes: a1 (general), a2..aD (specific factors in order; unloaded slopes fixed at 0).
#   - Item intercepts: d (2PL) or d1..d{K-1} (GRM, categories 0..K-1).
#   - Latent parameters: MEAN_1..MEAN_D -> mean_1..mean_D (fixed at 0), COV_11..COV_DD ->
#     var_1..var_D (fixed at 1); the factors are orthogonal (no free covariances).
#
# EAP: mirt's fscores() has no bifactor path; it integrates over the full D-dimensional product
# grid, which with independent normal weights equals the factorized posterior on the same 1-D grid.
# With D = 4 the product grid has quadpts^4 nodes, so EAPs are computed with eap_quadpts = 15
# (15^4 = 50,625 nodes) at mirt's estimates; the Python test scores at the same grid.
source(file.path("validation", "r", "setup.R"))  # mirt from GitHub; mirt_version / mirt_sha
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

run_case <- function(case, description, dat, specific, general, itemtype, n_cat, truth,
                     quadpts = 31L, eap_quadpts = 15L, theta_lim = c(-6, 6)) {
  items <- colnames(dat)
  spec_names <- names(specific)
  model <- rep(NA_integer_, ncol(dat))
  for (s in seq_along(specific)) model[match(specific[[s]], items)] <- s
  n_factors <- length(specific) + 1L
  t0 <- proc.time()[3L]
  m <- bfactor(
    dat, model,
    itemtype = itemtype,
    SE = TRUE,
    SE.type = "Oakes",
    quadpts = quadpts,
    TOL = 1e-10,
    accelerate = "none",
    optimizer = "NR",
    technical = list(NCYCLES = 20000L, theta_lim = theta_lim),
    verbose = FALSE
  )
  elapsed <- proc.time()[3L] - t0

  conv <- isTRUE(extract.mirt(m, "converged"))
  grid_check <- range(m@Model$Theta)
  stopifnot(conv, all(grid_check == theta_lim), nrow(m@Model$Theta) == quadpts^2)

  cf <- coef(m, printSE = TRUE)
  params <- lapply(items, function(i) extract_item(cf[[i]], itemtype, n_factors))
  names(params) <- items
  latent <- extract_latent(cf$GroupPars, n_factors)

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
  fnames <- c(general, spec_names)
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
    "%-16s loglik=%.6f iter=%d n_pat=%d time=%.1fs score_cols=%s\n",
    case, extract.mirt(m, "logLik"), extract.mirt(m, "iterations"), n_pat, elapsed,
    paste(score_cols, collapse = ",")
  ))

  list(
    meta = list(
      case = case,
      description = description,
      itemtype = itemtype,
      n_factors = n_factors,
      n_cat = n_cat,
      quadpts = quadpts,
      theta_lim = theta_lim,
      grid_check = grid_check,
      n_nodes = nrow(m@Model$Theta),
      TOL = 1e-10,
      accelerate = "none",
      optimizer = "NR",
      converged = conv,
      iterations = extract.mirt(m, "iterations")
    ),
    items = items,
    general = general,
    specific = specific,
    patterns = packed$patterns,
    counts = packed$counts,
    loglik = extract.mirt(m, "logLik"),
    estimates = list(params = params, latent = latent),
    eap = eap,
    truth = truth
  )
}

simulate_bifactor <- function(a, d, n, itemtype) {
  sigma <- diag(ncol(a))
  as.data.frame(simdata(a, d, n, itemtype = itemtype, sigma = sigma))
}

cases <- list()

# ---- V32a: bifactor 2PL, J = 15, three specific factors of four items, three general-only items --
set.seed(20261125)
J <- 15L; N <- 2000L
a <- matrix(0, nrow = J, ncol = 4L)
a[, 1] <- c(1.4, 1.2, 1.6, 1.0, 1.3, 1.5, 1.1, 1.4, 1.2, 1.7, 1.0, 1.3, 1.5, 1.1, 1.2)
a[1:4, 2] <- c(0.9, 1.1, 0.7, 1.0)
a[5:8, 3] <- c(1.2, 0.8, 1.0, 0.9)
a[9:12, 4] <- c(0.8, 1.0, 1.1, 0.7)
d <- matrix(seq(-1.2, 1.2, length.out = J))
dat <- simulate_bifactor(a, d, N, "2PL")
colnames(dat) <- sprintf("q%02d", seq_len(J))
cases$bifactor_2pl <- run_case(
  case = "bifactor_2pl",
  description = paste(
    "V32a: bifactor 2PL (general on q01-q15; s1 = q01-q04, s2 = q05-q08, s3 = q09-q12;",
    "q13-q15 general only)"
  ),
  dat = dat,
  specific = list(s1 = sprintf("q%02d", 1:4), s2 = sprintf("q%02d", 5:8), s3 = sprintf("q%02d", 9:12)),
  general = "G",
  itemtype = "2PL",
  n_cat = 2L,
  truth = list(a = a, d = as.numeric(d))
)

# ---- V32b: bifactor GRM (K = 4), J = 12, three testlets of four items --------------------------
set.seed(20261126)
J <- 12L; N <- 2000L; K <- 4L
a <- matrix(0, nrow = J, ncol = 4L)
a[, 1] <- c(1.3, 1.1, 1.5, 1.2, 1.4, 1.0, 1.6, 1.2, 1.1, 1.3, 1.5, 1.2)
a[1:4, 2] <- c(0.8, 1.0, 0.9, 1.1)
a[5:8, 3] <- c(1.0, 0.7, 1.2, 0.9)
a[9:12, 4] <- c(0.9, 1.1, 0.8, 1.0)
d <- t(sapply(seq_len(J), function(j) c(1.5, 0.2, -1.1) + (j - 6.5) * 0.08))
dat <- simulate_bifactor(a, d, N, "graded")
colnames(dat) <- sprintf("g%02d", seq_len(J))
stopifnot(all(apply(dat, 2, function(x) length(unique(x)) == K)))
cases$bifactor_grm <- run_case(
  case = "bifactor_grm",
  description = "V32b: bifactor GRM (K = 4; s1 = g01-g04, s2 = g05-g08, s3 = g09-g12)",
  dat = dat,
  specific = list(s1 = sprintf("g%02d", 1:4), s2 = sprintf("g%02d", 5:8), s3 = sprintf("g%02d", 9:12)),
  general = "G",
  itemtype = "graded",
  n_cat = K,
  truth = list(a = a, d = d)
)

bundle <- list(
  meta = list(
    case = "v32_bifactor",
    description = "V32: bifactor models (2PL with general-only items, GRM) by bfactor()",
    r_version = R.version.string,
    mirt_version = mirt_version_string(), mirt_sha = mirt_commit_sha(),
    TOL = 1e-10,
    accelerate = "none",
    optimizer = "NR",
    theta_lim = c(-6, 6),
    grid_check = c(-6, 6),
    converged = all(vapply(cases, function(c) isTRUE(c$meta$converged), logical(1)))
  ),
  cases = cases
)
stopifnot(isTRUE(bundle$meta$converged))

path <- file.path(out_dir, "v32_bifactor.json")
write_json(bundle, path, auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null")
cat("wrote", path, "\n")
