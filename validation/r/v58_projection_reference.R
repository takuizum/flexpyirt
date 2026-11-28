# Projective IRT reference values (VALIDATION.md V58, DESIGN.md §28) from R mirt's pirt().
# Run from the repository root:  Rscript validation/r/v58_projection_reference.R
# Writes tests/reference/v58_projection.json. CI does not run R; the JSON is committed.
#
# Parent models: bfactor() with the V32 settings (Gibbons-Hedeker dimension reduction, optimizer =
# "NR", accelerate = "none", TOL = 1e-10, SE.type = "Oakes", 31 points per dimension on [-6, 6]).
# The tight TOL makes the E-table mirt stores (its last E-step) equal to the E-step at the
# estimates, which flexpyirt uses, up to ~1e-10.
#
# Projections (pirt(mod, project = 1), onto the general factor):
#   - estimator = "MML" (Chalmers, Falk & Reise, 2026): a unidimensional model fitted to the
#     marginal E-table; SEs with SE.type = "Oakes" (pirt's default), "complete" and "Richardson"
#     (flexpyirt se = "oakes", "complete", "hessian"); TOL = 1e-10 is passed to mirt().
#   - estimator = "LKA" (Ip, 2010) and "FA" (Stucky et al., 2013): closed forms; "FA" also
#     returns delta-method SEs from the parent's Oakes covariance.
#
# Parameter mapping: a1 -> a, d (2PL) or d1..d{K-1} (GRM) unchanged; the projected latent
# distribution is N(0, 1) on the 31 general-factor nodes.
source(file.path("validation", "r", "setup.R"))  # mirt from GitHub; mirt_version / mirt_sha
suppressMessages({
  library(mirt)
  library(jsonlite)
})

out_dir <- file.path("tests", "reference")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

pack_patterns <- function(dat) {
  key <- apply(dat, 1, paste, collapse = ",")
  first <- !duplicated(key)
  pats <- as.matrix(dat[first, , drop = FALSE])
  list(patterns = unname(pats), counts = as.integer(table(key)[key[first]]))
}

intercept_names <- function(x, itemtype) {
  nm <- colnames(x)
  if (itemtype == "graded") nm[grepl("^d[1-9][0-9]*$", nm)] else "d"
}

run_case <- function(case, description, dat, specific, itemtype, n_cat, quadpts = 31L,
                     theta_lim = c(-6, 6)) {
  items <- colnames(dat)
  model <- rep(NA_integer_, ncol(dat))
  for (s in seq_along(specific)) model[match(specific[[s]], items)] <- s
  n_factors <- length(specific) + 1L
  t0 <- proc.time()[3L]
  m <- bfactor(
    dat, model,
    itemtype = itemtype, SE = TRUE, SE.type = "Oakes", quadpts = quadpts, TOL = 1e-10,
    accelerate = "none", optimizer = "NR",
    technical = list(NCYCLES = 20000L, theta_lim = theta_lim), verbose = FALSE
  )
  conv <- isTRUE(extract.mirt(m, "converged"))
  grid_check <- range(m@Model$Theta)
  stopifnot(conv, all(grid_check == theta_lim), nrow(m@Model$Theta) == quadpts^2)

  cf <- coef(m, printSE = TRUE)
  parent <- lapply(items, function(i) {
    keep <- c(paste0("a", seq_len(n_factors)), intercept_names(cf[[i]], itemtype))
    as.list(cf[[i]]["par", keep])
  })
  names(parent) <- items

  # MML-PIRT with each SE type: "Oakes" (pirt's default), "complete", "Richardson".
  se_types <- c(oakes = "Oakes", complete = "complete", hessian = "Richardson")
  fits <- lapply(se_types, function(st) {
    pm <- pirt(m, TOL = 1e-10, SE.type = st)
    stopifnot(isTRUE(extract.mirt(pm, "converged")))
    coef(pm, printSE = TRUE)
  })
  mml <- lapply(items, function(i) {
    keep <- c("a1", intercept_names(fits$oakes[[i]], itemtype))
    par <- fits$oakes[[i]]["par", keep]
    names(par) <- sub("^a1$", "a", keep)
    se <- lapply(fits, function(pcf) {
      v <- pcf[[i]]["SE", keep]
      names(v) <- names(par)
      as.list(v)
    })
    list(estimate = as.list(par), se = se)
  })
  names(mml) <- items

  closed <- function(mat, se = NULL) {
    out <- lapply(seq_along(items), function(k) {
      nm <- colnames(mat)
      est <- setNames(as.list(unname(mat[k, ])), nm)
      res <- list(estimate = est)
      if (!is.null(se)) res$se <- setNames(as.list(unname(se[k, ])), nm)
      res
    })
    names(out) <- items
    out
  }
  lka <- pirt(m, estimator = "LKA")
  fa <- pirt(m, estimator = "FA")
  stopifnot(is.list(fa), !is.null(fa$SE))
  colnames(fa$items) <- colnames(lka)
  elapsed <- proc.time()[3L] - t0

  packed <- pack_patterns(dat)
  cat(sprintf("%-16s loglik=%.6f iter=%d n_pat=%d time=%.1fs\n", case,
              extract.mirt(m, "logLik"), extract.mirt(m, "iterations"),
              nrow(packed$patterns), elapsed))
  list(
    meta = list(
      case = case, description = description, itemtype = itemtype, n_factors = n_factors,
      n_cat = n_cat, quadpts = quadpts, theta_lim = theta_lim, grid_check = grid_check,
      TOL = 1e-10, accelerate = "none", optimizer = "NR", converged = conv,
      iterations = extract.mirt(m, "iterations")
    ),
    items = items,
    specific = specific,
    patterns = packed$patterns,
    counts = packed$counts,
    parent = list(loglik = extract.mirt(m, "logLik"), params = parent),
    mml = mml,
    lka = closed(lka),
    fa = closed(fa$items, fa$SE)
  )
}

cases <- list()

# ---- V58a: Ip (2010) Table 3 bifactor 2PL (the pirt() example), 3% MCAR --------------------------
set.seed(20261128)
a <- cbind(rep(c(2.63, 1.63, 1.38), each = 5), c(rep(3.07, 5), numeric(10)),
           c(numeric(5), rep(1.36, 5), numeric(5)), c(numeric(10), rep(.69, 5)))
d <- round(rnorm(15, 0, sd = 1.5), 2)
dat <- as.data.frame(simdata(a, d, 2000L, itemtype = "2PL", sigma = diag(4)))
colnames(dat) <- sprintf("q%02d", seq_len(15))
dat[matrix(runif(prod(dim(dat))) < 0.03, nrow(dat))] <- NA
cases$bifactor_2pl <- run_case(
  case = "bifactor_2pl",
  description = "V58a: Ip (2010) Table 3 bifactor 2PL (s1 = q01-q05, s2 = q06-q10, s3 = q11-q15), 3% MCAR",
  dat = dat,
  specific = list(s1 = sprintf("q%02d", 1:5), s2 = sprintf("q%02d", 6:10), s3 = sprintf("q%02d", 11:15)),
  itemtype = "2PL", n_cat = 2L
)

# ---- V58b: bifactor GRM (K = 4), two testlets of four items and two general-only items ---------
set.seed(20261129)
J <- 10L; K <- 4L
a <- matrix(0, nrow = J, ncol = 3L)
a[, 1] <- c(1.3, 1.1, 1.5, 1.2, 1.4, 1.0, 1.6, 1.2, 1.1, 1.3)
a[1:4, 2] <- c(1.2, 1.4, 0.9, 1.1)
a[5:8, 3] <- c(0.8, 1.0, 1.3, 0.9)
d <- t(sapply(seq_len(J), function(j) c(1.5, 0.2, -1.1) + (j - 5.5) * 0.1))
dat <- as.data.frame(simdata(a, d, 1500L, itemtype = "graded", sigma = diag(3)))
colnames(dat) <- sprintf("g%02d", seq_len(J))
stopifnot(all(apply(dat, 2, function(x) length(unique(x)) == K)))
cases$bifactor_grm <- run_case(
  case = "bifactor_grm",
  description = "V58b: bifactor GRM (K = 4; s1 = g01-g04, s2 = g05-g08; g09, g10 general only)",
  dat = dat,
  specific = list(s1 = sprintf("g%02d", 1:4), s2 = sprintf("g%02d", 5:8)),
  itemtype = "graded", n_cat = K
)

bundle <- list(
  meta = list(
    case = "v58_projection",
    description = "V58: projective IRT, mirt pirt() (MML, LKA, FA) on bfactor() fits",
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

path <- file.path(out_dir, "v58_projection.json")
write_json(bundle, path, auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null")
cat("wrote", path, "\n")
