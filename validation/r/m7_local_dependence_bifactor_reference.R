# Local dependence for bifactor and two-tier models (VALIDATION.md V38, M7, issue #81).
# Run from the repository root:  Rscript validation/r/m7_local_dependence_bifactor_reference.R
# Writes tests/reference/v38_local_dependence_bifactor.json. CI does not run R; the JSON is
# committed.
#
# mirt residuals(type = "LD" / "LDG2" / "Q3") on bfactor() fits:
#   - a bifactor 2PL with three specific factors, two general-only items, one locally dependent
#     pair inside a specific factor and 3% MCAR (4 factors, 21 points per dimension);
#   - a bifactor GRM with 3 categories and two specific factors (3 factors, 21 points);
#   - a two-tier 2PL, bfactor(model2 = ), two correlated primary factors (one cross-loading
#     item) and two specific factors (4 factors, 15 points).
# Grids: residuals(type = "LD") builds the full product grid thetaComb(theta, nfact) with the
# fit's quadpts (21^4 = 194,481 and 15^4 = 50,625 nodes here; mirt warns above three factors) and
# the multivariate-normal weights on it. That sum equals flexpyirt's factorized sums on the same
# one-dimensional grid (DESIGN.md §17.3). Q3 uses fscores(), whose default grid does not depend on
# the fit, so Theta = fscores(m, quadpts = <the fit's>) is passed ("q3"); the default call is
# stored as "q3_default".
# The Python test puts mirt's estimates into a flexpyirt FitResult (every parameter fixed) and
# compares, so the comparison does not depend on how tightly either program converged (the
# estimates themselves are compared in V32 and V33).
suppressMessages({
  library(mirt)
  library(jsonlite)
})

LIM <- c(-6, 6)

rows_of <- function(mat) {
  lapply(seq_len(nrow(mat)), function(i) lapply(unname(mat[i, ]),
                                                function(v) if (is.na(v)) NULL else v))
}

# Lower triangle of residuals(type = "LD"): X2 (row > col); upper: signed Cramer's V.
pair_list <- function(ld, ldg2, q3, q3_default) {
  ld <- unclass(ld); ldg2 <- unclass(ldg2); q3 <- unclass(q3); q3_default <- unclass(q3_default)
  J <- ncol(ld)
  lower <- which(lower.tri(matrix(0, J, J)), arr.ind = TRUE)
  lapply(seq_len(nrow(lower)), function(k) {
    r <- lower[k, 1]; cl <- lower[k, 2]
    list(i = cl - 1L, j = r - 1L, x2 = ld[r, cl], cramer_v = ld[cl, r], g2 = ldg2[r, cl],
         q3 = q3[cl, r], q3_default = q3_default[cl, r])
  })
}

item_coefs <- function(cf) {
  keep <- setdiff(colnames(cf), c("g", "u"))
  out <- lapply(seq_len(nrow(cf)), function(k) as.list(cf[k, keep]))
  names(out) <- rownames(cf)
  out
}

latent_pars <- function(cf, nfact, n_primary) {
  out <- list()
  for (m in seq_len(nfact)) {
    out[[paste0("mean_", m)]] <- cf$means[[m]]
    out[[paste0("var_", m)]] <- cf$cov[m, m]
  }
  if (n_primary == 2L) out$cor_12 <- cf$cov[2, 1]
  # Everything else is orthogonal.
  off <- cf$cov[lower.tri(cf$cov)]
  stopifnot(sum(off != 0) == (n_primary == 2L), all(diag(cf$cov) == 1), all(cf$means == 0))
  out
}

sim_2pl <- function(eta, prefix = "q") {
  x <- (matrix(runif(length(eta)), nrow(eta)) < plogis(eta)) * 1L
  colnames(x) <- sprintf("%s%02d", prefix, seq_len(ncol(x)))
  x
}

add_missing <- function(x, rate) {
  x[matrix(runif(length(x)) < rate, nrow(x))] <- NA
  x
}

run_case <- function(name, kind, dat, specific_index, itemtype, quadpts, primary = NULL,
                     specific, model2 = NULL) {
  args <- list(dat, specific_index, itemtype = itemtype, quadpts = quadpts, TOL = 1e-6,
               verbose = FALSE, technical = list(NCYCLES = 20000, theta_lim = LIM))
  if (!is.null(model2)) args$model2 <- model2
  m <- do.call(bfactor, args)
  stopifnot(extract.mirt(m, "converged"), all(range(m@Model$Theta) == LIM))
  nfact <- extract.mirt(m, "nfact")
  n_primary <- nfact - length(specific)
  stopifnot(nrow(m@Model$Theta) == quadpts^(n_primary + 1L))
  # The product grid of more than three factors triggers mirt's QMC advice; the grid is wanted.
  ld <- suppressWarnings(residuals(m, type = "LD", verbose = FALSE))
  ldg2 <- suppressWarnings(residuals(m, type = "LDG2", verbose = FALSE))
  theta <- fscores(m, full.scores = TRUE, quadpts = quadpts, theta_lim = LIM)
  q3 <- residuals(m, type = "Q3", verbose = FALSE, Theta = theta)
  q3_default <- residuals(m, type = "Q3", verbose = FALSE)
  cf <- coef(m, simplify = TRUE)
  g <- list(group = NULL, items = item_coefs(cf$items),
            latent = latent_pars(cf, nfact, n_primary),
            pairs = pair_list(ld, ldg2, q3, q3_default))
  list(name = name, kind = kind, itemtype = itemtype, items = colnames(dat), n = nrow(dat),
       quadpts = quadpts, primary = primary, specific = specific, data = rows_of(dat),
       group = NULL, loglik = extract.mirt(m, "logLik"), groups = list(g))
}

set.seed(20261125)
cases <- list()

# (1) Bifactor 2PL: 11 items, specific factors of 3 items each, q10 and q11 general only;
#     q01 and q02 share an extra nuisance dimension (dependence beyond their specific factor);
#     3% MCAR.
n <- 900
theta <- matrix(rnorm(4 * n), n)
ag <- c(1.3, 1.1, 0.9, 1.5, 0.8, 1.2, 1.0, 1.4, 0.9, 1.1, 1.6)
as <- c(0.9, 0.7, 1.1, 0.8, 1.2, 0.6, 1.0, 0.7, 0.9, 0, 0)
d <- c(0.3, -0.5, 0.7, -0.9, 0.1, 1.0, -0.4, 0.4, -0.1, 0.6, -0.7)
block <- c(1, 1, 1, 2, 2, 2, 3, 3, 3, NA, NA)
eta <- sapply(1:11, function(j) {
  ag[j] * theta[, 1] + (if (is.na(block[j])) 0 else as[j] * theta[, 1 + block[j]]) + d[j]
})
eta[, 1:2] <- eta[, 1:2] + 1.0 * rnorm(n)
dat <- add_missing(sim_2pl(eta), 0.03)
specific <- list(S1 = colnames(dat)[1:3], S2 = colnames(dat)[4:6], S3 = colnames(dat)[7:9])
cases[[1]] <- run_case("bifactor_2pl", "bifactor", dat, block, "2PL", 21, specific = specific)

# (2) Bifactor GRM, 3 categories: 8 items, two specific factors of 4 items.
n <- 800
theta <- matrix(rnorm(3 * n), n)
ag <- c(1.3, 1.0, 1.6, 0.9, 1.2, 1.4, 1.1, 0.8)
as <- c(0.8, 1.0, 0.6, 0.9, 0.7, 1.1, 0.9, 0.6)
d1 <- c(1.0, 0.5, 1.2, 0.8, 1.5, 0.3, 0.9, 0.6)
d2 <- d1 - c(1.5, 1.3, 1.4, 1.7, 1.6, 1.5, 1.3, 1.3)
block <- rep(1:2, each = 4)
dat <- sapply(1:8, function(j) {
  lin <- ag[j] * theta[, 1] + as[j] * theta[, 1 + block[j]]
  u <- runif(n)
  (u < plogis(lin + d1[j])) + (u < plogis(lin + d2[j]))
})
colnames(dat) <- sprintf("g%02d", 1:8)
specific <- list(S1 = colnames(dat)[1:4], S2 = colnames(dat)[5:8])
cases[[2]] <- run_case("bifactor_grm", "bifactor", dat, block, "graded", 21,
                       specific = specific)

# (3) Two-tier 2PL: primary F1 on items 1-5, F2 on items 5-10 (item 5 cross-loads), correlation
#     0.5; specific S1 on items 1-3, S2 on items 6-8; items 4, 5, 9 and 10 have no specific
#     factor.
n <- 1000
z <- matrix(rnorm(2 * n), n) %*% chol(matrix(c(1, 0.5, 0.5, 1), 2))
sp <- matrix(rnorm(2 * n), n)
a1 <- c(1.3, 1.1, 0.9, 1.5, 0.8, 0, 0, 0, 0, 0)
a2 <- c(0, 0, 0, 0, 0.7, 1.2, 1.0, 1.4, 0.9, 1.1)
as <- c(0.9, 0.7, 1.1, 0, 0, 0.8, 1.0, 0.7, 0, 0)
d <- c(0.3, -0.5, 0.7, -0.9, 0.1, 1.0, -0.4, 0.4, -0.1, 0.6)
block <- c(1, 1, 1, NA, NA, 2, 2, 2, NA, NA)
eta <- z[, 1] %o% a1 + z[, 2] %o% a2 + matrix(d, n, 10, byrow = TRUE)
for (j in which(!is.na(block))) eta[, j] <- eta[, j] + as[j] * sp[, block[j]]
dat <- sim_2pl(eta, prefix = "t")
primary <- list(F1 = colnames(dat)[1:5], F2 = colnames(dat)[5:10])
specific <- list(S1 = colnames(dat)[1:3], S2 = colnames(dat)[6:8])
cases[[3]] <- run_case("two_tier_2pl", "two_tier", dat, block, "2PL", 15, primary = primary,
                       specific = specific,
                       model2 = mirt.model("F1 = 1-5\nF2 = 5-10\nCOV = F1*F2"))

out <- list(
  meta = list(case = "v38_local_dependence_bifactor", r_version = R.version.string,
              mirt_version = as.character(packageVersion("mirt")), converged = TRUE,
              theta_lim = LIM, grid_check = LIM),
  cases = cases
)
write_json(out, file.path("tests", "reference", "v38_local_dependence_bifactor.json"),
           auto_unbox = TRUE, digits = NA, null = "null")
cat("wrote tests/reference/v38_local_dependence_bifactor.json\n")
