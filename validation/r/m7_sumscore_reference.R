# M7 reference values (VALIDATION.md V36: summed-score distribution kernel) from R.
# Run from the repository root:  Rscript validation/r/m7_sumscore_reference.R
# Writes tests/reference/v36_sumscore.json. CI does not run R; the JSON is committed.
#
# Dichotomous items (Poisson-binomial distribution), package poibin (Hong, 2013):
#   - ppoibin(method = "RF")      recursive formula (exact)            -> cdf_rf
#   - ppoibin(method = "DFT-CF")  DFT of the characteristic function   -> cdf_dftcf
#   - dpoibin()                   pmf by DFT-CF                        -> pmf_dftcf
#   - ppoibin(method = "RNA")     refined normal approximation         -> cdf_rna
#   - ppoibin(method = "NA")      normal approximation                 -> cdf_na
#   - stats::dpois(s, sum(p))     Poisson approximation                -> pmf_poisson
# Polytomous items: mirt's summed-score table, fscores(method = "EAPsum", full.scores = FALSE),
# whose `expected` column is N * sum_q w_q L(s | theta_q) by mirt's own Lord-Wingersky recursion.
# The item category probabilities on the grid (probtrace) and the weights are stored, so the
# Python test needs only the kernel.
suppressMessages({
  library(poibin)
  library(mirt)
  library(jsonlite)
})

out_dir <- file.path("tests", "reference")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

poibin_case <- function(description, p) {
  kk <- 0:length(p)
  list(
    description = description,
    p = p,
    cdf_rf = ppoibin(kk, p, method = "RF"),
    cdf_dftcf = ppoibin(kk, p, method = "DFT-CF"),
    pmf_dftcf = dpoibin(kk, p),
    cdf_rna = ppoibin(kk, p, method = "RNA"),
    cdf_na = ppoibin(kk, p, method = "NA"),
    pmf_poisson = dpois(kk, sum(p))
  )
}

set.seed(20261129)
theta <- 0.4
a50 <- runif(50, 0.7, 2.0)
b50 <- rnorm(50)
dichotomous <- list(
  uniform_10 = poibin_case("J = 10, p ~ U(0.05, 0.95)", runif(10, 0.05, 0.95)),
  twopl_50 = poibin_case("J = 50, 2PL at theta = 0.4", plogis(a50 * (theta - b50))),
  extreme_200 = poibin_case(
    "J = 200, p from 2PL at theta = -3 (many near 0)",
    plogis(runif(200, 0.7, 2.0) * (-3 - rnorm(200)))
  ),
  rare_300 = poibin_case("J = 300, p ~ U(0.001, 0.02)", runif(300, 0.001, 0.02)),
  equal_40 = poibin_case("J = 40, p = 0.3 (binomial)", rep(0.3, 40))
)

eapsum_case <- function(description, dat, itemtype, quadpts = 61L, theta_lim = c(-6, 6)) {
  m <- mirt(dat, 1, itemtype = itemtype, TOL = 1e-8, verbose = FALSE,
            technical = list(NCYCLES = 5000L, theta_lim = theta_lim))
  stopifnot(isTRUE(extract.mirt(m, "converged")))
  tab <- fscores(m, method = "EAPsum", full.scores = FALSE, quadpts = quadpts,
                 theta_lim = theta_lim, verbose = FALSE)
  Theta <- matrix(seq(theta_lim[1], theta_lim[2], length.out = quadpts))
  w <- dnorm(Theta[, 1]); w <- w / sum(w)
  n_items <- ncol(dat)
  traces <- lapply(seq_len(n_items), function(j) unname(probtrace(extract.item(m, j), Theta)))
  list(
    description = description,
    n = nrow(dat),
    quadpts = quadpts,
    theta_lim = theta_lim,
    nodes = Theta[, 1],
    weights = w,
    traces = traces,  # per item: (Q, K_j)
    sum_scores = tab[, "Sum.Scores"],
    expected = tab[, "expected"],
    observed = tab[, "observed"],
    eap = tab[, "F1"],
    eap_sd = tab[, "SE_F1"]
  )
}

set.seed(20261130)
J <- 10L; N <- 1500L
a <- matrix(runif(J, 0.9, 1.8))
d_grm <- t(sapply(seq_len(J), function(j) c(1.4, 0.1, -1.2) + (j - 5.5) * 0.1))
dat_grm <- simdata(a, d_grm, N, itemtype = "graded")
d_mix <- matrix(NA_real_, J, 3)
d_mix[1:5, 1] <- seq(-1, 1, length.out = 5)
d_mix[6:10, ] <- d_grm[6:10, ]
types <- c(rep("2PL", 5), rep("graded", 5))
dat_mix <- simdata(a, d_mix, N, itemtype = c(rep("dich", 5), rep("graded", 5)))
polytomous <- list(
  grm_10 = eapsum_case("GRM, K = 4, J = 10, N = 1500", dat_grm, "graded"),
  mixed_10 = eapsum_case("five 2PL and five GRM (K = 4) items, N = 1500", dat_mix, types)
)

bundle <- list(
  meta = list(
    case = "v36_sumscore",
    description = "V36: summed-score distribution kernel vs poibin, dpois and mirt EAPsum tables",
    r_version = R.version.string,
    poibin_version = as.character(packageVersion("poibin")),
    mirt_version = as.character(packageVersion("mirt")),
    # The two mirt fits behind the EAPsum tables (asserted in eapsum_case()).
    converged = TRUE
  ),
  dichotomous = dichotomous,
  polytomous = polytomous
)
path <- file.path(out_dir, "v36_sumscore.json")
write_json(bundle, path, auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null")
cat("wrote", path, "\n")
