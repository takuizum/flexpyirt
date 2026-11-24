# Number-of-dimensions diagnostics (VALIDATION.md V25): EKC, MAP and parallel analysis.
# Run from the repository root (after m5_polychoric_reference.R):
#   Rscript validation/r/m5_dimensionality_reference.R
# Writes tests/reference/v25_dimensionality.json. CI does not run R; the JSON is committed.
#
# Deterministic parts use the lavaan polychoric matrices of V24 (tests/reference/v24_polychoric.json)
# as input, so they are compared exactly:
#   * EKC: EFAtools::EKC (Braeken & van Assen, 2017).
#   * MAP, squared partial correlations: psych::vss()$map (m >= 1) and EFA.dimensions::MAP (m >= 0).
#   * MAP, fourth power: an independent R implementation of O'Connor's (2000) elementwise rule.
#     EFA.dimensions 0.1.9.3 computes its "4th power" as (tr(R*^4) - p) / (p (p - 1)) with a
#     matrix power (values above 1 occur); it is recorded for documentation only.
#   * Observed PCA and SMC eigenvalues: eigen() of R and of R with SMCs on the diagonal.
# EFA.dimensions also exports PARALLEL() and EKC-like helpers, so EFAtools functions are
# called with their namespace.
# Stochastic part: means and 95th percentiles of reference eigenvalues of normal data
# (EFAtools::PARALLEL with many data sets), compared within Monte Carlo error.
suppressMessages({
  library(EFAtools)
  library(EFA.dimensions)
  library(psych)
  library(jsonlite)
})

v24 <- fromJSON(file.path("tests", "reference", "v24_polychoric.json"), simplifyVector = FALSE)

as_matrix <- function(rows) do.call(rbind, lapply(rows, unlist))

map_elementwise <- function(R, power) {
  p <- ncol(R)
  e <- eigen(R, symmetric = TRUE)
  loadings <- e$vectors %*% diag(sqrt(pmax(e$values, 0)))
  sapply(0:(p - 1), function(m) {
    partial <- if (m == 0) R else R - loadings[, 1:m, drop = FALSE] %*% t(loadings[, 1:m, drop = FALSE])
    d <- diag(1 / sqrt(diag(partial)))
    rstar <- d %*% partial %*% d
    (sum(rstar^power) - p) / (p * (p - 1))
  })
}

cases <- lapply(v24$cases, function(cs) {
  R <- as_matrix(cs$lavaan_rho)
  R <- (R + t(R)) / 2
  n <- cs$n
  p <- ncol(R)
  ekc <- EFAtools::EKC(R, N = n)$results[[1]]
  efad <- EFA.dimensions::MAP(R, corkind = "pearson", Ncases = n, verbose = FALSE)
  smc <- 1 - 1 / diag(solve(R))
  reduced <- R
  diag(reduced) <- smc
  list(
    name = cs$name, n = n, correlation = unname(R),
    ekc_n_factors = ekc$n_factors, ekc_eigenvalues = ekc$y, ekc_reference = ekc$reference,
    map2_psych = unname(vss(R, n = p - 1, n.obs = n, plot = FALSE)$map),
    map2_efa_dimensions = unname(efad$m_values[, 2]),
    map4_matrix_power_efa_dimensions = unname(efad$m_values[, 3]),
    map4_elementwise = map_elementwise(R, 4),
    map2_elementwise = map_elementwise(R, 2),
    map_n_factors_efa_dimensions = efad$NfactorsMAP,
    eigen_pca = eigen(R, symmetric = TRUE, only.values = TRUE)$values,
    eigen_smc = eigen(reduced, symmetric = TRUE, only.values = TRUE)$values
  )
})

set.seed(2501)
sizes <- list(c(300, 8), c(900, 6), c(150, 12))
n_datasets <- 20000
reference <- lapply(sizes, function(s) {
  r <- suppressMessages(EFAtools::PARALLEL(N = s[1], n_vars = s[2], n_datasets = n_datasets,
                                 eigen_type = c("PCA", "SMC"), percent = 95))
  list(n = s[1], p = s[2], n_datasets = n_datasets,
       pca_mean = r$results[[1]]$references$Means,
       pca_p95 = r$results[[1]]$references[["95 Percentile"]],
       smc_mean = r$results[[2]]$references$Means,
       smc_p95 = r$results[[2]]$references[["95 Percentile"]])
})

res <- list(
  meta = list(case = "v25_dimensionality", r_version = R.version.string,
              efatools_version = as.character(packageVersion("EFAtools")),
              efa_dimensions_version = as.character(packageVersion("EFA.dimensions")),
              psych_version = as.character(packageVersion("psych")),
              converged = TRUE),
  cases = cases,
  normal_reference = reference
)
write_json(res, file.path("tests", "reference", "v25_dimensionality.json"), auto_unbox = TRUE,
           digits = NA, null = "null", pretty = FALSE)
cat("wrote tests/reference/v25_dimensionality.json\n")
