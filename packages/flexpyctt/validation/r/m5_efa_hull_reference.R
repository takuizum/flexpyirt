# EFA (ULS / ML) and the Hull method (VALIDATION.md V28).
# Run from the repository root (after m5_polychoric_reference.R):
#   Rscript validation/r/m5_efa_hull_reference.R
# Writes tests/reference/v28_efa_hull.json. CI does not run R; the JSON is committed.
#
# EFAtools::EFA(method = "ULS" / "ML", rotation = "none") gives the unrotated loadings and fit
# indices (CAF, Bartlett-corrected chi-square, CFI, RMSEA) for every k up to HULL's maximum;
# EFAtools::HULL(method = "ULS" / "ML", gof = CAF, CFI, RMSEA) gives the Hull tables. HULL's
# maximum number of factors comes from parallel analysis (random), so it is recorded
# (settings$n_fac_max) and passed to flexpyirt explicitly. psych::fa(fm = "minres" / "ml") is
# recorded for one matrix as a second EFA reference.
suppressMessages({
  library(EFAtools)
  library(psych)
  library(jsonlite)
})

v24 <- fromJSON(file.path("tests", "reference", "v24_polychoric.json"), simplifyVector = FALSE)
as_matrix <- function(rows) do.call(rbind, lapply(rows, unlist))
rows_of <- function(m) lapply(seq_len(nrow(m)), function(i) unname(as.list(m[i, ])))

signed <- function(L) {
  L <- unclass(L)
  s <- ifelse(colSums(L) < 0, -1, 1)
  sweep(L, 2, s, "*")
}

efa_case <- function(R, n, k, method) {
  e <- suppressWarnings(suppressMessages(
    EFAtools::EFA(R, n_factors = k, N = n, method = method, rotation = "none")))
  fi <- e$fit_indices
  list(k = k, method = method, loadings = rows_of(signed(e$unrot_loadings)),
       h2 = unname(e$h2),
       fit = list(chi2 = fi$chi, df = fi$df, p = fi$p_chi, caf = fi$CAF, cfi = fi$CFI,
                  rmsea = fi$RMSEA))
}

hull_case <- function(R, n, method) {
  set.seed(2801)
  h <- suppressWarnings(suppressMessages(
    EFAtools::HULL(R, N = n, method = method, gof = c("CAF", "CFI", "RMSEA"))))
  J <- h$settings$n_fac_max
  tabs <- lapply(h$results, function(r) {
    list(gof = r$name, n_factors = r$n_factors, df = r$x, fit = r$y,
         st = lapply(r$st, function(v) if (is.na(v)) NULL else v), on_hull = r$on_hull)
  })
  list(method = method, max_factors = J, tables = tabs)
}

make <- function(name, R, n) {
  R <- (R + t(R)) / 2
  hulls <- lapply(c("ULS", "ML"), function(m) hull_case(R, n, m))
  J <- max(sapply(hulls, function(h) h$max_factors))
  efas <- list()
  for (m in c("ULS", "ML")) for (k in seq_len(J)) efas[[length(efas) + 1]] <- efa_case(R, n, k, m)
  list(name = name, n = n, correlation = rows_of(R), efa = efas, hull = hulls)
}

cases <- list()
for (cs in v24$cases) {
  R <- as_matrix(cs$lavaan_rho)
  if (ncol(R) >= 6) cases[[length(cases) + 1]] <- make(cs$name, R, cs$n)
}

sim_cor <- function(L, n, seed) {
  set.seed(seed)
  S <- L %*% t(L)
  diag(S) <- 1
  cor(MASS::mvrnorm(n, rep(0, nrow(L)), S))
}
L3 <- matrix(0, 12, 3)
L3[1:4, 1] <- c(.7, .6, .65, .55); L3[5:8, 2] <- c(.6, .7, .5, .6); L3[9:12, 3] <- c(.55, .65, .6, .7)
cases[[length(cases) + 1]] <- make("sim_3f", sim_cor(L3, 500, 2802), 500)
L2 <- matrix(0, 15, 2)
L2[1:8, 1] <- seq(.4, .7, length.out = 8); L2[9:15, 2] <- seq(.35, .6, length.out = 7)
cases[[length(cases) + 1]] <- make("sim_2f_weak", sim_cor(L2, 300, 2803), 300)

R3 <- as_matrix(cases[[length(cases) - 1]]$correlation)
psych_ref <- lapply(c("minres", "ml"), function(fm) {
  f <- suppressWarnings(psych::fa(R3, 3, n.obs = 500, fm = fm, rotate = "none"))
  list(fm = fm, loadings = rows_of(signed(f$loadings)), uniquenesses = unname(f$uniquenesses))
})

res <- list(
  meta = list(case = "v28_efa_hull", r_version = R.version.string,
              efatools_version = as.character(packageVersion("EFAtools")),
              psych_version = as.character(packageVersion("psych")), converged = TRUE),
  cases = cases, psych_sim_3f = psych_ref
)
write_json(res, file.path("tests", "reference", "v28_efa_hull.json"), auto_unbox = TRUE,
           digits = NA, null = "null", pretty = FALSE)
cat("wrote tests/reference/v28_efa_hull.json\n")
