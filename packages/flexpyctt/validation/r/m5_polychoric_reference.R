# Polychoric / tetrachoric correlations (VALIDATION.md V24) from lavaan's lavCor and psych.
# Run from the repository root:  Rscript packages/flexpyctt/validation/r/m5_polychoric_reference.R
# Writes packages/flexpyctt/tests/reference/v24_polychoric.json. CI does not run R; the JSON is committed.
#
# lavaan (two-step ML, nlminb with analytic gradient and Hessian) is the tight reference. psych's
# polychoric maximizes the same likelihood with optimize() at its default tolerance
# (~1.2e-4 in rho) and mnormt::sadmvn, so it is a loose cross-check only. psych is called with
# smooth = FALSE (smoothing is checked separately on cor.smooth) and its default global = TRUE,
# under which `correct` is not applied.
#
# lavaan's default zero.add = c(0.5, 0) with zero.keep.margins = TRUE corrects 2 x 2 tables with
# exactly one empty cell; flexpyctt's zero_cell_correction = 0.5 is the same rule.
suppressMessages({
  library(lavaan)
  library(psych)
  library(mirt)
  library(jsonlite)
})

lavaan_poly <- function(df, missing = "listwise") {
  # lavCor has no model to optimize; a per-pair estimation failure is a warning, made fatal here.
  fit <- withCallingHandlers(
    lavCor(df, ordered = names(df), missing = missing, output = "fit"),
    warning = function(w) if (grepl("converge", conditionMessage(w))) stop(w))
  ss <- lavInspect(fit, "sampstat")
  th <- ss$th
  list(rho = unname(unclass(ss$cov)), th = unname(as.numeric(th)), th_names = names(th))
}

psych_poly <- function(df) {
  r <- suppressWarnings(polychoric(df, smooth = FALSE, correct = 0, progress = FALSE))
  unname(r$rho)
}

as_rows <- function(df) {
  m <- as.matrix(df)
  lapply(seq_len(nrow(m)), function(i) {
    v <- as.list(unname(m[i, ]))
    lapply(v, function(x) if (is.na(x)) NULL else x)
  })
}

make_case <- function(name, df, missing = "listwise", note = "") {
  lv <- lavaan_poly(df, missing)
  list(name = name, note = note, items = names(df), n = nrow(df), data = as_rows(df),
       missing = missing, lavaan_rho = lv$rho, lavaan_th = lv$th, lavaan_th_names = lv$th_names,
       psych_rho = psych_poly(df))
}

# Ordinal data from a correlated latent normal with given thresholds.
simulate_ordinal <- function(n, sigma, cuts, seed) {
  set.seed(seed)
  z <- MASS::mvrnorm(n, rep(0, ncol(sigma)), sigma)
  out <- sapply(seq_len(ncol(z)), function(j) findInterval(z[, j], cuts[[j]]))
  colnames(out) <- paste0("y", seq_len(ncol(z)))
  as.data.frame(out)
}

one_factor <- function(lambda) {
  s <- tcrossprod(lambda)
  diag(s) <- 1
  s
}

cases <- list()

# 1. Tetrachoric, 8 binary items of varied difficulty.
lam <- c(0.8, 0.7, 0.6, 0.75, 0.5, 0.65, 0.55, 0.7)
cuts_bin <- as.list(c(-1.2, -0.6, -0.2, 0, 0.3, 0.7, 1.0, 1.4))
cases[[length(cases) + 1]] <- make_case(
  "tetrachoric_sim", simulate_ordinal(600, one_factor(lam), cuts_bin, 2401),
  note = "one-factor latent normal, 8 binary items")

# 2. Polychoric with mixed category counts (2, 3, 4, 5, 5, 4) and a two-factor structure.
s2 <- matrix(0.15, 6, 6)
s2[1:3, 1:3] <- 0.55
s2[4:6, 4:6] <- 0.5
diag(s2) <- 1
cuts_poly <- list(c(0.2), c(-0.5, 0.6), c(-1, 0, 1), c(-1.5, -0.5, 0.4, 1.2),
                  c(-1.2, -0.3, 0.5, 1.5), c(-0.8, 0.1, 0.9))
poly_df <- simulate_ordinal(900, s2, cuts_poly, 2402)
cases[[length(cases) + 1]] <- make_case("polychoric_mixed", poly_df,
                                        note = "two-factor latent normal, 2-5 categories")

# 3. Same data with 10% MCAR responses, pairwise available cases.
set.seed(2403)
miss_df <- poly_df
miss_df[matrix(runif(length(as.matrix(miss_df))) < 0.10, nrow(miss_df))] <- NA
cases[[length(cases) + 1]] <- make_case("polychoric_missing", miss_df, missing = "pairwise",
                                        note = "case 2 with 10% MCAR, pairwise")

# 4. Binary items where some 2 x 2 tables have exactly one empty cell (lavaan's zero.add).
zc <- simulate_ordinal(150, one_factor(c(0.9, 0.85, 0.8, 0.6)),
                       list(-2.2, 1.9, 0, -0.3), 2404)
cases[[length(cases) + 1]] <- make_case("tetrachoric_zero_cell", zc,
                                        note = "extreme items, 2 x 2 tables with one empty cell")

# 5-6. mirt's bundled data.
lsat <- expand.table(LSAT7)
cases[[length(cases) + 1]] <- make_case("lsat7", as.data.frame(lsat), note = "mirt LSAT7")
cases[[length(cases) + 1]] <- make_case("science", as.data.frame(Science), note = "mirt Science")

# Smoothing: psych::cor.smooth on a symmetric unit-diagonal matrix that is not positive definite.
bad <- matrix(c(1.0, 0.9, 0.9, 0.1,
                0.9, 1.0, 0.2, 0.8,
                0.9, 0.2, 1.0, 0.7,
                0.1, 0.8, 0.7, 1.0), 4, 4)
smooth_ref <- suppressWarnings(cor.smooth(bad))

res <- list(
  meta = list(case = "v24_polychoric", r_version = R.version.string,
              lavaan_version = as.character(packageVersion("lavaan")),
              psych_version = as.character(packageVersion("psych")),
              mirt_version = as.character(packageVersion("mirt")),
              converged = TRUE),
  cases = cases,
  smooth = list(input = unname(bad), output = unname(smooth_ref),
                min_eigenvalue = min(eigen(bad, only.values = TRUE)$values))
)
write_json(res, file.path("packages", "flexpyctt", "tests", "reference", "v24_polychoric.json"), auto_unbox = TRUE,
           digits = NA, null = "null", pretty = FALSE)
cat("wrote packages/flexpyctt/tests/reference/v24_polychoric.json\n")
