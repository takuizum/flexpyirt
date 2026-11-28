# mirt EM and QMCEM on the 3-D data of the Smolyak design study (DESIGN.md §12, item 6).
# Exploratory comparison, not a reference for tests:
#
#   uv run python -m benchmarks.sparse_grid_study --write-3d-data /tmp/d3.csv
#   Rscript benchmarks/sparse_grid_mirt.R /tmp/d3.csv
#
# Prints the estimates (slopes, intercepts, correlations) and logLik of each run, to be compared with
# the product Gauss-Hermite reference fit of benchmarks/sparse_grid_study.py.
suppressMessages(library(mirt))
args <- commandArgs(trailingOnly = TRUE)
y <- as.matrix(read.csv(args[1], header = FALSE))
spec <- mirt.model("F1 = 1-5\nF2 = 6-10\nF3 = 11-15\nCOV = F1*F2*F3")
show <- function(fit, label) {
  cf <- coef(fit, simplify = TRUE)
  r <- cf$cov[lower.tri(cf$cov)]
  cat(sprintf("%s (mirt %s): logLik = %.4f, time = %.1f s, points = %d\n", label,
              as.character(packageVersion("mirt")), extract.mirt(fit, "logLik"),
              sum(extract.mirt(fit, "time")), nrow(fit@Internals$Theta)))
  cat("a:", format(rowSums(cf$items[, 1:3]), digits = 6), "\n")
  cat("d:", format(cf$items[, "d"], digits = 6), "\n")
  cat("rho (21, 31, 32):", format(r, digits = 6), "\n")
}
show(mirt(y, spec, verbose = FALSE, TOL = 1e-6), "EM, 15^3 equispaced")
show(mirt(y, spec, method = "QMCEM", verbose = FALSE), "QMCEM, default points")
