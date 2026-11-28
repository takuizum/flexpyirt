# Smolyak sparse-grid rules from R SparseGrid (Heiss & Winschel, 2008), for the design study of
# DESIGN.md §12 (open question 6). Does not use mirt.
#
#   Rscript validation/r/sparse_grid_reference.R            # tests/reference/sparse_grid/gqn.json
#   Rscript validation/r/sparse_grid_reference.R <kpn_dir>  # also KPN rules as CSV for the study
#
# gqn.json holds the "GQN" (Gauss-Hermite, non-nested) rules that benchmarks/sparse_grid.py must
# reproduce (tests/test_sparse_grid.py). The KPN (nested Genz-Keister) CSVs are read by
# benchmarks/sparse_grid_study.py --kpn-dir and are not committed.
suppressMessages(library(SparseGrid))
library(jsonlite)

args <- commandArgs(trailingOnly = TRUE)
rule <- function(type, d, k) {
  g <- createSparseGrid(type, d, k)
  list(n_factors = d, level = k, nodes = unname(g$nodes), weights = unname(g$weights))
}
counts <- expand.grid(n_factors = 2:10, level = 3:5)
counts$n_points <- mapply(function(d, k) length(createSparseGrid("GQN", d, k)$weights),
                          counts$n_factors, counts$level)
out <- list(
  meta = list(
    package = "SparseGrid",
    package_version = as.character(packageVersion("SparseGrid")),
    r_version = paste(R.version$major, R.version$minor, sep = "."),
    rule = "GQN"
  ),
  rules = list(rule("GQN", 3, 4), rule("GQN", 5, 3)),
  counts = counts
)
dir.create(file.path("tests", "reference", "sparse_grid"), showWarnings = FALSE)
write_json(out, file.path("tests", "reference", "sparse_grid", "gqn.json"),
           digits = NA, auto_unbox = TRUE, pretty = FALSE)

if (length(args) >= 1) {
  dir.create(args[1], showWarnings = FALSE, recursive = TRUE)
  for (d in c(3, 5)) for (k in 3:8) {
    if (d == 5 && k > 7) next
    g <- createSparseGrid("KPN", d, k)
    write.csv(data.frame(g$nodes, w = g$weights), file.path(args[1], sprintf("kpn_%d_%d.csv", d, k)),
              row.names = FALSE)
  }
}
