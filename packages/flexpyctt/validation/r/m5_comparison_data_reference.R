# Comparison data (VALIDATION.md V29) from EFAtools.
# Run from the repository root (after m5_polychoric_reference.R and m5_efa_hull_reference.R):
#   Rscript packages/flexpyctt/validation/r/m5_comparison_data_reference.R
# Writes packages/flexpyctt/tests/reference/v29_comparison_data.json. CI does not run R; the JSON is committed.
#
# 1. Principal axis factoring as used by the generator: EFAtools:::.paf_iter from unit
#    communalities (criterion 0.001, absolute eigenvalues, max_iter 50).
# 2. The population generator EFAtools:::.simulate_cfm_empirical with one factor, whose random
#    draws (resampled marginals, shared and unique normal components) are reproduced here with
#    the same seed and exported, so flexpyctt can be fed the same draws. With one factor the
#    result does not depend on eigenvector signs (the generator flips the loadings when the first
#    is negative), so it is compared exactly.
# 3. EFAtools::CD on three data sets (suggested number and mean eigenvalue RMSE per k), and R's
#    wilcox.test(x, y, "less") on its RMSE columns.
suppressMessages({
  library(EFAtools)
  library(jsonlite)
})
ns <- asNamespace("EFAtools")
v24 <- fromJSON(file.path("packages", "flexpyctt", "tests", "reference", "v24_polychoric.json"), simplifyVector = FALSE)
v28 <- fromJSON(file.path("packages", "flexpyctt", "tests", "reference", "v28_efa_hull.json"), simplifyVector = FALSE)
as_matrix <- function(rows) do.call(rbind, lapply(rows, function(r) unlist(lapply(r, function(v) if (is.null(v)) NA else v))))
rows_of <- function(m) lapply(seq_len(nrow(m)), function(i) unname(as.list(m[i, ])))

# 1. PAF
R3 <- as_matrix(Filter(function(c) c$name == "sim_3f", v28$cases)[[1]]$correlation)
paf <- lapply(1:3, function(k) {
  L <- ns$.paf_iter(rep(1, ncol(R3)), 0.001, R3, k, TRUE, 1L, 50L)$L
  list(k = k, loadings = rows_of(as.matrix(L)))
})

# 2. Generator with one factor, draws exported.
mixed <- Filter(function(c) c$name == "polychoric_mixed", v24$cases)[[1]]
x <- as_matrix(mixed$data)
colnames(x) <- unlist(mixed$items)
R <- cor(x)
N <- 2000
seed <- 2901
set.seed(seed)
dists <- sapply(seq_len(ncol(x)), function(i) sort(sample(x[, i], size = N, replace = TRUE)))
shared <- matrix(rnorm(N * 1), N, 1)
unique <- matrix(rnorm(N * ncol(x)), N, ncol(x))
set.seed(seed)
pop <- ns$.simulate_cfm_empirical(R = R, x = x, n_factors = 1, N = N, max_iter = 50)
generator <- list(n_factors = 1, N = N, dists = rows_of(dists), shared = rows_of(shared),
                  unique = rows_of(unique), population = rows_of(pop))

# 3. CD
cd_case <- function(name, data, seed) {
  set.seed(seed)
  r <- suppressWarnings(EFAtools::CD(data, N_pop = 10000, N_samples = 500))
  rm <- r$results[[1]]$rmse_eigenvalues
  tested <- length(r$results[[1]]$y)
  wil <- if (tested >= 2) sapply(2:tested, function(k)
    stats::wilcox.test(rm[, k], rm[, k - 1], "less")$p.value) else numeric(0)
  list(name = name, data = rows_of(as.matrix(data)), n_factors = r$n_factors[[1]],
       mean_rmse = unname(r$results[[1]]$y), tested = tested,
       rmse = lapply(seq_len(tested), function(k) rm[, k]), wilcox_p = as.list(wil))
}
set.seed(2902)
L3 <- matrix(0, 12, 3)
L3[1:4, 1] <- .7; L3[5:8, 2] <- .6; L3[9:12, 3] <- .65
F <- matrix(rnorm(600 * 3), 600, 3)
sim3 <- F %*% t(L3) + matrix(rnorm(600 * 12), 600, 12) * 0.7
colnames(sim3) <- paste0("v", 1:12)
tet <- Filter(function(c) c$name == "tetrachoric_sim", v24$cases)[[1]]
xt <- as_matrix(tet$data)
colnames(xt) <- unlist(tet$items)
cds <- list(cd_case("sim_3f", sim3, 2903), cd_case("polychoric_mixed", x, 2904),
            cd_case("tetrachoric_sim", xt, 2905))

res <- list(
  meta = list(case = "v29_comparison_data", r_version = R.version.string,
              efatools_version = as.character(packageVersion("EFAtools")), converged = TRUE),
  paf = list(case = "sim_3f", fits = paf), generator = generator, cd = cds
)
write_json(res, file.path("packages", "flexpyctt", "tests", "reference", "v29_comparison_data.json"), auto_unbox = TRUE,
           digits = NA, null = "null", pretty = FALSE)
cat("wrote packages/flexpyctt/tests/reference/v29_comparison_data.json\n")
