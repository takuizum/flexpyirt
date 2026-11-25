# DETECT (VALIDATION.md V27) from sirt's conf.detect / expl.detect with use_sum_score = TRUE.
# Run from the repository root:  Rscript packages/flexpyctt/validation/r/m5_detect_reference.R
# Writes packages/flexpyctt/tests/reference/v27_detect.json. CI does not run R; the JSON is committed.
#
# use_sum_score = TRUE with bias_corr = TRUE is Zhang & Stout's (1999) estimator: the conditional
# covariance of each pair is the average of the covariances given the total score and given the
# rest score. expl.detect is called with an explicit estsample; its returned detect.unweighted
# holds validation-half indices only, so the estimation-half indices are recomputed with
# conf.detect on the estimation rows (same partitions).
suppressMessages({
  library(sirt)
  library(mirt)
  library(jsonlite)
})
grDevices::pdf(NULL)  # expl.detect() draws a dendrogram; do not write Rplots.pdf

quiet <- function(expr) {
  out <- NULL
  utils::capture.output(out <- suppressWarnings(expr))
  out
}

conf <- function(data, clusters, smooth = FALSE) {
  r <- quiet(conf.detect(data, score = rowSums(data), itemcluster = clusters,
                         use_sum_score = TRUE, bias_corr = TRUE, smooth = smooth,
                         progress = FALSE))
  list(indices = list(unweighted = unname(r$detect$unweighted), weighted = unname(r$detect$weighted)),
       ccov = lapply(seq_len(nrow(r$ccovtable$ccov.table)), function(k) {
         row <- r$ccovtable$ccov.table[k, ]
         list(i = row$item1ID - 1L, j = row$item2ID - 1L, ccov = row$ccov)
       }))
}

rows_of <- function(m) lapply(seq_len(nrow(m)), function(i) unname(as.list(m[i, ])))

simulate_2pl <- function(n, a, d, cor, seed, clusters) {
  set.seed(seed)
  k <- max(clusters)
  sigma <- matrix(cor, k, k)
  diag(sigma) <- 1
  theta <- MASS::mvrnorm(n, rep(0, k), sigma)
  eta <- theta[, clusters] * matrix(a, n, length(a), byrow = TRUE) +
    matrix(d, n, length(a), byrow = TRUE)
  y <- (matrix(runif(n * length(a)), n) < plogis(eta)) * 1L
  colnames(y) <- paste0("x", seq_along(a))
  y
}

cases <- list()
cl2 <- rep(1:2, each = 6)
y2 <- simulate_2pl(1500, rep(c(1.4, 1.0, 1.2), 4), seq(-1, 1, length.out = 12), 0.4, 2701, cl2)
wrong <- rep(1:2, 6)
cases[[1]] <- list(name = "sim_2d", items = colnames(y2), data = rows_of(y2),
                   partitions = list(true = cl2, wrong = wrong),
                   results = list(true = conf(y2, cl2), wrong = conf(y2, wrong)))

cl1 <- rep(1, 10)
y1 <- simulate_2pl(1000, rep(c(1.5, 1.0), 5), seq(-1.2, 1.2, length.out = 10), 0, 2702, cl1)
split <- rep(1:2, each = 5)
cases[[2]] <- list(name = "sim_1d", items = colnames(y1), data = rows_of(y1),
                   partitions = list(halves = split), results = list(halves = conf(y1, split)))

sci <- as.matrix(mirt::Science) - 1L
sp <- c(1, 2, 1, 2)
cases[[3]] <- list(name = "science", items = colnames(sci), data = rows_of(sci),
                   partitions = list(pairs = sp), results = list(pairs = conf(sci, sp)))

# Exploratory: three clusters of 5 items.
cl3 <- rep(1:3, each = 5)
y3 <- simulate_2pl(2000, rep(c(1.5, 1.2, 1.0, 1.3, 1.1), 3), seq(-1, 1, length.out = 15), 0.3,
                   2703, cl3)
set.seed(2704)
est <- sort(sample(seq_len(nrow(y3)), nrow(y3) %/% 2))
ex <- quiet(expl.detect(y3, score = rowSums(y3), nclusters = 4, use_sum_score = TRUE,
                        estsample = est))
val <- setdiff(seq_len(nrow(y3)), est)
explore <- lapply(2:4, function(k) {
  labels <- ex$itemcluster[, k]
  list(k = k, clusters = labels,
       estimation = conf(y3[est, ], labels)$indices$unweighted,
       validation = unname(ex$detect.unweighted[k - 1L, ]))
})
expl_case <- list(name = "sim_3d", items = colnames(y3), data = rows_of(y3),
                  estimation_rows = est - 1L, partitions = explore,
                  merge_heights = ex$clusterfit$height, merge = rows_of(ex$clusterfit$merge))

res <- list(
  meta = list(case = "v27_detect", r_version = R.version.string,
              sirt_version = as.character(packageVersion("sirt")), converged = TRUE),
  index_names = c("DETECT", "ASSI", "RATIO", "MADCOV100", "MCOV100"),
  cases = cases, exploratory = expl_case
)
write_json(res, file.path("packages", "flexpyctt", "tests", "reference", "v27_detect.json"), auto_unbox = TRUE,
           digits = NA, null = "null", pretty = FALSE)
cat("wrote packages/flexpyctt/tests/reference/v27_detect.json\n")
