# V55 reference values (VALIDATION.md V55, DESIGN.md §26, #79): many groups with mirt
# multipleGroup(). Run from the repository root:  Rscript validation/r/v55_many_groups_reference.R
# Writes tests/reference/v55_many_groups.json. CI does not run R; the JSON file is committed.
#
# 2PL, G = 10 groups of unequal size (200..470), 15 items, anchors q01-q06 invariant,
# q07-q15 free per group, focal means and variances free, 5% MCAR missing.
# As in m2_reference.R: optimizer = "NR", accelerate = "none", TOL = 1e-10, 81 points on [-8, 8].
source(file.path("validation", "r", "setup.R"))  # mirt from GitHub; mirt_version / mirt_sha
suppressMessages({
  library(mirt)
  library(jsonlite)
})

out_dir <- file.path("tests", "reference")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
common <- list(TOL = 1e-10, accelerate = "none", optimizer = "NR", verbose = FALSE)
quad <- list(quadpts = 81, theta_lim = c(-8, 8))
tech <- list(NCYCLES = 50000, theta_lim = quad$theta_lim)

clean_item <- function(x) {
  nm <- colnames(x)
  keep <- nm[nm == "a1" | nm == "d"]
  par <- x["par", keep]
  se <- if ("SE" %in% rownames(x)) x["SE", keep] else rep(NA, length(keep))
  names(par) <- sub("^a1$", "a", keep)
  names(se) <- paste0(names(par), "_se")
  c(as.list(par), as.list(se))
}
clean_group <- function(cf) {
  items <- setdiff(names(cf), "GroupPars")
  params <- lapply(items, function(i) clean_item(cf[[i]]))
  names(params) <- items
  gp <- cf$GroupPars
  se <- if ("SE" %in% rownames(gp)) gp["SE", ] else c(NA, NA)
  list(params = params,
       latent = list(mean = gp["par", "MEAN_1"], var = gp["par", "COV_11"],
                     mean_se = unname(se[1]), var_se = unname(se[2])))
}

set.seed(20261126)
G <- 10
J <- 15
groups <- sprintf("s%02d", seq_len(G))
sizes <- 200 + 30 * (seq_len(G) - 1)
mu <- c(0, round(rnorm(G - 1, 0, 0.5), 3))
sigma2 <- c(1, round(runif(G - 1, 0.6, 1.6), 3))
a <- rlnorm(J, 0, 0.3)
d <- rnorm(J)
anchors <- sprintf("q%02d", 1:6)
blocks <- vector("list", G)
for (g in seq_len(G)) {
  a_g <- a
  d_g <- d
  if (g > 1) {
    a_g[7:J] <- a[7:J] * exp(rnorm(J - 6, 0, 0.1))
    d_g[7:J] <- d[7:J] + rnorm(J - 6, 0, 0.3)
  }
  blocks[[g]] <- simdata(matrix(a_g), matrix(d_g), sizes[g], itemtype = "2PL",
                         mu = mu[g], sigma = matrix(sigma2[g]))
}
dat <- as.data.frame(do.call(rbind, blocks))
colnames(dat) <- sprintf("q%02d", 1:J)
dat[matrix(runif(nrow(dat) * J) < 0.05, nrow(dat), J)] <- NA
grp <- rep(groups, times = sizes)

fit_mg <- function(se) {
  do.call(multipleGroup, c(list(dat, 1, group = factor(grp, levels = groups), itemtype = "2PL",
                                invariance = c(anchors, "free_means", "free_var"),
                                SE = TRUE, SE.type = se, quadpts = quad$quadpts,
                                technical = tech), common))
}
elapsed <- system.time(m <- fit_mg("Oakes"))[["elapsed"]]
cf <- coef(m, printSE = TRUE)
names(cf) <- groups

key <- paste(grp, apply(dat, 1, paste, collapse = ","))
first <- !duplicated(key)
patterns <- as.matrix(dat[first, , drop = FALSE])
pattern_groups <- grp[first]
eap <- matrix(NA_real_, sum(first), 2)
for (g in seq_len(G)) {
  rows <- which(pattern_groups == groups[g])
  sc <- fscores(extract.group(m, g), method = "EAP", response.pattern = patterns[rows, , drop = FALSE],
                quadpts = quad$quadpts, theta_lim = quad$theta_lim)
  eap[rows, ] <- sc[, c("F1", "SE_F1")]
}

res <- list(
  meta = list(case = "v55_many_groups",
              description = paste("V55: 2PL, 10 groups (N = 200..470), anchors q01-q06,",
                                  "q07-q15 free per group, focal means/variances free, 5% missing"),
              r_version = R.version.string, mirt_version = mirt_version_string(), mirt_sha = mirt_commit_sha(),
              TOL = 1e-10, quadpts = quad$quadpts, theta_lim = quad$theta_lim,
              grid_check = range(m@Model$Theta),
              accelerate = "none", optimizer = "NR", SE_type = "Oakes",
              converged = extract.mirt(m, "converged"),
              iterations = extract.mirt(m, "iterations"),
              elapsed_seconds = elapsed),
  items = colnames(dat),
  groups = I(groups),
  anchors = I(anchors),
  patterns = unname(patterns),
  pattern_groups = pattern_groups,
  counts = as.integer(table(key)[key[first]]),
  loglik = extract.mirt(m, "logLik"),
  estimates = lapply(cf, clean_group),
  eap = list(theta = eap[, 1], se = eap[, 2])
)
stopifnot(isTRUE(res$meta$converged), all(res$meta$grid_check == quad$theta_lim))
path <- file.path(out_dir, "v55_many_groups.json")
write_json(res, path, auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null")
cat("wrote", path, " loglik =", res$loglik, " iter =", res$meta$iterations,
    " elapsed =", elapsed, "\n")
