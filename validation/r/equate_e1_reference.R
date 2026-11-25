# flexpyequate E1 reference values (packages/flexpyequate/docs/DESIGN.md §7, cases Q1 and Q2):
# mean, linear and equipercentile equating for the equivalent-groups and single-group designs.
# Run from the repository root:  Rscript validation/r/equate_e1_reference.R
# Writes packages/flexpyequate/tests/reference/e1_equate.json. CI does not run R; the JSON is
# committed.
#
# Two references:
# - R `equate` (Albano, 2016). It divides variances by N - 1 (flexpyequate: ddof = 1) and rounds
#   percentile ranks and cumulative proportions to 10 decimals inside equip(), so equipercentile
#   values agree to about 1e-7 only. Where the percentile rank is exactly 0 or 1 it returns the
#   end of the Y range instead of the mean of the upper and lower percentiles; the two differ
#   when Y has no observations at that end.
# - `kb_equipercentile()` below: an independent implementation of Kolen & Brennan (2014),
#   eq. 2.14-2.18, without rounding. This is the tight reference.
suppressMessages({
  library(equate)
  library(jsonlite)
})

# Percentile rank (as a proportion) at the possible scores: F(x - 1) + f(x) / 2.
kb_rank <- function(freq) {
  p <- freq / sum(freq)
  cumsum(p) - p / 2
}

# Kolen & Brennan eq. 2.15-2.18: mean of the upper and lower percentile at proportion `p`.
kb_percentile <- function(p, scores, freq) {
  inc <- scores[2] - scores[1]
  cum <- cumsum(freq) / sum(freq)
  k <- length(scores)
  one <- function(pp) {
    if (pp >= 1) {
      upper <- scores[k] + inc / 2
    } else {
      iu <- which(cum > pp)[1]
      below <- if (iu > 1) cum[iu - 1] else 0
      upper <- scores[iu] - inc / 2 + inc * (pp - below) / (cum[iu] - below)
    }
    if (pp <= 0) {
      lower <- scores[1] - inc / 2
    } else {
      il <- sum(cum < pp)  # 0 stands for the score below the lowest one
      at <- if (il > 0) cum[il] else 0
      lower <- scores[1] + inc * (il - 1) + inc / 2 + inc * (pp - at) / (cum[il + 1] - at)
    }
    (upper + lower) / 2
  }
  vapply(p, one, numeric(1))
}

kb_equipercentile <- function(xs, xf, ys, yf) kb_percentile(kb_rank(xf), ys, yf)

moments <- function(tab) {
  list(mean = mean(tab), sd_ddof1 = equate:::sd.freqtab(tab))
}

make_case <- function(name, design, xs, xf, ys, yf, joint = NULL, note = "") {
  xt <- as.freqtab(cbind(xs, xf))
  yt <- as.freqtab(cbind(ys, yf))
  # equate's "mean" type has slope (range of Y) / (range of X), which is 1 only for forms with
  # equal score ranges; sx = sy = 1 asks for the mean equating of Kolen & Brennan (slope 1).
  mean_eq <- equate(xt, yt, type = "mean", sx = 1, sy = 1)
  stopifnot(isTRUE(all.equal(unname(mean_eq$coefficients["slope"]), 1)))
  linear_eq <- equate(xt, yt, type = "linear")
  equip_eq <- equate(xt, yt, type = "equipercentile")
  out <- list(
    name = name, design = design, note = note,
    x_scores = xs, x_frequencies = xf, y_scores = ys, y_frequencies = yf,
    x_moments = moments(xt), y_moments = moments(yt),
    percentile_rank_x = as.numeric(px(xt)),
    percentile_rank_y = as.numeric(px(yt)),
    mean = list(equated = mean_eq$concordance$yx,
                intercept = unname(mean_eq$coefficients["intercept"]),
                slope = unname(mean_eq$coefficients["slope"])),
    linear = list(equated = linear_eq$concordance$yx,
                  intercept = unname(linear_eq$coefficients["intercept"]),
                  slope = unname(linear_eq$coefficients["slope"])),
    equipercentile = list(equated = equip_eq$concordance$yx),
    kb_equipercentile = list(equated = kb_equipercentile(xs, xf, ys, yf))
  )
  if (!is.null(joint)) {
    # Single group through equate's own bivariate path (y missing -> second margin).
    long <- cbind(expand.grid(x = xs, y = ys), n = as.vector(joint))
    xyt <- as.freqtab(long, scales = list(xs, ys), design = "sg")
    sg_linear <- equate(xyt, type = "linear")
    sg_equip <- equate(xyt, type = "equipercentile")
    stopifnot(isTRUE(all.equal(sg_linear$concordance$yx, linear_eq$concordance$yx)),
              isTRUE(all.equal(sg_equip$concordance$yx, equip_eq$concordance$yx)))
    out$joint_frequencies <- joint
    out$covariance_ddof1 <- equate:::cov.freqtab(xyt)[1, 2]
    out$correlation <- cor(rep(long$x, long$n), rep(long$y, long$n))
  }
  out
}

tabulate_scores <- function(v, scores) as.numeric(table(factor(v, levels = scores)))

cases <- list()

# Q1/Q2-a: ACT mathematics, two 40-item forms (Kolen & Brennan, 2014, Table 2.5; equate::ACTmath).
cases[[1]] <- make_case("act_math", "equivalent_groups", ACTmath$scale, ACTmath$xcount,
                        ACTmath$scale, ACTmath$ycount,
                        note = "equate::ACTmath, forms X and Y")

# Q1/Q2-b: forms of different length and unequal sample sizes.
set.seed(20261125)
xs <- 0:30
ys <- 0:50
xv <- rbinom(1800, 30, rbeta(1800, 5, 3.5))
yv <- rbinom(2500, 50, rbeta(2500, 4.5, 3.5))
cases[[2]] <- make_case("unequal_lengths", "equivalent_groups", xs, tabulate_scores(xv, xs),
                        ys, tabulate_scores(yv, ys),
                        note = "X 0..30 (N = 1800), Y 0..50 (N = 2500), beta-binomial")

# Q2-c: small samples, zero frequencies inside both distributions (the upper and lower
# percentiles of eq. 2.15-2.18 differ) but none at the ends of Y.
xs <- 0:20
ys <- 0:20
xf <- c(0, 0, 1, 3, 0, 5, 8, 0, 0, 11, 14, 9, 0, 7, 6, 0, 3, 2, 0, 1, 0)
yf <- c(1, 0, 0, 2, 4, 0, 7, 9, 0, 0, 0, 12, 10, 8, 0, 5, 0, 3, 0, 0, 2)
cases[[3]] <- make_case("zero_frequencies", "equivalent_groups", xs, xf, ys, yf,
                        note = "runs of zero frequencies inside X and Y; zeros at both ends of X")

# Q1/Q2-d: half-point scores (increment 0.5) on X, increment 2 on Y.
set.seed(4102)
xs <- seq(10, 30, by = 0.5)
ys <- seq(100, 200, by = 2)
xv <- 10 + rbinom(900, 40, 0.55) / 2
yv <- 100 + 2 * rbinom(1100, 50, 0.6)
cases[[4]] <- make_case("score_increments", "equivalent_groups", xs, tabulate_scores(xv, xs),
                        ys, tabulate_scores(yv, ys),
                        note = "X 10..30 by 0.5, Y 100..200 by 2")

# Q1/Q2-e: single group, correlated number-correct scores on two forms.
set.seed(777)
n <- 1500
theta <- rnorm(n)
xs <- 0:25
ys <- 0:30
xv <- rbinom(n, 25, plogis(0.9 * theta + 0.2))
yv <- rbinom(n, 30, plogis(1.1 * theta - 0.1))
joint <- unclass(table(factor(xv, levels = xs), factor(yv, levels = ys)))
dimnames(joint) <- NULL
cases[[5]] <- make_case("single_group", "single_group", xs, rowSums(joint), ys, colSums(joint),
                        joint = joint, note = "same examinees take X (0..25) and Y (0..30)")

out <- list(
  meta = list(
    generator = "validation/r/equate_e1_reference.R",
    r_version = R.version.string,
    equate_version = as.character(packageVersion("equate")),
    note = paste("sd_ddof1 and the linear results use N - 1 variances (equate).",
                 "kb_equipercentile is an independent, unrounded implementation of",
                 "Kolen & Brennan (2014) eq. 2.14-2.18.")
  ),
  cases = cases
)
path <- "packages/flexpyequate/tests/reference/e1_equate.json"
writeLines(toJSON(out, auto_unbox = TRUE, digits = NA, pretty = TRUE), path)
cat("wrote", path, "\n")
for (case in cases) {
  cat(sprintf("%-18s max |equate - K&B| = %.3e\n", case$name,
              max(abs(case$equipercentile$equated - case$kb_equipercentile$equated))))
}
