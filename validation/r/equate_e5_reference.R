# flexpyequate E5 reference values (packages/flexpyequate/docs/DESIGN.md §7, case Q7):
# equipercentile equating for non-equivalent groups with an anchor test (frequency estimation
# and chained), with and without log-linear presmoothing of the joint distributions.
# Run from the repository root:  Rscript validation/r/equate_e5_reference.R
# Writes packages/flexpyequate/tests/reference/e5_neat_equipercentile.json. CI does not run R.
#
# Two references:
# - R `equate` (Albano, 2016). Its synthetic distributions mix the populations as counts, so the
#   weight of population 1 is ws * N1 / (ws * N1 + (1 - ws) * N2) (`effective_weight`, passed to
#   flexpyequate). Its equipercentile routine rounds ranks to 10 decimals and returns the end of
#   the Y range where a rank is exactly 0 or 1 (see equate_e1_reference.R); the ranks are
#   recorded so that those scores can be left out.
# - `kb_*()`: Kolen & Brennan (2014, ch. 2 and 5) written out without rounding. Tight reference.
suppressMessages({
  library(equate)
  library(jsonlite)
})

# ----------------------------------------------------------------- independent implementation
# Percentile rank (proportion) of arbitrary scores `v`.
kb_rank_at <- function(v, scores, freq) {
  inc <- scores[2] - scores[1]
  cum <- cumsum(freq) / sum(freq)
  k <- length(scores)
  vapply(v, function(z) {
    if (z < scores[1] - inc / 2) return(0)
    if (z >= scores[k] + inc / 2) return(1)
    i <- min(max(floor((z - (scores[1] - inc / 2)) / inc) + 1, 1), k)
    below <- if (i > 1) cum[i - 1] else 0
    below + (z - (scores[i] - inc / 2)) / inc * (cum[i] - below)
  }, numeric(1))
}
# Mean of the upper and lower percentile at proportion `p`.
kb_percentile <- function(p, scores, freq) {
  inc <- scores[2] - scores[1]
  cum <- cumsum(freq) / sum(freq)
  k <- length(scores)
  vapply(p, function(pp) {
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
      il <- sum(cum < pp)
      at <- if (il > 0) cum[il] else 0
      lower <- scores[1] + inc * (il - 1) + inc / 2 + inc * (pp - at) / (cum[il + 1] - at)
    }
    (upper + lower) / 2
  }, numeric(1))
}
kb_equipercentile <- function(v, xs, xf, ys, yf) kb_percentile(kb_rank_at(v, xs, xf), ys, yf)

kb_synthetic <- function(joint1, joint2, w1) {
  p1 <- joint1 / sum(joint1)
  p2 <- joint2 / sum(joint2)
  h1 <- colSums(p1)
  h2 <- colSums(p2)
  stopifnot(all(h1 > 0), all(h2 > 0))
  list(x = w1 * rowSums(p1) + (1 - w1) * as.vector(sweep(p1, 2, h1, "/") %*% h2),
       y = w1 * as.vector(sweep(p2, 2, h2, "/") %*% h1) + (1 - w1) * rowSums(p2))
}
kb_frequency_estimation <- function(joint1, joint2, xs, ys, w1) {
  synth <- kb_synthetic(joint1, joint2, w1)
  list(equated = kb_equipercentile(xs, xs, synth$x, ys, synth$y),
       rank_x = kb_rank_at(xs, xs, synth$x), synthetic_x = synth$x, synthetic_y = synth$y)
}
kb_chained <- function(joint1, joint2, xs, ys, vs) {
  rank_x <- kb_rank_at(xs, xs, rowSums(joint1))
  on_anchor <- kb_percentile(rank_x, vs, colSums(joint1))
  rank_anchor <- kb_rank_at(on_anchor, vs, colSums(joint2))
  list(equated = kb_percentile(rank_anchor, ys, rowSums(joint2)), rank_x = rank_x,
       on_anchor = on_anchor, rank_anchor = rank_anchor)
}

# ----------------------------------------------------------------- cases
to_matrix <- function(data, total_scores, anchor_scores) {
  m <- unclass(table(factor(data[, 1], levels = total_scores),
                     factor(data[, 2], levels = anchor_scores)))
  dimnames(m) <- NULL
  m
}

make_case <- function(name, xdata, ydata, xs, ys, vs, note = "") {
  xt <- freqtab(xdata, scales = list(xs, vs))
  yt <- freqtab(ydata, scales = list(ys, vs))
  joint1 <- to_matrix(xdata, xs, vs)
  joint2 <- to_matrix(ydata, ys, vs)
  n1 <- sum(joint1)
  n2 <- sum(joint2)
  weights <- lapply(list(-1, 1, 0.5, 0.2), function(ws) {
    w1 <- if (ws == -1) n1 / (n1 + n2) else ws
    effective <- w1 * n1 / (w1 * n1 + (1 - w1) * n2)
    eq <- equate(xt, yt, type = "equipercentile", method = "frequency estimation", ws = ws)
    list(weight = w1, effective_weight = effective,
         equate = list(equated = eq$concordance$yx),
         kb = kb_frequency_estimation(joint1, joint2, xs, ys, w1),
         kb_at_effective_weight = kb_frequency_estimation(joint1, joint2, xs, ys, effective))
  })
  chained <- equate(xt, yt, type = "equipercentile", method = "chained")
  list(name = name, note = note, x_scores = xs, y_scores = ys, anchor_scores = vs,
       x_joint_frequencies = joint1, y_joint_frequencies = joint2,
       frequency_estimation = weights,
       chained = list(equate = list(equated = chained$concordance$yx),
                      kb = kb_chained(joint1, joint2, xs, ys, vs)))
}

cases <- list()
cases[[1]] <- make_case("kb_neat", KBneat$x, KBneat$y, 0:36, 0:36, 0:12,
                        note = "equate::KBneat (Kolen & Brennan, 2014), internal anchor")
set.seed(20261127)
simulate <- function(n, n_items, n_anchor, shift) {
  theta <- rnorm(n, shift, 1)
  cbind(total = rbinom(n, n_items, plogis(0.9 * theta + 0.2)),
        anchor = rbinom(n, n_anchor, plogis(0.8 * theta)))
}
cases[[2]] <- make_case("external_anchor", simulate(1200, 30, 12, 0.25), simulate(2000, 35, 12, -0.1),
                        0:30, 0:35, 0:12,
                        note = "external anchor 0..12; X 0..30 (N = 1200), Y 0..35 (N = 2000)")

# Presmoothed KBneat: bivariate log-linear models with degrees (6, 6) and the cross term x * v
# (equate: degrees = list(c(6, 6), c(1, 1)); glm default tolerance), then both methods.
xt <- freqtab(KBneat$x, scales = list(0:36, 0:12))
yt <- freqtab(KBneat$y, scales = list(0:36, 0:12))
degrees <- list(c(6, 6), c(1, 1))
xsmooth <- presmoothing(xt, smoothmethod = "loglinear", degrees = degrees)
ysmooth <- presmoothing(yt, smoothmethod = "loglinear", degrees = degrees)
smooth_matrix <- function(tab) matrix(as.numeric(tab), nrow = 37)
fe_smooth <- equate(xt, yt, type = "equipercentile", method = "frequency estimation", ws = 1,
                    smoothmethod = "loglinear", degrees = degrees)
chained_smooth <- equate(xt, yt, type = "equipercentile", method = "chained",
                         smoothmethod = "loglinear", degrees = degrees)
smoothed <- list(
  name = "kb_neat_smoothed", degree = c(6, 6), cross_terms = list(c(1, 1)), weight = 1,
  x_scores = 0:36, y_scores = 0:36, anchor_scores = 0:12,
  x_joint_frequencies = to_matrix(KBneat$x, 0:36, 0:12),
  y_joint_frequencies = to_matrix(KBneat$y, 0:36, 0:12),
  x_smoothed = smooth_matrix(xsmooth), y_smoothed = smooth_matrix(ysmooth),
  frequency_estimation = list(
    equate = list(equated = fe_smooth$concordance$yx),
    kb_on_equate_smoothing = kb_frequency_estimation(smooth_matrix(xsmooth),
                                                     smooth_matrix(ysmooth), 0:36, 0:36, 1)),
  chained = list(
    equate = list(equated = chained_smooth$concordance$yx),
    kb_on_equate_smoothing = kb_chained(smooth_matrix(xsmooth), smooth_matrix(ysmooth),
                                        0:36, 0:36, 0:12)))

out <- list(
  meta = list(
    generator = "validation/r/equate_e5_reference.R",
    r_version = R.version.string,
    equate_version = as.character(packageVersion("equate")),
    note = paste("equate: synthetic weight of population 1 is effective_weight; ranks rounded to",
                 "10 decimals. kb: Kolen & Brennan (2014) formulas without rounding.")
  ),
  cases = cases,
  smoothed = smoothed
)
path <- "packages/flexpyequate/tests/reference/e5_neat_equipercentile.json"
writeLines(toJSON(out, auto_unbox = TRUE, digits = NA, pretty = TRUE), path)
cat("wrote", path, "\n")
inside <- function(p) p > 0 & p < 1
for (case in cases) {
  for (w in case$frequency_estimation) {
    keep <- inside(w$kb_at_effective_weight$rank_x)
    cat(sprintf("%-16s FE w1 = %.3f  max |equate - kb(effective)| = %.2e (%d of %d scores)\n",
                case$name, w$weight,
                max(abs(w$equate$equated - w$kb_at_effective_weight$equated)[keep]), sum(keep),
                length(keep)))
  }
  keep <- inside(case$chained$kb$rank_x) & inside(case$chained$kb$rank_anchor)
  cat(sprintf("%-16s chained       max |equate - kb| = %.2e (%d of %d scores)\n", case$name,
              max(abs(case$chained$equate$equated - case$chained$kb$equated)[keep]), sum(keep),
              length(keep)))
}
cat(sprintf("smoothed FE %.2e  chained %.2e\n",
            max(abs(smoothed$frequency_estimation$equate$equated -
                      smoothed$frequency_estimation$kb_on_equate_smoothing$equated)),
            max(abs(smoothed$chained$equate$equated -
                      smoothed$chained$kb_on_equate_smoothing$equated))))
