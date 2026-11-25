# flexpyequate E4 reference values (packages/flexpyequate/docs/DESIGN.md §7, case Q6): linear
# equating for non-equivalent groups with an anchor test (Tucker, Levine observed and true score,
# Braun-Holland, chained linear).
# Run from the repository root:  Rscript validation/r/equate_e4_reference.R
# Writes packages/flexpyequate/tests/reference/e4_neat_linear.json. CI does not run R.
#
# Two references:
# - R `equate` (Albano, 2016). Variances and covariances are divided by N - 1 (flexpyequate:
#   ddof = 1). For Braun-Holland, equate mixes the two populations as *counts*,
#   ws * n1(x, v) + (1 - ws) * n1(x, v) / n1(v) * n2(v), so that the weight of population 1 in
#   its synthetic distributions is ws * N1 / (ws * N1 + (1 - ws) * N2) rather than ws. That
#   effective weight is recorded as `braun_holland_weight` and passed to flexpyequate.
# - `kb_*()` below: the formulas of Kolen & Brennan (2014, ch. 4 and 5) with population moments
#   (ddof = 0) and probability weights. Tight reference for the default settings.
suppressMessages({
  library(equate)
  library(jsonlite)
})

# ----------------------------------------------------------------- independent implementation
group_moments <- function(joint, total_scores, anchor_scores) {
  p <- joint / sum(joint)
  px <- rowSums(p)
  pv <- colSums(p)
  mx <- sum(total_scores * px)
  mv <- sum(anchor_scores * pv)
  list(mx = mx, mv = mv, vx = sum((total_scores - mx)^2 * px),
       vv = sum((anchor_scores - mv)^2 * pv),
       cxv = sum(outer(total_scores - mx, anchor_scores - mv) * p))
}
kb_gamma <- function(g, method, internal) {
  if (method == "tucker") g$cxv / g$vv
  else if (internal) g$vx / g$cxv
  else (g$vx + g$cxv) / (g$vv + g$cxv)
}
kb_synthetic_linear <- function(g1, g2, gamma1, gamma2, w1) {
  w2 <- 1 - w1
  msx <- g1$mx - w2 * gamma1 * (g1$mv - g2$mv)
  msy <- g2$mx + w1 * gamma2 * (g1$mv - g2$mv)
  vsx <- g1$vx - w2 * gamma1^2 * (g1$vv - g2$vv) + w1 * w2 * gamma1^2 * (g1$mv - g2$mv)^2
  vsy <- g2$vx + w1 * gamma2^2 * (g1$vv - g2$vv) + w1 * w2 * gamma2^2 * (g1$mv - g2$mv)^2
  slope <- sqrt(vsy / vsx)
  list(slope = slope, intercept = msy - slope * msx, gamma = c(gamma1, gamma2),
       synthetic_mean = c(msx, msy), synthetic_sd = sqrt(c(vsx, vsy)))
}
kb_levine_true <- function(g1, g2, gamma1, gamma2) {
  slope <- gamma2 / gamma1
  list(slope = slope, intercept = g2$mx + gamma2 * (g1$mv - g2$mv) - slope * g1$mx)
}
kb_chained <- function(g1, g2) {
  slope <- sqrt(g2$vx / g2$vv) * sqrt(g1$vv / g1$vx)
  list(slope = slope,
       intercept = g2$mx + sqrt(g2$vx / g2$vv) * (g1$mv - sqrt(g1$vv / g1$vx) * g1$mx - g2$mv))
}
kb_braun_holland <- function(joint1, joint2, xs, ys, w1) {
  p1 <- joint1 / sum(joint1)
  p2 <- joint2 / sum(joint2)
  h1 <- colSums(p1)
  h2 <- colSums(p2)
  stopifnot(all(h1 > 0), all(h2 > 0))
  fs <- w1 * rowSums(p1) + (1 - w1) * as.vector(sweep(p1, 2, h1, "/") %*% h2)
  gs <- w1 * as.vector(sweep(p2, 2, h2, "/") %*% h1) + (1 - w1) * rowSums(p2)
  msx <- sum(xs * fs)
  msy <- sum(ys * gs)
  ssx <- sqrt(sum((xs - msx)^2 * fs))
  ssy <- sqrt(sum((ys - msy)^2 * gs))
  list(slope = ssy / ssx, intercept = msy - ssy / ssx * msx, synthetic_mean = c(msx, msy),
       synthetic_sd = c(ssx, ssy), synthetic_x = fs, synthetic_y = gs)
}

# ----------------------------------------------------------------- equate
equate_fit <- function(xt, yt, method, ws, internal, lts = FALSE) {
  eq <- equate(xt, yt, type = "linear", method = method, ws = ws, internal = internal, lts = lts)
  out <- list(slope = unname(eq$coefficients["slope"]),
              intercept = unname(eq$coefficients["intercept"]),
              equated = eq$concordance$yx)
  if (!is.null(eq$gamma)) out$gamma <- unname(eq$gamma)
  if (!is.null(eq$synthstats)) {
    out$synthetic_mean <- eq$synthstats$mean
    out$synthetic_sd <- eq$synthstats$sd
  }
  out
}

make_case <- function(name, xdata, ydata, xs, ys, vs, internal, note = "") {
  xt <- freqtab(xdata, scales = list(xs, vs))
  yt <- freqtab(ydata, scales = list(ys, vs))
  joint1 <- unclass(table(factor(xdata[, 1], levels = xs), factor(xdata[, 2], levels = vs)))
  joint2 <- unclass(table(factor(ydata[, 1], levels = ys), factor(ydata[, 2], levels = vs)))
  dimnames(joint1) <- dimnames(joint2) <- NULL
  n1 <- sum(joint1)
  n2 <- sum(joint2)
  g1 <- group_moments(joint1, xs, vs)
  g2 <- group_moments(joint2, ys, vs)
  weights <- lapply(list(-1, 1, 0.5, 0.2), function(ws) {
    w1 <- if (ws == -1) n1 / (n1 + n2) else ws
    bh_weight <- w1 * n1 / (w1 * n1 + (1 - w1) * n2)
    list(
      weight = w1,
      braun_holland_weight = bh_weight,
      equate = list(
        tucker = equate_fit(xt, yt, "tucker", ws, internal),
        levine_observed = equate_fit(xt, yt, "levine", ws, internal),
        braun_holland = equate_fit(xt, yt, "braun/holland", ws, internal)),
      kb = list(
        tucker = kb_synthetic_linear(g1, g2, kb_gamma(g1, "tucker"), kb_gamma(g2, "tucker"), w1),
        levine_observed = kb_synthetic_linear(g1, g2, kb_gamma(g1, "levine", internal),
                                              kb_gamma(g2, "levine", internal), w1),
        braun_holland = kb_braun_holland(joint1, joint2, xs, ys, w1),
        braun_holland_at_equate_weight = kb_braun_holland(joint1, joint2, xs, ys, bh_weight)))
  })
  list(
    name = name, note = note, anchor = if (internal) "internal" else "external",
    x_scores = xs, y_scores = ys, anchor_scores = vs,
    x_joint_frequencies = joint1, y_joint_frequencies = joint2,
    weights = weights,
    equate = list(levine_true = equate_fit(xt, yt, "levine", -1, internal, lts = TRUE),
                  chained_linear = equate_fit(xt, yt, "chained", -1, internal)),
    kb = list(levine_true = kb_levine_true(g1, g2, kb_gamma(g1, "levine", internal),
                                           kb_gamma(g2, "levine", internal)),
              chained_linear = kb_chained(g1, g2)))
}

cases <- list()
# Q6-a: the anchor-test example of Kolen & Brennan (2014, ch. 4): 36-item forms with a 12-item
# internal anchor (equate::KBneat).
cases[[1]] <- make_case("kb_neat", KBneat$x, KBneat$y, 0:36, 0:36, 0:12, internal = TRUE,
                        note = "equate::KBneat, internal anchor")

# Q6-b: external anchor, unequal sample sizes and form lengths, populations of different ability.
set.seed(20261127)
simulate <- function(n, n_items, n_anchor, shift) {
  theta <- rnorm(n, shift, 1)
  cbind(total = rbinom(n, n_items, plogis(0.9 * theta + 0.2)),
        anchor = rbinom(n, n_anchor, plogis(0.8 * theta)))
}
cases[[2]] <- make_case("external_anchor", simulate(1200, 30, 12, 0.25), simulate(2000, 35, 12, -0.1),
                        0:30, 0:35, 0:12, internal = FALSE,
                        note = "external anchor 0..12; X 0..30 (N = 1200), Y 0..35 (N = 2000)")

out <- list(
  meta = list(
    generator = "validation/r/equate_e4_reference.R",
    r_version = R.version.string,
    equate_version = as.character(packageVersion("equate")),
    note = paste("equate: N - 1 variances (ddof = 1); its Braun-Holland weight is",
                 "braun_holland_weight. kb: Kolen & Brennan (2014) formulas with population",
                 "moments (ddof = 0).")
  ),
  cases = cases
)
path <- "packages/flexpyequate/tests/reference/e4_neat_linear.json"
writeLines(toJSON(out, auto_unbox = TRUE, digits = NA, pretty = TRUE), path)
cat("wrote", path, "\n")
for (case in cases) {
  w <- case$weights[[1]]
  cat(sprintf("%-16s w1 = %.4f\n", case$name, w$weight))
  for (m in c("tucker", "levine_observed", "braun_holland")) {
    cat(sprintf("  %-16s equate %.6f + %.6f x | kb %.6f + %.6f x\n", m, w$equate[[m]]$intercept,
                w$equate[[m]]$slope, w$kb[[m]]$intercept, w$kb[[m]]$slope))
  }
  cat(sprintf("  %-16s equate %.6f + %.6f x | kb(at equate weight) %.6f + %.6f x\n", "braun_holland",
              w$equate$braun_holland$intercept, w$equate$braun_holland$slope,
              w$kb$braun_holland_at_equate_weight$intercept,
              w$kb$braun_holland_at_equate_weight$slope))
  for (m in c("levine_true", "chained_linear")) {
    cat(sprintf("  %-16s equate %.6f + %.6f x | kb %.6f + %.6f x\n", m, case$equate[[m]]$intercept,
                case$equate[[m]]$slope, case$kb[[m]]$intercept, case$kb[[m]]$slope))
  }
}
