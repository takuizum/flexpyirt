# flexpyequate E7 reference values (packages/flexpyequate/docs/DESIGN.md §7, cases Q9-Q10):
# standard errors of the traditional equating methods.
# Run from the repository root:  Rscript validation/r/equate_e7_standard_errors_reference.R
# (about ten minutes, almost all of it equate::bootstrap)
# Reads the data of the E1 and E4 reference files and writes
# packages/flexpyequate/tests/reference/e7_standard_errors.json. CI does not run R.
#
# References:
# - "equate": the analytic standard errors of R `equate` (Albano, 2016): linear and
#   equipercentile equating for equivalent groups; Tucker, Levine observed-score and chained
#   linear equating with an anchor test, distribution-free ("g") and normal-theory ("n").
#   equate computes the anchor-test ones from population (N) moments but takes gamma and the
#   synthetic moments from its N - 1 equating, so "equate_scaled" repeats the call with all
#   frequencies times 1e6 (the two forms then coincide) and multiplies the result by 1e3.
# - "independent": the delta method written out below with a finite-difference Jacobian of the
#   equating function with respect to every cell probability and the multinomial covariance
#   (no analytic derivative is used), population moments (ddof = 0); and the normal-theory
#   covariance of the moments with a finite-difference Jacobian with respect to the moments.
# - "equate_bootstrap": equate::bootstrap(). Its random numbers cannot be reproduced in
#   Python; these values serve a statistical comparison only.
suppressMessages({
  library(equate)
  library(jsonlite)
})

reference_dir <- "packages/flexpyequate/tests/reference"
e1 <- fromJSON(file.path(reference_dir, "e1_equate.json"), simplifyDataFrame = FALSE)
e4 <- fromJSON(file.path(reference_dir, "e4_neat_linear.json"), simplifyDataFrame = FALSE)
SCALE <- 1e6
BOOTSTRAP_REPS <- 2000
as_table <- function(v) if (is.matrix(v)) v else do.call(rbind, v)

# ----------------------------------------------------------------- independent implementation
five_point <- function(f, v, i, step) {
  at <- function(t) { vv <- v; vv[i] <- vv[i] + t; f(vv) }
  (-at(2 * step) + 8 * at(step) - 8 * at(-step) + at(-2 * step)) / (12 * step)
}
# Variance of f(p) for multinomial proportions p of n examinees; f must not assume sum(p) = 1.
multinomial_variance <- function(f, p, n, step = 1e-6) {
  jac <- sapply(seq_along(p), function(i) if (p[i] > 0) five_point(f, p, i, step) else 0 * f(p))
  (rowSums(sweep(jac^2, 2, as.vector(p), "*")) - as.vector(jac %*% as.vector(p))^2) / n
}
# Moments (mean u, mean v, var u, cov, var v) of a joint table with unnormalized cells.
joint_moments <- function(m, u, v) {
  p <- m / sum(m)
  mu <- sum(u * rowSums(p))
  mv <- sum(v * colSums(p))
  c(mu, mv, sum((u - mu)^2 * rowSums(p)), sum(outer(u - mu, v - mv) * p),
    sum((v - mv)^2 * colSums(p)))
}
# Normal-theory covariance of those sample moments.
normal_covariance <- function(theta, n) {
  s11 <- theta[3]; s12 <- theta[4]; s22 <- theta[5]
  omega <- matrix(0, 5, 5)
  omega[1:2, 1:2] <- c(s11, s12, s12, s22)
  omega[3:5, 3:5] <- c(2 * s11^2, 2 * s11 * s12, 2 * s12^2,
                       2 * s11 * s12, s11 * s22 + s12^2, 2 * s12 * s22,
                       2 * s12^2, 2 * s12 * s22, 2 * s22^2)
  omega / n
}
moment_variance <- function(f, theta, omega, step = 1e-5) {
  jac <- sapply(seq_along(theta), function(i) five_point(f, theta, i, step * max(1, abs(theta[i]))))
  rowSums((jac %*% omega) * jac)
}
# Linear equating functions of the moments (Kolen & Brennan, 2014, ch. 2 and 4).
gamma_of <- function(t, method, internal) {
  if (method == "tucker") t[4] / t[5]
  else if (internal) t[3] / t[4]
  else (t[3] + t[4]) / (t[5] + t[4])
}
anchor_linear <- function(t1, t2, xs, method, w1, internal) {
  if (method == "chained_linear") {
    k <- sqrt(t2[3] / t2[5])
    b <- k * sqrt(t1[5] / t1[3])
    return(t2[1] + k * (t1[2] - t2[2]) - b * t1[1] + b * xs)
  }
  g1 <- gamma_of(t1, if (method == "tucker") "tucker" else "levine", internal)
  g2 <- gamma_of(t2, if (method == "tucker") "tucker" else "levine", internal)
  if (method == "levine_true") {
    b <- g2 / g1
    return(t2[1] + g2 * (t1[2] - t2[2]) - b * t1[1] + b * xs)
  }
  w2 <- 1 - w1
  dm <- t1[2] - t2[2]
  dv <- t1[5] - t2[5]
  msx <- t1[1] - w2 * g1 * dm
  msy <- t2[1] + w1 * g2 * dm
  vsx <- t1[3] - w2 * g1^2 * dv + w1 * w2 * g1^2 * dm^2
  vsy <- t2[3] + w1 * g2^2 * dv + w1 * w2 * g2^2 * dm^2
  b <- sqrt(vsy / vsx)
  msy - b * msx + b * xs
}
braun_holland <- function(m1, m2, xs, ys, w1) {
  p1 <- m1 / sum(m1)
  p2 <- m2 / sum(m2)
  h1 <- colSums(p1)
  h2 <- colSums(p2)
  fs <- w1 * rowSums(p1) + (1 - w1) * as.vector(sweep(p1, 2, h1, "/") %*% h2)
  gs <- w1 * as.vector(sweep(p2, 2, h2, "/") %*% h1) + (1 - w1) * rowSums(p2)
  msx <- sum(xs * fs)
  msy <- sum(ys * gs)
  b <- sqrt(sum((ys - msy)^2 * gs) / sum((xs - msx)^2 * fs))
  msy - b * msx + b * xs
}
mean_linear <- function(mx, vx, my, vy, xs, method) {
  b <- if (method == "linear") sqrt(vy / vx) else 1
  my - b * mx + b * xs
}
univariate_moments <- function(f, scores) {
  p <- f / sum(f)
  m <- sum(scores * p)
  c(m, sum((scores - m)^2 * p))
}

bootstrap_se <- function(...) {
  fit <- bootstrap(..., reps = BOOTSTRAP_REPS)
  list(reps = BOOTSTRAP_REPS, se = as.numeric(fit$se), mean = as.numeric(fit$mean))
}

# ----------------------------------------------------------------- equivalent groups
eg_case <- function(case) {
  xs <- as.numeric(case$x_scores)
  ys <- as.numeric(case$y_scores)
  fx <- as.numeric(case$x_frequencies)
  fy <- as.numeric(case$y_frequencies)
  xt <- as.freqtab(cbind(xs, fx))
  yt <- as.freqtab(cbind(ys, fy))
  nx <- sum(fx)
  ny <- sum(fy)
  unit_increment <- isTRUE(all.equal(diff(ys), rep(1, length(ys) - 1)))
  independent <- lapply(c(mean = "mean", linear = "linear"), function(method) {
    fx_fun <- function(f) { m <- univariate_moments(f, xs); my <- univariate_moments(fy, ys)
      mean_linear(m[1], m[2], my[1], my[2], xs, method) }
    fy_fun <- function(f) { m <- univariate_moments(fx, xs); my <- univariate_moments(f, ys)
      mean_linear(m[1], m[2], my[1], my[2], xs, method) }
    mx <- univariate_moments(fx, xs)
    my <- univariate_moments(fy, ys)
    f_theta <- function(t) mean_linear(t[1], t[2], t[3], t[4], xs, method)
    omega <- diag(c(mx[2] / nx, 2 * mx[2]^2 / nx, my[2] / ny, 2 * my[2]^2 / ny))
    list(se = sqrt(multinomial_variance(fx_fun, fx / nx, nx) +
                     multinomial_variance(fy_fun, fy / ny, ny)),
         se_normal = sqrt(moment_variance(f_theta, c(mx, my), omega)))
  })
  set.seed(7001)
  out <- list(
    name = case$name, x_scores = xs, y_scores = ys, x_frequencies = fx, y_frequencies = fy,
    equate = list(
      # N - 1 variances throughout: flexpyequate with ddof = 1.
      linear = equate(xt, yt, type = "linear", verbose = TRUE)$concordance$se,
      # In units of the form-Y score increment.
      equipercentile = equate(xt, yt, type = "equipercentile", verbose = TRUE)$concordance$se,
      equipercentile_in_increments = !unit_increment),
    independent = independent,
    equate_bootstrap = list(
      linear = bootstrap_se(xt, yt, type = "linear"),
      equipercentile = bootstrap_se(xt, yt, type = "equipercentile")))
  cat("EG", case$name, "done\n")
  out
}
eg_names <- c("act_math", "unequal_lengths", "score_increments")
eg_cases <- lapply(Filter(function(case) case$name %in% eg_names, e1$cases), eg_case)

# ----------------------------------------------------------------- single group
sg_source <- Filter(function(case) case$name == "single_group", e1$cases)[[1]]
single_group <- local({
  xs <- as.numeric(sg_source$x_scores)
  ys <- as.numeric(sg_source$y_scores)
  joint <- as_table(sg_source$joint_frequencies)
  n <- sum(joint)
  theta <- joint_moments(joint, xs, ys)
  independent <- lapply(c(mean = "mean", linear = "linear"), function(method) {
    f_cells <- function(m) { t <- joint_moments(m, xs, ys)
      mean_linear(t[1], t[3], t[2], t[5], xs, method) }
    f_theta <- function(t) mean_linear(t[1], t[3], t[2], t[5], xs, method)
    list(se = sqrt(multinomial_variance(f_cells, joint / n, n)),
         se_normal = sqrt(moment_variance(f_theta, theta, normal_covariance(theta, n))))
  })
  # Closed form under normality (Lord, 1950; Angoff, 1971; Kolen & Brennan, 2014, ch. 7).
  rho <- theta[4] / sqrt(theta[3] * theta[5])
  z <- (xs - theta[1]) / sqrt(theta[3])
  cells <- which(joint > 0, arr.ind = TRUE)
  xt <- freqtab(cbind(xs[cells[, 1]], ys[cells[, 2]])[rep(seq_len(nrow(cells)), joint[joint > 0]), ],
                scales = list(xs, ys), design = "sg")
  stopifnot(sum(xt) == n)
  set.seed(7002)
  out <- list(
    name = "single_group", x_scores = xs, y_scores = ys, joint_frequencies = joint,
    independent = independent,
    normal_closed_form_linear = sqrt(theta[5] * (1 - rho) * (z^2 * (1 + rho) + 2) / n),
    equate_bootstrap = list(linear = bootstrap_se(xt, type = "linear"),
                            equipercentile = bootstrap_se(xt, type = "equipercentile")))
  cat("SG done\n")
  out
})

# ----------------------------------------------------------------- anchor test
expand <- function(joint, total_scores, anchor_scores, factor = 1) {
  cells <- which(joint > 0, arr.ind = TRUE)
  as.freqtab(cbind(total_scores[cells[, 1]], anchor_scores[cells[, 2]], joint[joint > 0] * factor),
             scales = list(total_scores, anchor_scores))
}
neat_case <- function(case, bootstrap_methods) {
  xs <- as.numeric(case$x_scores)
  ys <- as.numeric(case$y_scores)
  vs <- as.numeric(case$anchor_scores)
  joint1 <- as_table(case$x_joint_frequencies)
  joint2 <- as_table(case$y_joint_frequencies)
  internal <- case$anchor == "internal"
  n1 <- sum(joint1)
  n2 <- sum(joint2)
  xt <- expand(joint1, xs, vs)
  yt <- expand(joint2, ys, vs)
  xt_scaled <- expand(joint1, xs, vs, SCALE)
  yt_scaled <- expand(joint2, ys, vs, SCALE)
  stopifnot(sum(xt) == n1, sum(yt) == n2, sum(xt_scaled) == n1 * SCALE)
  t1 <- joint_moments(joint1, xs, vs)
  t2 <- joint_moments(joint2, ys, vs)
  omega_normal <- matrix(0, 10, 10)
  omega_normal[1:5, 1:5] <- normal_covariance(t1, n1)
  omega_normal[6:10, 6:10] <- normal_covariance(t2, n2)
  equate_se <- function(a, b, method, ws, factor = 1) {
    se <- equate(a, b, type = "linear", method = method, ws = ws, internal = internal,
                 verbose = TRUE)$concordance$se
    list(general = se$g * factor, normal = se$n * factor)
  }
  independent_se <- function(method, w1) {
    f1 <- function(m) anchor_linear(joint_moments(m, xs, vs), t2, xs, method, w1, internal)
    f2 <- function(m) anchor_linear(t1, joint_moments(m, ys, vs), xs, method, w1, internal)
    f_theta <- function(t) anchor_linear(t[1:5], t[6:10], xs, method, w1, internal)
    list(se = sqrt(multinomial_variance(f1, joint1 / n1, n1) +
                     multinomial_variance(f2, joint2 / n2, n2)),
         se_normal = sqrt(moment_variance(f_theta, c(t1, t2), omega_normal)))
  }
  weights <- lapply(list(-1, 0.5, 1), function(ws) {
    w1 <- if (ws == -1) n1 / (n1 + n2) else ws
    b1 <- function(m) braun_holland(m, joint2, xs, ys, w1)
    b2 <- function(m) braun_holland(joint1, m, xs, ys, w1)
    list(
      weight = w1,
      equate = list(tucker = equate_se(xt, yt, "tucker", ws),
                    levine_observed = equate_se(xt, yt, "levine", ws)),
      equate_scaled = list(tucker = equate_se(xt_scaled, yt_scaled, "tucker", ws, sqrt(SCALE)),
                           levine_observed = equate_se(xt_scaled, yt_scaled, "levine", ws,
                                                       sqrt(SCALE))),
      independent = list(
        tucker = independent_se("tucker", w1),
        levine_observed = independent_se("levine_observed", w1),
        braun_holland = list(se = sqrt(multinomial_variance(b1, joint1 / n1, n1) +
                                         multinomial_variance(b2, joint2 / n2, n2)))))
  })
  set.seed(7003)
  boot <- list()
  for (method in bootstrap_methods) {
    boot[[method]] <- switch(
      method,
      tucker = bootstrap_se(xt, yt, type = "linear", method = "tucker", internal = internal),
      chained_linear = bootstrap_se(xt, yt, type = "linear", method = "chained"),
      frequency_estimation = bootstrap_se(xt, yt, type = "equipercentile",
                                          method = "frequency estimation"),
      chained_equipercentile = bootstrap_se(xt, yt, type = "equipercentile", method = "chained"))
    cat("  bootstrap", method, "done\n")
  }
  out <- list(
    name = case$name, anchor = case$anchor, x_scores = xs, y_scores = ys, anchor_scores = vs,
    x_joint_frequencies = joint1, y_joint_frequencies = joint2,
    all_anchor_scores_observed = all(colSums(joint1) > 0) && all(colSums(joint2) > 0),
    weights = weights,
    equate = list(chained_linear = equate_se(xt, yt, "chained", -1)),
    equate_scaled = list(chained_linear = equate_se(xt_scaled, yt_scaled, "chained", -1,
                                                    sqrt(SCALE))),
    independent = list(chained_linear = independent_se("chained_linear", NA),
                       levine_true = independent_se("levine_true", NA)),
    equate_bootstrap = boot)
  cat("NEAT", case$name, "done\n")
  out
}
neat_cases <- list(
  neat_case(e4$cases[[1]], c("tucker", "chained_linear", "frequency_estimation",
                             "chained_equipercentile")),
  neat_case(e4$cases[[2]], c("tucker")))

out <- list(
  meta = list(
    generator = "validation/r/equate_e7_standard_errors_reference.R",
    r_version = R.version.string,
    equate_version = as.character(packageVersion("equate")),
    frequency_factor = SCALE,
    note = paste("Data are those of e1_equate.json and e4_neat_linear.json. equate: R equate's",
                 "analytic standard errors (N - 1 moments in the equating function).",
                 "equate_scaled: frequencies times 1e6, standard errors times 1e3.",
                 "independent: delta method with finite-difference Jacobians, ddof = 0.",
                 "equate_bootstrap: equate::bootstrap, for statistical comparison only.")
  ),
  equivalent_groups = eg_cases,
  single_group = single_group,
  neat = neat_cases
)
path <- file.path(reference_dir, "e7_standard_errors.json")
writeLines(toJSON(out, auto_unbox = TRUE, digits = NA, pretty = TRUE), path)
cat("wrote", path, "\n")
for (case in neat_cases) {
  w <- case$weights[[1]]
  for (m in c("tucker", "levine_observed")) {
    cat(sprintf("%-16s %-16s scaled vs independent: g %.1e  n %.1e | ordinary vs scaled: %.1e\n",
                case$name, m, max(abs(w$equate_scaled[[m]]$general / w$independent[[m]]$se - 1)),
                max(abs(w$equate_scaled[[m]]$normal / w$independent[[m]]$se_normal - 1)),
                max(abs(w$equate[[m]]$general / w$equate_scaled[[m]]$general - 1))))
  }
  cat(sprintf("%-16s %-16s scaled vs independent: g %.1e  n %.1e\n", case$name, "chained_linear",
              max(abs(case$equate_scaled$chained_linear$general / case$independent$chained_linear$se - 1)),
              max(abs(case$equate_scaled$chained_linear$normal /
                        case$independent$chained_linear$se_normal - 1))))
}
