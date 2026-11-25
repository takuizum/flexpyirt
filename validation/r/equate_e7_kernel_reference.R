# flexpyequate E7 reference values (packages/flexpyequate/docs/DESIGN.md §7, cases Q11-Q12):
# kernel equating with the logistic and the uniform kernel (Lee & von Davier, 2011) in every
# design, and the bandwidth rules beyond the penalty (rule of thumb, double smoothing,
# leave-one-out cross-validation) for the Gaussian kernel.
# Run from the repository root:  Rscript validation/r/equate_e7_kernel_reference.R  (a few min)
# Reads the data of the E3 and E6 reference files and writes
# packages/flexpyequate/tests/reference/e7_kernels.json. CI does not run R.
#
# References (see equate_e3_reference.R for the reasons):
# - "kequate_book": kequate with probabilities, design matrices and sample sizes times 1e8
#   (removes its N / (N - 1) variance factor); SEE and SEED multiplied by 1e4.
# - "independent": the continuization written out below for a general kernel,
#   uniroot(tol = 1e-14), and a SEE from a finite-difference Jacobian with respect to every
#   cell probability (no analytic derivative is used).
# - SNSequate::bandwidth(): penalty bandwidths of the logistic and uniform kernels.
suppressMessages({
  library(kequate)
  library(SNSequate)
  library(jsonlite)
})

strict <- glm.control(epsilon = 1e-13, maxit = 200)
BIG <- 1e8
reference_dir <- "packages/flexpyequate/tests/reference"
e3 <- fromJSON(file.path(reference_dir, "e3_kernel.json"), simplifyDataFrame = FALSE)
e6 <- fromJSON(file.path(reference_dir, "e6_kernel_designs.json"), simplifyDataFrame = FALSE)
KERNELS <- c("logistic", "uniform")
as_table <- function(v) if (is.matrix(v)) v else do.call(rbind, v)

# ----------------------------------------------------------------- independent implementation
kernel_variance <- c(gaussian = 1, logistic = pi^2 / 3, uniform = 1 / 12)
kernel_p <- function(z, kernel) {
  switch(kernel, gaussian = pnorm(z), logistic = plogis(z), uniform = pmin(pmax(z + 0.5, 0), 1))
}
kernel_d <- function(z, kernel) {
  switch(kernel, gaussian = dnorm(z), logistic = dlogis(z),
         uniform = as.numeric(z >= -0.5 & z < 0.5))
}
kernel_dd <- function(z, kernel) {
  switch(kernel, gaussian = -z * dnorm(z), logistic = dlogis(z) * (1 - 2 * plogis(z)),
         uniform = 0 * z)
}
moments <- function(scores, p) {
  mu <- sum(scores * p)
  c(mu, sum((scores - mu)^2 * p))
}
standardized <- function(z, scores, p, h, kernel) {
  m <- moments(scores, p)
  a <- sqrt(m[2] / (m[2] + kernel_variance[[kernel]] * h^2))
  list(R = (z - a * scores - (1 - a) * m[1]) / (a * h), scale = a * h)
}
kernel_cdf <- function(v, scores, p, h, kernel) {
  vapply(v, function(z) sum(p * kernel_p(standardized(z, scores, p, h, kernel)$R, kernel)),
         numeric(1))
}
kernel_density <- function(v, scores, p, h, kernel) {
  vapply(v, function(z) {
    s <- standardized(z, scores, p, h, kernel)
    sum(p * kernel_d(s$R, kernel)) / s$scale
  }, numeric(1))
}
kernel_density_derivative <- function(v, scores, p, h, kernel) {
  vapply(v, function(z) {
    s <- standardized(z, scores, p, h, kernel)
    sum(p * kernel_dd(s$R, kernel)) / s$scale^2
  }, numeric(1))
}
kernel_quantile <- function(u, scores, p, h, kernel) {
  vapply(u, function(target) {
    uniroot(function(z) kernel_cdf(z, scores, p, h, kernel) - target,
            c(min(scores) - 100, max(scores) + 100), tol = 1e-14)$root
  }, numeric(1))
}
link <- function(v, from_scores, from_p, from_h, to_scores, to_p, to_h, kernel) {
  kernel_quantile(kernel_cdf(v, from_scores, from_p, from_h, kernel), to_scores, to_p, to_h, kernel)
}
# The penalty of von Davier et al. (2004, ch. 4) with both terms, for any kernel.
penalty <- function(h, scores, p, K, kernel) {
  pen1 <- sum((p - kernel_density(scores, scores, p, h, kernel))^2)
  left <- kernel_density_derivative(scores - 0.25, scores, p, h, kernel)
  right <- kernel_density_derivative(scores + 0.25, scores, p, h, kernel)
  pen1 + K * sum(left < 0 & right > 0)
}
design_sg <- function(P) list(r = rowSums(P), s = colSums(P))
design_cb <- function(P12, P21, wx, wy) {
  list(r = wx * rowSums(P12) + (1 - wx) * rowSums(P21),
       s = wy * colSums(P12) + (1 - wy) * colSums(P21))
}
design_pse <- function(P, Q, w) {
  tp <- colSums(P)
  tq <- colSums(Q)
  list(r = as.vector(P %*% (w + (1 - w) * tq / tp)),
       s = as.vector(Q %*% ((1 - w) + w * tp / tq)))
}
equate_two <- function(d, xs, ys, h, kernel) link(xs, xs, d$r, h[["x"]], ys, d$s, h[["y"]], kernel)
equate_ce <- function(P, Q, xs, ys, as, h, kernel) {
  on_anchor <- link(xs, xs, rowSums(P), h[["x"]], as, colSums(P), h[["anchor_x"]], kernel)
  link(on_anchor, as, colSums(Q), h[["anchor_y"]], ys, rowSums(Q), h[["y"]], kernel)
}
cell_covariance <- function(p, dm, n) {
  W <- diag(p) - tcrossprod(p)
  if (is.null(dm)) return(W / n)
  B <- qr.Q(qr(cbind(1, dm)))[, -1, drop = FALSE]
  W %*% B %*% solve(t(B) %*% W %*% B, t(B) %*% W) / n
}
delta_variance <- function(f, m, covariance, step = 1e-7) {
  jac <- sapply(seq_along(m), function(i) {
    at <- function(t) { mm <- m; mm[i] <- mm[i] + t; f(mm) }
    (-at(2 * step) + 8 * at(step) - 8 * at(-step) + at(-2 * step)) / (12 * step)
  })
  rowSums((jac %*% covariance) * jac)
}
smooth_univariate <- function(f, scores, degree) {
  fit <- glm(f ~ poly(scores, degree, raw = TRUE), family = poisson, x = TRUE, control = strict)
  stopifnot(fit$converged)
  list(p = as.vector(fitted(fit) / sum(f)), dm = fit$x[, -1, drop = FALSE], n = sum(f))
}
smooth_joint <- function(m, a_scores, b_scores, da, db) {
  cells <- data.frame(n = as.vector(m), a = rep(a_scores, times = length(b_scores)),
                      b = rep(b_scores, each = length(a_scores)))
  terms <- c(sprintf("I(a^%d)", seq_len(da)), sprintf("I(b^%d)", seq_len(db)), "I(a * b)")
  fit <- glm(as.formula(paste("n ~", paste(terms, collapse = " + "))), family = poisson,
             data = cells, x = TRUE, control = strict)
  stopifnot(fit$converged)
  list(P = matrix(fitted(fit) / sum(m), nrow = length(a_scores)),
       dm = fit$x[, -1, drop = FALSE], n = sum(m))
}
book <- function(fit) {
  list(equated = fit@equating$eqYx, see = fit@equating$SEEYx * sqrt(BIG),
       # kequate's linear function and SEED always use the Gaussian kernel.
       seed_gaussian_linear = fit@equating$SEEDYx * sqrt(BIG),
       linear_equated = fit@equating$eqYxLIN)
}
# Bandwidths: wide enough for the uniform kernel to leave no gap in the density.
BANDWIDTH <- list(logistic = c(x = 0.42, y = 0.47, anchor_x = 0.4, anchor_y = 0.45),
                  uniform = c(x = 1.2, y = 1.35, anchor_x = 1.25, anchor_y = 1.3))

# ----------------------------------------------------------------- EG (ACT mathematics)
act <- e3$cases[[which(vapply(e3$cases, function(case) case$name, "") == "act_math_smoothed")]]
xs_eg <- as.numeric(act$x_scores)
ys_eg <- as.numeric(act$y_scores)
eg_x <- smooth_univariate(as.numeric(act$x_frequencies), xs_eg, act$degree_x)
eg_y <- smooth_univariate(as.numeric(act$y_frequencies), ys_eg, act$degree_y)
PENALTY_GRID <- c(0.3, 0.45, 0.6, 0.9, 1.0, 1.3, 2.0)
equivalent_groups <- list(source = "e3_kernel.json: act_math_smoothed", penalty_grid = PENALTY_GRID)
for (kernel in KERNELS) {
  h <- BANDWIDTH[[kernel]]
  fit <- kequate("EG", xs_eg, ys_eg, eg_x$p, eg_y$p, eg_x$dm, eg_y$dm, eg_x$n * BIG, eg_y$n * BIG,
                 hx = h[["x"]], hy = h[["y"]], kernel = kernel)
  auto <- kequate("EG", xs_eg, ys_eg, eg_x$p, eg_y$p, eg_x$dm, eg_y$dm, eg_x$n * BIG,
                  eg_y$n * BIG, kernel = kernel)
  fr <- function(r) link(xs_eg, xs_eg, r, h[["x"]], ys_eg, eg_y$p, h[["y"]], kernel)
  fs <- function(s) link(xs_eg, xs_eg, eg_x$p, h[["x"]], ys_eg, s, h[["y"]], kernel)
  mx <- moments(xs_eg, eg_x$p)
  optimum <- function(p, scores, K) {
    grid <- exp(seq(log(0.05), log(10), length.out = 2000))
    values <- vapply(grid, penalty, numeric(1), scores = scores, p = p, K = K, kernel = kernel)
    i <- which.min(values)
    best <- optimize(penalty, c(grid[max(i - 1, 1)], grid[min(i + 1, length(grid))]),
                     scores = scores, p = p, K = K, kernel = kernel, tol = 1e-12)
    if (best$objective <= values[i]) best$minimum else grid[i]
  }
  equivalent_groups[[kernel]] <- list(
    bandwidth_x = h[["x"]], bandwidth_y = h[["y"]],
    kequate_book = c(book(fit), list(
      cdf_x = fit@scores$X$cdfx, cdf_y = fit@scores$Y$cdfy, pre = fit@PRE$PREYx,
      penalty_bandwidth_x = auto@h$hx, penalty_bandwidth_y = auto@h$hy,
      penalty_x_weight0 = vapply(PENALTY_GRID, function(hh) {
        kequate:::PEN(hh, eg_x$p, xs_eg, mx[2], mx[1], 0.25, 0, kernel, 1, 0.5)
      }, numeric(1)))),
    independent = list(
      equated = fr(eg_x$p),
      see = sqrt(delta_variance(fr, eg_x$p, cell_covariance(eg_x$p, eg_x$dm, eg_x$n)) +
                   delta_variance(fs, eg_y$p, cell_covariance(eg_y$p, eg_y$dm, eg_y$n))),
      cdf_x = kernel_cdf(xs_eg, xs_eg, eg_x$p, h[["x"]], kernel),
      density_x = kernel_density(xs_eg, xs_eg, eg_x$p, h[["x"]], kernel),
      penalty_x_weight0 = vapply(PENALTY_GRID, penalty, numeric(1), scores = xs_eg, p = eg_x$p,
                                 K = 0, kernel = kernel),
      penalty_x_weight1 = vapply(PENALTY_GRID, penalty, numeric(1), scores = xs_eg, p = eg_x$p,
                                 K = 1, kernel = kernel),
      penalty_bandwidth_x_weight0 = optimum(eg_x$p, xs_eg, 0),
      penalty_bandwidth_y_weight0 = optimum(eg_y$p, ys_eg, 0),
      penalty_bandwidth_x_weight1 = optimum(eg_x$p, xs_eg, 1),
      penalty_bandwidth_y_weight1 = optimum(eg_y$p, ys_eg, 1)),
    snsequate = list(
      penalty_bandwidth_x_weight1 = bandwidth(scores = as.numeric(act$x_frequencies),
                                              kert = substr(kernel, 1, if (kernel == "uniform") 4 else 5),
                                              design = "EG", Kp = 1, r = eg_x$p)$h,
      penalty_bandwidth_y_weight1 = bandwidth(scores = as.numeric(act$y_frequencies),
                                              kert = substr(kernel, 1, if (kernel == "uniform") 4 else 5),
                                              design = "EG", Kp = 1, r = eg_y$p)$h))
  cat("EG", kernel, "done\n")
}

# ----------------------------------------------------------------- SG
sgd <- e6$single_group
xs <- as.numeric(sgd$x_scores)
ys <- as.numeric(sgd$y_scores)
sg_table <- as_table(sgd$joint_frequencies)
sg <- smooth_joint(sg_table, xs, ys, sgd$degree[1], sgd$degree[2])
single_group <- list(source = "e6_kernel_designs.json: single_group")
for (kernel in KERNELS) {
  h <- BANDWIDTH[[kernel]]
  fit <- kequate("SG", xs, ys, sg$P, sg$dm, sg$n * BIG, hx = h[["x"]], hy = h[["y"]],
                 kernel = kernel)
  f <- function(P) equate_two(design_sg(P), xs, ys, h, kernel)
  single_group[[kernel]] <- list(
    bandwidth_x = h[["x"]], bandwidth_y = h[["y"]],
    kequate_book = c(book(fit), list(pre = fit@PRE$PREYx)),
    independent = list(
      equated = f(sg$P),
      see = sqrt(delta_variance(f, sg$P, cell_covariance(as.vector(sg$P), sg$dm, sg$n)))))
  cat("SG", kernel, "done\n")
}

# ----------------------------------------------------------------- CB
cbd <- e6$counterbalanced
cb12_table <- as_table(cbd$frequencies_12)
cb21_table <- as_table(cbd$frequencies_21)
cb12 <- smooth_joint(cb12_table, xs, ys, cbd$degree[1], cbd$degree[2])
cb21 <- smooth_joint(cb21_table, xs, ys, cbd$degree[1], cbd$degree[2])
counterbalanced <- list(source = "e6_kernel_designs.json: counterbalanced")
for (kernel in KERNELS) {
  h <- BANDWIDTH[[kernel]]
  cb_case <- function(wx, wy, with_kequate) {
    f12 <- function(P) equate_two(design_cb(P, cb21$P, wx, wy), xs, ys, h, kernel)
    f21 <- function(P) equate_two(design_cb(cb12$P, P, wx, wy), xs, ys, h, kernel)
    out <- list(
      weights = c(wx, wy),
      independent = list(
        equated = f12(cb12$P),
        see = sqrt(
          delta_variance(f12, cb12$P, cell_covariance(as.vector(cb12$P), cb12$dm, cb12$n)) +
            delta_variance(f21, cb21$P, cell_covariance(as.vector(cb21$P), cb21$dm, cb21$n)))))
    if (with_kequate) {
      fit <- kequate("CB", xs, ys, cb12$P, cb21$P, cb12$dm, cb21$dm, cb12$n * BIG, cb21$n * BIG,
                     hx = h[["x"]], hy = h[["y"]], wcb = wx, kernel = kernel)
      out$kequate_book <- book(fit)
    }
    out
  }
  counterbalanced[[kernel]] <- list(bandwidth_x = h[["x"]], bandwidth_y = h[["y"]],
                                    cases = list(cb_case(0.5, 0.5, TRUE), cb_case(0.3, 0.8, FALSE)))
  cat("CB", kernel, "done\n")
}

# ----------------------------------------------------------------- NEAT
nd <- e6$neat
xs_n <- as.numeric(nd$x_scores)
ys_n <- as.numeric(nd$y_scores)
as_n <- as.numeric(nd$anchor_scores)
p_table <- as_table(nd$x_joint_frequencies)
q_table <- as_table(nd$y_joint_frequencies)
sp <- smooth_joint(p_table, xs_n, as_n, nd$x_degree[1], nd$x_degree[2])
sq <- smooth_joint(q_table, ys_n, as_n, nd$y_degree[1], nd$y_degree[2])
cov_p <- cell_covariance(as.vector(sp$P), sp$dm, sp$n)
cov_q <- cell_covariance(as.vector(sq$P), sq$dm, sq$n)
neat <- list(source = "e6_kernel_designs.json: neat")
for (kernel in KERNELS) {
  h <- BANDWIDTH[[kernel]]
  pse_case <- function(w) {
    fp <- function(P) equate_two(design_pse(P, sq$P, w), xs_n, ys_n, h, kernel)
    fq <- function(Q) equate_two(design_pse(sp$P, Q, w), xs_n, ys_n, h, kernel)
    fit <- kequate("NEAT_PSE", xs_n, ys_n, sp$P, sq$P, sp$dm, sq$dm, sp$n * BIG, sq$n * BIG,
                   w = w, hx = h[["x"]], hy = h[["y"]], kernel = kernel)
    list(weight = w, kequate_book = c(book(fit), list(pre = fit@PRE$PREYx)),
         independent = list(
           equated = fp(sp$P),
           see = sqrt(delta_variance(fp, sp$P, cov_p) + delta_variance(fq, sq$P, cov_q))))
  }
  ce_fit <- kequate("NEAT_CE", xs_n, ys_n, as_n, sp$P, sq$P, sp$dm, sq$dm, sp$n * BIG,
                    sq$n * BIG, hxP = h[["x"]], hyQ = h[["y"]], haP = h[["anchor_x"]],
                    haQ = h[["anchor_y"]], kernel = kernel)
  ce_p <- function(P) equate_ce(P, sq$P, xs_n, ys_n, as_n, h, kernel)
  ce_q <- function(Q) equate_ce(sp$P, Q, xs_n, ys_n, as_n, h, kernel)
  neat[[kernel]] <- list(
    bandwidths = as.list(h),
    post_stratification = list(pse_case(0.5), pse_case(0.2)),
    chained = list(
      kequate_book = list(equated = ce_fit@equating$eqYx, see = ce_fit@equating$SEEYx * sqrt(BIG),
                          pre_x_to_anchor = ce_fit@PRE$PREAx, pre_anchor_to_y = ce_fit@PRE$PREYa),
      independent = list(equated = ce_p(sp$P),
                         see = sqrt(delta_variance(ce_p, sp$P, cov_p) +
                                      delta_variance(ce_q, sq$P, cov_q)))))
  cat("NEAT", kernel, "done\n")
}

# ----------------------------------------------------------------- bandwidth rules (Gaussian)
# Written out independently of kequate (scores 0, 1, 2, ...; population variance).
weights_matrix <- function(points, scores, p, h) {
  t(vapply(points, function(z) {
    s <- standardized(z, scores, p, h, "gaussian")
    dnorm(s$R) / s$scale
  }, numeric(length(scores))))
}
double_smoothing <- function(h, scores, p, pilot = 4) {
  grid <- seq(min(scores), max(scores), by = 0.5)
  at_scores <- seq(1, length(grid), by = 2)
  pilot_density <- as.vector(weights_matrix(grid, scores, p, pilot) %*% p)
  target <- pilot_density
  target[at_scores] <- p
  mean((as.vector(weights_matrix(grid, scores, p, h) %*% pilot_density[at_scores]) - target)^2)
}
cross_validation <- function(h, scores, p) {
  w <- weights_matrix(scores, scores, p, h)
  diag(w) <- 0
  mean((p - as.vector(w %*% p) / rowSums(w))^2)
}
rule_of_thumb <- function(scores, p, n) 9 * sqrt(moments(scores, p)[2]) / sqrt(100 * n^(2 / 5) - 81)
minimize <- function(f, ..., upper = 10) {
  grid <- exp(seq(log(0.05), log(upper), length.out = 2000))
  values <- vapply(grid, f, numeric(1), ...)
  i <- which.min(values)
  optimize(f, c(grid[max(i - 1, 1)], grid[min(i + 1, length(grid))]), ..., tol = 1e-12)$minimum
}
RULE_GRID <- c(0.3, 0.5, 0.8, 1.2, 2.0)
rule_case <- function(name, scores, p, n) {
  m <- moments(scores, p)
  inflated <- list(list(p, scores, n * BIG, diag(length(p))))
  ordinary <- list(list(p, scores, n, diag(length(p))))
  list(
    name = name, scores = scores, probabilities = p, n = n, grid = RULE_GRID,
    # kequate's criteria with the population variance passed in.
    kequate = list(
      double_smoothing_values = vapply(RULE_GRID, function(h) {
        kequate:::DS(h, p, scores, m[2], m[1], 4, diag(length(p)))
      }, numeric(1)),
      cross_validation_values = vapply(RULE_GRID, function(h) {
        kequate:::CV(h, p, scores, m[2], m[1])
      }, numeric(1)),
      # Sample size times 1e8: no N / (N - 1) variance factor.
      double_smoothing = kequate:::bandwidth.select(inflated, "DS"),
      cross_validation = kequate:::bandwidth.select(inflated, "CV"),
      # Ordinary sample size: kequate's sigma has the factor sqrt(N / (N - 1)).
      rule_of_thumb = kequate:::bandwidth.select(ordinary, "altopt")),
    independent = list(
      double_smoothing_values = vapply(RULE_GRID, double_smoothing, numeric(1), scores = scores, p = p),
      cross_validation_values = vapply(RULE_GRID, cross_validation, numeric(1), scores = scores, p = p),
      # Candidates stay below the pilot bandwidth (4), as in kequate.
      double_smoothing = minimize(double_smoothing, scores = scores, p = p, upper = 4),
      cross_validation = minimize(cross_validation, scores = scores, p = p),
      rule_of_thumb = rule_of_thumb(scores, p, n)))
}
d_sg <- design_sg(sg$P)
bandwidth_rules <- list(
  rule_case("eg_x", xs_eg, eg_x$p, eg_x$n), rule_case("eg_y", ys_eg, eg_y$p, eg_y$n),
  rule_case("sg_x", xs, d_sg$r, sg$n), rule_case("sg_y", ys, d_sg$s, sg$n),
  rule_case("neat_x", xs_n, rowSums(sp$P), sp$n), rule_case("neat_anchor_x", as_n, colSums(sp$P), sp$n),
  rule_case("neat_y", ys_n, rowSums(sq$P), sq$n), rule_case("neat_anchor_y", as_n, colSums(sq$P), sq$n))
# End to end through kequate (sample sizes times 1e8): bandwidths selected inside the call.
end_to_end <- local({
  eg <- function(...) kequate("EG", xs_eg, ys_eg, eg_x$p, eg_y$p, eg_x$dm, eg_y$dm, eg_x$n * BIG,
                              eg_y$n * BIG, ...)
  ce <- function(...) kequate("NEAT_CE", xs_n, ys_n, as_n, sp$P, sq$P, sp$dm, sq$dm, sp$n * BIG,
                              sq$n * BIG, ...)
  pack <- function(fit) list(equated = fit@equating$eqYx, see = fit@equating$SEEYx * sqrt(BIG),
                             bandwidths = as.list(fit@h))
  list(equivalent_groups = list(double_smoothing = pack(eg(DS = TRUE)),
                                cross_validation = pack(eg(CV = TRUE))),
       chained = list(double_smoothing = pack(ce(DS = TRUE)),
                      cross_validation = pack(ce(CV = TRUE))))
})
cat("bandwidth rules done\n")

out <- list(
  meta = list(
    generator = "validation/r/equate_e7_kernel_reference.R",
    r_version = R.version.string,
    kequate_version = as.character(packageVersion("kequate")),
    snsequate_version = as.character(packageVersion("SNSequate")),
    sample_size_factor = BIG,
    note = paste("Data and log-linear models are those of e3_kernel.json (act_math_smoothed)",
                 "and e6_kernel_designs.json. kequate_book: sample sizes times 1e8, SEE and",
                 "SEED rescaled; kequate's linear function and SEED use the Gaussian kernel.",
                 "independent: general-kernel continuization with a finite-difference SEE.")
  ),
  equivalent_groups = equivalent_groups,
  single_group = single_group,
  counterbalanced = counterbalanced,
  neat = neat,
  bandwidth_rules = list(pilot_bandwidth = 4, cases = bandwidth_rules, end_to_end = end_to_end)
)
path <- file.path(reference_dir, "e7_kernels.json")
writeLines(toJSON(out, auto_unbox = TRUE, digits = NA, pretty = TRUE), path)
cat("wrote", path, "\n")
for (kernel in KERNELS) {
  for (block in list(list("EG", equivalent_groups[[kernel]]), list("SG", single_group[[kernel]]),
                     list("CB", counterbalanced[[kernel]]$cases[[1]]),
                     list("PSE", neat[[kernel]]$post_stratification[[1]]),
                     list("CE", neat[[kernel]]$chained))) {
    k <- block[[2]]$kequate_book
    i <- block[[2]]$independent
    cat(sprintf("%-8s %-3s kequate vs independent: eq %.1e  see rel %.1e\n", kernel, block[[1]],
                max(abs(k$equated - i$equated)), max(abs(k$see / i$see - 1))))
  }
}
for (case in bandwidth_rules) {
  cat(sprintf("%-14s DS %.6f / %.6f  CV %.6f / %.6f  RoT %.6f / %.6f\n", case$name,
              case$kequate$double_smoothing, case$independent$double_smoothing,
              case$kequate$cross_validation, case$independent$cross_validation,
              case$kequate$rule_of_thumb, case$independent$rule_of_thumb))
}
