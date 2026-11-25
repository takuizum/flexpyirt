# flexpyequate E6 reference values (packages/flexpyequate/docs/DESIGN.md §7, case Q8): kernel
# equating for the single-group (SG), counterbalanced (CB) and anchor-test designs (chained, CE,
# and post-stratification, PSE), with SEE and SEED.
# Run from the repository root:  Rscript validation/r/equate_e6_reference.R   (about a minute)
# Writes packages/flexpyequate/tests/reference/e6_kernel_designs.json. CI does not run R.
#
# References (see equate_e3_reference.R for the reasons):
# - "kequate_book": kequate with probability matrices, design matrices and sample sizes times
#   1e8 (removes its N / (N - 1) variance factor); SEE and SEED multiplied by 1e4.
# - "kequate_default": the ordinary call with glm objects.
# - "independent": the design functions and the continuization of von Davier, Holland & Thayer
#   (2004) written out below, uniroot(tol = 1e-14), and a SEE from a finite-difference Jacobian
#   with respect to every cell probability (no analytic derivative is used).
suppressMessages({
  library(kequate)
  library(jsonlite)
})

strict <- glm.control(epsilon = 1e-13, maxit = 200)
BIG <- 1e8

# ----------------------------------------------------------------- independent implementation
moments <- function(scores, p) {
  mu <- sum(scores * p)
  c(mu, sum((scores - mu)^2 * p))
}
kernel_cdf <- function(v, scores, p, h) {
  m <- moments(scores, p)
  a <- sqrt(m[2] / (m[2] + h^2))
  vapply(v, function(z) sum(p * pnorm((z - a * scores - (1 - a) * m[1]) / (a * h))), numeric(1))
}
kernel_quantile <- function(u, scores, p, h) {
  vapply(u, function(target) {
    uniroot(function(z) kernel_cdf(z, scores, p, h) - target,
            c(min(scores) - 100, max(scores) + 100), tol = 1e-14)$root
  }, numeric(1))
}
link <- function(v, from_scores, from_p, from_h, to_scores, to_p, to_h) {
  kernel_quantile(kernel_cdf(v, from_scores, from_p, from_h), to_scores, to_p, to_h)
}
# Design functions. Matrices have the first score in rows and the second in columns.
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
equate_two <- function(d, xs, ys, hx, hy) link(xs, xs, d$r, hx, ys, d$s, hy)
equate_ce <- function(P, Q, xs, ys, as, h) {
  on_anchor <- link(xs, xs, rowSums(P), h["x"], as, colSums(P), h["anchor_x"])
  link(on_anchor, as, colSums(Q), h["anchor_y"], ys, rowSums(Q), h["y"])
}
# Covariance of the fitted cell probabilities of a log-linear model whose design matrix spans
# the columns of `dm` and the constant (Holland & Thayer, 1987); multinomial if dm is NULL.
cell_covariance <- function(p, dm, n) {
  W <- diag(p) - tcrossprod(p)
  if (is.null(dm)) return(W / n)
  B <- qr.Q(qr(cbind(1, dm)))[, -1, drop = FALSE]
  W %*% B %*% solve(t(B) %*% W %*% B, t(B) %*% W) / n
}
# Variance of f(p) by the delta method with a five-point finite-difference Jacobian in every
# cell of the matrix `m` (not renormalized; the covariance has zero row sums).
delta_variance <- function(f, m, covariance, step = 1e-7) {
  jac <- sapply(seq_along(m), function(i) {
    at <- function(t) { mm <- m; mm[i] <- mm[i] + t; f(mm) }
    (-at(2 * step) + 8 * at(step) - 8 * at(-step) + at(-2 * step)) / (12 * step)
  })
  rowSums((jac %*% covariance) * jac)
}

# ----------------------------------------------------------------- data and smoothing
joint_table <- function(a, b, a_scores, b_scores) {
  m <- unclass(table(factor(a, levels = a_scores), factor(b, levels = b_scores)))
  dimnames(m) <- NULL
  m
}
# Bivariate log-linear model with raw powers (kequate needs the design matrix): degrees (da, db)
# and the cross term a * b. Cells in column-major order (first score fastest), as kequate wants.
smooth_joint <- function(m, a_scores, b_scores, da, db) {
  cells <- data.frame(n = as.vector(m), a = rep(a_scores, times = length(b_scores)),
                      b = rep(b_scores, each = length(a_scores)))
  terms <- c(sprintf("I(a^%d)", seq_len(da)), sprintf("I(b^%d)", seq_len(db)), "I(a * b)")
  fit <- glm(as.formula(paste("n ~", paste(terms, collapse = " + "))), family = poisson,
             data = cells, x = TRUE, control = strict)
  stopifnot(fit$converged)
  list(glm = fit, P = matrix(fitted(fit) / sum(m), nrow = length(a_scores)),
       dm = fit$x[, -1, drop = FALSE], n = sum(m))
}
book <- function(fit) list(equated = fit@equating$eqYx, see = fit@equating$SEEYx * sqrt(BIG))
with_seed <- function(out, fit) {
  out$seed_linear <- fit@equating$SEEDYx * sqrt(BIG)
  out$linear_equated <- fit@equating$eqYxLIN
  out$linear_see <- fit@equating$SEEYxLIN * sqrt(BIG)
  out
}

set.seed(20261128)
n_sg <- 2000
theta <- rnorm(n_sg)
xs <- 0:15
ys <- 0:18
sg_table <- joint_table(rbinom(n_sg, 15, plogis(theta + 0.1)), rbinom(n_sg, 18, plogis(0.9 * theta - 0.2)),
                        xs, ys)
sg <- smooth_joint(sg_table, xs, ys, 4, 4)
h_sg <- c(x = 0.6, y = 0.55)

# ----------------------------------------------------------------- SG
sg_book <- kequate("SG", xs, ys, sg$P, sg$dm, sg$n * BIG, hx = h_sg["x"], hy = h_sg["y"])
sg_auto <- kequate("SG", xs, ys, sg$P, sg$dm, sg$n * BIG)
sg_default <- kequate("SG", xs, ys, sg$glm, hx = h_sg["x"], hy = h_sg["y"])
sg_f <- function(P) equate_two(design_sg(P), xs, ys, h_sg["x"], h_sg["y"])
single_group <- list(
  x_scores = xs, y_scores = ys, joint_frequencies = sg_table, degree = c(4, 4),
  cross_terms = list(c(1, 1)), bandwidth_x = unname(h_sg["x"]), bandwidth_y = unname(h_sg["y"]),
  fitted_probabilities = sg$P,
  kequate_book = c(with_seed(book(sg_book), sg_book),
                   list(penalty_bandwidth_x = sg_auto@h$hx, penalty_bandwidth_y = sg_auto@h$hy,
                        pre = sg_book@PRE$PREYx)),
  kequate_default = list(equated = sg_default@equating$eqYx, see = sg_default@equating$SEEYx),
  independent = list(
    equated = sg_f(sg$P),
    see = sqrt(delta_variance(sg_f, sg$P, cell_covariance(as.vector(sg$P), sg$dm, sg$n)))),
  # Without presmoothing: observed proportions and the multinomial covariance.
  observed = local({
    P <- sg_table / sum(sg_table)
    list(equated = sg_f(P),
         see = sqrt(delta_variance(sg_f, P, cell_covariance(as.vector(P), NULL, sum(sg_table)))))
  }))
cat("SG done\n")

# ----------------------------------------------------------------- CB
n12 <- 1500
n21 <- 1400
t12 <- rnorm(n12)
t21 <- rnorm(n21, 0.1)
cb12_table <- joint_table(rbinom(n12, 15, plogis(t12 + 0.1)), rbinom(n12, 18, plogis(0.9 * t12 - 0.1)),
                          xs, ys)
cb21_table <- joint_table(rbinom(n21, 15, plogis(t21 + 0.2)), rbinom(n21, 18, plogis(0.9 * t21 - 0.2)),
                          xs, ys)
cb12 <- smooth_joint(cb12_table, xs, ys, 3, 3)
cb21 <- smooth_joint(cb21_table, xs, ys, 3, 3)
h_cb <- c(x = 0.6, y = 0.6)
cb_case <- function(wx, wy, with_kequate) {
  f12 <- function(P) equate_two(design_cb(P, cb21$P, wx, wy), xs, ys, h_cb["x"], h_cb["y"])
  f21 <- function(P) equate_two(design_cb(cb12$P, P, wx, wy), xs, ys, h_cb["x"], h_cb["y"])
  out <- list(
    weights = c(wx, wy),
    independent = list(
      equated = f12(cb12$P),
      see = sqrt(delta_variance(f12, cb12$P, cell_covariance(as.vector(cb12$P), cb12$dm, cb12$n)) +
                   delta_variance(f21, cb21$P, cell_covariance(as.vector(cb21$P), cb21$dm, cb21$n)))))
  if (with_kequate) {
    fit <- kequate("CB", xs, ys, cb12$P, cb21$P, cb12$dm, cb21$dm, cb12$n * BIG, cb21$n * BIG,
                   hx = h_cb["x"], hy = h_cb["y"], wcb = wx)
    out$kequate_book <- with_seed(book(fit), fit)
  }
  out
}
counterbalanced <- list(
  x_scores = xs, y_scores = ys, frequencies_12 = cb12_table, frequencies_21 = cb21_table,
  degree = c(3, 3), cross_terms = list(c(1, 1)), bandwidth_x = unname(h_cb["x"]),
  bandwidth_y = unname(h_cb["y"]),
  # kequate has one weight wcb: r = wcb r1 + (1 - wcb) r2 and s = wcb s1 + (1 - wcb) s2. Its
  # SEE, however, weights the Y margins the other way round (1 - wcb for sample 12), so it
  # belongs to these r and s only for wcb = 1/2. With wcb = 0.3 its equated scores are those of
  # the weights (0.3, 0.3) and its SEE that of nothing consistent; both are recorded.
  cases = list(cb_case(0.5, 0.5, TRUE), cb_case(0.3, 0.8, FALSE), cb_case(0.3, 0.3, FALSE)),
  kequate_wcb_0.3 = local({
    fit <- kequate("CB", xs, ys, cb12$P, cb21$P, cb12$dm, cb21$dm, cb12$n * BIG, cb21$n * BIG,
                   hx = h_cb["x"], hy = h_cb["y"], wcb = 0.3)
    book(fit)
  }))
cat("CB done\n")

# ----------------------------------------------------------------- NEAT
np <- 3000
nq <- 3500
tp <- rnorm(np, 0.2)
tq <- rnorm(nq, -0.1)
xs_n <- 0:20
ys_n <- 0:22
as_n <- 0:8
anchor_p <- rbinom(np, 8, plogis(tp))
anchor_q <- rbinom(nq, 8, plogis(tq))
p_table <- joint_table(anchor_p + rbinom(np, 12, plogis(tp)), anchor_p, xs_n, as_n)
q_table <- joint_table(anchor_q + rbinom(nq, 14, plogis(tq - 0.1)), anchor_q, ys_n, as_n)
sp <- smooth_joint(p_table, xs_n, as_n, 4, 3)
sq <- smooth_joint(q_table, ys_n, as_n, 4, 3)
cov_p <- cell_covariance(as.vector(sp$P), sp$dm, sp$n)
cov_q <- cell_covariance(as.vector(sq$P), sq$dm, sq$n)

h_pse <- c(x = 0.62, y = 0.58)
pse_case <- function(w, with_default) {
  fp <- function(P) equate_two(design_pse(P, sq$P, w), xs_n, ys_n, h_pse["x"], h_pse["y"])
  fq <- function(Q) equate_two(design_pse(sp$P, Q, w), xs_n, ys_n, h_pse["x"], h_pse["y"])
  fit <- kequate("NEAT_PSE", xs_n, ys_n, sp$P, sq$P, sp$dm, sq$dm, sp$n * BIG, sq$n * BIG, w = w,
                 hx = h_pse["x"], hy = h_pse["y"])
  auto <- kequate("NEAT_PSE", xs_n, ys_n, sp$P, sq$P, sp$dm, sq$dm, sp$n * BIG, sq$n * BIG, w = w)
  d <- design_pse(sp$P, sq$P, w)
  out <- list(
    weight = w, r = d$r, s = d$s,
    kequate_book = c(with_seed(book(fit), fit),
                     list(penalty_bandwidth_x = auto@h$hx, penalty_bandwidth_y = auto@h$hy,
                          pre = fit@PRE$PREYx)),
    independent = list(equated = fp(sp$P),
                       see = sqrt(delta_variance(fp, sp$P, cov_p) + delta_variance(fq, sq$P, cov_q))))
  if (with_default) {
    default <- kequate("NEAT_PSE", xs_n, ys_n, sp$glm, sq$glm, w = w, hx = h_pse["x"],
                       hy = h_pse["y"])
    out$kequate_default <- list(equated = default@equating$eqYx, see = default@equating$SEEYx)
  }
  out
}
post_stratification <- list(bandwidth_x = unname(h_pse["x"]), bandwidth_y = unname(h_pse["y"]),
                            cases = list(pse_case(0.5, TRUE), pse_case(0.2, FALSE),
                                         pse_case(1, FALSE)))
cat("PSE done\n")

h_ce <- c(x = 0.6, y = 0.58, anchor_x = 0.55, anchor_y = 0.5)
ce_fit <- kequate("NEAT_CE", xs_n, ys_n, as_n, sp$P, sq$P, sp$dm, sq$dm, sp$n * BIG, sq$n * BIG,
                  hxP = h_ce["x"], hyQ = h_ce["y"], haP = h_ce["anchor_x"], haQ = h_ce["anchor_y"])
ce_auto <- kequate("NEAT_CE", xs_n, ys_n, as_n, sp$P, sq$P, sp$dm, sq$dm, sp$n * BIG, sq$n * BIG)
ce_default <- kequate("NEAT_CE", xs_n, ys_n, as_n, sp$glm, sq$glm, hxP = h_ce["x"],
                      hyQ = h_ce["y"], haP = h_ce["anchor_x"], haQ = h_ce["anchor_y"])
ce_p <- function(P) equate_ce(P, sq$P, xs_n, ys_n, as_n, h_ce)
ce_q <- function(Q) equate_ce(sp$P, Q, xs_n, ys_n, as_n, h_ce)
chained <- list(
  bandwidths = as.list(h_ce),
  kequate_book = c(with_seed(book(ce_fit), ce_fit),
                   list(penalty_bandwidths = list(x = ce_auto@h$hxP, y = ce_auto@h$hyQ,
                                                  anchor_x = ce_auto@h$haP,
                                                  anchor_y = ce_auto@h$haQ),
                        pre_x_to_anchor = ce_fit@PRE$PREAx, pre_anchor_to_y = ce_fit@PRE$PREYa)),
  kequate_default = list(equated = ce_default@equating$eqYx, see = ce_default@equating$SEEYx),
  independent = list(equated = ce_p(sp$P),
                     see = sqrt(delta_variance(ce_p, sp$P, cov_p) +
                                  delta_variance(ce_q, sq$P, cov_q))))
cat("CE done\n")

out <- list(
  meta = list(
    generator = "validation/r/equate_e6_reference.R",
    r_version = R.version.string,
    kequate_version = as.character(packageVersion("kequate")),
    sample_size_factor = BIG,
    note = paste("Matrices have the first score (X, Y or the total) in rows and the second (Y or",
                 "the anchor) in columns. kequate_book: sample sizes times 1e8, SEE and SEED",
                 "rescaled. independent: finite-difference SEE.")
  ),
  single_group = single_group,
  counterbalanced = counterbalanced,
  neat = list(x_scores = xs_n, y_scores = ys_n, anchor_scores = as_n,
              x_joint_frequencies = p_table, y_joint_frequencies = q_table,
              x_degree = c(4, 3), y_degree = c(4, 3), cross_terms = list(c(1, 1)),
              x_fitted_probabilities = sp$P, y_fitted_probabilities = sq$P,
              post_stratification = post_stratification, chained = chained)
)
path <- "packages/flexpyequate/tests/reference/e6_kernel_designs.json"
writeLines(toJSON(out, auto_unbox = TRUE, digits = NA, pretty = TRUE), path)
cat("wrote", path, "\n")
report <- function(label, k, i) {
  cat(sprintf("%-10s |kequate - independent| equated %.1e  see rel %.1e\n", label,
              max(abs(k$equated - i$equated)), max(abs(k$see / i$see - 1))))
}
report("SG", single_group$kequate_book, single_group$independent)
report("CB", counterbalanced$cases[[1]]$kequate_book, counterbalanced$cases[[1]]$independent)
report("CB w=.3", counterbalanced[["kequate_wcb_0.3"]], counterbalanced$cases[[3]]$independent)
for (case in post_stratification$cases) report(sprintf("PSE w=%.1f", case$weight), case$kequate_book,
                                               case$independent)
report("CE", chained$kequate_book, chained$independent)
print(names(sg_book@equating)); print(names(ce_fit@equating)); print(names(ce_fit@h))
