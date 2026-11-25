# flexpyequate E3 reference values (packages/flexpyequate/docs/DESIGN.md §7, cases Q4 and Q5):
# kernel equating for the equivalent-groups design with a Gaussian kernel.
# Run from the repository root:  Rscript validation/r/equate_e3_reference.R
# Writes packages/flexpyequate/tests/reference/e3_kernel.json. CI does not run R.
#
# References:
# - kequate (Andersson, Branberg & Wiberg, 2013). It multiplies the variance in the continuization
#   by N / (N - 1) while keeping the derivative formulas of von Davier, Holland & Thayer (2004),
#   which assume the variance sum_j r_j (x_j - mu)^2. flexpyequate follows the book, so:
#     * "kequate_book": kequate called with probability vectors, design matrices and sample
#       sizes multiplied by 1e8, which makes the factor negligible. Its SEE is proportional to
#       N^(-1/2) exactly, so it is multiplied by 1e4 afterwards. Tight comparison.
#     * "kequate_default": the ordinary call with glm objects (N / (N - 1) variance). Loose.
#   kequate solves G(y) = F(x) with uniroot(tol = 1e-6).
# - "independent": the formulas of the book written out below, with uniroot(tol = 1e-14), and a
#   SEE whose Jacobian is a finite difference (it does not use the analytic derivative formulas
#   of the book).
# - SNSequate (Gonzalez, 2014): ker.eq() and bandwidth(), which use the book's variance and the
#   book's second penalty term.
suppressMessages({
  library(equate)   # ACTmath data only
  library(kequate)
  library(SNSequate)
  library(jsonlite)
})

strict <- glm.control(epsilon = 1e-13, maxit = 200)
BIG <- 1e8

# ----------------------------------------------------------------- independent implementation
kernel_cdf <- function(v, scores, p, h) {
  mu <- sum(scores * p)
  s2 <- sum((scores - mu)^2 * p)
  a <- sqrt(s2 / (s2 + h^2))
  vapply(v, function(z) sum(p * pnorm((z - a * scores - (1 - a) * mu) / (a * h))), numeric(1))
}
kernel_density <- function(v, scores, p, h) {
  mu <- sum(scores * p)
  s2 <- sum((scores - mu)^2 * p)
  a <- sqrt(s2 / (s2 + h^2))
  vapply(v, function(z) sum(p * dnorm((z - a * scores - (1 - a) * mu) / (a * h))) / (a * h),
         numeric(1))
}
kernel_density_derivative <- function(v, scores, p, h) {
  mu <- sum(scores * p)
  s2 <- sum((scores - mu)^2 * p)
  a <- sqrt(s2 / (s2 + h^2))
  vapply(v, function(z) {
    rr <- (z - a * scores - (1 - a) * mu) / (a * h)
    -sum(p * rr * dnorm(rr)) / (a * h)^2
  }, numeric(1))
}
kernel_equate <- function(xs, r, hx, ys, s, hy) {
  u <- kernel_cdf(xs, xs, r, hx)
  vapply(u, function(target) {
    uniroot(function(y) kernel_cdf(y, ys, s, hy) - target,
            c(min(ys) - 100, max(ys) + 100), tol = 1e-14)$root
  }, numeric(1))
}
# Covariance of the fitted probabilities of a log-linear model with design matrix B (no
# intercept): W B (B' W B)^-1 B' W / N, W = D_r - r r' (Holland & Thayer, 1987).
loglinear_covariance <- function(p, B, n) {
  W <- diag(p) - tcrossprod(p)
  W %*% B %*% solve(t(B) %*% W %*% B, t(B) %*% W) / n
}
# SEE with a finite-difference Jacobian (five-point stencil; a two-point difference with step
# 1e-6 is off by 1e-3 where a score probability is as small as 5e-6). Perturbed probabilities
# are not renormalized (the covariance has zero row sums, so the direction along the constraint
# does not matter).
numeric_see <- function(xs, r, hx, cov_r, ys, s, hy, cov_s, step = 1e-7) {
  stencil <- function(f) (-f(2 * step) + 8 * f(step) - 8 * f(-step) + f(-2 * step)) / (12 * step)
  shifted <- function(p, j, t) { p[j] <- p[j] + t; p }
  jac_r <- sapply(seq_along(r), function(j) {
    stencil(function(t) kernel_equate(xs, shifted(r, j, t), hx, ys, s, hy))
  })
  jac_s <- sapply(seq_along(s), function(k) {
    stencil(function(t) kernel_equate(xs, r, hx, ys, shifted(s, k, t), hy))
  })
  sqrt(rowSums((jac_r %*% cov_r) * jac_r) + rowSums((jac_s %*% cov_s) * jac_s))
}
# Penalty of ch. 4 (scores with unit increment).
penalty <- function(h, scores, p, weight) {
  pen1 <- sum((p - kernel_density(scores, scores, p, h))^2)
  left <- kernel_density_derivative(scores - 0.25, scores, p, h)
  right <- kernel_density_derivative(scores + 0.25, scores, p, h)
  pen1 + weight * sum(left < 0 & right > 0)
}
penalty_minimum <- function(scores, p, weight) {
  grid <- exp(seq(log(0.05), log(10), length.out = 2000))
  values <- vapply(grid, penalty, numeric(1), scores = scores, p = p, weight = weight)
  i <- which.min(values)
  opt <- optimize(penalty, c(grid[max(i - 1, 1)], grid[min(i + 1, length(grid))]),
                  scores = scores, p = p, weight = weight, tol = 1e-12)
  if (opt$objective <= values[i]) opt$minimum else grid[i]
}

# ----------------------------------------------------------------- cases
tabulate_scores <- function(v, scores) as.numeric(table(factor(v, levels = scores)))

make_case <- function(name, xs, xf, ys, yf, degree_x, degree_y, hx, hy, sns = FALSE, note = "") {
  n <- sum(xf)
  m <- sum(yf)
  out <- list(name = name, note = note, x_scores = xs, x_frequencies = xf, y_scores = ys,
              y_frequencies = yf, bandwidth_x = hx, bandwidth_y = hy)
  smoothed <- !is.null(degree_x)
  if (smoothed) {
    glm_x <- glm(xf ~ poly(xs, degree_x, raw = TRUE), family = poisson, x = TRUE, control = strict)
    glm_y <- glm(yf ~ poly(ys, degree_y, raw = TRUE), family = poisson, x = TRUE, control = strict)
    r <- unname(fitted(glm_x)) / n
    s <- unname(fitted(glm_y)) / m
    dm_x <- glm_x$x[, -1, drop = FALSE]
    dm_y <- glm_y$x[, -1, drop = FALSE]
    # Orthogonal polynomials span the same space as the raw powers given to kequate and keep
    # B' W B well conditioned.
    cov_r <- loglinear_covariance(r, poly(xs, degree_x), n)
    cov_s <- loglinear_covariance(s, poly(ys, degree_y), m)
    out$degree_x <- degree_x
    out$degree_y <- degree_y
    book <- kequate("EG", xs, ys, r, s, dm_x, dm_y, n * BIG, m * BIG, hx = hx, hy = hy)
    book_auto <- kequate("EG", xs, ys, r, s, dm_x, dm_y, n * BIG, m * BIG)
    default_fixed <- kequate("EG", xs, ys, glm_x, glm_y, hx = hx, hy = hy)
    default_auto <- kequate("EG", xs, ys, glm_x, glm_y)
    # kequate's covariance of the fitted probabilities, C C'.
    c_r <- kequate:::cmatrixSG(r, dm_x, n)
    out$covariance_r <- c_r %*% t(c_r)
  } else {
    r <- xf / n
    s <- yf / m
    cov_r <- (diag(r) - tcrossprod(r)) / n
    cov_s <- (diag(s) - tcrossprod(s)) / m
    book <- kequate("EG", xs, ys, r, s, N = n * BIG, M = m * BIG, hx = hx, hy = hy,
                    smoothed = FALSE)
    book_auto <- kequate("EG", xs, ys, r, s, N = n * BIG, M = m * BIG, smoothed = FALSE)
    default_fixed <- kequate("EG", xs, ys, r, s, N = n, M = m, hx = hx, hy = hy,
                             smoothed = FALSE)
    default_auto <- kequate("EG", xs, ys, r, s, N = n, M = m, smoothed = FALSE)
  }
  out$r <- r
  out$s <- s
  out$kequate_book <- list(
    equated = book@equating$eqYx, see = book@equating$SEEYx * sqrt(BIG),
    cdf_x = book@scores$X$cdfx, cdf_y = book@scores$Y$cdfy, pre = book@PRE$PREYx,
    penalty_bandwidth_x = book_auto@h$hx, penalty_bandwidth_y = book_auto@h$hy)
  out$kequate_default <- list(
    equated = default_fixed@equating$eqYx, see = default_fixed@equating$SEEYx,
    penalty_bandwidth_x = default_auto@h$hx, penalty_bandwidth_y = default_auto@h$hy,
    auto_equated = default_auto@equating$eqYx, auto_see = default_auto@equating$SEEYx)
  h_grid <- c(0.3, 0.45, 0.6, 0.8, 1.2, 2.5)
  out$independent <- list(
    equated = kernel_equate(xs, r, hx, ys, s, hy),
    see = numeric_see(xs, r, hx, cov_r, ys, s, hy, cov_s),
    density_x = kernel_density(xs, xs, r, hx),
    penalty_grid = h_grid,
    penalty_x_weight0 = vapply(h_grid, penalty, numeric(1), scores = xs, p = r, weight = 0),
    penalty_x_weight1 = vapply(h_grid, penalty, numeric(1), scores = xs, p = r, weight = 1),
    penalty_bandwidth_x_weight0 = penalty_minimum(xs, r, 0),
    penalty_bandwidth_y_weight0 = penalty_minimum(ys, s, 0),
    penalty_bandwidth_x_weight1 = penalty_minimum(xs, r, 1),
    penalty_bandwidth_y_weight1 = penalty_minimum(ys, s, 1))
  if (sns) {
    # SNSequate presmooths internally with the same polynomial degrees.
    # ker.eq() searches the root in (-1, 101) only and stops when an equated score falls below
    # -1; its equated scores and SEE are recorded when it succeeds.
    sns_fit <- tryCatch(
      ker.eq(scores = cbind(xf, yf), kert = "gauss", hx = hx, hy = hy,
             degree = c(degree_x, degree_y), design = "EG"),
      error = function(e) NULL)
    out$snsequate <- list(
      equated = if (is.null(sns_fit)) NULL else as.numeric(sns_fit$eqYx),
      see = if (is.null(sns_fit)) NULL else as.numeric(sns_fit$SEEYx),
      penalty_bandwidth_x_weight1 = bandwidth(scores = xf, kert = "gauss", degree = degree_x,
                                              design = "EG", Kp = 1)$h,
      penalty_bandwidth_y_weight1 = bandwidth(scores = yf, kert = "gauss", degree = degree_y,
                                              design = "EG", Kp = 1)$h,
      penalty_bandwidth_x_weight0 = bandwidth(scores = xf, kert = "gauss", degree = degree_x,
                                              design = "EG", Kp = 0)$h)
  }
  out
}

cases <- list()
cases[[1]] <- make_case("act_math_smoothed", ACTmath$scale, ACTmath$xcount, ACTmath$scale,
                        ACTmath$ycount, 6, 6, 0.62, 0.58, sns = TRUE,
                        note = "equate::ACTmath, log-linear degree 6 for both forms")
set.seed(20261126)
xs <- 0:30
ys <- 0:50
xf <- tabulate_scores(rbinom(1800, 30, rbeta(1800, 5, 3.5)), xs)
yf <- tabulate_scores(rbinom(2500, 50, rbeta(2500, 4.5, 3.5)), ys)
cases[[2]] <- make_case("unequal_lengths_smoothed", xs, xf, ys, yf, 4, 5, 0.55, 0.7,
                        note = "X 0..30 (N = 1800, degree 4), Y 0..50 (N = 2500, degree 5)")
cases[[3]] <- make_case("act_math_observed", ACTmath$scale, ACTmath$xcount, ACTmath$scale,
                        ACTmath$ycount, NULL, NULL, 0.62, 0.58,
                        note = "equate::ACTmath without presmoothing (multinomial covariance)")
set.seed(5)
xs <- 0:20
xf <- tabulate_scores(rbinom(600, 20, rbeta(600, 3, 2)), xs)
yf <- tabulate_scores(rbinom(650, 20, rbeta(650, 2.5, 2.5)), xs)
cases[[4]] <- make_case("small_sample_smoothed", xs, xf, xs, yf, 3, 3, 0.5, 2.0, sns = TRUE,
                        note = "N = 600 / 650, degree 3; unequal bandwidths")

out <- list(
  meta = list(
    generator = "validation/r/equate_e3_reference.R",
    r_version = R.version.string,
    kequate_version = as.character(packageVersion("kequate")),
    snsequate_version = as.character(packageVersion("SNSequate")),
    sample_size_factor = BIG,
    note = paste("kequate_book: kequate with sample sizes times 1e8 (variance factor",
                 "N / (N - 1) negligible), SEE rescaled by 1e4. kequate_default: ordinary call.",
                 "independent: formulas of von Davier et al. (2004) with a strict root",
                 "tolerance and a finite-difference SEE.")
  ),
  cases = cases
)
path <- "packages/flexpyequate/tests/reference/e3_kernel.json"
writeLines(toJSON(out, auto_unbox = TRUE, digits = NA, pretty = TRUE), path)
cat("wrote", path, "\n")
for (case in cases) {
  cat(sprintf("%-26s |keq - indep| eq %.1e  see rel %.1e | default eq %.1e see rel %.1e\n",
              case$name,
              max(abs(case$kequate_book$equated - case$independent$equated)),
              max(abs(case$kequate_book$see / case$independent$see - 1)),
              max(abs(case$kequate_default$equated - case$independent$equated)),
              max(abs(case$kequate_default$see / case$independent$see - 1))))
  cat(sprintf("   h(K=0): kequate %.6f / %.6f  indep %.6f / %.6f | h(K=1) indep %.6f / %.6f\n",
              case$kequate_book$penalty_bandwidth_x, case$kequate_book$penalty_bandwidth_y,
              case$independent$penalty_bandwidth_x_weight0,
              case$independent$penalty_bandwidth_y_weight0,
              case$independent$penalty_bandwidth_x_weight1,
              case$independent$penalty_bandwidth_y_weight1))
  if (!is.null(case$snsequate)) {
    if (is.null(case$snsequate$equated)) cat("   SNSequate ker.eq failed\n")
    else cat(sprintf("   SNSequate: eq %.1e  see rel %.1e\n",
                     max(abs(case$snsequate$equated - case$independent$equated)),
                     max(abs(case$snsequate$see / case$independent$see - 1))))
    cat(sprintf("   SNSequate: h(K=1) %.6f / %.6f  h(K=0) x %.6f\n",
                case$snsequate$penalty_bandwidth_x_weight1,
                case$snsequate$penalty_bandwidth_y_weight1,
                case$snsequate$penalty_bandwidth_x_weight0))
  }
}
