# flexpyequate E2 reference values (packages/flexpyequate/docs/DESIGN.md §7, case Q3):
# polynomial log-linear presmoothing of univariate and bivariate score distributions.
# Run from the repository root:  Rscript validation/r/equate_e2_reference.R
# Writes packages/flexpyequate/tests/reference/e2_presmoothing.json. CI does not run R.
#
# Two references:
# - stats::glm(family = poisson) with a strict convergence tolerance (epsilon = 1e-13) on
#   orthogonal polynomials (the fitted values do not depend on the polynomial basis). Tight.
# - equate::presmoothing(smoothmethod = "loglinear") and equate(..., smoothmethod = "loglinear"),
#   which call glm with its default tolerance (epsilon = 1e-8 on the deviance). Looser.
suppressMessages({
  library(equate)
  library(jsonlite)
})

strict <- glm.control(epsilon = 1e-13, maxit = 200)

glm_summary <- function(fit) {
  n <- fit$y
  m <- fitted(fit)
  list(fitted = unname(m),
       likelihood_ratio = deviance(fit),
       pearson = sum((n - m)^2 / m),
       freeman_tukey = sum((sqrt(n) + sqrt(n + 1) - sqrt(4 * m + 1))^2),
       degrees_of_freedom = df.residual(fit),
       log_likelihood = as.numeric(logLik(fit)),
       aic = AIC(fit),
       converged = fit$converged)
}

univariate_case <- function(name, scores, freq, degrees, note = "") {
  tab <- as.freqtab(cbind(scores, freq))
  fits <- lapply(degrees, function(d) {
    fit <- glm(freq ~ poly(scores, d), family = poisson, control = strict)
    out <- glm_summary(fit)
    out$degree <- d
    out$equate_fitted <- as.numeric(presmoothing(tab, smoothmethod = "loglinear", degrees = d))
    out
  })
  list(name = name, note = note, scores = scores, frequencies = freq, fits = fits)
}

tabulate_scores <- function(v, scores) as.numeric(table(factor(v, levels = scores)))

univariate <- list()
univariate[[1]] <- univariate_case("act_math_x", ACTmath$scale, ACTmath$xcount, c(2, 4, 6, 8),
                                   note = "equate::ACTmath form X")
univariate[[2]] <- univariate_case("act_math_y", ACTmath$scale, ACTmath$ycount, c(2, 4, 6, 8),
                                   note = "equate::ACTmath form Y")
set.seed(31)
scores <- seq(10, 30, by = 0.5)
univariate[[3]] <- univariate_case(
  "half_points", scores, tabulate_scores(10 + rbinom(700, 40, rbeta(700, 6, 4)) / 2, scores),
  c(3, 5), note = "scores 10..30 by 0.5, N = 700, zero frequencies in the tails")

# Equipercentile equating of the presmoothed ACT forms (Kolen & Brennan, 2014, section 3.1).
xt <- as.freqtab(cbind(ACTmath$scale, ACTmath$xcount))
yt <- as.freqtab(cbind(ACTmath$scale, ACTmath$ycount))
smoothed_equating <- lapply(c(4, 6), function(d) {
  eq <- equate(xt, yt, type = "equipercentile", smoothmethod = "loglinear", degrees = d)
  list(degree = d, equated = eq$concordance$yx)
})

# Bivariate: the same examinees take X (0..20) and Y (0..25).
set.seed(99)
n <- 2000
theta <- rnorm(n)
xs <- 0:20
ys <- 0:25
xv <- rbinom(n, 20, plogis(0.9 * theta + 0.3))
yv <- rbinom(n, 25, plogis(1.0 * theta - 0.2))
joint <- unclass(table(factor(xv, levels = xs), factor(yv, levels = ys)))
dimnames(joint) <- NULL
# Row-major cells (X slowest), as flexpyequate flattens the table.
cells <- data.frame(x = rep(xs, each = length(ys)), y = rep(ys, times = length(xs)),
                    n = as.vector(t(joint)))
bivariate_fit <- function(formula, degree, cross_terms) {
  fit <- glm(formula, family = poisson, data = cells, control = strict)
  out <- glm_summary(fit)
  out$fitted <- matrix(out$fitted, nrow = length(xs), byrow = TRUE)
  out$degree <- degree
  out$cross_terms <- cross_terms
  out
}
bivariate_fits <- list(
  bivariate_fit(n ~ poly(x, 4) + poly(y, 4) + I(x * y), c(4, 4), list(c(1, 1))),
  bivariate_fit(n ~ poly(x, 3) + poly(y, 5), c(3, 5), list()),
  bivariate_fit(n ~ poly(x, 5) + poly(y, 5) + I(x * y) + I(x^2 * y) + I(x * y^2) + I(x^2 * y^2),
                c(5, 5), list(c(1, 1), c(2, 1), c(1, 2), c(2, 2)))
)
# equate's bivariate presmoothing for the first model: degrees = list(c(4, 4), c(1, 1)).
long <- cbind(expand.grid(x = xs, y = ys), n = as.vector(joint))
xyt <- as.freqtab(long, scales = list(xs, ys), design = "sg")
equate_joint <- presmoothing(xyt, smoothmethod = "loglinear", degrees = list(c(4, 4), c(1, 1)))
bivariate_fits[[1]]$equate_fitted <- matrix(as.numeric(equate_joint), nrow = length(xs))

out <- list(
  meta = list(
    generator = "validation/r/equate_e2_reference.R",
    r_version = R.version.string,
    equate_version = as.character(packageVersion("equate")),
    glm_epsilon = strict$epsilon,
    note = paste("fitted etc. come from stats::glm with epsilon = 1e-13;",
                 "equate_fitted from equate::presmoothing (glm default tolerance).")
  ),
  univariate = univariate,
  smoothed_equating = list(x_scores = ACTmath$scale, x_frequencies = ACTmath$xcount,
                           y_scores = ACTmath$scale, y_frequencies = ACTmath$ycount,
                           results = smoothed_equating),
  bivariate = list(name = "single_group", x_scores = xs, y_scores = ys,
                   joint_frequencies = joint, fits = bivariate_fits)
)
path <- "packages/flexpyequate/tests/reference/e2_presmoothing.json"
writeLines(toJSON(out, auto_unbox = TRUE, digits = NA, pretty = TRUE), path)
cat("wrote", path, "\n")
for (case in univariate) for (fit in case$fits) {
  cat(sprintf("%-12s degree %d converged %s  max rel |equate - glm| = %.2e\n", case$name,
              fit$degree, fit$converged, max(abs(fit$equate_fitted / fit$fitted - 1))))
}
for (fit in bivariate_fits) cat("bivariate converged", fit$converged, "\n")
cat(sprintf("bivariate max rel |equate - glm| = %.2e\n",
            max(abs(bivariate_fits[[1]]$equate_fitted / bivariate_fits[[1]]$fitted - 1))))
