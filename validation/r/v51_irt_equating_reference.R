# V51 reference values: IRT true-score and observed-score equating and raw-to-scale tables
# (VALIDATION.md V51, DESIGN.md §20, issue #106).
# Run from the repository root:  Rscript validation/r/v51_irt_equating_reference.R
# Writes tests/reference/v51_irt_equating.json. CI does not run R; the JSON is committed.
#
# No parameters are estimated: the item parameters are chosen here and stored, and the Python
# tests fix a flexpyirt FitResult at them (latent N(0, 1), 61 equispaced nodes on [-6, 6]). The
# link puts the new form's scale onto the old one's: theta_old = A * theta_new + B.
#
# Case "dichotomous" (20 + 20 items, 3PL and 2PL, D = 1):
#   - true-score equating (TSE): equateIRT's own solver (equateIRT:::score_compute, Newton to
#     |d theta| < 1e-6, i.e. ~1e-12 after the quadratic convergence) for the scores inside the
#     range of the new form's test characteristic curve, and SNSequate::irt.eq(method = "TS")
#     for every score (its Kolen-Brennan linear interpolation below the sum of the guessing
#     parameters, and the maximum score; its uniroot() has the default tolerance ~1.2e-4 in
#     theta, so its interior values are a loose cross-check only).
#   - observed-score equating (OSE): equateIRT:::score_compute(method = "OSE") with the fitted
#     grid and weights for both populations and synthetic weights w = 1, 0, 0.5.
#     equateIRT works on the new form's scale (X) with Y's items transformed by
#     theta_X = A_e theta_Y + B_e, A_e = 1 / A, B_e = -B / A; equated scores do not depend on
#     the common scale.
# Case "polytomous" (new: 4 GRM K = 4 + 6 2PL; old: 3 GPCM K = 3 + 2 GRM K = 4 + 5 3PL + 3 2PL):
#   no package implements IRT equating for polytomous items, so the reference is built in R from
#   mirt (mirt objects at the chosen parameters, TOL = NaN):
#   - TSE: uniroot(tol = 1e-13) on mirt::expected.test() of the new form, then expected.test() of
#     the old form at A * theta + B.
#   - Model-implied summed-score distributions: probtrace() on the population nodes and the
#     Lord-Wingersky recursion written here, checked against mirt's
#     fscores(method = "EAPsum", full.scores = FALSE) `expected` / N in the forms' own
#     populations (stored as well).
#   - OSE: equate::equate(type = "equipercentile") on the two synthetic distributions.
#   - Raw-to-scale table of the old form: true score (uniroot on expected.test(), SE from
#     testinfo()) and EAPsum (fscores), then the linear transformation, clipping and round().
suppressMessages({
  library(mirt)
  library(equateIRT)
  library(SNSequate)
  library(equate)
  library(jsonlite)
})

out_dir <- file.path("tests", "reference")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

theta_lim <- c(-6, 6)
quadpts <- 61L
nodes <- seq(theta_lim[1], theta_lim[2], length.out = quadpts)
weights <- dnorm(nodes)
weights <- weights / sum(weights)

# ---------------------------------------------------------------------------------------------
# Case 1: dichotomous 3PL / 2PL
# ---------------------------------------------------------------------------------------------
set.seed(20261126)
dich_form <- function(prefix, n, n3) {
  a <- runif(n, 0.6, 2.0)
  b <- rnorm(n, 0, 1)
  g <- c(runif(n3, 0.10, 0.25), rep(0, n - n3))
  names(a) <- names(b) <- names(g) <- sprintf("%s%02d", prefix, seq_len(n))
  list(a = a, b = b, c = g)
}
dx <- dich_form("x", 20, 12)
dy <- dich_form("y", 20, 9)
d_link <- c(A = 1.15, B = -0.25)
A_e <- 1 / d_link[["A"]]
B_e <- -d_link[["B"]] / d_link[["A"]]

dich_params <- function(f) {
  lapply(setNames(seq_along(f$a), names(f$a)), function(j) {
    p <- list(a = f$a[[j]], d = -f$a[[j]] * f$b[[j]])
    if (f$c[[j]] > 0) p <- c(p, list(g = f$c[[j]], u = 1))
    p
  })
}
dich_types <- function(f) as.list(setNames(ifelse(f$c > 0, "3PL", "2PL"), names(f$a)))

itm_prepare <- list(
  diffX = dx$b, discrX = dx$a, guessX = dx$c,
  diffY = dy$b, discrY = dy$a, guessY = dy$c,
  diffY2X = dy$b * A_e + B_e, discrY2X = dy$a / A_e, guessY2X = dy$c
)
score_compute <- getFromNamespace("score_compute", "equateIRT")
tse_e <- score_compute(
  method = "TSE", itm_prepare = itm_prepare, D = 1, varFull = NULL, partial = NULL,
  varAB = NULL, itmp = 3, A = A_e, B = B_e, scores = NULL, se = FALSE, nq = 30, w = 0.5,
  theta = NULL, weights = NULL, names = c("new", "old"), printmessages = FALSE
)
ose_e <- lapply(c(new = 1, old = 0, synthetic = 0.5), function(w1) {
  out <- score_compute(
    method = "OSE", itm_prepare = itm_prepare, D = 1, varFull = NULL, partial = NULL,
    varAB = NULL, itmp = 3, A = A_e, B = B_e, scores = NULL, se = FALSE, nq = 30, w = w1,
    theta = nodes, weights = weights, names = c("new", "old"), printmessages = FALSE
  )
  list(w = w1, equated = out$old)
})
tse_sns <- irt.eq(
  20, dx, dy, A = d_link[["A"]], B = d_link[["B"]], method = "TS", D = 1
)

dichotomous <- list(
  new = list(items = names(dx$a), itemtypes = dich_types(dx), params = dich_params(dx)),
  old = list(items = names(dy$a), itemtypes = dich_types(dy), params = dich_params(dy)),
  link = as.list(d_link),
  sum_guessing = list(new = sum(dx$c), old = sum(dy$c)),
  tse_equateirt = list(
    score = tse_e$new,
    theta_new_scale = tse_e$theta,
    equated = tse_e$old
  ),
  tse_snsequate = list(
    score = 0:20,
    theta = tse_sns$theta_equivalent,
    equated = tse_sns$tau_y
  ),
  ose_equateirt = ose_e
)

# ---------------------------------------------------------------------------------------------
# Case 2: polytomous and mixed formats, built from mirt
# ---------------------------------------------------------------------------------------------
set.seed(20261127)
grm_d <- function(k) sort(rnorm(k - 1, 0, 1.2), decreasing = TRUE)
new_spec <- c(
  lapply(1:4, function(j) list(type = "graded", a = runif(1, 0.8, 2.0), d = grm_d(4))),
  lapply(1:6, function(j) list(type = "2PL", a = runif(1, 0.7, 1.8), d = rnorm(1)))
)
names(new_spec) <- sprintf("n%02d", seq_along(new_spec))
old_spec <- c(
  lapply(1:3, function(j) list(type = "gpcm", a = runif(1, 0.6, 1.6), d = c(0.4, 0.1) + rnorm(2, 0, 0.5))),
  lapply(1:2, function(j) list(type = "graded", a = runif(1, 0.8, 2.0), d = grm_d(4))),
  lapply(1:5, function(j) list(type = "3PL", a = runif(1, 0.8, 2.0), d = rnorm(1), g = runif(1, 0.1, 0.25))),
  lapply(1:3, function(j) list(type = "2PL", a = runif(1, 0.7, 1.8), d = rnorm(1)))
)
names(old_spec) <- sprintf("o%02d", seq_along(old_spec))
p_link <- c(A = 0.9, B = 0.3)

n_cats <- function(s) if (s$type %in% c("2PL", "3PL")) 2L else length(s$d) + 1L

mirt_at <- function(spec) {
  # Data only fixes the categories; every parameter is set below and TOL = NaN skips the EM.
  dat <- as.data.frame(lapply(spec, function(s) {
    k <- n_cats(s)
    sample(rep(0:(k - 1), length.out = 400))
  }))
  types <- vapply(spec, `[[`, "", "type")
  sv <- mirt(dat, 1, itemtype = types, pars = "values")
  for (it in names(spec)) {
    s <- spec[[it]]
    set <- function(name, value) sv$value[sv$item == it & sv$name == name] <<- value
    set("a1", s$a)
    if (s$type %in% c("2PL", "3PL")) {
      set("d", s$d)
      if (s$type == "3PL") set("g", s$g)
    } else {
      for (k in seq_along(s$d)) set(sprintf("d%d", k), s$d[k])
    }
  }
  mirt(dat, 1, itemtype = types, pars = sv, TOL = NaN, quadpts = quadpts,
       technical = list(theta_lim = theta_lim), verbose = FALSE)
}
m_new <- mirt_at(new_spec)
m_old <- mirt_at(old_spec)

flex_params <- function(spec) {
  lapply(spec, function(s) {
    if (s$type == "2PL") return(list(a = s$a, d = s$d))
    if (s$type == "3PL") return(list(a = s$a, d = s$d, g = s$g, u = 1))
    c(list(a = s$a), setNames(as.list(s$d), sprintf("d%d", seq_along(s$d))))
  })
}
flex_types <- function(spec) {
  lapply(spec, function(s) switch(s$type, graded = "GRM", gpcm = "GPCM", s$type))
}

tcc <- function(m, theta) as.numeric(expected.test(m, matrix(theta)))
max_new <- sum(vapply(new_spec, n_cats, 1L) - 1L)
max_old <- sum(vapply(old_spec, n_cats, 1L) - 1L)
lo_old <- sum(vapply(old_spec, function(s) if (s$type == "3PL") s$g else 0, 0))

solve_tcc <- function(m, x) {
  uniroot(function(t) tcc(m, t) - x, c(-40, 40), tol = 1e-13, maxiter = 1000)$root
}
interior_new <- 1:(max_new - 1)
theta_new <- vapply(interior_new, function(x) solve_tcc(m_new, x), 0)
theta_common <- p_link[["A"]] * theta_new + p_link[["B"]]
tse_poly <- list(
  score = 0:max_new,
  theta = c(-Inf, theta_common, Inf),
  equated = c(0, tcc(m_old, theta_common), max_old)
)

lord_wingersky <- function(m, theta) {
  traces <- lapply(seq_len(extract.mirt(m, "nitems")), function(j) {
    probtrace(extract.item(m, j), matrix(theta))
  })
  lik <- matrix(1, length(theta), 1)
  for (p in traces) {
    k <- ncol(p)
    new <- matrix(0, length(theta), ncol(lik) + k - 1)
    for (c in seq_len(k)) {
      cols <- seq_len(ncol(lik)) + c - 1
      new[, cols] <- new[, cols] + lik * p[, c]
    }
    lik <- new
  }
  lik
}
dist_at <- function(m, theta) as.numeric(colSums(lord_wingersky(m, theta) * weights))
eapsum <- function(m) {
  fscores(m, method = "EAPsum", full.scores = FALSE, quadpts = quadpts, theta_lim = theta_lim,
          verbose = FALSE)
}
tab_new <- eapsum(m_new)
tab_old <- eapsum(m_old)
dist <- list(
  new_in_new = dist_at(m_new, nodes),
  new_in_old = dist_at(m_new, (nodes - p_link[["B"]]) / p_link[["A"]]),
  old_in_new = dist_at(m_old, p_link[["A"]] * nodes + p_link[["B"]]),
  old_in_old = dist_at(m_old, nodes)
)
n_obs <- nrow(extract.mirt(m_new, "data"))
stopifnot(max(abs(dist$new_in_new - tab_new[, "expected"] / n_obs)) < 1e-12)
stopifnot(max(abs(dist$old_in_old - tab_old[, "expected"] / nrow(extract.mirt(m_old, "data")))) < 1e-12)

ose_poly <- lapply(c(new = 1, old = 0, synthetic = 0.5), function(w1) {
  f <- w1 * dist$new_in_new + (1 - w1) * dist$new_in_old
  g <- w1 * dist$old_in_new + (1 - w1) * dist$old_in_old
  eq <- equate(as.freqtab(cbind(0:max_new, f * 1e6)), as.freqtab(cbind(0:max_old, g * 1e6)),
               type = "equipercentile")
  list(w = w1, f = f, g = g, equated = eq$concordance$yx)
})

# Raw-to-scale table of the old form on its own scale.
slope <- 10; intercept <- 50; lowest <- 25; highest <- 75
interior_old <- (floor(lo_old) + 1):(max_old - 1)
theta_ts <- vapply(interior_old, function(x) solve_tcc(m_old, x), 0)
se_ts <- 1 / sqrt(as.numeric(testinfo(m_old, matrix(theta_ts))))
below <- 0:floor(lo_old)
ts_theta <- c(rep(-Inf, length(below)), theta_ts, Inf)
ts_se <- c(rep(NaN, length(below)), se_ts, NaN)
transform <- function(theta) pmin(pmax(slope * theta + intercept, lowest), highest)
scale_old <- list(
  slope = slope, intercept = intercept, lowest = lowest, highest = highest,
  true_score = list(
    score = 0:max_old, theta = ts_theta, theta_se = ts_se,
    scale_score = transform(ts_theta), scale_score_se = slope * ts_se,
    scale_score_rounded = round(transform(ts_theta))
  ),
  eapsum = list(
    score = tab_old[, "Sum.Scores"], theta = tab_old[, "F1"], theta_se = tab_old[, "SE_F1"],
    scale_score = transform(tab_old[, "F1"]), scale_score_se = slope * tab_old[, "SE_F1"],
    scale_score_rounded = round(transform(tab_old[, "F1"]))
  )
)

polytomous <- list(
  new = list(items = names(new_spec), itemtypes = flex_types(new_spec),
             params = flex_params(new_spec)),
  old = list(items = names(old_spec), itemtypes = flex_types(old_spec),
             params = flex_params(old_spec)),
  link = as.list(p_link),
  max_score = list(new = max_new, old = max_old),
  sum_guessing = list(new = 0, old = lo_old),
  tse = tse_poly,
  eapsum_expected_over_n = list(
    new = as.numeric(tab_new[, "expected"]) / n_obs,
    old = as.numeric(tab_old[, "expected"]) / nrow(extract.mirt(m_old, "data"))
  ),
  distributions = dist,
  ose = ose_poly,
  scale_old = scale_old
)

meta <- list(
  case = "v51_irt_equating",
  converged = TRUE,  # no estimation: every parameter is fixed
  theta_lim = theta_lim,
  quadpts = quadpts,
  mirt = as.character(packageVersion("mirt")),
  equateIRT = as.character(packageVersion("equateIRT")),
  SNSequate = as.character(packageVersion("SNSequate")),
  equate = as.character(packageVersion("equate")),
  R = R.version.string
)
out <- list(meta = meta, nodes = nodes, weights = weights,
            dichotomous = dichotomous, polytomous = polytomous)
write_json(out, file.path(out_dir, "v51_irt_equating.json"), auto_unbox = TRUE, digits = NA,
           pretty = TRUE)
cat("Wrote", file.path(out_dir, "v51_irt_equating.json"), "\n")
