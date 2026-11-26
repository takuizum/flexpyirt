# Weighted and cluster-robust sandwich covariances (VALIDATION.md V57, DESIGN.md §17.6).
# Run from the repository root:  Rscript validation/r/v57_weighted_cluster_se_reference.R [case ...]
# Writes tests/reference/v57_<case>.json. CI does not run R; the JSON files are committed.
#
# mirt has no design-based weighted sandwich and no cluster-robust sandwich: with survey.weights,
# mirt 1.48 builds the middle matrix of SE.type = "crossprod" / "sandwich" from Data$Freq, the sums
# of the weights (dpars.cpp computeInfo: add2outer(Igrad, grad, rs(g, pat))), i.e.
# sum_i w_i s_i s_i^T. The design-based middle matrix is sum_i w_i^2 s_i s_i^T (Binder, 1983). So
# every case stores
#   - mirt's estimates and its SE.type = "Oakes" covariance with survey.weights (the bread,
#     H_w^-1),
#   - mirt's SE.type = "crossprod" and "sandwich" with survey.weights (checks of flexpyirt's
#     se = "crossprod" with weights, and of the reading of mirt's sandwich above),
#   - per-person scores s_i: numDeriv::jacobian of each person's marginal log-likelihood on mirt's
#     own quadrature grid and prior (independent of mirt's derivative code), checked against
#     mirt's estfun.AllModelClass(weights = 1),
#   - the reference covariances, computed here:
#       weighted:  V = H^-1 (sum_i w_i^2 s_i s_i^T) H^-1
#       cluster:   V = H^-1 (G / (G - 1) sum_c u_c u_c^T) H^-1,  u_c = sum_{i in c} w_i s_i
#       unweighted cluster (w_i = 1), and singleton clusters (= N / (N - 1) times weighted).
# Covariance rows are labelled by the first parnum of each (possibly constrained) parameter as
# in vcov(), so the tests can map them to flexpyirt's free_id.
suppressMessages({
  library(mirt)
  library(jsonlite)
  library(numDeriv)
})

out_dir <- file.path("tests", "reference")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
TOL <- 1e-10
quad <- list(quadpts = 61, theta_lim = c(-6, 6))
common <- list(TOL = TOL, accelerate = "none", optimizer = "NR", verbose = FALSE,
               quadpts = quad$quadpts, technical = list(NCYCLES = 100000, theta_lim = quad$theta_lim))

parnum_of_rows <- function(v) {
  as.integer(vapply(strsplit(rownames(v), ".", fixed = TRUE), function(x) x[2], ""))
}

# Per-person marginal log-likelihood of a 2PL with a normal prior on mirt's grid.
person_loglik <- function(theta, y, a, d, mu, var) {
  prior <- dnorm(theta, mu, sqrt(var))
  prior <- prior / sum(prior)
  logp <- rep(0, length(theta))
  for (j in seq_along(a)) {
    if (is.na(y[j])) next
    p <- plogis(a[j] * theta + d[j])
    logp <- logp + if (y[j] == 1) log(p) else log1p(-p)
  }
  m <- max(logp)
  m + log(sum(prior * exp(logp - m)))
}

# Scores of every person with respect to every parnum (zero outside the person's group).
numderiv_scores <- function(m, dat, grp) {
  values <- mod2values(m)
  theta <- as.vector(m@Model$Theta)
  stopifnot(all(range(theta) == quad$theta_lim), length(theta) == quad$quadpts)
  items <- colnames(dat)
  groups <- unique(as.character(values$group))
  if (is.null(grp)) grp <- rep(groups[1], nrow(dat))
  n_par <- max(values$parnum)
  scores <- matrix(0, nrow(dat), n_par)
  for (g in groups) {
    vg <- values[values$group == g, ]
    pick_par <- function(item, name) vg[vg$item == item & vg$name == name, ]
    a_rows <- lapply(items, pick_par, name = "a1")
    d_rows <- lapply(items, pick_par, name = "d")
    mu_row <- pick_par("GROUP", "MEAN_1"); var_row <- pick_par("GROUP", "COV_11")
    par_rows <- c(a_rows, d_rows, list(mu_row, var_row))
    x0 <- vapply(par_rows, function(r) r$value, 0)
    pn <- vapply(par_rows, function(r) r$parnum, 0L)
    J <- length(items)
    rows <- which(grp == g)
    for (i in rows) {
      f <- function(x) person_loglik(theta, dat[i, ], x[1:J], x[J + 1:J], x[2 * J + 1], x[2 * J + 2])
      scores[i, pn] <- jacobian(f, x0, method = "Richardson",
                                method.args = list(eps = 1e-4, d = 1e-4, r = 6))
    }
  }
  scores
}

# Collapse parnum-space scores to the rows of vcov() (sum over each equality constraint).
collapse_scores <- function(m, scores, v) {
  constrain <- extract.mirt(m, "constrain")
  first <- parnum_of_rows(v)
  out <- matrix(0, nrow(scores), length(first))
  for (k in seq_along(first)) {
    members <- first[k]
    for (cset in constrain) if (first[k] %in% cset) members <- cset
    out[, k] <- rowSums(scores[, members, drop = FALSE])
  }
  out
}

fit_all <- function(fit_fun) {
  types <- c("Oakes", "crossprod", "sandwich")
  fits <- lapply(types, function(t) suppressMessages(fit_fun(t)))
  names(fits) <- types
  for (t in types) {
    stopifnot(isTRUE(extract.mirt(fits[[t]], "converged")))
    stopifnot(max(abs(mod2values(fits[[t]])$value - mod2values(fits$Oakes)$value)) < 1e-10)
  }
  fits
}

block <- function(v, first) {
  stopifnot(all(is.finite(v)))
  list(parnum = first, matrix = unname(v))
}

write_case <- function(case, description, dat, grp, weights, cluster, fit_fun, extra = list()) {
  fits <- fit_all(fit_fun)
  m <- fits$Oakes
  v_oakes <- vcov(m)
  first <- parnum_of_rows(v_oakes)
  for (t in c("crossprod", "sandwich")) stopifnot(identical(parnum_of_rows(vcov(fits[[t]])), first))
  s_full <- numderiv_scores(m, dat, grp)
  s <- collapse_scores(m, s_full, v_oakes)
  # Check against mirt's analytic person scores (estfun, unweighted) and the score equation.
  s_mirt <- estfun.AllModelClass(m, weights = rep(1, nrow(dat)))
  colnames(s_mirt) <- NULL
  score_diff <- max(abs(s - s_mirt))
  stopifnot(score_diff < 1e-6)
  gradient <- max(abs(colSums(weights * s)))
  stopifnot(gradient < 1e-4)
  # mirt's sandwich with survey.weights is H^-1 (sum_i w_i s_i s_i^T) H^-1 (see the header).
  mirt_meat <- crossprod(sqrt(weights) * s)
  mirt_reading <- max(abs(v_oakes %*% mirt_meat %*% v_oakes - vcov(fits$sandwich))) /
    max(abs(vcov(fits$sandwich)))
  stopifnot(mirt_reading < 1e-5)
  sandwich <- function(meat) { out <- v_oakes %*% meat %*% v_oakes; (out + t(out)) / 2 }
  ws <- weights * s
  totals <- rowsum(ws, cluster, reorder = FALSE)
  G <- nrow(totals)
  utotals <- rowsum(s, cluster, reorder = FALSE)
  N <- nrow(dat)
  res <- c(list(
    meta = list(case = case, description = description, r_version = R.version.string,
                mirt_version = as.character(packageVersion("mirt")),
                numDeriv_version = as.character(packageVersion("numDeriv")), TOL = TOL,
                quadpts = quad$quadpts, theta_lim = quad$theta_lim,
                grid_check = range(m@Model$Theta), accelerate = "none", optimizer = "NR",
                converged = TRUE, iterations = extract.mirt(m, "iterations"),
                max_abs_numderiv_minus_estfun = score_diff, max_abs_weighted_score = gradient,
                mirt_sandwich_reading_rel = mirt_reading, n_clusters = G, small_sample = "G/(G-1)"),
    items = colnames(dat),
    responses = unname(dat),
    group = if (is.null(grp)) NULL else grp,
    weights = weights,
    cluster = cluster,
    loglik = extract.mirt(m, "logLik"),
    values = lapply(seq_len(nrow(mod2values(m))), function(i) {
      v <- mod2values(m)[i, ]
      list(group = as.character(v$group), item = as.character(v$item), name = as.character(v$name),
           parnum = v$parnum, value = v$value, est = v$est)
    }),
    vcov = list(
      oakes = block(v_oakes, first),
      mirt_crossprod = block(vcov(fits$crossprod), first),
      mirt_sandwich = block(vcov(fits$sandwich), first),
      weighted = block(sandwich(crossprod(ws)), first),
      cluster = block(sandwich(G / (G - 1) * crossprod(totals)), first),
      unweighted_cluster_meat = block(G / (G - 1) * crossprod(utotals), first),
      singleton = block(sandwich(N / (N - 1) * crossprod(ws)), first)
    )
  ), extra)
  path <- file.path(out_dir, paste0("v57_", case, ".json"))
  write_json(res, path, auto_unbox = TRUE, digits = NA, pretty = FALSE, na = "null")
  cat(sprintf("wrote %s loglik=%.6f iter=%d G=%d numDeriv-estfun=%.2e mirt-reading=%.2e\n",
              path, res$loglik, res$meta$iterations, G, score_diff, mirt_reading))
  print(rbind(oakes = sqrt(diag(v_oakes))[1:4], weighted = sqrt(diag(res$vcov$weighted$matrix))[1:4],
              cluster = sqrt(diag(res$vcov$cluster$matrix))[1:4]))
}

RUN <- commandArgs(trailingOnly = TRUE)
run <- function(case) length(RUN) == 0 || case %in% RUN

# ---- 2PL, 40 schools of 25, school effect in theta, unequal weights, 3% MCAR ----------------
if (run("school_2pl")) {
  set.seed(20261201)
  J <- 10; G <- 40; n_per <- 25; N <- G * n_per
  school <- rep(sprintf("s%02d", 1:G), each = n_per)
  theta <- rep(rnorm(G, 0, 0.6), each = n_per) + rnorm(N, 0, 0.8)
  a <- runif(J, 0.8, 2.0); d <- rnorm(J, 0, 0.9)
  dat <- simdata(matrix(a), matrix(d), N, itemtype = "2PL", Theta = matrix(theta))
  dat[matrix(runif(N * J) < 0.03, N, J)] <- NA
  colnames(dat) <- sprintf("q%02d", 1:J)
  w <- rep(runif(G, 0.5, 2.5), each = n_per) * runif(N, 0.5, 1.5)
  write_case("school_2pl", "2PL, 40 schools x 25, school effect, unequal weights, 3% MCAR, J=10",
             dat, NULL, w, school,
             function(t) do.call(mirt, c(list(dat, 1, itemtype = "2PL", SE = TRUE, SE.type = t,
                                              survey.weights = w), common)))
}

# ---- two groups with anchors, clusters that span the groups ---------------------------------
if (run("twogroup")) {
  set.seed(20261202)
  J <- 8; G <- 30; n_per <- 40; N <- G * n_per
  school <- rep(sprintf("s%02d", 1:G), each = n_per)
  grp <- ifelse(runif(N) < 0.5, "ref", "focal")
  theta <- rep(rnorm(G, 0, 0.5), each = n_per) + rnorm(N, 0, 0.85) + ifelse(grp == "focal", 0.3, 0)
  a <- runif(J, 0.8, 2.0); d <- rnorm(J, 0, 0.9)
  d_f <- d; d_f[6:J] <- d[6:J] + rnorm(J - 5, 0, 0.4)
  p <- plogis(outer(theta, a) + ifelse(grp == "focal", 1, 0) %o% d_f + ifelse(grp == "ref", 1, 0) %o% d)
  dat <- (matrix(runif(N * J), N, J) < p) * 1
  colnames(dat) <- sprintf("q%02d", 1:J)
  w <- rep(runif(G, 0.5, 2.0), each = n_per) * ifelse(grp == "focal", 1.5, 1) * runif(N, 0.7, 1.3)
  anchors <- sprintf("q%02d", 1:5)
  write_case("twogroup", "2PL, ref/focal, anchors q01-q05, focal mean/var free, 30 schools spanning groups",
             dat, grp, w, school,
             function(t) do.call(multipleGroup, c(list(dat, 1, group = factor(grp, levels = c("ref", "focal")),
                                                       itemtype = "2PL",
                                                       invariance = c(anchors, "free_means", "free_var"),
                                                       SE = TRUE, SE.type = t, survey.weights = w),
                                                  common)),
             extra = list(anchors = anchors))
}
