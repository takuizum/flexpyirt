# M5 normal-mixture latent distribution reference values (VALIDATION.md V22) from R mirt.
# Run from the repository root:  Rscript validation/r/m5_mixture_reference.R
# Writes tests/reference/v22_mixture_*.json. CI does not run R; JSON files are committed.
#
# flexpyirt's NormalMixture (DESIGN.md §16.7) uses mirt's class parameterization for the shape
# (raw class 1 = N(0, 1); class c >= 2 has MEAN_1 = rmean_c, COV_11 = rvar_c, PI = logit_c =
# log(pi_c / pi_1)) but standardizes the whole mixture analytically: with raw moments
# M = sum_c pi_c rmean_c and V = sum_c pi_c (rvar_c + rmean_c^2) - M^2, theta = (z - M) / sqrt(V)
# has mean 0 and variance 1. Each component is normalized on the grid and then mixed, as mirt's
# dentype = "mixture-k" does. Hence flexpyirt's model on the grid theta_q (61 points on [-6, 6])
# is exactly mirt's multipleGroup(dentype = "mixture-k", invariance = <all items>, free_means,
# free_var) on the grid z_q = M + sqrt(V) theta_q, with
#   a = a_mirt sqrt(V),  d_k = d_k,mirt + s_k a_mirt M  (s_k = 1 for 2PL/GRM, s_k = k for GPCM).
# Each case records:
#   1. exact: flexpyirt's discretized MMLE re-coded independently in R (S parameterization on
#      [-6, 6]), reached by Newton polishing (central-difference gradient, Richardson Hessian) from
#      mirt's solution, with Richardson-Hessian SEs and EAPs computed in R.
#   2. mirt: built-in mirt(dentype = "mixture-k") refitted on z_q = M* + sqrt(V*) theta_q (the
#      image of flexpyirt's grid at the exact MMLE; asymmetric theta_lim), optimizer = "NR",
#      accelerate = "none" (V22c: 121 points and "squarem", see below), TOL = 1e-10, SE.type = "Oakes"; logLik, estimates and EAPs mapped to
#      flexpyirt's scale, and the Oakes covariance mapped by the delta method (numerical Jacobian
#      of the map above). R's own mirt-scale log-likelihood at mirt's estimates is stored as an
#      independent check of the discretization. mirt's Oakes SEs for mixtures are NOT the
#      observed-information SEs (they are close to the complete-data SEs with known classes), so
#      the SE reference is the R Richardson Hessian; mirt's Oakes and "complete" SEs and the R
#      Hessian SEs on mirt's own scale are stored as evidence (VALIDATION.md V22).
source(file.path("validation", "r", "setup.R"))  # mirt from GitHub; mirt_version / mirt_sha
suppressMessages({
  library(mirt)
  library(jsonlite)
})

out_dir <- file.path("tests", "reference")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
TOL <- 1e-10
LIM <- c(-6, 6)
# Grid and mirt acceleration are set per case below (QUADPTS, NODES, ACCEL are read at call time).
set_grid <- function(quadpts, accel) {
  QUADPTS <<- as.integer(quadpts)
  NODES <<- seq(LIM[1L], LIM[2L], length.out = QUADPTS)
  ACCEL <<- accel
}
set_grid(61L, "none")
CASES <- strsplit(Sys.getenv("V22_CASES", "abc"), "")[[1L]]

# --- mixture helpers ----------------------------------------------------------------------------

raw_shape <- function(shape) {
  sh <- matrix(shape, ncol = 3L, byrow = TRUE)
  rm <- c(0, sh[, 1L]); rv <- c(1, sh[, 2L]); lg <- c(0, sh[, 3L])
  p <- exp(lg - max(lg)); p <- p / sum(p)
  M <- sum(p * rm)
  V <- sum(p * (rv + rm^2)) - M^2
  list(rm = rm, rv = rv, pi = p, M = M, V = V)
}

# Components normalized on the grid, then mixed (mirt's per-class normalized densities).
mix_logw_from <- function(nodes, mu, s2, p) {
  if (any(!is.finite(s2)) || any(s2 <= 0)) return(rep(NaN, length(nodes)))
  w <- numeric(length(nodes))
  for (c in seq_along(mu)) {
    lk <- dnorm(nodes, mu[c], sqrt(s2[c]), log = TRUE)
    w <- w + p[c] * exp(lk - max(lk)) / sum(exp(lk - max(lk)))
  }
  log(w)
}

# flexpyirt's standardized mixture (mean 0, var 1) on flexpyirt's grid
mix_logw_std <- function(nodes, shape) {
  r <- raw_shape(shape)
  if (any(r$rv <= 0) || !(r$V > 0)) return(rep(NaN, length(nodes)))
  mix_logw_from(nodes, (r$rm - r$M) / sqrt(r$V), r$rv / r$V, r$pi)
}

# mirt's raw mixture (class 1 = N(0, 1)) on mirt's grid
mix_logw_raw <- function(nodes, shape) {
  r <- raw_shape(shape)
  if (any(r$rv <= 0)) return(rep(NaN, length(nodes)))
  mix_logw_from(nodes, r$rm, r$rv, r$pi)
}

# Category log-probabilities (n_cat x Q) of one item
item_logp <- function(itemtype, a, d, nodes) {
  if (itemtype == "2PL") {
    z <- a * nodes + d
    return(rbind(plogis(-z, log.p = TRUE), plogis(z, log.p = TRUE)))
  }
  if (itemtype == "graded") {
    if (length(d) > 1L && any(diff(d) >= 0)) return(NULL)
    n_d <- length(d)
    cumP <- rbind(1, t(sapply(d, function(dk) plogis(a * nodes + dk))), 0)
    if (n_d == 1L) cumP <- rbind(1, plogis(a * nodes + d), 0)
    probs <- cumP[-nrow(cumP), , drop = FALSE] - cumP[-1L, , drop = FALSE]
    if (any(probs <= 0)) return(NULL)
    return(log(probs))
  }
  # gpcm: P(k) ∝ exp(k a theta + d_k), d_0 = 0
  K <- length(d) + 1L
  eta <- outer(0:(K - 1L), nodes) * a + c(0, d)
  mx <- apply(eta, 2L, max)
  sweep(eta, 2L, mx + log(colSums(exp(sweep(eta, 2L, mx)))))
}

d_shift <- function(itemtype, n_d) if (itemtype == "gpcm") seq_len(n_d) else rep(1, n_d)

# eta layout: per item (a, d_1..d_{K-1}), then shape (rmean_c, rvar_c, logit_c), c = 2..k
make_mll <- function(Yp, cnt, nodes, itemtype, n_cat, k, logw_fn) {
  J <- ncol(Yp)
  n_d <- n_cat - 1L
  n_item <- J * (1L + n_d)
  function(eta) {
    shape <- eta[(n_item + 1L):length(eta)]
    logw <- logw_fn(nodes, shape)
    if (any(!is.finite(logw))) return(-Inf)
    ell <- matrix(0, nrow(Yp), length(nodes))
    for (j in seq_len(J)) {
      pos <- (j - 1L) * (1L + n_d)
      lp <- item_logp(itemtype, eta[pos + 1L], eta[pos + 1L + seq_len(n_d)], nodes)
      if (is.null(lp)) return(-Inf)
      ell <- ell + lp[Yp[, j] + 1L, , drop = FALSE]
    }
    lj <- sweep(ell, 2L, logw, "+")
    mx <- apply(lj, 1L, max)
    sum(cnt * (mx + log(rowSums(exp(lj - mx)))))
  }
}

eap_from <- function(Yp, nodes, itemtype, n_cat, eta, logw) {
  J <- ncol(Yp); n_d <- n_cat - 1L
  ell <- matrix(0, nrow(Yp), length(nodes))
  for (j in seq_len(J)) {
    pos <- (j - 1L) * (1L + n_d)
    lp <- item_logp(itemtype, eta[pos + 1L], eta[pos + 1L + seq_len(n_d)], nodes)
    ell <- ell + lp[Yp[, j] + 1L, , drop = FALSE]
  }
  lj <- sweep(ell, 2L, logw, "+")
  h <- exp(lj - apply(lj, 1L, max)); h <- h / rowSums(h)
  m <- as.numeric(h %*% nodes)
  list(theta = m, se = sqrt(pmax(as.numeric(h %*% nodes^2) - m^2, 0)))
}

# Map between mirt's raw scale (xi) and flexpyirt's standardized scale (eta); same layout.
mirt_to_std <- function(xi, J, n_d, itemtype) {
  n_item <- J * (1L + n_d)
  r <- raw_shape(xi[(n_item + 1L):length(xi)])
  eta <- xi
  for (j in seq_len(J)) {
    pos <- (j - 1L) * (1L + n_d)
    a_m <- xi[pos + 1L]
    eta[pos + 1L] <- a_m * sqrt(r$V)
    eta[pos + 1L + seq_len(n_d)] <- xi[pos + 1L + seq_len(n_d)] + d_shift(itemtype, n_d) * a_m * r$M
  }
  eta
}

std_to_mirt <- function(eta, J, n_d, itemtype) {
  n_item <- J * (1L + n_d)
  r <- raw_shape(eta[(n_item + 1L):length(eta)])
  xi <- eta
  for (j in seq_len(J)) {
    pos <- (j - 1L) * (1L + n_d)
    a_m <- eta[pos + 1L] / sqrt(r$V)
    xi[pos + 1L] <- a_m
    xi[pos + 1L + seq_len(n_d)] <- eta[pos + 1L + seq_len(n_d)] - d_shift(itemtype, n_d) * a_m * r$M
  }
  xi
}

richardson_hessian <- function(f, x, h = 1e-4) {
  n <- length(x)
  H <- matrix(0, n, n)
  f0 <- f(x)
  for (i in seq_len(n)) {
    ei <- numeric(n); ei[i] <- h
    H[i, i] <- (f(x + ei) - 2 * f0 + f(x - ei)) / (h * h)
    if (i < n) {
      for (j in (i + 1L):n) {
        ej <- numeric(n); ej[j] <- h
        H[i, j] <- H[j, i] <-
          (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h * h)
      }
    }
  }
  H
}

grad_fd <- function(f, x, eps = 1e-6) {
  vapply(seq_along(x), function(i) {
    e <- numeric(length(x)); e[i] <- eps
    (f(x + e) - f(x - e)) / (2 * eps)
  }, numeric(1))
}

jacobian_fd <- function(fn, x, eps = 1e-7) {
  f0 <- fn(x)
  Jm <- matrix(0, length(f0), length(x))
  for (i in seq_along(x)) {
    e <- numeric(length(x)); e[i] <- eps
    Jm[, i] <- (fn(x + e) - fn(x - e)) / (2 * eps)
  }
  Jm
}

# --- mirt interface -----------------------------------------------------------------------------

d_names_mirt <- function(itemtype, n_d) {
  if (itemtype == "2PL") return("d")
  if (itemtype == "graded") return(paste0("d", seq_len(n_d)))
  paste0("d", seq_len(n_d))  # gpcm: d0 fixed at 0, d1..d_{K-1} free
}

fit_mirt_mixture <- function(dat, itemtype, k, theta_lim, xi_start, SE) {
  items <- colnames(dat)
  J <- length(items)
  n_d <- if (itemtype == "2PL") 1L else max(dat) - min(dat)
  inv <- c(items, "free_means", "free_var")
  # multipleGroup's third positional argument is `group`, so itemtype must be named.
  sv <- suppressWarnings(multipleGroup(dat, 1, itemtype = itemtype,
                                       dentype = paste0("mixture-", k),
                                       invariance = inv, pars = "values"))
  mirt_class <- c("2PL" = "dich", graded = "graded", gpcm = "gpcm")[[itemtype]]
  stopifnot(all(sv$class[sv$item %in% items] == mirt_class))
  dn <- d_names_mirt(itemtype, n_d)
  for (j in seq_len(J)) {
    pos <- (j - 1L) * (1L + n_d)
    sel <- sv$item == items[j]
    sv[sel & sv$name == "a1", "value"] <- xi_start[pos + 1L]
    for (m in seq_len(n_d)) sv[sel & sv$name == dn[m], "value"] <- xi_start[pos + 1L + m]
  }
  n_item <- J * (1L + n_d)
  for (c in 2:k) {
    sh <- xi_start[n_item + 3L * (c - 2L) + 1:3]
    g <- sv$group == paste0("MIXTURE_", c) & sv$item == "GROUP"
    sv[g & sv$name == "MEAN_1", "value"] <- sh[1L]
    sv[g & sv$name == "COV_11", "value"] <- sh[2L]
    sv[g & sv$name == "PI", "value"] <- sh[3L]
  }
  # invariance + dentype = "mixture" emits a harmless recycling warning (logged as quirk 22).
  m <- suppressWarnings(multipleGroup(
    dat, 1, itemtype = itemtype, dentype = paste0("mixture-", k), invariance = inv, pars = sv,
    quadpts = QUADPTS, optimizer = "NR", accelerate = ACCEL, TOL = TOL,
    SE = SE, SE.type = "Oakes",
    technical = list(theta_lim = theta_lim, NCYCLES = 100000L), verbose = FALSE
  ))
  stopifnot(isTRUE(extract.mirt(m, "converged")))
  cf <- coef(m)
  xi <- c()
  for (j in seq_len(J)) {
    x <- cf$MIXTURE_1[[items[j]]]
    xi <- c(xi, unname(x[1L, "a1"]), unname(x[1L, dn]))
  }
  for (c in 2:k) {
    gp <- cf[[paste0("MIXTURE_", c)]]$GroupPars
    xi <- c(xi, unname(gp[1L, c("MEAN_1", "COV_11", "PI")]))
  }
  list(model = m, xi = xi)
}

# --- one case -----------------------------------------------------------------------------------

solve_case <- function(dat, itemtype, n_cat, k, xi_true_start, eap_pats) {
  items <- colnames(dat)
  J <- length(items)
  n_d <- n_cat - 1L
  key <- apply(dat, 1L, paste, collapse = ",")
  first <- !duplicated(key)
  Yp <- unname(as.matrix(dat[first, , drop = FALSE]))
  cnt <- as.numeric(table(key)[key[first]])

  message(format(Sys.time()), ": mirt fit on [-6, 6]")
  # 1. mirt on [-6, 6] from the generating values (a starting point for the polish).
  t0 <- proc.time()[["elapsed"]]
  f0 <- fit_mirt_mixture(dat, itemtype, k, LIM, xi_true_start, SE = FALSE)
  time_first <- proc.time()[["elapsed"]] - t0
  eta <- mirt_to_std(f0$xi, J, n_d, itemtype)

  message(format(Sys.time()), ": Newton polish")
  # 2. Newton polish on flexpyirt's standardized discretized log-likelihood (independent R code).
  mll <- make_mll(Yp, cnt, NODES, itemtype, n_cat, k, mix_logw_std)
  grad <- grad_fd(mll, eta)
  max_grad_start <- max(abs(grad))
  n_polish <- 0L
  while (max(abs(grad)) > 2e-5 && n_polish < 20L) {
    H <- -richardson_hessian(mll, eta)
    H <- 0.5 * (H + t(H))
    step <- solve(H, grad)
    t_step <- 1
    ll0 <- mll(eta)
    while (!(mll(eta + t_step * step) >= ll0 - 1e-9) && t_step > 1e-6) t_step <- t_step / 2
    eta <- eta + t_step * step
    grad <- grad_fd(mll, eta)
    n_polish <- n_polish + 1L
  }
  max_grad <- max(abs(grad))
  stopifnot(max_grad < 1e-4)
  H <- -richardson_hessian(mll, eta)
  H <- 0.5 * (H + t(H))
  se_h <- sqrt(diag(solve(H)))
  ll_exact <- mll(eta)
  eap_exact <- eap_from(eap_pats, NODES, itemtype, n_cat, eta,
                        mix_logw_std(NODES, eta[(J * (1L + n_d) + 1L):length(eta)]))

  message(format(Sys.time()), ": mirt refit on the image grid")
  # 3. mirt on the image of flexpyirt's grid at the exact MMLE.
  n_item <- J * (1L + n_d)
  r_star <- raw_shape(eta[(n_item + 1L):length(eta)])
  lim_star <- r_star$M + sqrt(r_star$V) * LIM
  t0 <- proc.time()[["elapsed"]]
  f1 <- fit_mirt_mixture(dat, itemtype, k, lim_star, std_to_mirt(eta, J, n_d, itemtype), SE = TRUE)
  time_second <- proc.time()[["elapsed"]] - t0
  m <- f1$model
  grid_star <- as.numeric(m@Model$Theta)
  stopifnot(max(abs(range(grid_star) - lim_star)) < 1e-12)
  xi_m <- f1$xi
  eta_m <- mirt_to_std(xi_m, J, n_d, itemtype)
  ll_mirt <- extract.mirt(m, "logLik")
  mll_mirt <- make_mll(Yp, cnt, grid_star, itemtype, n_cat, k, mix_logw_raw)
  ll_r_mirt_scale <- mll_mirt(xi_m)
  max_grad_mirt_scale <- max(abs(grad_fd(mll_mirt, xi_m)))
  v_m <- vcov(m)
  stopifnot(nrow(v_m) == length(xi_m))
  # mirt 1.46.1's Oakes SEs for dentype = "mixture" are far below the observed-information SEs
  # (close to the complete-data SEs with known class membership; logged as a mirt quirk). Keep
  # both on mirt's own scale as evidence: R Richardson Hessian of mirt's discretized likelihood.
  H_m <- -richardson_hessian(mll_mirt, xi_m)
  se_hessian_mirt_scale <- sqrt(diag(solve(0.5 * (H_m + t(H_m)))))
  m_complete <- suppressWarnings(multipleGroup(
    dat, 1, itemtype = itemtype, dentype = paste0("mixture-", k),
    invariance = c(items, "free_means", "free_var"), pars = mod2values(m), TOL = NaN,
    quadpts = QUADPTS, SE = TRUE, SE.type = "complete",
    technical = list(theta_lim = lim_star), verbose = FALSE
  ))
  se_complete_mirt_scale <- sqrt(diag(vcov(m_complete)))
  Jac <- jacobian_fd(function(x) mirt_to_std(x, J, n_d, itemtype), xi_m)
  se_m <- sqrt(diag(Jac %*% v_m %*% t(Jac)))
  r_m <- raw_shape(xi_m[(n_item + 1L):length(xi_m)])
  fs <- fscores(m, method = "EAP", response.pattern = eap_pats, full.scores.SE = TRUE)
  fs_r <- eap_from(eap_pats, grid_star, itemtype, n_cat, xi_m,
                   mix_logw_raw(grid_star, xi_m[(n_item + 1L):length(xi_m)]))
  stopifnot(max(abs(fs[, 1L] - fs_r$theta)) < 1e-8)

  unpack <- function(v, se = NULL) {
    out_items <- list()
    for (j in seq_len(J)) {
      pos <- (j - 1L) * (1L + n_d)
      dn <- if (n_d == 1L) "d" else paste0("d", seq_len(n_d))
      vals <- c(a = v[pos + 1L], setNames(v[pos + 1L + seq_len(n_d)], dn))
      out_items[[items[j]]] <- as.list(vals)
    }
    lat <- list()
    for (c in 2:k) {
      sh <- v[n_item + 3L * (c - 2L) + 1:3]
      lat[[paste0("rmean_", c)]] <- sh[1L]
      lat[[paste0("rvar_", c)]] <- sh[2L]
      lat[[paste0("logit_", c)]] <- sh[3L]
    }
    list(items = out_items, latent = lat)
  }
  ex <- unpack(eta)
  ex_se <- unpack(se_h)
  mi <- unpack(eta_m)
  mi_se <- unpack(se_m)
  list(
    exact = list(
      loglik = ll_exact,
      max_grad_start = max_grad_start,
      max_grad = max_grad,
      newton_polish_steps = n_polish,
      items = ex$items,
      latent = ex$latent,
      items_se_hessian = ex_se$items,
      latent_se_hessian = ex_se$latent,
      raw_moments = list(M = r_star$M, V = r_star$V),
      eap = eap_exact
    ),
    mirt = list(
      loglik = ll_mirt,
      loglik_r_mirt_scale = ll_r_mirt_scale,
      max_grad_mirt_scale = max_grad_mirt_scale,
      loglik_std_at_mirt = mll(eta_m),
      max_grad_std_at_mirt = max(abs(grad_fd(mll, eta_m))),
      theta_lim = lim_star,
      iterations = extract.mirt(m, "iterations"),
      iterations_first_fit = extract.mirt(f0$model, "iterations"),
      seconds = c(first = time_first, second = time_second),
      converged = extract.mirt(m, "converged"),
      raw_moments = list(M = r_m$M, V = r_m$V),
      items = mi$items,
      latent = mi$latent,
      items_se_oakes = mi_se$items,
      latent_se_oakes = mi_se$latent,
      xi = xi_m,
      se_oakes_mirt_scale = sqrt(diag(v_m)),
      se_complete_mirt_scale = se_complete_mirt_scale,
      se_hessian_mirt_scale = se_hessian_mirt_scale,
      eap = list(theta = (fs[, 1L] - r_m$M) / sqrt(r_m$V), se = fs[, 2L] / sqrt(r_m$V))
    )
  )
}

write_v22 <- function(case, description, dat, itemtype, n_cat, k, res, extra = list()) {
  key <- apply(dat, 1L, paste, collapse = ",")
  first <- !duplicated(key)
  out <- c(list(
    meta = list(
      case = case,
      description = description,
      r_version = R.version.string,
      mirt_version = mirt_version_string(), mirt_sha = mirt_commit_sha(),
      TOL = TOL,
      quadpts = QUADPTS,
      theta_lim = LIM,
      grid_check = LIM,
      accelerate = ACCEL,
      optimizer = "NR",
      itemtype = itemtype,
      n_cat = n_cat,
      n_components = k,
      converged = isTRUE(res$mirt$converged)
    ),
    items = colnames(dat),
    patterns = unname(as.matrix(dat[first, , drop = FALSE])),
    counts = as.integer(table(key)[key[first]])
  ), res, extra)
  path <- file.path(out_dir, paste0(case, ".json"))
  write_json(out, path, auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null")
  cat(sprintf(
    "wrote %s: exact ll=%.8f (|grad| %.1e, %d polish), mirt ll=%.8f (R %.8f, std %.8f), iters %d/%d\n",
    path, res$exact$loglik, res$exact$max_grad, res$exact$newton_polish_steps,
    res$mirt$loglik, res$mirt$loglik_r_mirt_scale, res$mirt$loglik_std_at_mirt,
    res$mirt$iterations_first_fit, res$mirt$iterations
  ))
}

# Start values on mirt's raw scale from the generating model: raw class 1 = the N(0, 1) class.
true_xi <- function(a, d, itemtype, shape) {
  xi <- c()
  for (j in seq_along(a)) {
    dj <- if (is.matrix(d)) d[j, ] else d[j]
    if (itemtype == "gpcm") dj <- dj[-1L]
    xi <- c(xi, a[j], dj)
  }
  c(xi, shape)
}

draw_mixture <- function(n, p, mu, sd) {
  cls <- sample.int(length(p), n, replace = TRUE, prob = p)
  rnorm(n, mu[cls], sd[cls])
}

# ---- V22a: 2PL, k = 2, clearly bimodal ---------------------------------------------------------
if ("a" %in% CASES) {
  set_grid(61L, "none")
  set.seed(20261122)
  N1 <- 3000L; J1 <- 20L
  a1 <- runif(J1, 0.8, 2.0); d1 <- seq(-1.5, 2.5, length.out = J1)
  p1 <- c(0.65, 0.35); mu1 <- c(0, 2.5); sd1 <- c(1, 0.5)
  th1 <- draw_mixture(N1, p1, mu1, sd1)
  dat1 <- as.data.frame(simdata(matrix(a1), d1, N1, itemtype = "2PL", Theta = matrix(th1)))
  colnames(dat1) <- sprintf("q%02d", seq_len(J1))
  eap1 <- rbind(unname(as.matrix(dat1[seq_len(60L), ])), rep(0L, J1), rep(1L, J1))
  shape1 <- c(mu1[2L], sd1[2L]^2, log(p1[2L] / p1[1L]))
  r1 <- solve_case(dat1, "2PL", 2L, 2L, true_xi(a1, d1, "2PL", shape1), eap1)
  write_v22("v22_mixture_2pl_k2",
            "V22a: 2PL with NormalMixture(2): 0.65 N(0,1) + 0.35 N(2.5, 0.5^2), J=20, N=3000",
            dat1, "2PL", 2L, 2L, r1,
            extra = list(eap_patterns = eap1, true = list(pi = p1, mu = mu1, sd = sd1)))
}

# ---- V22b: GRM (K = 4), k = 2, small left component -------------------------------------------
if ("b" %in% CASES) {
  set_grid(61L, "none")
  set.seed(20261123)
  N2 <- 2500L; J2 <- 12L
  a2 <- runif(J2, 0.9, 2.2)
  d2 <- cbind(runif(J2, 1.0, 2.2), runif(J2, -0.4, 0.4), runif(J2, -2.2, -1.0))
  p2 <- c(0.8, 0.2); mu2 <- c(0, -2.2); sd2 <- c(1, 0.4)
  th2 <- draw_mixture(N2, p2, mu2, sd2)
  dat2 <- as.data.frame(simdata(matrix(a2), d2, N2, itemtype = "graded", Theta = matrix(th2)))
  colnames(dat2) <- sprintf("q%02d", seq_len(J2))
  eap2 <- rbind(unname(as.matrix(dat2[seq_len(60L), ])), rep(0L, J2), rep(3L, J2))
  shape2 <- c(mu2[2L], sd2[2L]^2, log(p2[2L] / p2[1L]))
  r2 <- solve_case(dat2, "graded", 4L, 2L, true_xi(a2, d2, "graded", shape2), eap2)
  write_v22("v22_mixture_grm_k2",
            "V22b: GRM (K=4) with NormalMixture(2): 0.8 N(0,1) + 0.2 N(-2.2, 0.4^2), J=12, N=2500",
            dat2, "graded", 4L, 2L, r2,
            extra = list(eap_patterns = eap2, true = list(pi = p2, mu = mu2, sd = sd2)))
}

# ---- V22c: GPCM (K = 3), k = 3, trimodal ------------------------------------------------------
if ("c" %in% CASES) {
  # Narrow components and 20 three-category items make the posteriors about as narrow as the
  # 61-point spacing (0.2). There mirt's fixed grid and flexpyirt's standardized grid (which
  # moves with the shape) gave discretized maxima 2.6e-4 log-likelihood points and up to 6e-3
  # in the parameters apart; 121 points on [-6, 6] (spacing 0.1) remove that discretization
  # effect. mirt runs with accelerate = "squarem" (same fixed point; it still needs ~25,000 EM
  # cycles at TOL = 1e-10, ~30,000 without).
  set_grid(121L, "squarem")
  set.seed(20261124)
  N3 <- 4000L; J3 <- 20L
  a3 <- runif(J3, 0.8, 1.8)
  d3 <- cbind(0, runif(J3, 0.0, 1.5), 0)
  d3[, 3L] <- d3[, 2L] + runif(J3, -1.5, 0.5)
  p3 <- c(0.5, 0.3, 0.2); mu3 <- c(0, 2.3, -2.3); sd3 <- c(1, 0.45, 0.5)
  th3 <- draw_mixture(N3, p3, mu3, sd3)
  dat3 <- as.data.frame(simdata(matrix(a3), d3, N3, itemtype = "gpcm", Theta = matrix(th3)))
  colnames(dat3) <- sprintf("q%02d", seq_len(J3))
  eap3 <- rbind(unname(as.matrix(dat3[seq_len(60L), ])), rep(0L, J3), rep(2L, J3))
  shape3 <- c(mu3[2L], sd3[2L]^2, log(p3[2L] / p3[1L]), mu3[3L], sd3[3L]^2, log(p3[3L] / p3[1L]))
  r3 <- solve_case(dat3, "gpcm", 3L, 3L, true_xi(a3, d3, "gpcm", shape3), eap3)
  write_v22("v22_mixture_gpcm_k3",
            "V22c: GPCM (K=3) with NormalMixture(3): 0.5 N(0,1) + 0.3 N(2.3, 0.45^2) + 0.2 N(-2.3, 0.5^2), J=20, N=4000",
            dat3, "gpcm", 3L, 3L, r3,
            extra = list(eap_patterns = eap3, true = list(pi = p3, mu = mu3, sd = sd3)))
}
