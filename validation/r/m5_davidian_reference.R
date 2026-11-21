# M5 Davidian-curve reference values (VALIDATION.md V19) from R mirt + dcurver.
# Run from the repository root:  Rscript validation/r/m5_davidian_reference.R
# Writes tests/reference/v19_davidian_*.json. CI does not run R; JSON files are committed.
#
# Built-in mirt(..., dentype = "Davidian-k") in mirt 1.46.1 applies Woods' linear-interpolation
# standardizeQuadrature to expected counts before running nlminb on unnormalized dcurver::ddc,
# disables SE = TRUE, and forbids freeing MEAN_1 / COV_11 (DESIGN.md §16.4, VALIDATION.md V19).
# Each case therefore records:
#   1. Built-in mirt dentype = "Davidian-k" (V19a, V19b): logLik, estimates, PHI_1..k,
#      dcurver moments (mu_z, var_z), equivalent cosine-first angles, Prior, and EAP
#      (fscores with use_dentype_estimate = TRUE).
#   2. Exact standardized Davidian MMLE evaluated inside mirt via createGroup() (using
#      dcurver::ddc evaluated at z = mu_z + sd_z * (theta - mean) / sqrt(var)) with
#      SE = TRUE, SE.type = "Oakes", fscores(..., use_dentype_estimate = TRUE), plus exact R
#      central-difference Hessian SEs (mirt 1.46.1's BL.LL has a bug for customGroup that uses
#      pars instead of pars2, making mirt's SE.type = "Richardson" singular).
suppressMessages({
  library(mirt)
  library(dcurver)
  library(jsonlite)
})

out_dir <- file.path("tests", "reference")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
TOL <- 1e-10

# --- Hermite / spherical coordinate helpers ---------------------------------------------------

# dcurver sine-first spherical chart -> unit Hermite vector c (length k + 1)
phi_dc_to_c <- function(phi) {
  k <- length(phi)
  s <- sin(phi); cc <- cos(phi)
  cvec <- numeric(k + 1)
  for (j in 0:(k - 1)) cvec[j + 1] <- (if (j > 0) prod(cc[1:j]) else 1) * s[j + 1]
  cvec[k + 1] <- prod(cc)
  cvec
}

# Cosine-first spherical chart (flexpyirt) -> unit Hermite vector c (length k + 1)
phi_cos_to_c <- function(phi) {
  k <- length(phi)
  s <- sin(phi); cc <- cos(phi)
  cvec <- numeric(k + 1)
  cvec[1] <- prod(cc)
  for (j in 1:k) cvec[j + 1] <- s[j] * (if (j < k) prod(cc[(j + 1):k]) else 1)
  cvec
}

# Unit Hermite vector c (length k + 1) -> cosine-first angles phi (length k)
c_to_phi_cos <- function(cvec) {
  # Fix sign so c[1] >= 0 when possible (P^2 is invariant to c -> -c).
  if (cvec[1] < 0) cvec <- -cvec
  k <- length(cvec) - 1L
  phi <- numeric(k)
  rem <- 1.0
  for (j in seq(k, 1L, by = -1L)) {
    val <- max(-1.0, min(1.0, cvec[j + 1L] / rem))
    phi[j] <- asin(val)
    rem <- rem * cos(phi[j])
    if (abs(rem) < 1e-15) break
  }
  phi
}

# Unit Hermite vector c -> dcurver sine-first angles phi (length k)
c_to_phi_dc <- function(cvec) {
  k <- length(cvec) - 1L
  if (cvec[k + 1L] < 0) cvec <- -cvec
  phi <- numeric(k)
  rem <- 1.0
  for (j in seq_len(k)) {
    val <- max(-1.0, min(1.0, cvec[j] / rem))
    phi[j] <- asin(val)
    rem <- rem * cos(phi[j])
    if (abs(rem) < 1e-15) break
  }
  phi
}

# Closed-form continuous moments (mean, var) of h_k(z) = P_k(z)^2 * dnorm(z)
dc_moments_c <- function(cvec) {
  k <- length(cvec) - 1L
  mu <- 2.0 * sum(sqrt(seq_len(k)) * cvec[seq_len(k)] * cvec[seq_len(k) + 1L])
  e2 <- sum((2.0 * (0:k) + 1.0) * cvec^2)
  if (k >= 2L) {
    e2 <- e2 + 2.0 * sum(sqrt(seq_len(k - 1L) * (seq_len(k - 1L) + 1L)) *
                           cvec[seq_len(k - 1L)] * cvec[seq_len(k - 1L) + 2L])
  }
  c(mean = mu, var = e2 - mu^2)
}

# Standardized Davidian grid weights using dcurver::ddc (normalized on the quadrature grid so
# mirt's customGroup M-step and SE.type = "Oakes" match the exact discretized marginal likelihood).
dc_std_weights <- function(theta, mean, var, phi_cos) {
  if (!is.finite(var) || var <= 0) return(rep(NaN, length(theta)))
  cvec <- phi_cos_to_c(phi_cos)
  mom <- dc_moments_c(cvec)
  z <- mom["mean"] + sqrt(mom["var"]) * (theta - mean) / sqrt(var)
  phi_dc <- c_to_phi_dc(cvec)
  d <- dcurver::ddc(z, phi_dc)
  s <- sum(d)
  if (!is.finite(s) || s <= 0) return(rep(NaN, length(theta)))
  d / s
}

make_custom_group <- function(degree, mean = 0, var = 1, phi_cos = rep(0, degree),
                              free_mean = FALSE, free_var = FALSE) {
  par <- c(mean = mean, var = var, setNames(phi_cos, paste0("phi_", seq_len(degree))))
  est <- c(free_mean, free_var, rep(TRUE, degree))
  lbound <- c(-Inf, 1e-8, rep(-Inf, degree))
  ubound <- rep(Inf, 2L + degree)
  den <- function(obj, Theta) {
    p <- obj@par
    dc_std_weights(as.numeric(Theta), p[1L], p[2L], p[-(1:2)])
  }
  createGroup(par = par, est = est, den = den, nfact = 1L, lbound = lbound, ubound = ubound,
              derivType = "central")
}

# Central finite-difference Hessian (Richardson order 2) of scalar function f at x
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
        val <- (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h * h)
        H[i, j] <- H[j, i] <- val
      }
    }
  }
  H
}

# Exact marginal log-likelihood evaluator in R (2PL or GRM)
make_mll <- function(Yp, cnt, nodes, itemtype, n_cat, degree,
                     fixed_items = integer(), fixed_a = numeric(), fixed_d = list(),
                     free_mean = FALSE, free_var = FALSE) {
  J <- ncol(Yp)
  free_items <- setdiff(seq_len(J), fixed_items)
  n_d <- n_cat - 1L
  function(eta) {
    pos <- 1L
    a <- numeric(J)
    d_mat <- matrix(0, J, n_d)
    if (length(fixed_items) > 0L) {
      a[fixed_items] <- fixed_a
      for (idx in seq_along(fixed_items)) d_mat[fixed_items[idx], ] <- fixed_d[[idx]]
    }
    for (j in free_items) {
      a[j] <- eta[pos]; pos <- pos + 1L
      d_mat[j, ] <- eta[pos:(pos + n_d - 1L)]; pos <- pos + n_d
    }
    mu <- if (free_mean) { v <- eta[pos]; pos <- pos + 1L; v } else 0.0
    vr <- if (free_var)  { v <- eta[pos]; pos <- pos + 1L; v } else 1.0
    phi <- eta[pos:(pos + degree - 1L)]
    if (vr <= 0) return(-Inf)
    w <- dc_std_weights(nodes, mu, vr, phi)
    if (any(!is.finite(w)) || any(w < 0)) return(-Inf)
    logw <- log(w)

    ell <- matrix(0, nrow(Yp), length(nodes))
    if (itemtype == "2PL") {
      z <- outer(a, nodes) + d_mat[, 1L]
      lp1 <- plogis(z, log.p = TRUE)
      lp0 <- plogis(-z, log.p = TRUE)
      ell <- Yp %*% lp1 + (1 - Yp) %*% lp0
    } else {
      # GRM
      for (j in seq_len(J)) {
        dj <- d_mat[j, ]
        if (any(diff(dj) >= 0)) return(-Inf)
        # P(y >= k) for k = 1..n_d
        cumP <- matrix(0, n_d, length(nodes))
        for (k in seq_len(n_d)) cumP[k, ] <- plogis(a[j] * nodes + dj[k])
        probs <- matrix(0, n_cat, length(nodes))
        probs[1L, ] <- 1.0 - cumP[1L, ]
        if (n_d >= 2L) for (k in 2:n_d) probs[k, ] <- cumP[k - 1L, ] - cumP[k, ]
        probs[n_cat, ] <- cumP[n_d, ]
        if (any(probs <= 0)) return(-Inf)
        lp <- log(probs)
        ell <- ell + lp[Yp[, j] + 1L, , drop = FALSE]
      }
    }
    lj <- sweep(ell, 2L, logw, "+")
    mx <- apply(lj, 1L, max)
    sum(cnt * (mx + log(rowSums(exp(lj - mx)))))
  }
}

# Build exact_mle block: polish with R BFGS, pass into mirt(customGroup) for Oakes SEs & EAP,
# and compute exact R Richardson Hessian SEs.
solve_exact_case <- function(dat, itemtype, n_cat, degree, quad,
                             fixed_items = integer(), bank = NULL,
                             free_mean = FALSE, free_var = FALSE,
                             eap_pats, phi_start) {
  J <- ncol(dat)
  items <- colnames(dat)
  key <- apply(dat, 1L, paste, collapse = ",")
  first <- !duplicated(key)
  Yp <- unname(as.matrix(dat[first, , drop = FALSE]))
  cnt <- as.numeric(table(key)[key[first]])
  nodes <- seq(quad$theta_lim[1L], quad$theta_lim[2L], length.out = quad$quadpts)

  free_items <- setdiff(seq_len(J), fixed_items)
  n_d <- n_cat - 1L
  d_names <- if (n_d == 1L) "d" else paste0("d", seq_len(n_d))

  # Fit directly in mirt via createGroup() with normalized grid weights (exact discretized MMLE).
  cg <- make_custom_group(degree, mean = 0.0, var = 1.0, phi_cos = phi_start,
                          free_mean = free_mean, free_var = free_var)
  sv <- mirt(dat, 1L, itemtype, customGroup = cg, pars = "values", quadpts = quad$quadpts,
             technical = list(theta_lim = quad$theta_lim))
  if (!is.null(bank)) {
    for (k in seq_len(nrow(bank))) {
      sel <- sv$item == bank$item[k]
      sv[sel & sv$name == "a1", c("value", "est")] <- list(bank$a[k], FALSE)
      sv[sel & sv$name == "d", c("value", "est")] <- list(bank$d[k], FALSE)
    }
  }
  m_oakes <- mirt(dat, 1L, itemtype, customGroup = cg, pars = sv, quadpts = quad$quadpts,
                  optimizer = "BFGS", accelerate = "none", TOL = TOL,
                  SE = TRUE, SE.type = "Oakes",
                  technical = list(NCYCLES = 5000, theta_lim = quad$theta_lim),
                  control = list(reltol = 1e-13, maxit = 500L), verbose = FALSE)
  stopifnot(
    isTRUE(extract.mirt(m_oakes, "converged")),
    all(range(m_oakes@Model$Theta) == quad$theta_lim)
  )
  cf_o <- coef(m_oakes, printSE = TRUE)
  # Extract mirt Oakes estimates and SEs, and build eta_hat for exact R Richardson Hessian.
  est_o <- list(); se_o <- list()
  eta_hat <- c()
  for (j in free_items) {
    nm <- items[j]
    x <- cf_o[[nm]]
    a_v <- unname(x["par", "a1"])
    d_v <- unname(x["par", d_names])
    eta_hat <- c(eta_hat, a_v, d_v)
    est_o[[nm]] <- c(list(a = a_v), as.list(setNames(d_v, d_names)))
    se_o[[nm]] <- c(list(a_se = unname(x["SE", "a1"])),
                    as.list(setNames(unname(x["SE", d_names]), paste0(d_names, "_se"))))
  }
  gp_o <- cf_o$GroupPars
  phi_names <- paste0("phi_", seq_len(degree))
  mu_val <- unname(gp_o["par", "mean"])
  vr_val <- unname(gp_o["par", "var"])
  phi_val <- unname(gp_o["par", phi_names])
  if (free_mean) eta_hat <- c(eta_hat, mu_val)
  if (free_var)  eta_hat <- c(eta_hat, vr_val)
  eta_hat <- c(eta_hat, phi_val)

  fixed_a <- if (!is.null(bank)) bank$a else numeric()
  fixed_d <- if (!is.null(bank)) lapply(bank$d, function(x) x) else list()
  mll <- make_mll(Yp, cnt, nodes, itemtype, n_cat, degree,
                  fixed_items = fixed_items, fixed_a = fixed_a, fixed_d = fixed_d,
                  free_mean = free_mean, free_var = free_var)
  eps <- 1e-6
  grad <- numeric(length(eta_hat))
  for (i in seq_along(eta_hat)) {
    e <- numeric(length(eta_hat)); e[i] <- eps
    grad[i] <- (mll(eta_hat + e) - mll(eta_hat - e)) / (2 * eps)
  }
  max_grad <- max(abs(grad))
  H <- -richardson_hessian(mll, eta_hat, h = 1e-4)
  H <- 0.5 * (H + t(H))
  se_hess <- sqrt(diag(solve(H)))

  # Unpack exact R Richardson Hessian SEs
  pos <- 1L
  item_se_h <- list()
  for (j in free_items) {
    nm <- items[j]
    a_sh <- se_hess[pos]; pos <- pos + 1L
    d_sh <- se_hess[pos:(pos + n_d - 1L)]; pos <- pos + n_d
    item_se_h[[nm]] <- c(list(a_se = a_sh), as.list(setNames(d_sh, paste0(d_names, "_se"))))
  }
  mu_se_h <- if (free_mean) { s <- se_hess[pos]; pos <- pos + 1L; s } else NA_real_
  vr_se_h <- if (free_var)  { s <- se_hess[pos]; pos <- pos + 1L; s } else NA_real_
  phi_se_h <- se_hess[pos:(pos + degree - 1L)]

  lat_o <- c(
    list(mean = mu_val, var = vr_val),
    as.list(setNames(phi_val, phi_names)),
    list(mean_se = unname(gp_o["SE", "mean"]), var_se = unname(gp_o["SE", "var"])),
    as.list(setNames(unname(gp_o["SE", phi_names]), paste0(phi_names, "_se")))
  )
  lat_h <- c(
    list(mean = mu_val, var = vr_val),
    as.list(setNames(phi_val, phi_names)),
    list(mean_se = mu_se_h, var_se = vr_se_h),
    as.list(setNames(phi_se_h, paste0(phi_names, "_se")))
  )
  eap_o <- fscores(m_oakes, method = "EAP", use_dentype_estimate = TRUE,
                   response.pattern = eap_pats, full.scores.SE = TRUE,
                   quadpts = quad$quadpts, theta_lim = quad$theta_lim)

  list(
    loglik = extract.mirt(m_oakes, "logLik"),
    loglik_r_opt = mll(eta_hat),
    max_grad_r = max_grad,
    converged = extract.mirt(m_oakes, "converged"),
    iterations = extract.mirt(m_oakes, "iterations"),
    grid_check = range(m_oakes@Model$Theta),
    items = est_o,
    items_se_oakes = se_o,
    items_se_hessian = item_se_h,
    latent = lat_o,
    latent_hessian = lat_h,
    eap = list(theta = unname(eap_o[, "F1"]), se = unname(eap_o[, "SE_F1"]))
  )
}

extract_builtin_dc <- function(dat, itemtype, n_cat, degree, quad, eap_pats) {
  m <- mirt(dat, 1L, itemtype, dentype = paste0("Davidian-", degree),
            quadpts = quad$quadpts, optimizer = "NR", accelerate = "none", TOL = TOL,
            technical = list(NCYCLES = 5000, theta_lim = quad$theta_lim), verbose = FALSE)
  stopifnot(
    isTRUE(extract.mirt(m, "converged")),
    all(range(m@Model$Theta) == quad$theta_lim)
  )
  cf <- coef(m, simplify = TRUE)$items
  gp <- coef(m)$GroupPars[1L, ]
  phi_dc <- unname(as.numeric(gp[paste0("PHI_", seq_len(degree))]))
  cvec <- phi_dc_to_c(phi_dc)
  mom <- dc_moments_c(cvec)
  phi_cos <- c_to_phi_cos(cvec)
  n_d <- n_cat - 1L
  d_names <- if (n_d == 1L) "d" else paste0("d", seq_len(n_d))
  items_out <- lapply(colnames(dat), function(nm) {
    c(list(a = unname(cf[nm, "a1"])),
      as.list(setNames(unname(cf[nm, d_names]), d_names)))
  })
  names(items_out) <- colnames(dat)
  eap <- fscores(m, method = "EAP", use_dentype_estimate = TRUE,
                 response.pattern = eap_pats, full.scores.SE = TRUE,
                 quadpts = quad$quadpts, theta_lim = quad$theta_lim)
  list(
    loglik = extract.mirt(m, "logLik"),
    converged = extract.mirt(m, "converged"),
    iterations = extract.mirt(m, "iterations"),
    grid_check = range(m@Model$Theta),
    items = items_out,
    phi_dcurver = phi_dc,
    phi_cos = phi_cos,
    mu_z = unname(mom["mean"]),
    var_z = unname(mom["var"]),
    prior = as.numeric(m@Internals$Prior[[1L]]),
    eap = list(theta = unname(eap[, "F1"]), se = unname(eap[, "SE_F1"]))
  )
}

write_v19 <- function(case, description, dat, itemtype, n_cat, degree, quad,
                      builtin, exact, extra = list()) {
  key <- apply(dat, 1L, paste, collapse = ",")
  first <- !duplicated(key)
  patterns <- unname(as.matrix(dat[first, , drop = FALSE]))
  counts <- as.integer(table(key)[key[first]])
  res <- c(list(
    meta = list(
      case = case,
      description = description,
      r_version = R.version.string,
      mirt_version = as.character(packageVersion("mirt")),
      dcurver_version = as.character(packageVersion("dcurver")),
      TOL = TOL,
      quadpts = quad$quadpts,
      theta_lim = quad$theta_lim,
      grid_check = exact$grid_check,
      accelerate = "none",
      optimizer = if (!is.null(builtin)) "NR (builtin) / BFGS (customGroup)" else "BFGS (customGroup)",
      degree = degree,
      itemtype = itemtype,
      n_cat = n_cat,
      converged = exact$converged
    ),
    items = colnames(dat),
    patterns = patterns,
    counts = counts,
    exact = exact
  ), extra)
  if (!is.null(builtin)) res$mirt_builtin <- builtin
  path <- file.path(out_dir, paste0(case, ".json"))
  write_json(res, path, auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null")
  cat(sprintf("wrote %s: exact ll=%.6f (|grad|=%.2e)%s\n",
              path, exact$loglik, exact$max_grad_r,
              if (!is.null(builtin)) sprintf(", builtin ll=%.6f (diff=%+.4f)", builtin$loglik, exact$loglik - builtin$loglik) else ""))
}

skewed <- function(n) (rchisq(n, df = 4) - 4) / sqrt(8)

# ---- V19a: 2PL, degree=3, N=3000, J=20, standardized chi^2_4 trait ---------------------------
set.seed(20261131)
N1 <- 3000L; J1 <- 20L
quad61 <- list(quadpts = 61L, theta_lim = c(-6, 6))
a1 <- runif(J1, 0.8, 2.0); d1 <- seq(-2.0, 2.0, length.out = J1)
dat1 <- as.data.frame(simdata(matrix(a1), d1, N1, itemtype = "2PL", Theta = matrix(skewed(N1))))
colnames(dat1) <- sprintf("q%02d", seq_len(J1))
eap_pats1 <- unname(as.matrix(dat1[seq_len(100L), , drop = FALSE]))

b1 <- extract_builtin_dc(dat1, "2PL", 2L, 3L, quad61, eap_pats1)
e1 <- solve_exact_case(dat1, "2PL", 2L, 3L, quad61, eap_pats = eap_pats1,
                       phi_start = c(-0.1, 0.1, 0.0))
write_v19("v19_davidian_2pl_k3",
          "V19a: 2PL with DavidianCurve(degree=3), skewed chi^2_4 trait, J=20, N=3000",
          dat1, "2PL", 2L, 3L, quad61, b1, e1,
          extra = list(eap_patterns = eap_pats1, true_a = a1, true_d = d1))

# ---- V19b: GRM (K=4), degree=2, N=3000, J=10, standardized chi^2_4 trait ---------------------
set.seed(20261132)
N2 <- 3000L; J2 <- 10L
a2 <- runif(J2, 0.9, 1.9)
d2 <- cbind(
  runif(J2, 1.0, 1.8),
  runif(J2, -0.3, 0.3),
  runif(J2, -1.8, -1.0)
)
dat2 <- as.data.frame(simdata(matrix(a2), d2, N2, itemtype = "graded", Theta = matrix(skewed(N2))))
colnames(dat2) <- sprintf("q%02d", seq_len(J2))
eap_pats2 <- unname(as.matrix(dat2[seq_len(100L), , drop = FALSE]))

b2 <- extract_builtin_dc(dat2, "graded", 4L, 2L, quad61, eap_pats2)
e2 <- solve_exact_case(dat2, "graded", 4L, 2L, quad61, eap_pats = eap_pats2,
                       phi_start = c(-0.1, 0.1))
write_v19("v19_davidian_grm_k2",
          "V19b: GRM (K=4) with DavidianCurve(degree=2), skewed chi^2_4 trait, J=10, N=3000",
          dat2, "graded", 4L, 2L, quad61, b2, e2,
          extra = list(eap_patterns = eap_pats2))

# ---- V19c: FIPC 2PL, degree=3, free mean & var, q01..q10 fixed, 81 pts [-8, 8] --------------
set.seed(20261133)
N3 <- 3000L; J3 <- 20L
quad81 <- list(quadpts = 81L, theta_lim = c(-8, 8))
a3 <- runif(J3, 0.8, 1.9); d3 <- seq(-1.8, 1.8, length.out = J3)
th3 <- 0.4 + 1.2 * skewed(N3)
dat3 <- as.data.frame(simdata(matrix(a3), d3, N3, itemtype = "2PL", Theta = matrix(th3)))
colnames(dat3) <- sprintf("q%02d", seq_len(J3))
bank3 <- data.frame(item = colnames(dat3)[1:10], a = a3[1:10], d = d3[1:10])
eap_pats3 <- unname(as.matrix(dat3[seq_len(100L), , drop = FALSE]))

e3 <- solve_exact_case(dat3, "2PL", 2L, 3L, quad81,
                       fixed_items = 1:10, bank = bank3,
                       free_mean = TRUE, free_var = TRUE,
                       eap_pats = eap_pats3, phi_start = c(-0.1, 0.1, 0.0))
write_v19("v19_davidian_fipc_k3",
          "V19c: FIPC 2PL (q01-q10 fixed) with DavidianCurve.free(degree=3), shifted skewed trait, J=20, N=3000",
          dat3, "2PL", 2L, 3L, quad81, NULL, e3,
          extra = list(bank = bank3, eap_patterns = eap_pats3))
