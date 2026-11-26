# Differentiable IRT scale linking reference values (VALIDATION.md V20).
# Run from the repository root:  Rscript validation/r/m5_linking_reference.R
# Writes tests/reference/v20_linking.json. CI does not run R; the JSON is committed.
#
# mirt does not implement IRT scale linking (Haebara / Stocking-Lord). Each form is calibrated
# separately with mirt (optimizer = "NR", accelerate = "none", TOL = 1e-10, 61 equispaced
# quadrature points on [-6, 6], SE.type = "Oakes"; latent N(0, 1) except the source form of V20e,
# which is calibrated under a fixed N(0.3, 1.5) so that the symmetric criterion's reverse term uses
# a non-standard source grid), and the linking constants (A, B), criterion values, and Ogasawara
# (2001) delta-method SEs are computed by an independent base-R implementation of the papers'
# formulas (Haebara, 1980; Stocking & Lord, 1983; Kim & Lee, 2006) using mirt's probtrace() /
# expected.item() for the characteristic curves and optim(BFGS) + nlm() for tight optimization.
# Haebara sums squared differences over all categories of each item (Kim & Lee, 2006), including
# both categories of a dichotomous item.
source(file.path("validation", "r", "setup.R"))  # mirt from GitHub; mirt_version / mirt_sha
suppressMessages({
  library(mirt)
  library(jsonlite)
})

quadpts <- 61L
theta_lim <- c(-6, 6)
common_mirt <- list(
  TOL = 1e-10,
  accelerate = "none",
  optimizer = "NR",
  SE = TRUE,
  SE.type = "Oakes",
  quadpts = quadpts,
  verbose = FALSE,
  technical = list(NCYCLES = 50000L, theta_lim = theta_lim)
)

# ---- Helpers for extracting mirt item parameters & natural-scale vcov --------------------------

clean_item <- function(x, itemtype) {
  nm <- colnames(x)
  if (itemtype == "3PL") {
    par_raw <- x["par", ]
    se_raw <- if ("SE" %in% rownames(x)) x["SE", ] else setNames(rep(NA_real_, ncol(x)), nm)
    g_nat <- plogis(unname(par_raw[["logit(g)"]]))
    u_nat <- plogis(unname(par_raw[["logit(u)"]]))
    g_se <- if (is.na(se_raw[["logit(g)"]])) NA_real_ else g_nat * (1 - g_nat) * se_raw[["logit(g)"]]
    u_se <- if (is.na(se_raw[["logit(u)"]])) NA_real_ else u_nat * (1 - u_nat) * se_raw[["logit(u)"]]
    return(list(
      a = unname(par_raw[["a1"]]),
      d = unname(par_raw[["d"]]),
      g = g_nat,
      u = u_nat,
      a_se = unname(se_raw[["a1"]]),
      d_se = unname(se_raw[["d"]]),
      g_se = unname(g_se),
      u_se = unname(u_se)
    ))
  }
  keep <- nm[nm == "a1" | grepl("^d[1-9][0-9]*$|^d$", nm)]
  par <- x["par", keep]
  se <- if ("SE" %in% rownames(x)) x["SE", keep] else rep(NA_real_, length(keep))
  names(par) <- sub("^a1$", "a", keep)
  names(se) <- paste0(names(par), "_se")
  c(as.list(par), as.list(se))
}

extract_form <- function(m, dat, itemtypes, n_cats) {
  cf <- coef(m, printSE = TRUE)
  items <- setdiff(names(cf), "GroupPars")
  params <- lapply(items, function(i) clean_item(cf[[i]], itemtypes[[i]]))
  names(params) <- items
  key <- apply(dat, 1, paste, collapse = ",")
  first <- !duplicated(key)
  patterns <- unname(as.matrix(dat[first, , drop = FALSE]))
  counts <- as.integer(table(key)[key[first]])
  cs <- coef(m, simplify = TRUE)
  list(
    items = items,
    n_obs = nrow(dat),
    patterns = patterns,
    counts = counts,
    loglik = extract.mirt(m, "logLik"),
    converged = extract.mirt(m, "converged"),
    iterations = extract.mirt(m, "iterations"),
    grid_check = range(m@Model$Theta),
    latent_mean = unname(cs$means[1]),
    latent_var = unname(cs$cov[1, 1]),
    itemtypes = as.list(itemtypes[items]),
    n_cats = as.list(n_cats[items]),
    params = params
  )
}

# ---- Independent base-R linking criteria using mirt's probtrace() and expected.item() ----------
# Forward term: target curves on the target grid (theta, w_tgt) against source curves at
# (theta - B) / A. Reverse (symmetric) term: source curves on the source grid (theta, w_src)
# against target curves at A * theta + B (Kim & Lee, 2006).

make_criterion <- function(m_src, m_tgt, common_items, theta, w_tgt, w_src, method, symmetric) {
  src_ext <- lapply(common_items, function(nm) extract.item(m_src, nm))
  tgt_ext <- lapply(common_items, function(nm) extract.item(m_tgt, nm))
  n_common <- length(common_items)

  if (method == "haebara") {
    p_tgt_fixed <- lapply(tgt_ext, function(it) probtrace(it, matrix(theta)))
    p_src_fixed <- if (symmetric) lapply(src_ext, function(it) probtrace(it, matrix(theta))) else NULL
    return(function(par) {
      A <- par[1]
      B <- par[2]
      if (!is.finite(A) || !is.finite(B) || A <= 0) return(1e30)
      th_src <- matrix((theta - B) / A)
      diff_fwd <- numeric(length(theta))
      for (j in seq_len(n_common)) {
        d <- p_tgt_fixed[[j]] - probtrace(src_ext[[j]], th_src)
        diff_fwd <- diff_fwd + rowSums(d * d)
      }
      val <- sum(w_tgt * diff_fwd)
      if (symmetric) {
        th_tgt <- matrix(A * theta + B)
        diff_bwd <- numeric(length(theta))
        for (j in seq_len(n_common)) {
          d <- p_src_fixed[[j]] - probtrace(tgt_ext[[j]], th_tgt)
          diff_bwd <- diff_bwd + rowSums(d * d)
        }
        val <- val + sum(w_src * diff_bwd)
      }
      val
    })
  }

  # Stocking-Lord: squared difference of common-item test characteristic curves (TCC).
  # Verify expected.item() matches sum_k k * P_k(theta) with 0-based scores.
  for (j in seq_len(n_common)) {
    pt <- probtrace(tgt_ext[[j]], matrix(theta))
    ei <- expected.item(tgt_ext[[j]], matrix(theta))
    manual <- as.numeric(pt %*% (seq_len(ncol(pt)) - 1L))
    stopifnot(max(abs(ei - manual)) < 1e-13)
  }
  tcc_tgt_fixed <- Reduce(`+`, lapply(tgt_ext, function(it) expected.item(it, matrix(theta))))
  tcc_src_fixed <- if (symmetric) {
    Reduce(`+`, lapply(src_ext, function(it) expected.item(it, matrix(theta))))
  } else NULL

  function(par) {
    A <- par[1]
    B <- par[2]
    if (!is.finite(A) || !is.finite(B) || A <= 0) return(1e30)
    th_src <- matrix((theta - B) / A)
    tcc_src_moved <- Reduce(`+`, lapply(src_ext, function(it) expected.item(it, th_src)))
    val <- sum(w_tgt * (tcc_tgt_fixed - tcc_src_moved)^2)
    if (symmetric) {
      th_tgt <- matrix(A * theta + B)
      tcc_tgt_moved <- Reduce(`+`, lapply(tgt_ext, function(it) expected.item(it, th_tgt)))
      val <- val + sum(w_src * (tcc_src_fixed - tcc_tgt_moved)^2)
    }
    val
  }
}

solve_base_r <- function(fn, start = c(1.0, 0.0)) {
  grad_fd <- function(x, eps = 1e-6) {
    c(
      (fn(x + c(eps, 0)) - fn(x - c(eps, 0))) / (2 * eps),
      (fn(x + c(0, eps)) - fn(x - c(0, eps))) / (2 * eps)
    )
  }
  opt <- optim(
    start, fn, gr = grad_fd, method = "BFGS",
    control = list(reltol = 1e-15, abstol = 1e-15, maxit = 1000)
  )
  pol <- nlm(fn, opt$par, ndigit = 15, gradtol = 1e-13, steptol = 1e-13, iterlim = 1000)
  ab <- if (fn(pol$estimate) <= fn(opt$par)) pol$estimate else opt$par
  # Polish with a few damped central-FD Newton steps so |grad| reaches ~1e-12.
  for (iter in seq_len(10L)) {
    g <- grad_fd(ab, eps = 1e-6)
    if (max(abs(g)) < 1e-12) break
    e <- 1e-5
    H <- cbind(
      (grad_fd(ab + c(e, 0), 1e-6) - grad_fd(ab - c(e, 0), 1e-6)) / (2 * e),
      (grad_fd(ab + c(0, e), 1e-6) - grad_fd(ab - c(0, e), 1e-6)) / (2 * e)
    )
    H <- 0.5 * (H + t(H))
    step <- tryCatch(-solve(H, g), error = function(err) rep(0, 2))
    cand <- ab + step
    if (all(is.finite(cand)) && cand[1] > 0 && max(abs(grad_fd(cand, 1e-6))) < max(abs(g))) {
      ab <- cand
    } else {
      break
    }
  }
  val <- fn(ab)
  grad <- grad_fd(ab, eps = 1e-6)
  stopifnot(all(is.finite(ab)), ab[1] > 0, max(abs(grad)) < 1e-9)
  list(A = ab[1], B = ab[2], criterion = val, grad_norm = max(abs(grad)), nlm_code = pol$code)
}

# ---- Pure-R Ogasawara (2001) delta-method SEs via central FD of the criterion ------------------

item_probs_r <- function(par_vec, theta, itemtype, n_cat) {
  if (itemtype == "2PL") {
    p1 <- plogis(par_vec[1] * theta + par_vec[2])
    return(cbind(1 - p1, p1))
  }
  if (itemtype == "3PL") {
    g <- par_vec[3]
    p1 <- g + (1 - g) * plogis(par_vec[1] * theta + par_vec[2])
    return(cbind(1 - p1, p1))
  }
  if (itemtype == "graded") {
    a <- par_vec[1]
    ds <- par_vec[-1]
    cum <- sapply(ds, function(dk) plogis(a * theta + dk))
    out <- matrix(0, nrow = length(theta), ncol = n_cat)
    out[, 1] <- 1 - cum[, 1]
    if (n_cat > 2) {
      for (k in 2:(n_cat - 1)) out[, k] <- cum[, k - 1] - cum[, k]
    }
    out[, n_cat] <- cum[, n_cat - 1]
    return(out)
  }
  if (itemtype == "gpcm") {
    a <- par_vec[1]
    ds <- c(0, par_vec[-1])
    eta <- outer(a * theta, 0:(n_cat - 1)) + rep(ds, each = length(theta))
    m <- apply(eta, 1, max)
    e <- exp(eta - m)
    return(e / rowSums(e))
  }
  stop("Unsupported itemtype: ", itemtype)
}

item_param_names <- function(itemtype, n_cat) {
  if (itemtype == "2PL") return(c("a1", "d"))
  if (itemtype == "3PL") return(c("a1", "d", "g"))
  c("a1", paste0("d", seq_len(n_cat - 1)))
}

# Common-item free parameters (natural scale, item-major in common_items order) and their
# covariance block from vcov(m).
extract_common <- function(m, common_items, itemtypes, n_cats) {
  sv <- mod2values(m)
  vc <- vcov(m)
  # Match order of free parameters in vcov(m) (parnum order where est == TRUE).
  free_rows <- sv[sv$est, , drop = FALSE]
  stopifnot(nrow(free_rows) == nrow(vc))
  # mod2values(m)$value stores g on the natural scale (0..1), while vcov(m) is on the logit
  # scale for g; convert vcov(m) to natural scale with Jacobian g * (1 - g).
  jac <- rep(1.0, nrow(vc))
  is_g <- free_rows$name == "g"
  if (any(is_g)) {
    g_nat <- free_rows$value[is_g]
    jac[is_g] <- g_nat * (1 - g_nat)
  }
  vc_nat <- (jac %o% jac) * vc
  idx <- integer(0)
  vals <- numeric(0)
  labels <- list()
  for (it in common_items) {
    for (pn in item_param_names(itemtypes[[it]], n_cats[[it]])) {
      pos <- which(free_rows$item == it & free_rows$name == pn)
      stopifnot(length(pos) == 1L)
      idx <- c(idx, pos)
      vals <- c(vals, free_rows$value[pos])
      labels[[length(labels) + 1L]] <- c(it, sub("^a1$", "a", pn))
    }
  }
  list(vals = vals, vcov = vc_nat[idx, idx, drop = FALSE], labels = labels)
}

ogasawara_se_r <- function(cs, ct, common_items, itemtypes, n_cats, theta, w_tgt, w_src,
                           method, symmetric, ab_hat) {
  n_common <- length(common_items)
  p_sizes <- vapply(
    common_items, function(it) length(item_param_names(itemtypes[[it]], n_cats[[it]])), 1L
  )
  ends <- cumsum(p_sizes)
  starts <- ends - p_sizes + 1L

  crit_from_vecs <- function(ab, vs, vt) {
    A <- ab[1]; B <- ab[2]
    if (!is.finite(A) || !is.finite(B) || A <= 0) return(1e30)
    th_s <- (theta - B) / A
    th_t <- A * theta + B
    if (method == "haebara") {
      dfwd <- numeric(length(theta))
      dbwd <- numeric(length(theta))
      for (j in seq_len(n_common)) {
        rng <- starts[j]:ends[j]
        it <- itemtypes[[common_items[j]]]
        kj <- n_cats[[common_items[j]]]
        d <- item_probs_r(vt[rng], theta, it, kj) - item_probs_r(vs[rng], th_s, it, kj)
        dfwd <- dfwd + rowSums(d * d)
        if (symmetric) {
          d2 <- item_probs_r(vs[rng], theta, it, kj) - item_probs_r(vt[rng], th_t, it, kj)
          dbwd <- dbwd + rowSums(d2 * d2)
        }
      }
      return(sum(w_tgt * dfwd) + if (symmetric) sum(w_src * dbwd) else 0)
    }
    tcc_t <- numeric(length(theta))
    tcc_s_on_t <- numeric(length(theta))
    tcc_s <- numeric(length(theta))
    tcc_t_on_s <- numeric(length(theta))
    for (j in seq_len(n_common)) {
      rng <- starts[j]:ends[j]
      it <- itemtypes[[common_items[j]]]
      kj <- n_cats[[common_items[j]]]
      cats <- 0:(kj - 1L)
      tcc_t <- tcc_t + as.numeric(item_probs_r(vt[rng], theta, it, kj) %*% cats)
      tcc_s_on_t <- tcc_s_on_t + as.numeric(item_probs_r(vs[rng], th_s, it, kj) %*% cats)
      if (symmetric) {
        tcc_s <- tcc_s + as.numeric(item_probs_r(vs[rng], theta, it, kj) %*% cats)
        tcc_t_on_s <- tcc_t_on_s + as.numeric(item_probs_r(vt[rng], th_t, it, kj) %*% cats)
      }
    }
    val <- sum(w_tgt * (tcc_t - tcc_s_on_t)^2)
    if (symmetric) val <- val + sum(w_src * (tcc_s - tcc_t_on_s)^2)
    val
  }

  # Compute H_cc and H_{c, eta} by central finite differences of the gradient.
  eps_c <- 1e-5
  eps_e <- 1e-5
  grad_c <- function(ab, vs, vt) {
    c(
      (crit_from_vecs(ab + c(eps_c, 0), vs, vt) - crit_from_vecs(ab - c(eps_c, 0), vs, vt)) / (2 * eps_c),
      (crit_from_vecs(ab + c(0, eps_c), vs, vt) - crit_from_vecs(ab - c(0, eps_c), vs, vt)) / (2 * eps_c)
    )
  }
  H_cc <- cbind(
    (grad_c(ab_hat + c(eps_c, 0), cs$vals, ct$vals) - grad_c(ab_hat - c(eps_c, 0), cs$vals, ct$vals)) / (2 * eps_c),
    (grad_c(ab_hat + c(0, eps_c), cs$vals, ct$vals) - grad_c(ab_hat - c(0, eps_c), cs$vals, ct$vals)) / (2 * eps_c)
  )
  H_cc <- 0.5 * (H_cc + t(H_cc))
  H_cs <- matrix(0, nrow = 2, ncol = length(cs$vals))
  for (i in seq_along(cs$vals)) {
    dp <- numeric(length(cs$vals)); dp[i] <- eps_e
    H_cs[, i] <- (grad_c(ab_hat, cs$vals + dp, ct$vals) - grad_c(ab_hat, cs$vals - dp, ct$vals)) / (2 * eps_e)
  }
  H_ct <- matrix(0, nrow = 2, ncol = length(ct$vals))
  for (i in seq_along(ct$vals)) {
    dp <- numeric(length(ct$vals)); dp[i] <- eps_e
    H_ct[, i] <- (grad_c(ab_hat, cs$vals, ct$vals + dp) - grad_c(ab_hat, cs$vals, ct$vals - dp)) / (2 * eps_e)
  }
  J_s <- -solve(H_cc, H_cs)
  J_t <- -solve(H_cc, H_ct)
  cov_ab <- J_s %*% cs$vcov %*% t(J_s) + J_t %*% ct$vcov %*% t(J_t)
  cov_ab <- 0.5 * (cov_ab + t(cov_ab))
  list(
    A_se = sqrt(cov_ab[1, 1]),
    B_se = sqrt(cov_ab[2, 2]),
    vcov = unname(cov_ab)
  )
}

# itemtypes_* / n_cats_*: named per-item vectors (column names of each form's data).
# src_latent: fixed (mean, var) of the source calibration's latent distribution (N(0, 1) by
# default); the source-grid weights of the symmetric reverse term follow it.
run_linking_case <- function(case_name, description, itemtypes_tgt, itemtypes_src, n_cats_tgt,
                             n_cats_src, dat_src, dat_tgt, common_items, src_latent = c(0, 1)) {
  m_tgt <- do.call(
    mirt, c(list(dat_tgt, 1, itemtype = unname(itemtypes_tgt[colnames(dat_tgt)])), common_mirt)
  )
  sv_src <- mirt(dat_src, 1, itemtype = unname(itemtypes_src[colnames(dat_src)]), pars = "values")
  sv_src$value[sv_src$name == "MEAN_1"] <- src_latent[1]
  sv_src$value[sv_src$name == "COV_11"] <- src_latent[2]
  stopifnot(!any(sv_src$est[sv_src$item == "GROUP"]))
  m_src <- do.call(mirt, c(
    list(dat_src, 1, itemtype = unname(itemtypes_src[colnames(dat_src)]), pars = sv_src),
    common_mirt
  ))
  stopifnot(extract.mirt(m_tgt, "converged"), extract.mirt(m_src, "converged"))
  stopifnot(all(itemtypes_tgt[common_items] == itemtypes_src[common_items]))

  theta <- as.numeric(m_tgt@Model$Theta[, 1])
  w_tgt <- dnorm(theta, 0, 1)
  w_tgt <- w_tgt / sum(w_tgt)
  w_src <- dnorm(theta, src_latent[1], sqrt(src_latent[2]))
  w_src <- w_src / sum(w_src)
  stopifnot(
    all(range(theta) == theta_lim),
    length(theta) == quadpts,
    max(abs(theta - as.numeric(m_src@Model$Theta[, 1]))) < 1e-14,
    max(abs(w_tgt - as.numeric(m_tgt@Internals$Prior[[1]]))) < 1e-13,
    max(abs(w_src - as.numeric(m_src@Internals$Prior[[1]]))) < 1e-13
  )

  # For 3PL, confirm all estimated guessing parameters are strictly interior (no Heywood).
  if (any(itemtypes_tgt == "3PL")) {
    for (m in list(m_tgt, m_src)) {
      cf <- coef(m, simplify = TRUE)$items
      g_vals <- cf[, "g"]
      stopifnot(all(g_vals > 0.01 & g_vals < 0.99))
    }
  }

  it_common <- as.list(itemtypes_tgt[common_items])
  k_common <- as.list(n_cats_tgt[common_items])
  cs <- extract_common(m_src, common_items, it_common, k_common)
  ct <- extract_common(m_tgt, common_items, it_common, k_common)

  variants <- list(
    stocking_lord = list(method = "stocking_lord", symmetric = FALSE),
    stocking_lord_sym = list(method = "stocking_lord", symmetric = TRUE),
    haebara = list(method = "haebara", symmetric = FALSE),
    haebara_sym = list(method = "haebara", symmetric = TRUE)
  )
  linking_out <- list()
  for (vname in names(variants)) {
    v <- variants[[vname]]
    fn <- make_criterion(m_src, m_tgt, common_items, theta, w_tgt, w_src, v$method, v$symmetric)
    sol <- solve_base_r(fn, start = c(1.2, 0.5))
    se_res <- ogasawara_se_r(
      cs, ct, common_items, it_common, k_common, theta, w_tgt, w_src,
      v$method, v$symmetric, c(sol$A, sol$B)
    )
    linking_out[[vname]] <- c(
      list(method = v$method, symmetric = v$symmetric),
      sol,
      se_res
    )
    cat(sprintf(
      "  %-18s %-18s A=%.6f (SE=%.6f) B=%.6f (SE=%.6f) crit=%.6e |g|=%.2e\n",
      case_name, vname, sol$A, se_res$A_se, sol$B, se_res$B_se, sol$criterion, sol$grad_norm
    ))
  }

  types_common <- unique(unname(unlist(it_common)))
  list(
    meta = list(
      case = case_name,
      description = description,
      itemtype = if (length(types_common) == 1L) types_common else "mixed",
      n_cat = max(unlist(k_common)),
      quadpts = quadpts,
      theta_lim = theta_lim,
      grid_check = range(m_tgt@Model$Theta),
      source_latent = list(mean = src_latent[1], var = src_latent[2]),
      converged = TRUE
    ),
    common_items = common_items,
    # Common-item free parameters (natural scale; g not logit) and mirt's Oakes covariance
    # block, item-major, for checking the delta-method SEs on identical inputs.
    common_vcov = list(
      source = list(labels = cs$labels, values = cs$vals, vcov = unname(cs$vcov)),
      target = list(labels = ct$labels, values = ct$vals, vcov = unname(ct$vcov))
    ),
    target = extract_form(m_tgt, dat_tgt, itemtypes_tgt, n_cats_tgt),
    source = extract_form(m_src, dat_src, itemtypes_src, n_cats_src),
    linking = linking_out
  )
}

run_uniform_case <- function(case_name, description, itemtype, n_cat, dat_src, dat_tgt,
                             common_items, src_latent = c(0, 1)) {
  types_of <- function(dat) setNames(rep(itemtype, ncol(dat)), colnames(dat))
  cats_of <- function(dat) setNames(rep(as.integer(n_cat), ncol(dat)), colnames(dat))
  run_linking_case(
    case_name, description, types_of(dat_tgt), types_of(dat_src), cats_of(dat_tgt),
    cats_of(dat_src), dat_src, dat_tgt, common_items, src_latent
  )
}

# ---- Simulate and calibrate all cases -----------------------------------------------------------
# True population parameters: target ~ N(0, 1), source ~ N(0.5, 1.2^2) -> true A = 1.2, B = 0.5
# (for source calibrations on N(0, 1)).

cases <- list()

# 1. V20a: 2PL (J = 20 per form, 10 common, N = 2000 per group)
set.seed(20261201)
J_c <- 10L; J_u <- 10L; N <- 2000L
common_2pl <- sprintf("c%02d", seq_len(J_c))
tgt_only_2pl <- sprintf("t%02d", seq_len(J_u))
src_only_2pl <- sprintf("s%02d", seq_len(J_u))
a_all <- rlnorm(J_c + 2L * J_u, meanlog = 0.1, sdlog = 0.25)
d_all <- rnorm(J_c + 2L * J_u, mean = 0, sd = 0.9)
idx_c <- seq_len(J_c)
idx_t <- (J_c + 1L):(J_c + J_u)
idx_s <- (J_c + J_u + 1L):(J_c + 2L * J_u)
dat_tgt_2pl <- as.data.frame(simdata(
  matrix(a_all[c(idx_c, idx_t)]), matrix(d_all[c(idx_c, idx_t)]), N, itemtype = "2PL",
  mu = 0.0, sigma = matrix(1.0)
))
colnames(dat_tgt_2pl) <- c(common_2pl, tgt_only_2pl)
dat_src_2pl <- as.data.frame(simdata(
  matrix(a_all[c(idx_c, idx_s)]), matrix(d_all[c(idx_c, idx_s)]), N, itemtype = "2PL",
  mu = 0.5, sigma = matrix(1.2^2)
))
colnames(dat_src_2pl) <- c(common_2pl, src_only_2pl)
cases[["v20a_2pl"]] <- run_uniform_case(
  "v20a_2pl",
  "V20a: 2PL, J=20/form, 10 common, N=2000/group, target N(0,1), source N(0.5, 1.2^2)",
  "2PL", 2L, dat_src_2pl, dat_tgt_2pl, common_2pl
)

# 2. V20b: GRM K=4 (J = 16 per form, 8 common, N = 2000 per group)
set.seed(20261202)
J_c <- 8L; J_u <- 8L; N <- 2000L; K <- 4L
J_tot <- J_c + 2L * J_u
common_p <- sprintf("c%02d", seq_len(J_c))
tgt_only_p <- sprintf("t%02d", seq_len(J_u))
src_only_p <- sprintf("s%02d", seq_len(J_u))
a_grm <- rlnorm(J_tot, meanlog = 0.1, sdlog = 0.2)
b_mid <- rnorm(J_tot, 0, 0.5)
d_grm <- cbind(
  -a_grm * (b_mid - 1.1),
  -a_grm * b_mid,
  -a_grm * (b_mid + 1.1)
)
idx_c <- seq_len(J_c); idx_t <- (J_c + 1L):(J_c + J_u); idx_s <- (J_c + J_u + 1L):J_tot
dat_tgt_grm <- as.data.frame(simdata(
  matrix(a_grm[c(idx_c, idx_t)]), d_grm[c(idx_c, idx_t), ], N, itemtype = "graded",
  mu = 0.0, sigma = matrix(1.0)
))
colnames(dat_tgt_grm) <- c(common_p, tgt_only_p)
dat_src_grm <- as.data.frame(simdata(
  matrix(a_grm[c(idx_c, idx_s)]), d_grm[c(idx_c, idx_s), ], N, itemtype = "graded",
  mu = 0.5, sigma = matrix(1.2^2)
))
colnames(dat_src_grm) <- c(common_p, src_only_p)
cases[["v20b_grm"]] <- run_uniform_case(
  "v20b_grm",
  "V20b: GRM (K=4), J=16/form, 8 common, N=2000/group, target N(0,1), source N(0.5, 1.2^2)",
  "graded", K, dat_src_grm, dat_tgt_grm, common_p
)

# 3. V20c: GPCM K=4 (J = 16 per form, 8 common, N = 2000 per group)
set.seed(20261203)
a_gpcm <- rlnorm(J_tot, meanlog = 0.0, sdlog = 0.2)
step1 <- rnorm(J_tot, -0.9, 0.3)
step2 <- rnorm(J_tot, 0.0, 0.3)
step3 <- rnorm(J_tot, 0.9, 0.3)
# mirt simdata for gpcm uses d = [0, d1, d2, d3] where d_k = -a * sum_{c=1}^k b_c.
d_gpcm <- cbind(
  0,
  -a_gpcm * step1,
  -a_gpcm * (step1 + step2),
  -a_gpcm * (step1 + step2 + step3)
)
dat_tgt_gpcm <- as.data.frame(simdata(
  matrix(a_gpcm[c(idx_c, idx_t)]), d_gpcm[c(idx_c, idx_t), ], N, itemtype = "gpcm",
  mu = 0.0, sigma = matrix(1.0)
))
colnames(dat_tgt_gpcm) <- c(common_p, tgt_only_p)
dat_src_gpcm <- as.data.frame(simdata(
  matrix(a_gpcm[c(idx_c, idx_s)]), d_gpcm[c(idx_c, idx_s), ], N, itemtype = "gpcm",
  mu = 0.5, sigma = matrix(1.2^2)
))
colnames(dat_src_gpcm) <- c(common_p, src_only_p)
cases[["v20c_gpcm"]] <- run_uniform_case(
  "v20c_gpcm",
  "V20c: GPCM (K=4), J=16/form, 8 common, N=2000/group, target N(0,1), source N(0.5, 1.2^2)",
  "gpcm", K, dat_src_gpcm, dat_tgt_gpcm, common_p
)

# 4. V20d: 3PL (J = 16 per form, 8 common, N = 4000 per group, interior guessing parameters)
set.seed(20261204)
N_3pl <- 4000L
a_3pl <- rlnorm(J_tot, meanlog = 0.45, sdlog = 0.15)
d_3pl <- runif(J_tot, -1.8, -0.2)
g_3pl <- runif(J_tot, 0.16, 0.25)
dat_tgt_3pl <- as.data.frame(simdata(
  matrix(a_3pl[c(idx_c, idx_t)]), matrix(d_3pl[c(idx_c, idx_t)]), N_3pl,
  guess = g_3pl[c(idx_c, idx_t)], itemtype = "3PL",
  mu = 0.0, sigma = matrix(1.0)
))
colnames(dat_tgt_3pl) <- c(common_p, tgt_only_p)
dat_src_3pl <- as.data.frame(simdata(
  matrix(a_3pl[c(idx_c, idx_s)]), matrix(d_3pl[c(idx_c, idx_s)]), N_3pl,
  guess = g_3pl[c(idx_c, idx_s)], itemtype = "3PL",
  mu = 0.5, sigma = matrix(1.2^2)
))
colnames(dat_src_3pl) <- c(common_p, src_only_p)
cases[["v20d_3pl"]] <- run_uniform_case(
  "v20d_3pl",
  "V20d: 3PL, J=16/form, 8 common, N=4000/group, target N(0,1), source N(0.5, 1.2^2)",
  "3PL", 2L, dat_src_3pl, dat_tgt_3pl, common_p
)

# 5. V20e: V20a data with the source calibrated under a fixed N(0.3, 1.5) latent distribution,
#    so the symmetric criterion's reverse term uses a non-standard source grid.
cases[["v20e_2pl_srcgrid"]] <- run_uniform_case(
  "v20e_2pl_srcgrid",
  "V20e: V20a data; source calibrated with fixed N(0.3, 1.5) (non-standard source grid)",
  "2PL", 2L, dat_src_2pl, dat_tgt_2pl, common_2pl, src_latent = c(0.3, 1.5)
)

# 6. V20f: mixed format, 2PL + GPCM (K=4). Each form has 5 dichotomous + 3 polytomous common
#    items and 5 + 3 unique items (J = 16 per form, N = 2000 per group).
set.seed(20261206)
N <- 2000L
n_d <- 15L; n_p <- 9L
a_d <- rlnorm(n_d, meanlog = 0.1, sdlog = 0.25)
d_d <- rnorm(n_d, mean = 0, sd = 0.9)
a_p <- rlnorm(n_p, meanlog = 0.0, sdlog = 0.2)
st <- cbind(rnorm(n_p, -0.9, 0.3), rnorm(n_p, 0.0, 0.3), rnorm(n_p, 0.9, 0.3))
d_p <- cbind(0, -a_p * st[, 1], -a_p * (st[, 1] + st[, 2]), -a_p * rowSums(st))
mixed_form <- function(id, ip, mu, s2, names_d, names_p) {
  th <- matrix(rnorm(N, mu, sqrt(s2)))
  dd <- simdata(matrix(a_d[id]), matrix(d_d[id]), N, itemtype = "2PL", Theta = th)
  dp <- simdata(matrix(a_p[ip]), d_p[ip, , drop = FALSE], N, itemtype = "gpcm", Theta = th)
  out <- as.data.frame(cbind(dd, dp))
  colnames(out) <- c(names_d, names_p)
  out
}
common_mix_d <- sprintf("cd%02d", 1:5)
common_mix_p <- sprintf("cp%02d", 1:3)
dat_tgt_mix <- mixed_form(
  c(1:5, 6:10), c(1:3, 4:6), 0.0, 1.0,
  c(common_mix_d, sprintf("td%02d", 1:5)), c(common_mix_p, sprintf("tp%02d", 1:3))
)
dat_src_mix <- mixed_form(
  c(1:5, 11:15), c(1:3, 7:9), 0.5, 1.2^2,
  c(common_mix_d, sprintf("sd%02d", 1:5)), c(common_mix_p, sprintf("sp%02d", 1:3))
)
mix_types <- function(dat) {
  setNames(ifelse(grepl("^.d", colnames(dat)), "2PL", "gpcm"), colnames(dat))
}
mix_cats <- function(dat) {
  setNames(ifelse(grepl("^.d", colnames(dat)), 2L, 4L), colnames(dat))
}
cases[["v20f_mixed"]] <- run_linking_case(
  "v20f_mixed",
  "V20f: mixed 2PL + GPCM (K=4), J=16/form, 5+3 common, N=2000/group, source N(0.5, 1.2^2)",
  mix_types(dat_tgt_mix), mix_types(dat_src_mix), mix_cats(dat_tgt_mix), mix_cats(dat_src_mix),
  dat_src_mix, dat_tgt_mix, c(common_mix_d, common_mix_p)
)

res <- list(
  meta = list(
    case = "v20_linking",
    description = paste(
      "V20: characteristic-curve IRT scale linking (Haebara / Stocking-Lord, non-symmetric and",
      "symmetric) for 2PL, GRM, GPCM, 3PL, a non-standard source grid, and mixed 2PL + GPCM."
    ),
    note = paste(
      "mirt does not implement IRT scale linking; each form is calibrated with mirt and the",
      "linking criteria, optima (A, B), and Ogasawara (2001) delta-method SEs are computed in",
      "base R directly from Haebara (1980), Stocking & Lord (1983), and Kim & Lee (2006) using",
      "mirt's probtrace() and expected.item() for characteristic curves."
    ),
    r_version = R.version.string,
    mirt_version = mirt_version_string(), mirt_sha = mirt_commit_sha(),
    TOL = 1e-10,
    quadpts = quadpts,
    theta_lim = theta_lim,
    grid_check = theta_lim,
    accelerate = "none",
    optimizer = "NR",
    SE_type = "Oakes",
    converged = TRUE
  ),
  cases = cases
)

path <- file.path("tests", "reference", "v20_linking.json")
write_json(res, path, auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null")
cat("wrote", path, "\n")
