# Scale linking for 4PL, nominal and multidimensional models (VALIDATION.md V40).
# Run from the repository root:  Rscript validation/r/m5_linking_extensions_reference.R
# Writes tests/reference/v40_linking_unidim.json (4PL, nominal, 2PL + nominal) and
# tests/reference/v40_linking_multidim.json (two- and three-factor confirmatory models).
# CI does not run R; the JSON files are committed. Set V40_PART=unidim or V40_PART=multidim to
# regenerate only one file.
#
# mirt does not implement IRT scale linking. As in m5_linking_reference.R (V20), each form is
# calibrated separately with mirt (accelerate = "none", TOL = 1e-10, SE.type = "Oakes"; latent
# means 0 and variances 1 in both calibrations, factor correlations free), and the linking
# constants, criterion values and Ogasawara (2001) delta-method covariance matrices are computed by
# an independent base-R implementation of the formulas:
#   - criteria (Haebara, 1980; Stocking & Lord, 1983; Kim & Lee, 2006; multidimensional forms
#     Oshima, Davey & Lee, 2000) from mirt's probtrace() / expected.item() on the calibration grid,
#     minimized with optim(BFGS) + nlm() and a central-finite-difference Newton polish;
#   - delta-method covariances from category probabilities written out in R (item_probs_r) and
#     nested central finite differences.
#
# Transformation theta* = A theta + beta (source scale -> target scale):
#   forward term:  target curves at the target grid theta_q against source curves at
#                  A^{-1} (theta_q - beta);
#   reverse term (symmetric): source curves at the source grid against target curves at
#                  A theta_q + beta.
# Parameterizations of A (D = number of factors; D = 1 is the scalar (A, B) of V20):
#   "diagonal":   par = (A_1..A_D, beta_1..beta_D),            A = diag(A_m), A_m > 0;
#   "oblique":    par = (A_11, A_12, .., A_DD (row-major), beta), det(A) > 0;
#   "orthogonal": par = (k, phi, beta), A = k R(phi), k > 0, R a rotation: for D = 2
#                 R = [[cos phi, -sin phi], [sin phi, cos phi]]; for D = 3 Rodrigues' formula for
#                 exp(S), S[i, j] = -phi_ij, S[j, i] = phi_ij (i < j; phi = (phi_12, phi_13, phi_23)).
# Covariances are reported for (vec(A) row-major, beta) in every parameterization.
suppressMessages({
  library(mirt)
  library(jsonlite)
})

part <- Sys.getenv("V40_PART", "all")

mirt_args <- function(quadpts, theta_lim, optimizer = "NR", control = NULL) {
  out <- list(
    TOL = 1e-10, accelerate = "none", optimizer = optimizer, SE = TRUE, SE.type = "Oakes",
    quadpts = quadpts, verbose = FALSE,
    technical = list(NCYCLES = 50000L, theta_lim = theta_lim)
  )
  if (!is.null(control)) out$control <- control
  out
}

# ---- Transformation parameterizations -----------------------------------------------------------

rotation <- function(phi, D) {
  if (D == 2L) {
    return(matrix(c(cos(phi), sin(phi), -sin(phi), cos(phi)), 2L, 2L))
  }
  stopifnot(D == 3L)
  S <- matrix(0, 3L, 3L)
  S[1, 2] <- -phi[1]; S[2, 1] <- phi[1]
  S[1, 3] <- -phi[2]; S[3, 1] <- phi[2]
  S[2, 3] <- -phi[3]; S[3, 2] <- phi[3]
  t <- sqrt(sum(phi^2))
  if (t < 1e-12) return(diag(3L) + S)
  diag(3L) + sin(t) / t * S + (1 - cos(t)) / t^2 * (S %*% S)
}

unpack <- function(par, D, transformation) {
  if (transformation == "diagonal") {
    return(list(A = diag(par[seq_len(D)], D), beta = par[D + seq_len(D)],
                ok = all(par[seq_len(D)] > 0)))
  }
  if (transformation == "oblique") {
    A <- matrix(par[seq_len(D * D)], D, D, byrow = TRUE)
    return(list(A = A, beta = par[D * D + seq_len(D)], ok = det(A) > 0))
  }
  n_phi <- D * (D - 1L) / 2L
  list(A = par[1] * rotation(par[1L + seq_len(n_phi)], D),
       beta = par[1L + n_phi + seq_len(D)], ok = par[1] > 0)
}

# (vec(A) row-major, beta) as a function of par.
constants <- function(par, D, transformation) {
  tr <- unpack(par, D, transformation)
  c(as.numeric(t(tr$A)), tr$beta)
}

to_source <- function(Theta, tr) sweep(Theta, 2L, tr$beta) %*% t(solve(tr$A))
to_target <- function(Theta, tr) sweep(Theta %*% t(tr$A), 2L, tr$beta, `+`)

# ---- Criteria from mirt's probtrace() / expected.item() -----------------------------------------

make_criterion <- function(m_src, m_tgt, common_items, Theta, w_tgt, w_src, method, symmetric,
                           D, transformation) {
  src_ext <- lapply(common_items, function(nm) extract.item(m_src, nm))
  tgt_ext <- lapply(common_items, function(nm) extract.item(m_tgt, nm))
  n_common <- length(common_items)
  curves <- function(ext, th) {
    if (method == "haebara") return(probtrace(ext, th))
    matrix(expected.item(ext, th))
  }
  # expected.item() scores categories 0..K-1 (also for nominal items).
  for (j in seq_len(n_common)) {
    pt <- probtrace(tgt_ext[[j]], Theta)
    manual <- as.numeric(pt %*% (seq_len(ncol(pt)) - 1L))
    stopifnot(max(abs(expected.item(tgt_ext[[j]], Theta) - manual)) < 1e-12)
  }
  direction <- function(fixed, moving_ext, th_moved, w) {
    if (method == "haebara") {
      acc <- numeric(nrow(th_moved))
      for (j in seq_len(n_common)) {
        d <- fixed[[j]] - curves(moving_ext[[j]], th_moved)
        acc <- acc + rowSums(d * d)
      }
      return(sum(w * acc))
    }
    tcc_fixed <- Reduce(`+`, fixed)
    tcc_moved <- Reduce(`+`, lapply(moving_ext, function(it) curves(it, th_moved)))
    sum(w * (tcc_fixed - tcc_moved)^2)
  }
  tgt_fixed <- lapply(tgt_ext, function(it) curves(it, Theta))
  src_fixed <- lapply(src_ext, function(it) curves(it, Theta))
  function(par) {
    if (!all(is.finite(par))) return(1e30)
    tr <- unpack(par, D, transformation)
    if (!tr$ok) return(1e30)
    val <- direction(tgt_fixed, src_ext, to_source(Theta, tr), w_tgt)
    if (symmetric) val <- val + direction(src_fixed, tgt_ext, to_target(Theta, tr), w_src)
    val
  }
}

grad_fd <- function(fn, x, eps = 1e-6) {
  vapply(seq_along(x), function(i) {
    e <- numeric(length(x)); e[i] <- eps
    (fn(x + e) - fn(x - e)) / (2 * eps)
  }, numeric(1))
}

hess_fd <- function(fn, x, eps_outer = 1e-5, eps_inner = 1e-6) {
  H <- vapply(seq_along(x), function(i) {
    e <- numeric(length(x)); e[i] <- eps_outer
    (grad_fd(fn, x + e, eps_inner) - grad_fd(fn, x - e, eps_inner)) / (2 * eps_outer)
  }, numeric(length(x)))
  0.5 * (H + t(H))
}

solve_base_r <- function(fn, start) {
  gr <- function(x) grad_fd(fn, x)
  opt <- optim(start, fn, gr = gr, method = "BFGS",
               control = list(reltol = 1e-15, abstol = 1e-15, maxit = 2000))
  pol <- suppressWarnings(
    nlm(fn, opt$par, ndigit = 15, gradtol = 1e-13, steptol = 1e-13, iterlim = 2000)
  )
  x <- if (fn(pol$estimate) <= fn(opt$par)) pol$estimate else opt$par
  for (iter in seq_len(20L)) {
    g <- gr(x)
    if (max(abs(g)) < 1e-12) break
    step <- tryCatch(-solve(hess_fd(fn, x), g), error = function(err) rep(0, length(x)))
    cand <- x + step
    if (all(is.finite(cand)) && fn(cand) < 1e29 && max(abs(gr(cand))) < max(abs(g))) {
      x <- cand
    } else {
      break
    }
  }
  g <- gr(x)
  stopifnot(all(is.finite(x)), fn(x) < 1e29, max(abs(g)) < 1e-9)
  # The reported optimum must be a strict local minimum.
  stopifnot(min(eigen(hess_fd(fn, x), symmetric = TRUE, only.values = TRUE)$values) > 0)
  list(par = x, criterion = fn(x), grad_norm = max(abs(g)))
}

# ---- Category probabilities written out in R (for the delta method) ----------------------------
# `full` is the item's named mod2values() vector (natural scale): a1..aD, then d, g, u
# (dichotomous), d1.. (graded), or ak0.., d0.. (gpcm / nominal).

item_probs_r <- function(full, Theta, itemtype, D) {
  lin <- as.numeric(Theta %*% full[paste0("a", seq_len(D))])
  if (itemtype %in% c("2PL", "3PL", "4PL")) {
    p1 <- full[["g"]] + (full[["u"]] - full[["g"]]) * plogis(lin + full[["d"]])
    return(cbind(1 - p1, p1))
  }
  if (itemtype == "graded") {
    ds <- full[grepl("^d[0-9]+$", names(full))]
    cum <- cbind(1, vapply(ds, function(dk) plogis(lin + dk), numeric(length(lin))), 0)
    return(cum[, -ncol(cum), drop = FALSE] - cum[, -1L, drop = FALSE])
  }
  stopifnot(itemtype %in% c("gpcm", "nominal"))
  ak <- full[grepl("^ak[0-9]+$", names(full))]
  ds <- full[grepl("^d[0-9]+$", names(full))]
  eta <- outer(lin, ak) + rep(ds, each = length(lin))
  e <- exp(eta - apply(eta, 1L, max))
  e / rowSums(e)
}

# Common items' full parameter vectors, their free entries (natural scale, item-major) and the
# matching block of vcov(m). mirt's vcov is on the logit scale for g and u; it is converted with
# the Jacobian p (1 - p).
extract_common <- function(m, common_items, D) {
  sv <- mod2values(m)
  vc <- vcov(m)
  free_rows <- sv[sv$est, , drop = FALSE]
  stopifnot(nrow(free_rows) == nrow(vc))
  jac <- rep(1.0, nrow(vc))
  is_p <- free_rows$name %in% c("g", "u")
  jac[is_p] <- free_rows$value[is_p] * (1 - free_rows$value[is_p])
  vc_nat <- (jac %o% jac) * vc
  full <- list(); where <- list(); idx <- integer(0); labels <- list()
  for (it in common_items) {
    rows <- sv[sv$item == it, , drop = FALSE]
    full[[it]] <- setNames(rows$value, rows$name)
    free_names <- rows$name[rows$est]
    pos <- vapply(free_names, function(pn) {
      which(free_rows$item == it & free_rows$name == pn)
    }, integer(1))
    where[[it]] <- list(names = free_names, at = length(idx) + seq_along(pos))
    idx <- c(idx, pos)
    for (pn in free_names) {
      labels[[length(labels) + 1L]] <- c(it, if (D == 1L) sub("^a1$", "a", pn) else pn)
    }
  }
  list(full = full, where = where, vals = free_rows$value[idx],
       vcov = unname(vc_nat[idx, idx, drop = FALSE]), labels = labels)
}

ogasawara_r <- function(cs, ct, common_items, itemtypes, Theta, w_tgt, w_src, method, symmetric,
                        D, transformation, par_hat) {
  plug <- function(side, vals) {
    lapply(setNames(common_items, common_items), function(it) {
      full <- side$full[[it]]
      full[side$where[[it]]$names] <- vals[side$where[[it]]$at]
      full
    })
  }
  curves <- function(pars, th) {
    lapply(common_items, function(it) {
      p <- item_probs_r(pars[[it]], th, itemtypes[[it]], D)
      if (method == "haebara") p else p %*% (seq_len(ncol(p)) - 1L)
    })
  }
  direction <- function(fixed, moved, w) {
    if (method == "haebara") {
      return(sum(w * Reduce(`+`, Map(function(f, m) rowSums((f - m)^2), fixed, moved))))
    }
    sum(w * (Reduce(`+`, fixed) - Reduce(`+`, moved))^2)
  }
  crit <- function(par, vs, vt) {
    tr <- unpack(par, D, transformation)
    stopifnot(tr$ok)
    ps <- plug(cs, vs); pt <- plug(ct, vt)
    val <- direction(curves(pt, Theta), curves(ps, to_source(Theta, tr)), w_tgt)
    if (symmetric) {
      val <- val + direction(curves(ps, Theta), curves(pt, to_target(Theta, tr)), w_src)
    }
    val
  }
  eps <- 1e-5
  grad_c <- function(par, vs, vt) grad_fd(function(x) crit(x, vs, vt), par, eps)
  p <- length(par_hat)
  H_cc <- vapply(seq_len(p), function(i) {
    e <- numeric(p); e[i] <- eps
    (grad_c(par_hat + e, cs$vals, ct$vals) - grad_c(par_hat - e, cs$vals, ct$vals)) / (2 * eps)
  }, numeric(p))
  H_cc <- 0.5 * (H_cc + t(H_cc))
  cross <- function(vals, side) {
    vapply(seq_along(vals), function(i) {
      e <- numeric(length(vals)); e[i] <- eps
      if (side == "s") {
        (grad_c(par_hat, vals + e, ct$vals) - grad_c(par_hat, vals - e, ct$vals)) / (2 * eps)
      } else {
        (grad_c(par_hat, cs$vals, vals + e) - grad_c(par_hat, cs$vals, vals - e)) / (2 * eps)
      }
    }, numeric(p))
  }
  J_s <- -solve(H_cc, cross(cs$vals, "s"))
  J_t <- -solve(H_cc, cross(ct$vals, "t"))
  cov_par <- J_s %*% cs$vcov %*% t(J_s) + J_t %*% ct$vcov %*% t(J_t)
  cov_par <- 0.5 * (cov_par + t(cov_par))
  # Jacobian of (vec(A), beta) with respect to par (central differences).
  G <- vapply(seq_len(p), function(i) {
    e <- numeric(p); e[i] <- 1e-6
    (constants(par_hat + e, D, transformation) - constants(par_hat - e, D, transformation)) / 2e-6
  }, numeric(D * D + D))
  cov_const <- G %*% cov_par %*% t(G)
  list(vcov_par = unname(cov_par), vcov = unname(0.5 * (cov_const + t(cov_const))))
}

# ---- Extraction for the JSON file ---------------------------------------------------------------

flex_params <- function(full, itemtype, D) {
  slopes <- full[paste0("a", seq_len(D))]
  names(slopes) <- if (D == 1L) "a" else paste0("a", seq_len(D))
  K <- sum(grepl("^d[0-9]+$", names(full)))
  rest <- switch(itemtype,
    "2PL" = full["d"],
    "3PL" = full[c("d", "g", "u")],
    "4PL" = full[c("d", "g", "u")],
    "graded" = full[paste0("d", seq_len(K))],
    "gpcm" = full[paste0("d", seq_len(K - 1L))],
    "nominal" = full[c(if (K > 2L) paste0("ak", seq_len(K - 2L)), paste0("d", seq_len(K - 1L)))]
  )
  as.list(c(slopes, rest))
}

extract_form <- function(m, dat, itemtypes, D, factors) {
  sv <- mod2values(m)
  items <- colnames(dat)
  params <- lapply(setNames(items, items), function(it) {
    rows <- sv[sv$item == it, , drop = FALSE]
    flex_params(setNames(rows$value, rows$name), itemtypes[[it]], D)
  })
  gp <- sv[sv$item == "GROUP", , drop = FALSE]
  gv <- setNames(gp$value, gp$name)
  latent <- list()
  if (D == 1L) {
    latent$mean <- gv[["MEAN_1"]]; latent$var <- gv[["COV_11"]]
  } else {
    for (i in seq_len(D)) latent[[sprintf("mean_%d", i)]] <- gv[[sprintf("MEAN_%d", i)]]
    for (i in seq_len(D)) latent[[sprintf("var_%d", i)]] <- gv[[sprintf("COV_%d%d", i, i)]]
    for (i in seq_len(D - 1L)) for (j in seq(i + 1L, D)) {
      latent[[sprintf("cor_%d%d", i, j)]] <- gv[[sprintf("COV_%d%d", j, i)]]
    }
  }
  key <- apply(dat, 1, paste, collapse = ",")
  first <- !duplicated(key)
  list(
    items = items,
    n_obs = nrow(dat),
    patterns = unname(as.matrix(dat[first, , drop = FALSE])),
    counts = as.integer(table(key)[key[first]]),
    loglik = extract.mirt(m, "logLik"),
    converged = extract.mirt(m, "converged"),
    iterations = extract.mirt(m, "iterations"),
    itemtypes = as.list(itemtypes[items]),
    n_cats = as.list(setNames(extract.mirt(m, "K"), items)),
    factors = factors,
    latent = latent,
    params = params
  )
}

grid_weights <- function(m, D) {
  Theta <- m@Model$Theta
  gv <- coef(m, simplify = TRUE)
  Sigma <- gv$cov
  mu <- as.numeric(gv$means)
  z <- sweep(Theta, 2L, mu)
  w <- exp(-0.5 * rowSums((z %*% solve(Sigma)) * z))
  w <- w / sum(w)
  stopifnot(max(abs(w - as.numeric(m@Internals$Prior[[1]]))) < 1e-12)
  w
}

# ---- One case -----------------------------------------------------------------------------------
# model_*: 1 (unidimensional) or a mirt.model() string; factors_*: named list factor -> items
# (NULL for unidimensional); start: named list transformation -> starting par for the solver;
# truth: list(A = D x D matrix, beta = vector) of the simulation.

run_case <- function(case_name, description, dat_src, dat_tgt, itemtypes_src, itemtypes_tgt,
                     common_items, D, quadpts, theta_lim, transformations, truth,
                     model_src = 1, model_tgt = 1, factors_src = NULL, factors_tgt = NULL,
                     pars_fn = NULL, optimizer = "NR", control = NULL) {
  fit_form <- function(dat, model, itemtypes) {
    model_obj <- if (is.character(model)) mirt.model(model) else model
    args <- c(list(dat, model_obj, itemtype = unname(itemtypes[colnames(dat)])),
              mirt_args(quadpts, theta_lim, optimizer, control))
    if (!is.null(pars_fn)) args$pars <- pars_fn(dat, model_obj, itemtypes)
    t0 <- proc.time()[3L]
    m <- do.call(mirt, args)
    cat(sprintf("  %-22s fit: logLik=%.4f iter=%d time=%.0fs\n", case_name,
                extract.mirt(m, "logLik"), extract.mirt(m, "iterations"), proc.time()[3L] - t0))
    stopifnot(extract.mirt(m, "converged"), all(range(m@Model$Theta) == theta_lim),
              nrow(m@Model$Theta) == quadpts^D)
    m
  }
  m_tgt <- fit_form(dat_tgt, model_tgt, itemtypes_tgt)
  m_src <- fit_form(dat_src, model_src, itemtypes_src)
  stopifnot(all(itemtypes_tgt[common_items] == itemtypes_src[common_items]))
  # Asymptotes of every 3PL / 4PL item must be interior (no boundary solution).
  for (m in list(m_tgt, m_src)) {
    sv <- mod2values(m)
    gu <- sv[sv$est & sv$name %in% c("g", "u"), , drop = FALSE]
    stopifnot(all(gu$value > 0.01 & gu$value < 0.99))
  }
  Theta <- m_tgt@Model$Theta
  stopifnot(max(abs(Theta - m_src@Model$Theta)) < 1e-14)
  w_tgt <- grid_weights(m_tgt, D)
  w_src <- grid_weights(m_src, D)

  cs <- extract_common(m_src, common_items, D)
  ct <- extract_common(m_tgt, common_items, D)
  it_common <- as.list(itemtypes_tgt[common_items])

  linking_out <- list()
  for (transformation in transformations) {
    true_const <- c(as.numeric(t(truth$A)), truth$beta)
    start <- switch(transformation,
      diagonal = c(diag(truth$A), truth$beta),
      oblique = true_const,
      orthogonal = c(mean(diag(truth$A)), rep(0, D * (D - 1L) / 2L), truth$beta)
    )
    for (method in c("stocking_lord", "haebara")) for (symmetric in c(FALSE, TRUE)) {
      vname <- paste0(method, if (symmetric) "_sym" else "")
      if (D > 1L) vname <- paste0(transformation, "_", vname)
      fn <- make_criterion(m_src, m_tgt, common_items, Theta, w_tgt, w_src, method, symmetric,
                           D, transformation)
      sol <- solve_base_r(fn, start)
      se <- ogasawara_r(cs, ct, common_items, it_common, Theta, w_tgt, w_src, method, symmetric,
                        D, transformation, sol$par)
      tr <- unpack(sol$par, D, transformation)
      linking_out[[vname]] <- list(
        method = method, symmetric = symmetric, transformation = transformation,
        par = sol$par, A = tr$A, beta = tr$beta, criterion = sol$criterion,
        grad_norm = sol$grad_norm, vcov_par = se$vcov_par, vcov = se$vcov
      )
      cat(sprintf("  %-22s %-32s A=[%s] beta=[%s] crit=%.4e |g|=%.1e\n", case_name, vname,
                  paste(sprintf("%.4f", as.numeric(t(tr$A))), collapse = " "),
                  paste(sprintf("%.4f", tr$beta), collapse = " "), sol$criterion, sol$grad_norm))
    }
  }

  list(
    meta = list(
      case = case_name, description = description, n_factors = D, quadpts = quadpts,
      theta_lim = theta_lim, grid_check = range(Theta), optimizer = optimizer, converged = TRUE
    ),
    truth = truth,
    common_items = common_items,
    common_vcov = list(
      source = list(labels = cs$labels, values = cs$vals, vcov = cs$vcov),
      target = list(labels = ct$labels, values = ct$vals, vcov = ct$vcov)
    ),
    target = extract_form(m_tgt, dat_tgt, itemtypes_tgt, D, factors_tgt),
    source = extract_form(m_src, dat_src, itemtypes_src, D, factors_src),
    linking = linking_out
  )
}

write_bundle <- function(cases, file, description) {
  res <- list(
    meta = list(
      case = sub("\\.json$", "", file),
      description = description,
      note = paste(
        "mirt does not implement IRT scale linking; each form is calibrated with mirt and the",
        "criteria, optima and Ogasawara (2001) delta-method covariances are computed in base R",
        "from the published formulas using mirt's probtrace() and expected.item()."
      ),
      r_version = R.version.string,
      mirt_version = as.character(packageVersion("mirt")),
      TOL = 1e-10, accelerate = "none", SE_type = "Oakes",
      theta_lim = c(-6, 6), grid_check = c(-6, 6), converged = TRUE
    ),
    cases = cases
  )
  path <- file.path("tests", "reference", file)
  write_json(res, path, auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null")
  cat("wrote", path, "\n")
}

uni_truth <- list(A = matrix(1.2), beta = 0.5)
named <- function(value, dat) setNames(rep(value, ncol(dat)), colnames(dat))

# =================================================================================================
# Part 1: unidimensional 4PL, nominal and 2PL + nominal (61 points on [-6, 6]).
# Target ~ N(0, 1), source ~ N(0.5, 1.2^2), both calibrated under N(0, 1): true A = 1.2, B = 0.5.
# =================================================================================================
if (part %in% c("all", "unidim")) {
  cases <- list()

  # ---- V40a: 4PL, J = 12 per form (8 common), N = 20000 per group ------------------------------
  # Design and start (a = 2) as in V11b: unregularized 4PL needs both asymptotes observed.
  set.seed(20261210)
  J_c <- 8L; J_u <- 4L; N <- 20000L; J_tot <- J_c + 2L * J_u
  a4 <- rlnorm(J_tot, 0.6, 0.15); d4 <- runif(J_tot, -0.5, 0.5)
  g4 <- runif(J_tot, 0.10, 0.22); u4 <- runif(J_tot, 0.78, 0.90)
  idx_c <- seq_len(J_c); idx_t <- J_c + seq_len(J_u); idx_s <- J_c + J_u + seq_len(J_u)
  sim4 <- function(idx, mu, s2, nms) {
    out <- as.data.frame(simdata(matrix(a4[idx]), matrix(d4[idx]), N, itemtype = "4PL",
                                 guess = g4[idx], upper = u4[idx], mu = mu, sigma = matrix(s2)))
    colnames(out) <- nms
    out
  }
  common4 <- sprintf("c%02d", idx_c)
  dat_tgt <- sim4(c(idx_c, idx_t), 0.0, 1.0, c(common4, sprintf("t%02d", seq_len(J_u))))
  dat_src <- sim4(c(idx_c, idx_s), 0.5, 1.2^2, c(common4, sprintf("s%02d", seq_len(J_u))))
  start_a2 <- function(dat, model, itemtypes) {
    sv <- mirt(dat, model, itemtype = unname(itemtypes[colnames(dat)]), pars = "values")
    sv$value[sv$name == "a1"] <- 2.0
    sv
  }
  cases[["v40a_4pl"]] <- run_case(
    "v40a_4pl",
    "V40a: 4PL, J=12/form, 8 common, N=20000/group, target N(0,1), source N(0.5, 1.2^2)",
    dat_src, dat_tgt, named("4PL", dat_src), named("4PL", dat_tgt), common4,
    D = 1L, quadpts = 61L, theta_lim = c(-6, 6), transformations = "diagonal",
    truth = uni_truth, pars_fn = start_a2
  )

  # ---- V40b: nominal K = 4, J = 12 per form (6 common), N = 3000 per group ---------------------
  # Scoring coefficients include unordered and negative values. NR starts at the GPCM point
  # (ak_k = k, d_k = log(p_k / p_0)) as in m5_nominal_reference.R.
  nominal_start <- function(dat, model, itemtypes) {
    sv <- mirt(dat, model, itemtype = unname(itemtypes[colnames(dat)]), pars = "values")
    for (itm in unique(as.character(sv$item[sv$class == "nominal"]))) {
      col <- dat[[itm]]
      K <- max(col) + 1L
      for (k in seq_len(K - 2L)) sv$value[sv$item == itm & sv$name == paste0("ak", k)] <- k
      p <- pmax(as.numeric(table(factor(col, levels = 0:(K - 1L)))) / length(col), 0.01)
      for (k in seq_len(K - 1L)) {
        sv$value[sv$item == itm & sv$name == paste0("d", k)] <- log(p[k + 1L] / p[1L])
      }
    }
    sv
  }
  sim_nominal <- function(a, ak, d, theta) {
    out <- matrix(0L, length(theta), length(a))
    for (j in seq_along(a)) {
      eta <- outer(a[j] * theta, ak[j, ]) + rep(d[j, ], each = length(theta))
      p <- exp(eta - apply(eta, 1L, max)); p <- p / rowSums(p)
      out[, j] <- rowSums(runif(length(theta)) > t(apply(p, 1L, cumsum))[, -ncol(p), drop = FALSE])
    }
    out
  }
  set.seed(20261211)
  J_c <- 6L; J_u <- 6L; N <- 3000L; J_tot <- J_c + 2L * J_u
  a_n <- rlnorm(J_tot, 0.0, 0.2)
  ak_n <- cbind(0, runif(J_tot, -0.6, 1.4), runif(J_tot, 1.2, 2.6), 3)
  d_n <- cbind(0, rnorm(J_tot, 0.4, 0.4), rnorm(J_tot, 0.3, 0.4), rnorm(J_tot, -0.3, 0.5))
  idx_c <- seq_len(J_c); idx_t <- J_c + seq_len(J_u); idx_s <- J_c + J_u + seq_len(J_u)
  common_n <- sprintf("c%02d", idx_c)
  form_n <- function(idx, mu, sd, nms) {
    out <- as.data.frame(sim_nominal(a_n[idx], ak_n[idx, ], d_n[idx, ], rnorm(N, mu, sd)))
    colnames(out) <- nms
    out
  }
  dat_tgt <- form_n(c(idx_c, idx_t), 0.0, 1.0, c(common_n, sprintf("t%02d", seq_len(J_u))))
  dat_src <- form_n(c(idx_c, idx_s), 0.5, 1.2, c(common_n, sprintf("s%02d", seq_len(J_u))))
  cases[["v40b_nominal"]] <- run_case(
    "v40b_nominal",
    "V40b: nominal (K=4), J=12/form, 6 common, N=3000/group, target N(0,1), source N(0.5, 1.2^2)",
    dat_src, dat_tgt, named("nominal", dat_src), named("nominal", dat_tgt), common_n,
    D = 1L, quadpts = 61L, theta_lim = c(-6, 6), transformations = "diagonal",
    truth = uni_truth, pars_fn = nominal_start
  )

  # ---- V40c: mixed 2PL + nominal (K = 3); 6 + 3 common, 4 + 2 unique per form, N = 3000 --------
  set.seed(20261212)
  N <- 3000L; n_d <- 14L; n_p <- 7L
  a_d <- rlnorm(n_d, 0.1, 0.25); d_d <- rnorm(n_d, 0, 0.9)
  a_p <- rlnorm(n_p, 0.0, 0.2)
  ak_p <- cbind(0, runif(n_p, -0.4, 1.2), 2)
  d_p <- cbind(0, rnorm(n_p, 0.3, 0.4), rnorm(n_p, -0.2, 0.5))
  form_mix <- function(id, ip, mu, sd, names_d, names_p) {
    th <- rnorm(N, mu, sd)
    dd <- simdata(matrix(a_d[id]), matrix(d_d[id]), N, itemtype = "2PL", Theta = matrix(th))
    out <- as.data.frame(cbind(dd, sim_nominal(a_p[ip], ak_p[ip, ], d_p[ip, ], th)))
    colnames(out) <- c(names_d, names_p)
    out
  }
  cd <- sprintf("cd%02d", 1:6); cp <- sprintf("cp%02d", 1:3)
  dat_tgt <- form_mix(c(1:6, 7:10), c(1:3, 4:5), 0.0, 1.0,
                      c(cd, sprintf("td%02d", 1:4)), c(cp, sprintf("tp%02d", 1:2)))
  dat_src <- form_mix(c(1:6, 11:14), c(1:3, 6:7), 0.5, 1.2,
                      c(cd, sprintf("sd%02d", 1:4)), c(cp, sprintf("sp%02d", 1:2)))
  mix_types <- function(dat) {
    setNames(ifelse(grepl("^.d", colnames(dat)), "2PL", "nominal"), colnames(dat))
  }
  cases[["v40c_mixed"]] <- run_case(
    "v40c_mixed",
    "V40c: mixed 2PL + nominal (K=3), J=15/form, 6+3 common, N=3000/group, source N(0.5, 1.2^2)",
    dat_src, dat_tgt, mix_types(dat_src), mix_types(dat_tgt), c(cd, cp),
    D = 1L, quadpts = 61L, theta_lim = c(-6, 6), transformations = "diagonal",
    truth = uni_truth, pars_fn = nominal_start
  )

  write_bundle(cases, "v40_linking_unidim.json", paste(
    "V40 (unidimensional): characteristic-curve scale linking (Haebara / Stocking-Lord,",
    "non-symmetric and symmetric) for 4PL, nominal and mixed 2PL + nominal common items."
  ))
}
