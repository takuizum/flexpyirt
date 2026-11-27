# V56 reference values (VALIDATION.md V56, DESIGN.md §27, #75): DavidianCurve and NormalMixture
# latent distributions per group in multiple-group models.
# Run from the repository root:  Rscript validation/r/v56_multigroup_shapes_reference.R
# Writes tests/reference/v56{a,b,c}_*.json. CI does not run R; the JSON files are committed.
#
# mirt's multipleGroup() cannot free the mean and variance of a Davidian group ("not currently
# supported"), and its dentype = "mixture-k" is a latent-class model. As V19 / V22 do for one group,
# each case therefore
#   1. fits flexpyirt's exact discretized model in mirt with multipleGroup(customGroup = ...),
#      whose density is the standardized Davidian curve (dcurver::ddc) or normal mixture (each
#      component normalized on the grid, then mixed) normalized on the grid, optimizer = "BFGS";
#   2. polishes mirt's solution by Newton steps on an independent R marginal log-likelihood
#      (central-difference gradient, Richardson Hessian) until max |grad| < 2e-6 or 40 steps
#      (the numerical gradient's noise floor is ~1e-5; max |grad| < 1e-4 is required and recorded),
#      and records the R Hessian SEs (the reference SEs) and the log-likelihood;
#   3. re-runs multipleGroup at the polished values with TOL = NaN for mirt's Oakes SEs (evidence);
#   4. computes per-group EAPs of every unique pattern in R from the polished parameters;
#   5. computes S-X2 per group with itemfit(use_dentype_estimate = TRUE) on a single-group mirt
#      object whose items and customGroup density are fixed at that group's polished values
#      (TOL = NaN); without use_dentype_estimate, itemfit's S-X2 integrates over a normal prior.
# Cases: (a) 2PL, 3 groups (bimodal, right- and left-skewed), Davidian-2, anchors q01-q06,
# q07-q10 free per group, focal means and variances free, 3% missing; (b) 2PL (J = 15), 2 groups,
# NormalMixture-2, every item invariant, focal mean / variance free; (c) FIPC: 2PL items fixed at bank values, 2 groups, Davidian-2, every group's mean and
# variance free.
source(file.path("validation", "r", "setup.R"))  # mirt from GitHub; mirt_version / mirt_sha
suppressMessages({
  library(mirt)
  library(dcurver)
  library(jsonlite)
})

out_dir <- file.path("tests", "reference")
LIM <- c(-6, 6)
QUADPTS <- 61L
NODES <- seq(LIM[1L], LIM[2L], length.out = QUADPTS)

# --- Davidian density (flexpyirt's cosine-first angles; DESIGN.md §16.4) ------------------------
phi_cos_to_c <- function(phi) {
  k <- length(phi)
  s <- sin(phi); cc <- cos(phi)
  cvec <- numeric(k + 1)
  cvec[1] <- prod(cc)
  for (j in 1:k) cvec[j + 1] <- s[j] * (if (j < k) prod(cc[(j + 1):k]) else 1)
  cvec
}
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
davidian_weights <- function(theta, p) {
  mean <- p[1L]; var <- p[2L]; phi <- p[-(1:2)]
  if (!is.finite(var) || var <= 0) return(rep(NaN, length(theta)))
  cvec <- phi_cos_to_c(phi)
  mom <- dc_moments_c(cvec)
  z <- mom["mean"] + sqrt(mom["var"]) * (theta - mean) / sqrt(var)
  d <- dcurver::ddc(z, c_to_phi_dc(cvec))
  s <- sum(d)
  if (!is.finite(s) || s <= 0) return(rep(NaN, length(theta)))
  d / s
}

# --- Normal mixture density (mirt's class parameterization, standardized; DESIGN.md §16.7) -----
mixture_weights <- function(theta, p) {
  mean <- p[1L]; var <- p[2L]
  sh <- matrix(p[-(1:2)], ncol = 3L, byrow = TRUE)
  rm <- c(0, sh[, 1L]); rv <- c(1, sh[, 2L]); lg <- c(0, sh[, 3L])
  if (!is.finite(var) || var <= 0 || any(rv <= 0)) return(rep(NaN, length(theta)))
  pi <- exp(lg - max(lg)); pi <- pi / sum(pi)
  M <- sum(pi * rm)
  V <- sum(pi * (rv + rm^2)) - M^2
  mu_c <- mean + sqrt(var) * (rm - M) / sqrt(V)
  s2_c <- var * rv / V
  w <- numeric(length(theta))
  for (c in seq_along(mu_c)) {
    lk <- -0.5 * (theta - mu_c[c])^2 / s2_c[c]
    e <- exp(lk - max(lk))
    w <- w + pi[c] * e / sum(e)
  }
  w
}

make_custom_group <- function(kind, k, start) {
  if (kind == "davidian") {
    nm <- c("mean", "var", paste0("phi_", seq_len(k)))
    lb <- c(-Inf, 1e-8, rep(-Inf, k))
    den_fn <- davidian_weights
  } else {
    nm <- c("mean", "var", as.vector(t(outer(2:k, c("rmean_", "rvar_", "logit_"),
                                                function(c, p) paste0(p, c)))))
    lb <- c(-Inf, 1e-8, rep(c(-Inf, 1e-8, -Inf), k - 1L))
    den_fn <- mixture_weights
  }
  par <- setNames(start, nm)
  den <- function(obj, Theta) den_fn(as.numeric(Theta), obj@par)
  createGroup(par = par, est = c(FALSE, FALSE, rep(TRUE, length(nm) - 2L)), den = den,
              nfact = 1L, lbound = lb, ubound = rep(Inf, length(nm)), derivType = "central")
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
        H[i, j] <- H[j, i] <- (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) +
                                 f(x - ei - ej)) / (4 * h * h)
      }
    }
  }
  H
}

# --- one case ---------------------------------------------------------------------------------
run_case <- function(case, description, dat, grp, groups, kind, k, start, anchors,
                     focal_free = TRUE, bank = NULL) {
  if (length(anchors) == 0L) anchors <- ""
  dat <- as.data.frame(dat)
  items <- colnames(dat)
  J <- length(items)
  den_fn <- if (kind == "davidian") davidian_weights else mixture_weights
  cg <- make_custom_group(kind, k, start)
  gfac <- factor(grp, levels = groups)
  tech <- list(NCYCLES = 20000L, theta_lim = LIM)
  sv <- multipleGroup(dat, 1L, group = gfac, itemtype = "2PL", customGroup = cg,
                      invariance = anchors, pars = "values", quadpts = QUADPTS, technical = tech)
  lat_names <- names(cg@par)
  for (g in groups) {
    sel <- sv$group == g & sv$item == "GROUP" & sv$name %in% c("mean", "var")
    sv[sel, "est"] <- if (is.null(bank)) (g != groups[1L] && focal_free) else TRUE
  }
  if (!is.null(bank)) {
    for (r in seq_len(nrow(bank))) {
      sel <- sv$item == bank$item[r]
      sv[sel & sv$name == "a1", c("value", "est")] <- list(bank$a[r], FALSE)
      sv[sel & sv$name == "d", c("value", "est")] <- list(bank$d[r], FALSE)
    }
  }
  fit_mg <- function(pars, tol, se) {
    multipleGroup(dat, 1L, group = gfac, itemtype = "2PL", customGroup = cg,
                  invariance = anchors, pars = pars, quadpts = QUADPTS, optimizer = "BFGS",
                  accelerate = "none", TOL = tol, SE = se, SE.type = "Oakes", technical = tech,
                  control = list(reltol = 1e-13, maxit = 1000L), verbose = FALSE)
  }
  elapsed <- system.time(m <- fit_mg(sv, 1e-10, FALSE))[["elapsed"]]
  stopifnot(isTRUE(extract.mirt(m, "converged")), all(range(m@Model$Theta) == LIM))
  iterations <- extract.mirt(m, "iterations")

  # Free parameters, with equality constraints (anchors) merged into one coordinate each.
  vals <- mod2values(m)
  free_rows <- which(vals$est)
  cons <- extract.mirt(m, "constrain")
  coord <- setNames(seq_along(free_rows), vals$parnum[free_rows])
  for (cl in cons) {
    keep <- coord[as.character(cl[1L])]
    for (pn in cl[-1L]) coord[as.character(pn)] <- keep
  }
  uniq <- sort(unique(coord))
  coord[] <- match(coord, uniq)
  n_free <- length(uniq)
  first_row <- free_rows[match(seq_len(n_free), coord)]
  eta0 <- vals$value[first_row]
  coord_of <- function(r) unname(coord[as.character(vals$parnum[r])])

  # Unique (group, pattern) rows; missing responses (NA) are skipped in the likelihood.
  key <- paste(grp, apply(dat, 1L, paste, collapse = ","))
  first <- !duplicated(key)
  pat <- as.matrix(dat[first, , drop = FALSE])
  pat_grp <- grp[first]
  cnt <- as.numeric(table(key)[key[first]])

  values_at <- function(eta) {
    v <- vals$value
    v[free_rows] <- eta[coord]
    v
  }
  group_par <- function(v, g) {
    sel <- vals$group == g
    a <- sapply(items, function(it) v[sel & vals$item == it & vals$name == "a1"])
    d <- sapply(items, function(it) v[sel & vals$item == it & vals$name == "d"])
    lat <- sapply(lat_names, function(nm) v[sel & vals$item == "GROUP" & vals$name == nm])
    list(a = a, d = d, lat = lat)
  }
  group_post <- function(v, g) {
    gp <- group_par(v, g)
    w <- den_fn(NODES, gp$lat)
    if (any(!is.finite(w)) || any(w < 0)) return(NULL)
    rows <- which(pat_grp == g)
    y <- pat[rows, , drop = FALSE]
    z <- outer(gp$a, NODES) + gp$d
    lp1 <- plogis(z, log.p = TRUE); lp0 <- plogis(-z, log.p = TRUE)
    obs <- !is.na(y)
    y0 <- ifelse(obs, y, 0)
    ell <- (y0 * obs) %*% lp1 + ((1 - y0) * obs) %*% lp0
    lj <- sweep(ell, 2L, log(w), "+")
    mx <- apply(lj, 1L, max)
    lse <- mx + log(rowSums(exp(lj - mx)))
    list(rows = rows, lse = lse, post = exp(lj - lse))
  }
  mll <- function(eta) {
    v <- values_at(eta)
    total <- 0
    for (g in groups) {
      gp <- group_post(v, g)
      if (is.null(gp)) return(-Inf)
      total <- total + sum(cnt[gp$rows] * gp$lse)
    }
    total
  }
  eps <- 1e-5
  grad_fn <- function(x) {
    vapply(seq_along(x), function(i) {
      e <- numeric(length(x)); e[i] <- eps
      (mll(x + e) - mll(x - e)) / (2 * eps)
    }, numeric(1))
  }
  eta <- eta0
  grad <- grad_fn(eta)
  max_grad_mirt <- max(abs(grad))
  n_polish <- 0L
  while (max(abs(grad)) > 2e-6 && n_polish < 40L) {
    H <- richardson_hessian(mll, eta)
    H <- 0.5 * (H + t(H))
    step <- solve(-H, grad)
    t_step <- 1.0
    ll0 <- mll(eta)
    while (mll(eta + t_step * step) < ll0 - 1e-9 && t_step > 1e-4) t_step <- t_step / 2
    eta <- eta + t_step * step
    grad <- grad_fn(eta)
    n_polish <- n_polish + 1L
    cat(case, "polish", n_polish, "max|grad| =", max(abs(grad)), "step =", t_step, "\n")
  }
  max_grad <- max(abs(grad))
  stopifnot(max_grad < 1e-4)
  H <- richardson_hessian(mll, eta)
  se_r <- sqrt(diag(solve(-0.5 * (H + t(H)))))

  # mirt Oakes SEs at the polished values (no EM cycles).
  sv2 <- vals
  sv2$value <- values_at(eta)
  m2 <- fit_mg(sv2, NaN, TRUE)
  vals2 <- mod2values(m2)
  stopifnot(max(abs(vals2$value[free_rows] - eta[coord])) < 1e-10)
  oakes_se <- rep(NA_real_, n_free)
  for (g in groups) {
    cf <- coef(m2, printSE = TRUE)[[g]]
    for (r in which(vals$group == g & vals$est)) {
      it <- vals$item[r]; nm <- vals$name[r]
      tab <- if (it == "GROUP") cf$GroupPars else cf[[it]]
      oakes_se[coord_of(r)] <- tab["SE", nm]
    }
  }

  v_hat <- values_at(eta)
  table_rows <- lapply(seq_len(nrow(vals)), function(r) {
    it <- vals$item[r]
    if (!(it %in% c(items, "GROUP")) || !(vals$name[r] %in% c("a1", "d", lat_names))) return(NULL)
    free <- vals$est[r]
    list(group = vals$group[r], item = if (it == "GROUP") "__latent__" else it,
         param = if (vals$name[r] == "a1") "a" else vals$name[r], value = v_hat[r],
         free = free, se = if (free) se_r[coord_of(r)] else NA_real_,
         oakes_se = if (free) oakes_se[coord_of(r)] else NA_real_)
  })
  table_rows <- Filter(Negate(is.null), table_rows)

  # EAP per unique pattern, and S-X2 per group from single-group mirt objects.
  eap <- numeric(nrow(pat)); eap_sd <- numeric(nrow(pat))
  sx2 <- list()
  for (g in groups) {
    gp <- group_post(v_hat, g)
    eap[gp$rows] <- gp$post %*% NODES
    eap_sd[gp$rows] <- sqrt(pmax(gp$post %*% NODES^2 - eap[gp$rows]^2, 0))
    par_g <- group_par(v_hat, g)
    cg_g <- make_custom_group(kind, k, unname(par_g$lat))
    dat_g <- dat[grp == g, , drop = FALSE]
    svg <- mirt(dat_g, 1L, "2PL", customGroup = cg_g, pars = "values", quadpts = QUADPTS,
                technical = list(theta_lim = LIM))
    for (it in items) {
      svg[svg$item == it & svg$name == "a1", "value"] <- par_g$a[[it]]
      svg[svg$item == it & svg$name == "d", "value"] <- par_g$d[[it]]
    }
    svg$est <- FALSE
    mg <- mirt(dat_g, 1L, "2PL", customGroup = cg_g, pars = svg, quadpts = QUADPTS, TOL = NaN,
               technical = list(theta_lim = LIM), verbose = FALSE)
    stopifnot(abs(extract.mirt(mg, "logLik") - sum(cnt[gp$rows] * gp$lse)) < 1e-6)
    if (!anyNA(dat_g)) {
      # use_dentype_estimate = TRUE: the custom density, not mirt's default normal prior.
      it_fit <- itemfit(mg, fit_stats = "S_X2", mincell = 1, use_dentype_estimate = TRUE,
                        quadpts = QUADPTS, theta_lim = LIM)
      sx2[[g]] <- list(item = it_fit$item, s_x2 = it_fit$S_X2, df_mirt = it_fit$df.S_X2)
    }
  }

  list(
    meta = list(case = case, description = description, kind = kind, k = k,
                r_version = R.version.string, mirt_version = mirt_version_string(), mirt_sha = mirt_commit_sha(),
                dcurver_version = as.character(packageVersion("dcurver")),
                quadpts = QUADPTS, theta_lim = LIM, grid_check = range(m@Model$Theta),
                converged = extract.mirt(m, "converged"), iterations = iterations,
                optimizer = "BFGS (customGroup)", elapsed_seconds = elapsed,
                max_grad_mirt = max_grad_mirt, max_grad = max_grad, n_polish = n_polish),
    items = items, groups = I(groups), anchors = I(anchors),
    bank = bank,
    patterns = unname(pat), pattern_groups = pat_grp, counts = as.integer(cnt),
    loglik = mll(eta), loglik_mirt = extract.mirt(m, "logLik"),
    parameters = table_rows,
    eap = list(theta = as.numeric(eap), sd = as.numeric(eap_sd)),
    s_x2 = sx2
  )
}

two_pl <- function(theta, a, d) {
  p <- plogis(outer(theta, a) + rep(d, each = length(theta)))
  matrix(as.integer(runif(length(p)) < p), nrow(p))
}

# V56_CASES selects cases (default "abc"); each case is written to its own JSON file.
SELECTED <- strsplit(Sys.getenv("V56_CASES", "abc"), "")[[1L]]
cases <- list()

# (a) Davidian-2, three groups, anchors q01-q06 ------------------------------------------------
set.seed(20261127)
J <- 10
a <- rlnorm(J, 0, 0.25); d <- rnorm(J, 0, 0.8)
th <- list(c(rnorm(350, -0.9, 0.5), rnorm(350, 0.9, 0.5)),
           0.3 + (rgamma(650, 3, 2) - 1.5),
           -0.4 - (rgamma(600, 3, 2) - 1.5))
blocks <- lapply(seq_along(th), function(g) {
  a_g <- a; d_g <- d
  if (g > 1L) d_g[7:J] <- d[7:J] + rnorm(J - 6, 0, 0.3)
  two_pl(th[[g]], a_g, d_g)
})
dat_a <- do.call(rbind, blocks); colnames(dat_a) <- sprintf("q%02d", 1:J)
dat_a[matrix(runif(length(dat_a)) < 0.03, nrow(dat_a))] <- NA
grp_a <- rep(c("g1", "g2", "g3"), times = lengths(th))
if ("a" %in% SELECTED) cases$v56a_davidian_three_groups <- run_case(
  "v56a_davidian_three_groups",
  "V56a: 2PL, 3 groups, DavidianCurve(2), anchors q01-q06, focal mean/var free, 3% missing",
  dat_a, grp_a, c("g1", "g2", "g3"), "davidian", 2L, c(0, 1, 0.2, 0.2), sprintf("q%02d", 1:6))

# (b) NormalMixture-2, two groups, every item invariant ------------------------------------------
set.seed(20261128)
J <- 15
a <- rlnorm(J, 0.35, 0.2); d <- rnorm(J, 0, 0.8)
th <- list(c(rnorm(800, -1.1, 0.5), rnorm(700, 1.1, 0.5)),
           c(rnorm(500, -0.6, 0.55), rnorm(900, 1.5, 0.5)))
dat_b <- do.call(rbind, lapply(th, two_pl, a = a, d = d)); colnames(dat_b) <- sprintf("q%02d", 1:J)
grp_b <- rep(c("ref", "foc"), times = lengths(th))
if ("b" %in% SELECTED) cases$v56b_mixture_two_groups <- run_case(
  "v56b_mixture_two_groups",
  "V56b: 2PL (J = 15), 2 groups (1500, 1400), NormalMixture(2), every item invariant, focal mean/var free",
  dat_b, grp_b, c("ref", "foc"), "mixture", 2L, c(0, 1, 2.0, 1.0, 0.0), colnames(dat_b))

# (c) FIPC, Davidian-2, two groups, items fixed at bank values -----------------------------------
set.seed(20261129)
J <- 10
a <- rlnorm(J, 0, 0.25); d <- rnorm(J, 0, 0.8)
th <- list(0.4 + rgamma(700, 4, 2.5) - 1.6, c(rnorm(300, -1.2, 0.5), rnorm(500, 0.8, 0.6)))
dat_c <- do.call(rbind, lapply(th, two_pl, a = a, d = d)); colnames(dat_c) <- sprintf("q%02d", 1:J)
grp_c <- rep(c("y2025", "y2026"), times = lengths(th))
bank_c <- data.frame(item = colnames(dat_c), a = a, d = d)
if ("c" %in% SELECTED) cases$v56c_fipc_davidian <- run_case(
  "v56c_fipc_davidian",
  "V56c: FIPC, 2PL items fixed at bank values, 2 groups, DavidianCurve(2), all means/vars free",
  dat_c, grp_c, c("y2025", "y2026"), "davidian", 2L, c(0, 1, 0.2, 0.2), character(0),
  bank = bank_c)

for (nm in names(cases)) {
  path <- file.path(out_dir, paste0(nm, ".json"))
  write_json(cases[[nm]], path, auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null")
  cat("wrote", path, " loglik =", cases[[nm]]$loglik, " mirt =", cases[[nm]]$loglik_mirt,
      " polish =", cases[[nm]]$meta$n_polish, " max_grad_mirt =", cases[[nm]]$meta$max_grad_mirt,
      "\n")
}
