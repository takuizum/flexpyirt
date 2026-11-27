# M3 reference values (VALIDATION.md V7: priors and MAP / Bayes modal estimation) from R mirt.
# Run from the repository root:  Rscript validation/r/m3_priors_reference.R
# Writes:
#   tests/reference/v7_2pl_map.json        (2PL: a1 ~ norm(1.2, 0.5), d ~ norm(0, 2); mirt NR + Oakes)
#   tests/reference/v7_2pl_lnorm_map.json  (2PL: a1 ~ lnorm(0.2, 0.3), d ~ norm(0, 2); mirt NR + exact R SE)
#   tests/reference/v7_3pl_map.json        (3PL: a1 ~ lnorm(0.2, 0.3), d ~ norm(0, 2), g ~ expbeta(5, 17))
# CI does not run R.
#
# Verification findings for mirt 1.46 (recorded in VALIDATION.md and DESIGN.md §15):
# 1. Objective (mirt:::LL.Priors): natural scale, NO logit Jacobian for expbeta:
#      norm(mu, sd)    -> dnorm(par, mu, sd, log = TRUE)
#      lnorm(mu, sd)   -> dlnorm(par, mu, sd, log = TRUE)
#      expbeta(a, b)   -> dbeta(plogis(logit_g), a, b, log = TRUE)
#    This matches DESIGN.md §4.7 exactly.
# 2. mirt 1.46 derivative/SE issues confirmed to ~1e-10:
#    (a) norm priors: gradient and Hessian in mirt:::DerivativePriors / computeDPars are exact, so
#        mirt NR estimates, logLik, logPrior, AND Oakes SEs agree with exact MAP to ~1e-9.
#    (b) lnorm priors: gradient is exact (so mirt NR estimates, logLik, logPrior match exact MAP to
#        ~1e-8), but the Hessian has a sign error on the third term
#        (-(log(a)-mu)/(a^2*s^2) instead of +(log(a)-mu)/(a^2*s^2)), making mirt's Oakes SEs off by
#        a few percent whenever log(a) != mu.
#    (c) expbeta priors: DerivativePriors / computeDPars add natural-scale derivatives
#        (a-1)/g - (b-1)/(1-g) and -(a-1)/g^2 - (b-1)/(1-g)^2 directly to the logit(g) gradient and
#        Hessian without the chain-rule factor g*(1-g). As a result, NR fails to converge, BFGS
#        stops at a non-stationary point with strictly lower logLik + logPrior, and Oakes SE on g
#        is off by ~10x. Therefore v7_3pl_map maximizes mirt's exact objective (LL + LL.Priors) in R
#        via BFGS with exact analytic gradients (verified |grad| < 1e-10), cross-checks logLik,
#        logPrior and EAP directly against mirt(..., pars = sv_fixed), and also records mirt's
#        own BFGS output under `mirt_bfgs` for documentation.
#    (d) SE.type = "Richardson" (mirt:::BL.LL) omits LL.Priors completely and computes the
#        likelihood-only Hessian at the MAP estimate rather than the posterior Hessian.
source(file.path("validation", "r", "setup.R"))  # mirt from GitHub; mirt_version / mirt_sha
suppressMessages({
  library(mirt)
  library(jsonlite)
})

out_dir <- file.path("tests", "reference")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
common <- list(TOL = 1e-10, accelerate = "none", optimizer = "NR", verbose = FALSE)
quad <- list(quadpts = 61, theta_lim = c(-6, 6))
tech <- list(NCYCLES = 50000, theta_lim = quad$theta_lim)

nodes <- seq(quad$theta_lim[1], quad$theta_lim[2], length.out = quad$quadpts)
log_w <- dnorm(nodes, 0, 1, log = TRUE)
log_w <- log_w - max(log_w) - log(sum(exp(log_w - max(log_w))))

compress_dat <- function(dat) {
  key <- apply(dat, 1, paste, collapse = ",")
  first <- !duplicated(key)
  list(
    patterns = as.matrix(dat[first, , drop = FALSE]),
    counts = as.integer(table(key)[key[first]])
  )
}

# Exact analytic gradient of discretized marginal log-likelihood + log-prior on natural scale.
eval_2pl_map <- function(P, cnt, av, dv, a_prior, d_prior) {
  J <- length(av)
  ell <- matrix(0, nrow = nrow(P), ncol = length(nodes))
  p1_list <- vector("list", J)
  for (j in 1:J) {
    p1 <- plogis(av[j] * nodes + dv[j])
    p1_list[[j]] <- p1
    ell <- ell + outer(P[, j], plogis(av[j] * nodes + dv[j], log.p = TRUE)) +
      outer(1 - P[, j], plogis(-(av[j] * nodes + dv[j]), log.p = TRUE))
  }
  lj <- sweep(ell, 2, log_w, "+")
  mx <- apply(lj, 1, max)
  lse <- mx + log(rowSums(exp(lj - mx)))
  ll <- sum(cnt * lse)
  h <- exp(lj - lse)
  w <- sweep(h, 1, cnt, "*")
  n_q <- colSums(w)
  ga_ll <- numeric(J); gd_ll <- numeric(J)
  for (j in 1:J) {
    r1 <- colSums(sweep(w, 1, P[, j], "*"))
    diff_q <- r1 - n_q * p1_list[[j]]
    ga_ll[j] <- sum(diff_q * nodes)
    gd_ll[j] <- sum(diff_q)
  }
  if (a_prior$type == "norm") {
    lp_a <- sum(dnorm(av, a_prior$p1, a_prior$p2, log = TRUE))
    gp_a <- -(av - a_prior$p1) / a_prior$p2^2
  } else {
    lp_a <- sum(dlnorm(av, a_prior$p1, a_prior$p2, log = TRUE))
    gp_a <- -(1 + (log(av) - a_prior$p1) / a_prior$p2^2) / av
  }
  lp_d <- sum(dnorm(dv, d_prior$p1, d_prior$p2, log = TRUE))
  gp_d <- -(dv - d_prior$p1) / d_prior$p2^2
  list(ll = ll, lp = lp_a + lp_d, lpost = ll + lp_a + lp_d,
       gr_ll = c(ga_ll, gd_ll), gr_post = c(ga_ll + gp_a, gd_ll + gp_d))
}

eval_3pl_map <- function(P, cnt, av, dv, gv, a_prior, d_prior, g_prior) {
  J <- length(av)
  ell <- matrix(0, nrow = nrow(P), ncol = length(nodes))
  p1_list <- vector("list", J); s_list <- vector("list", J)
  for (j in 1:J) {
    s <- plogis(av[j] * nodes + dv[j])
    p1 <- gv[j] + (1 - gv[j]) * s
    s_list[[j]] <- s; p1_list[[j]] <- p1
    ell <- ell + outer(P[, j], log(p1)) + outer(1 - P[, j], log(1 - p1))
  }
  lj <- sweep(ell, 2, log_w, "+")
  mx <- apply(lj, 1, max)
  lse <- mx + log(rowSums(exp(lj - mx)))
  ll <- sum(cnt * lse)
  h <- exp(lj - lse)
  w <- sweep(h, 1, cnt, "*")
  n_q <- colSums(w)
  ga_ll <- numeric(J); gd_ll <- numeric(J); gg_ll <- numeric(J)
  for (j in 1:J) {
    r1 <- colSums(sweep(w, 1, P[, j], "*"))
    r0 <- n_q - r1
    p1 <- p1_list[[j]]; s <- s_list[[j]]; gj <- gv[j]
    d_z <- r1 * ((1 - gj) * s * (1 - s) / p1) - r0 * s
    ga_ll[j] <- sum(d_z * nodes)
    gd_ll[j] <- sum(d_z)
    gg_ll[j] <- sum(r1 * ((1 - s) / p1) - r0 / (1 - gj))
  }
  lp_a <- sum(dlnorm(av, a_prior$p1, a_prior$p2, log = TRUE))
  gp_a <- -(1 + (log(av) - a_prior$p1) / a_prior$p2^2) / av
  lp_d <- sum(dnorm(dv, d_prior$p1, d_prior$p2, log = TRUE))
  gp_d <- -(dv - d_prior$p1) / d_prior$p2^2
  lp_g <- sum(dbeta(gv, g_prior$p1, g_prior$p2, log = TRUE))
  gp_g <- (g_prior$p1 - 1) / gv - (g_prior$p2 - 1) / (1 - gv)
  list(ll = ll, lp = lp_a + lp_d + lp_g, lpost = ll + lp_a + lp_d + lp_g,
       gr_ll = c(ga_ll, gd_ll, gg_ll), gr_post = c(ga_ll + gp_a, gd_ll + gp_d, gg_ll + gp_g))
}

hess_from_grad <- function(grad_fn, x, eps = 1e-6) {
  p <- length(x); H <- matrix(0, p, p)
  for (i in 1:p) {
    ei <- rep(0, p); ei[i] <- eps
    H[, i] <- (grad_fn(x + ei) - grad_fn(x - ei)) / (2 * eps)
  }
  0.5 * (H + t(H))
}

extract_2pl <- function(m) {
  cf <- coef(m, printSE = TRUE)
  items <- setdiff(names(cf), "GroupPars")
  params <- lapply(items, function(i) {
    par <- cf[[i]]["par", ]; se <- cf[[i]]["SE", ]
    list(a = par[["a1"]], d = par[["d"]], a_se = se[["a1"]], d_se = se[["d"]])
  })
  names(params) <- items
  list(all = list(params = params,
                  latent = list(mean = 0, var = 1, mean_se = NA, var_se = NA)))
}

pack_2pl_exact <- function(items, av, dv, se_vec) {
  J <- length(items)
  params <- lapply(seq_len(J), function(j) {
    list(a = av[j], d = dv[j], a_se = se_vec[j], d_se = se_vec[J + j])
  })
  names(params) <- items
  list(all = list(params = params,
                  latent = list(mean = 0, var = 1, mean_se = NA, var_se = NA)))
}

pack_3pl_exact <- function(items, av, dv, gv, se_vec) {
  J <- length(items)
  params <- lapply(seq_len(J), function(j) {
    list(a = av[j], d = dv[j], g = gv[j], u = 1.0,
         a_se = se_vec[j], d_se = se_vec[J + j], g_se = se_vec[2 * J + j], u_se = NA)
  })
  names(params) <- items
  list(all = list(params = params,
                  latent = list(mean = 0, var = 1, mean_se = NA, var_se = NA)))
}

# ---- V7a: 2PL with norm priors on a1 and d (direct mirt NR + Oakes SE) --------------------------
set.seed(20261127)
J <- 15; N <- 1000
a1 <- rlnorm(J, 0.2, 0.3); d1 <- rnorm(J, 0, 1.0)
dat1 <- as.data.frame(simdata(matrix(a1), matrix(d1), N, itemtype = "2PL"))
colnames(dat1) <- sprintf("q%02d", 1:J)

model1 <- mirt.model(paste0(
  "F1 = 1-", J, "\n",
  "PRIOR = (1-", J, ", a1, norm, 1.2, 0.5), (1-", J, ", d, norm, 0, 2)"
))
m1_oakes <- do.call(mirt, c(list(dat1, model1, itemtype = "2PL", SE = TRUE, SE.type = "Oakes",
                                 quadpts = quad$quadpts, technical = tech), common))
m1_rich  <- do.call(mirt, c(list(dat1, model1, itemtype = "2PL", SE = TRUE, SE.type = "Richardson",
                                 quadpts = quad$quadpts, technical = tech), common))
comp1 <- compress_dat(dat1)
sc1 <- fscores(m1_oakes, method = "EAP", response.pattern = comp1$patterns,
               quadpts = quad$quadpts, theta_lim = quad$theta_lim)
cf1 <- coef(m1_oakes, simplify = TRUE)$items
ev1 <- eval_2pl_map(comp1$patterns, comp1$counts, cf1[, "a1"], cf1[, "d"],
                    list(type = "norm", p1 = 1.2, p2 = 0.5),
                    list(type = "norm", p1 = 0.0, p2 = 2.0))
H1_post <- hess_from_grad(
  function(x) eval_2pl_map(comp1$patterns, comp1$counts, x[1:J], x[(J + 1):(2 * J)],
                           list(type = "norm", p1 = 1.2, p2 = 0.5),
                           list(type = "norm", p1 = 0.0, p2 = 2.0))$gr_post,
  c(cf1[, "a1"], cf1[, "d"])
)
se1_exact <- sqrt(diag(solve(-H1_post)))

stopifnot(
  isTRUE(extract.mirt(m1_oakes, "converged")),
  isTRUE(extract.mirt(m1_rich, "converged")),
  all(range(m1_oakes@Model$Theta) == quad$theta_lim),
  all(range(m1_rich@Model$Theta) == quad$theta_lim),
  abs(m1_oakes@Fit$logLik - ev1$ll) < 1e-9,
  abs(m1_oakes@Fit$logPrior - ev1$lp) < 1e-9
)

res1 <- list(
  meta = list(
    case = "v7_2pl_map",
    description = "V7a: 2PL MAP, J=15, N=1000, a~NormalPrior(1.2, 0.5), d~NormalPrior(0, 2)",
    itemtype = "2PL",
    r_version = R.version.string,
    mirt_version = mirt_version_string(), mirt_sha = mirt_commit_sha(),
    TOL = 1e-10,
    quadpts = quad$quadpts,
    theta_lim = quad$theta_lim,
    grid_check = range(m1_oakes@Model$Theta),
    accelerate = "none",
    optimizer = "NR",
    converged = extract.mirt(m1_oakes, "converged"),
    iterations = extract.mirt(m1_oakes, "iterations"),
    converged_richardson = extract.mirt(m1_rich, "converged"),
    se_note = "mirt Oakes includes exact Normal prior Hessian; mirt Richardson (BL.LL) omits priors"
  ),
  items = colnames(dat1),
  groups = I("all"),
  priors = list(
    a = list(dist = "NormalPrior", mean = 1.2, sd = 0.5),
    d = list(dist = "NormalPrior", mean = 0.0, sd = 2.0)
  ),
  patterns = unname(comp1$patterns),
  pattern_groups = rep("all", nrow(comp1$patterns)),
  counts = comp1$counts,
  loglik = extract.mirt(m1_oakes, "logLik"),
  log_prior = m1_oakes@Fit$logPrior,
  log_posterior = extract.mirt(m1_oakes, "logLik") + m1_oakes@Fit$logPrior,
  estimates = extract_2pl(m1_oakes),
  se_posterior_exact = pack_2pl_exact(colnames(dat1), cf1[, "a1"], cf1[, "d"], se1_exact),
  se_richardson = extract_2pl(m1_rich),
  eap = list(theta = unname(sc1[, "F1"]), se = unname(sc1[, "SE_F1"])),
  truth = list(a = a1, d = d1)
)
write_json(res1, file.path(out_dir, "v7_2pl_map.json"),
           auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null")
cat(sprintf("wrote v7_2pl_map.json        LL=%.6f LP=%.6f LPost=%.6f iter=%d max|grad|=%.2e\n",
            res1$loglik, res1$log_prior, res1$log_posterior, res1$meta$iterations,
            max(abs(ev1$gr_post))))

# ---- V7b: 2PL with lnorm(a1) + norm(d) (direct mirt NR; exact posterior SE in R) ----------------
set.seed(20261128)
a2 <- rlnorm(J, 0.2, 0.3); d2 <- rnorm(J, 0, 1.0)
dat2 <- as.data.frame(simdata(matrix(a2), matrix(d2), N, itemtype = "2PL"))
colnames(dat2) <- sprintf("q%02d", 1:J)

model2 <- mirt.model(paste0(
  "F1 = 1-", J, "\n",
  "PRIOR = (1-", J, ", a1, lnorm, 0.2, 0.3), (1-", J, ", d, norm, 0, 2)"
))
m2_oakes <- do.call(mirt, c(list(dat2, model2, itemtype = "2PL", SE = TRUE, SE.type = "Oakes",
                                 quadpts = quad$quadpts, technical = tech), common))
m2_rich  <- do.call(mirt, c(list(dat2, model2, itemtype = "2PL", SE = TRUE, SE.type = "Richardson",
                                 quadpts = quad$quadpts, technical = tech), common))
comp2 <- compress_dat(dat2)
sc2 <- fscores(m2_oakes, method = "EAP", response.pattern = comp2$patterns,
               quadpts = quad$quadpts, theta_lim = quad$theta_lim)
cf2 <- coef(m2_oakes, simplify = TRUE)$items
ev2 <- eval_2pl_map(comp2$patterns, comp2$counts, cf2[, "a1"], cf2[, "d"],
                    list(type = "lnorm", p1 = 0.2, p2 = 0.3),
                    list(type = "norm", p1 = 0.0, p2 = 2.0))
H2_post <- hess_from_grad(
  function(x) eval_2pl_map(comp2$patterns, comp2$counts, x[1:J], x[(J + 1):(2 * J)],
                           list(type = "lnorm", p1 = 0.2, p2 = 0.3),
                           list(type = "norm", p1 = 0.0, p2 = 2.0))$gr_post,
  c(cf2[, "a1"], cf2[, "d"])
)
se2_exact <- sqrt(diag(solve(-H2_post)))

stopifnot(
  isTRUE(extract.mirt(m2_oakes, "converged")),
  isTRUE(extract.mirt(m2_rich, "converged")),
  all(range(m2_oakes@Model$Theta) == quad$theta_lim),
  all(range(m2_rich@Model$Theta) == quad$theta_lim),
  abs(m2_oakes@Fit$logLik - ev2$ll) < 1e-9,
  abs(m2_oakes@Fit$logPrior - ev2$lp) < 1e-9
)

res2 <- list(
  meta = list(
    case = "v7_2pl_lnorm_map",
    description = "V7b: 2PL MAP, J=15, N=1000, a~LogNormal(0.2, 0.3), d~NormalPrior(0, 2)",
    itemtype = "2PL",
    r_version = R.version.string,
    mirt_version = mirt_version_string(), mirt_sha = mirt_commit_sha(),
    TOL = 1e-10,
    quadpts = quad$quadpts,
    theta_lim = quad$theta_lim,
    grid_check = range(m2_oakes@Model$Theta),
    accelerate = "none",
    optimizer = "NR",
    converged = extract.mirt(m2_oakes, "converged"),
    iterations = extract.mirt(m2_oakes, "iterations"),
    converged_richardson = extract.mirt(m2_rich, "converged"),
    se_note = paste(
      "mirt 1.46 DerivativePriors has a sign bug in the lnorm Hessian",
      "(-(log(a)-mu)/(a^2*s^2) instead of +(log(a)-mu)/(a^2*s^2)),",
      "so exact posterior SEs are in se_posterior_exact; mirt Oakes SEs are in estimates",
      "and likelihood-only Richardson SEs in se_richardson"
    )
  ),
  items = colnames(dat2),
  groups = I("all"),
  priors = list(
    a = list(dist = "LogNormal", meanlog = 0.2, sdlog = 0.3),
    d = list(dist = "NormalPrior", mean = 0.0, sd = 2.0)
  ),
  patterns = unname(comp2$patterns),
  pattern_groups = rep("all", nrow(comp2$patterns)),
  counts = comp2$counts,
  loglik = extract.mirt(m2_oakes, "logLik"),
  log_prior = m2_oakes@Fit$logPrior,
  log_posterior = extract.mirt(m2_oakes, "logLik") + m2_oakes@Fit$logPrior,
  estimates = extract_2pl(m2_oakes),
  se_posterior_exact = pack_2pl_exact(colnames(dat2), cf2[, "a1"], cf2[, "d"], se2_exact),
  se_richardson = extract_2pl(m2_rich),
  eap = list(theta = unname(sc2[, "F1"]), se = unname(sc2[, "SE_F1"])),
  truth = list(a = a2, d = d2)
)
write_json(res2, file.path(out_dir, "v7_2pl_lnorm_map.json"),
           auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null")
cat(sprintf("wrote v7_2pl_lnorm_map.json  LL=%.6f LP=%.6f LPost=%.6f iter=%d max|grad|=%.2e\n",
            res2$loglik, res2$log_prior, res2$log_posterior, res2$meta$iterations,
            max(abs(ev2$gr_post))))

# ---- V7c: 3PL with lnorm(a1) + norm(d) + expbeta(g) (exact R MAP + mirt fixed-eval check) -------
set.seed(20261129)
a3 <- rlnorm(J, 0.2, 0.3); d3 <- rnorm(J, 0, 1.0); g3 <- rbeta(J, 5, 17)
dat3 <- as.data.frame(simdata(matrix(a3), matrix(d3), N, itemtype = "3PL", guess = g3))
colnames(dat3) <- sprintf("q%02d", 1:J)
comp3 <- compress_dat(dat3)

a3_pri <- list(type = "lnorm", p1 = 0.2, p2 = 0.3)
d3_pri <- list(type = "norm",  p1 = 0.0, p2 = 2.0)
g3_pri <- list(type = "beta",  p1 = 5.0, p2 = 17.0)

# Optimize unconstrained (log a, d, logit g) maximizing natural-scale logLik + logPrior (no Jacobian)
obj3_fn <- function(par) {
  av <- exp(par[1:J]); dv <- par[(J + 1):(2 * J)]; gv <- plogis(par[(2 * J + 1):(3 * J)])
  -eval_3pl_map(comp3$patterns, comp3$counts, av, dv, gv, a3_pri, d3_pri, g3_pri)$lpost
}
obj3_gr <- function(par) {
  av <- exp(par[1:J]); dv <- par[(J + 1):(2 * J)]; gv <- plogis(par[(2 * J + 1):(3 * J)])
  g_nat <- eval_3pl_map(comp3$patterns, comp3$counts, av, dv, gv, a3_pri, d3_pri, g3_pri)$gr_post
  -c(g_nat[1:J] * av, g_nat[(J + 1):(2 * J)], g_nat[(2 * J + 1):(3 * J)] * gv * (1 - gv))
}
opt3 <- optim(c(log(a3), d3, qlogis(g3)), obj3_fn, obj3_gr, method = "BFGS",
              control = list(reltol = 1e-15, abstol = 1e-15, maxit = 5000))
# Polish with exact Newton steps on natural scale so max|grad| < 1e-11
x3 <- c(exp(opt3$par[1:J]), opt3$par[(J + 1):(2 * J)], plogis(opt3$par[(2 * J + 1):(3 * J)]))
for (step in 1:6) {
  ev3 <- eval_3pl_map(comp3$patterns, comp3$counts,
                      x3[1:J], x3[(J + 1):(2 * J)], x3[(2 * J + 1):(3 * J)],
                      a3_pri, d3_pri, g3_pri)
  if (max(abs(ev3$gr_post)) < 1e-11) break
  H3 <- hess_from_grad(
    function(x) eval_3pl_map(comp3$patterns, comp3$counts,
                             x[1:J], x[(J + 1):(2 * J)], x[(2 * J + 1):(3 * J)],
                             a3_pri, d3_pri, g3_pri)$gr_post,
    x3
  )
  x3 <- x3 - as.vector(solve(H3, ev3$gr_post))
}
a3_hat <- x3[1:J]; d3_hat <- x3[(J + 1):(2 * J)]; g3_hat <- x3[(2 * J + 1):(3 * J)]
ev3 <- eval_3pl_map(comp3$patterns, comp3$counts, a3_hat, d3_hat, g3_hat, a3_pri, d3_pri, g3_pri)
H3_post <- hess_from_grad(
  function(x) eval_3pl_map(comp3$patterns, comp3$counts,
                           x[1:J], x[(J + 1):(2 * J)], x[(2 * J + 1):(3 * J)],
                           a3_pri, d3_pri, g3_pri)$gr_post,
  x3
)
H3_ll <- hess_from_grad(
  function(x) eval_3pl_map(comp3$patterns, comp3$counts,
                           x[1:J], x[(J + 1):(2 * J)], x[(2 * J + 1):(3 * J)],
                           a3_pri, d3_pri, g3_pri)$gr_ll,
  x3
)
se3_post <- sqrt(diag(solve(-H3_post)))

# Cross-check logLik, logPrior, grid and EAP directly with mirt (fixed at exact MAP)
model3 <- mirt.model(paste0(
  "F1 = 1-", J, "\n",
  "PRIOR = (1-", J, ", a1, lnorm, 0.2, 0.3),\n",
  "        (1-", J, ", d, norm, 0, 2),\n",
  "        (1-", J, ", g, expbeta, 5, 17)"
))
sv3 <- mirt(dat3, model3, itemtype = "3PL", pars = "values",
            quadpts = quad$quadpts, technical = tech)
sv3[sv3$name == "a1", "value"] <- a3_hat
sv3[sv3$name == "d",  "value"] <- d3_hat
sv3[sv3$name == "g",  "value"] <- g3_hat
sv3$est <- FALSE
m3_eval <- mirt(dat3, model3, itemtype = "3PL", pars = sv3,
                quadpts = quad$quadpts, technical = tech, verbose = FALSE)
sc3 <- fscores(m3_eval, method = "EAP", response.pattern = comp3$patterns,
               quadpts = quad$quadpts, theta_lim = quad$theta_lim)

# Also record mirt 1.46's BFGS run to document the expbeta derivative bug
m3_bfgs <- mirt(dat3, model3, itemtype = "3PL", SE = FALSE,
                quadpts = quad$quadpts, TOL = 1e-10, accelerate = "none", optimizer = "BFGS",
                technical = tech, verbose = FALSE)
cf3_bfgs <- coef(m3_bfgs, simplify = TRUE)$items

stopifnot(
  max(abs(ev3$gr_post)) < 1e-10,
  all(range(m3_eval@Model$Theta) == quad$theta_lim),
  abs(m3_eval@Fit$logLik - ev3$ll) < 1e-10,
  abs(m3_eval@Fit$logPrior - ev3$lp) < 1e-10
)

res3 <- list(
  meta = list(
    case = "v7_3pl_map",
    description = "V7c: 3PL MAP, J=15, N=1000, a~LogNormal(0.2, 0.3), d~NormalPrior(0, 2), g~Beta(5, 17)",
    itemtype = "3PL",
    r_version = R.version.string,
    mirt_version = mirt_version_string(), mirt_sha = mirt_commit_sha(),
    TOL = 1e-10,
    quadpts = quad$quadpts,
    theta_lim = quad$theta_lim,
    grid_check = range(m3_eval@Model$Theta),
    accelerate = "none",
    optimizer = "R exact BFGS+Newton on mirt objective (cross-checked via mirt fixed eval)",
    converged = TRUE,
    max_abs_grad = max(abs(ev3$gr_post)),
    mirt_eval_loglik_diff = m3_eval@Fit$logLik - ev3$ll,
    mirt_eval_logprior_diff = m3_eval@Fit$logPrior - ev3$lp,
    se_scale = "natural (exact posterior Hessian in R)",
    mirt_bfgs_diagnostic = list(
      converged = extract.mirt(m3_bfgs, "converged"),
      iterations = extract.mirt(m3_bfgs, "iterations"),
      loglik = extract.mirt(m3_bfgs, "logLik"),
      log_prior = m3_bfgs@Fit$logPrior,
      log_posterior = extract.mirt(m3_bfgs, "logLik") + m3_bfgs@Fit$logPrior,
      log_posterior_deficit = ev3$lpost - (extract.mirt(m3_bfgs, "logLik") + m3_bfgs@Fit$logPrior),
      max_g_diff = max(abs(cf3_bfgs[, "g"] - g3_hat))
    )
  ),
  items = colnames(dat3),
  groups = I("all"),
  priors = list(
    a = list(dist = "LogNormal", meanlog = 0.2, sdlog = 0.3),
    d = list(dist = "NormalPrior", mean = 0.0, sd = 2.0),
    g = list(dist = "Beta", a = 5.0, b = 17.0)
  ),
  patterns = unname(comp3$patterns),
  pattern_groups = rep("all", nrow(comp3$patterns)),
  counts = comp3$counts,
  loglik = m3_eval@Fit$logLik,
  log_prior = m3_eval@Fit$logPrior,
  log_posterior = m3_eval@Fit$logLik + m3_eval@Fit$logPrior,
  estimates = pack_3pl_exact(colnames(dat3), a3_hat, d3_hat, g3_hat, se3_post),
  se_posterior_exact = pack_3pl_exact(colnames(dat3), a3_hat, d3_hat, g3_hat, se3_post),
  eap = list(theta = unname(sc3[, "F1"]), se = unname(sc3[, "SE_F1"])),
  truth = list(a = a3, d = d3, g = g3)
)
write_json(res3, file.path(out_dir, "v7_3pl_map.json"),
           auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null")
cat(sprintf(
  "wrote v7_3pl_map.json        LL=%.6f LP=%.6f LPost=%.6f max|grad|=%.2e mirt_eval_diff=(%.1e, %.1e) mirt_BFGS_deficit=+%.4f\n",
  res3$loglik, res3$log_prior, res3$log_posterior, max(abs(ev3$gr_post)),
  res3$meta$mirt_eval_loglik_diff, res3$meta$mirt_eval_logprior_diff,
  res3$meta$mirt_bfgs_diagnostic$log_posterior_deficit
))
