# V45 reference values (VALIDATION.md V45, DESIGN.md §15.48): mirt's `norm` prior on the 3PL
# lower asymptote, which acts on mirt's internal logit(g) without a Jacobian. flexpyirt reproduces
# it with LogitNormalPrior(mean, sd, mode_scale = "logit").
# Run from the repository root:  Rscript validation/r/m7_logit_norm_prior_reference.R
# Writes:
#   tests/reference/v45_3pl_logit_norm_map.json
#     (3PL, J = 8, N = 1200: g ~ norm(-1.3863, 0.6) on logit(g), d ~ norm(0, 3); mirt NR + Oakes)
# CI does not run R.
#
# Checks done here (stopifnot):
# 1. mirt's logPrior equals sum(dnorm(qlogis(g), mu, s)) + sum(dnorm(d, 0, 3)): the objective is
#    the logit-metric one (no -log g - log(1 - g) term).
# 2. mirt's estimate is a stationary point of logLik + logPrior on (a, d, logit g), with the
#    gradient written here (max |grad| < 1e-6).
suppressMessages({
  library(mirt)
  library(jsonlite)
})

out_dir <- file.path("tests", "reference")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
quad <- list(quadpts = 61, theta_lim = c(-6, 6))
tech <- list(NCYCLES = 50000, theta_lim = quad$theta_lim)
mu_g <- -1.3863
sd_g <- 0.6
sd_d <- 3

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

# Gradient of logLik + logit-metric logPrior with respect to (a, d, lambda = logit g).
grad_logit_objective <- function(P, cnt, av, dv, gv) {
  J <- length(av)
  ell <- matrix(0, nrow = nrow(P), ncol = length(nodes))
  s_list <- vector("list", J); p1_list <- vector("list", J)
  for (j in 1:J) {
    s <- plogis(av[j] * nodes + dv[j])
    p1 <- gv[j] + (1 - gv[j]) * s
    s_list[[j]] <- s; p1_list[[j]] <- p1
    ell <- ell + outer(P[, j], log(p1)) + outer(1 - P[, j], log(1 - p1))
  }
  lj <- sweep(ell, 2, log_w, "+")
  mx <- apply(lj, 1, max)
  lse <- mx + log(rowSums(exp(lj - mx)))
  w <- sweep(exp(lj - lse), 1, cnt, "*")
  n_q <- colSums(w)
  ga <- numeric(J); gd <- numeric(J); gl <- numeric(J)
  for (j in 1:J) {
    r1 <- colSums(sweep(w, 1, P[, j], "*"))
    r0 <- n_q - r1
    p1 <- p1_list[[j]]; s <- s_list[[j]]; gj <- gv[j]
    d_z <- r1 * ((1 - gj) * s * (1 - s) / p1) - r0 * s
    ga[j] <- sum(d_z * nodes)
    gd[j] <- sum(d_z) - dv[j] / sd_d^2
    gg <- sum(r1 * ((1 - s) / p1) - r0 / (1 - gj))
    gl[j] <- gg * gj * (1 - gj) - (qlogis(gj) - mu_g) / sd_g^2
  }
  list(ll = sum(cnt * lse), grad = c(ga, gd, gl))
}

set.seed(20261126)
J <- 8; N <- 1200
a <- rlnorm(J, 0.2, 0.2); d <- rnorm(J, 0, 0.8); g <- rbeta(J, 4, 16)
dat <- as.data.frame(simdata(matrix(a), matrix(d), N, itemtype = "3PL", guess = g))
colnames(dat) <- sprintf("q%02d", 1:J)
comp <- compress_dat(dat)

model <- mirt.model(paste0(
  "F1 = 1-", J, "\n",
  "PRIOR = (1-", J, ", g, norm, ", mu_g, ", ", sd_g, "), (1-", J, ", d, norm, 0, ", sd_d, ")"
))
m <- mirt(dat, model, itemtype = "3PL", SE = TRUE, SE.type = "Oakes", quadpts = quad$quadpts,
          TOL = 1e-10, accelerate = "none", optimizer = "NR", technical = tech, verbose = FALSE)
cf <- coef(m, simplify = TRUE)$items
cf_se <- coef(m, printSE = TRUE)
items <- colnames(dat)
lp_check <- sum(dnorm(qlogis(cf[, "g"]), mu_g, sd_g, log = TRUE)) +
  sum(dnorm(cf[, "d"], 0, sd_d, log = TRUE))
ev <- grad_logit_objective(comp$patterns, comp$counts, cf[, "a1"], cf[, "d"], cf[, "g"])
sc <- fscores(m, method = "EAP", response.pattern = comp$patterns,
              quadpts = quad$quadpts, theta_lim = quad$theta_lim)

stopifnot(
  isTRUE(extract.mirt(m, "converged")),
  all(range(m@Model$Theta) == quad$theta_lim),
  abs(m@Fit$logPrior - lp_check) < 1e-9,
  abs(m@Fit$logLik - ev$ll) < 1e-8,
  max(abs(ev$grad)) < 1e-6
)

# coef(printSE = TRUE) reports logit(g) and its SE on the logit scale; g_se is the delta-method SE
# g (1 - g) SE(logit g) (DESIGN.md §15.48).
params <- lapply(items, function(i) {
  par <- cf_se[[i]]["par", ]; se <- cf_se[[i]]["SE", ]
  gi <- plogis(par[["logit(g)"]])
  list(a = par[["a1"]], d = par[["d"]], g = gi, u = 1.0,
       a_se = se[["a1"]], d_se = se[["d"]], g_se = gi * (1 - gi) * se[["logit(g)"]], u_se = NA,
       logit_g = par[["logit(g)"]], logit_g_se = se[["logit(g)"]])
})
names(params) <- items

res <- list(
  meta = list(
    case = "v45_3pl_logit_norm_map",
    description = paste0(
      "V45: 3PL MAP, J=8, N=1200, mirt norm(", mu_g, ", ", sd_g, ") on logit(g) ",
      "(LogitNormalPrior mode_scale='logit'), d~NormalPrior(0, 3)"
    ),
    itemtype = "3PL",
    r_version = R.version.string,
    mirt_version = as.character(packageVersion("mirt")),
    TOL = 1e-10,
    quadpts = quad$quadpts,
    theta_lim = quad$theta_lim,
    grid_check = range(m@Model$Theta),
    accelerate = "none",
    optimizer = "NR",
    converged = extract.mirt(m, "converged"),
    iterations = extract.mirt(m, "iterations"),
    max_abs_grad_logit_objective = max(abs(ev$grad)),
    se_note = paste(
      "mirt Oakes SEs from coef(printSE = TRUE); g_se is the delta-method SE",
      "g(1 - g) * SE(logit g) of mirt's logit_g_se"
    )
  ),
  items = items,
  groups = I("all"),
  priors = list(
    g = list(dist = "LogitNormalPrior", mean_logit = mu_g, sd_logit = sd_g, mode_scale = "logit"),
    d = list(dist = "NormalPrior", mean = 0.0, sd = sd_d)
  ),
  patterns = unname(comp$patterns),
  pattern_groups = rep("all", nrow(comp$patterns)),
  counts = comp$counts,
  loglik = extract.mirt(m, "logLik"),
  log_prior = m@Fit$logPrior,
  log_posterior = extract.mirt(m, "logLik") + m@Fit$logPrior,
  estimates = list(all = list(params = params,
                              latent = list(mean = 0, var = 1, mean_se = NA, var_se = NA))),
  eap = list(theta = unname(sc[, "F1"]), se = unname(sc[, "SE_F1"])),
  truth = list(a = a, d = d, g = g)
)
write_json(res, file.path(out_dir, "v45_3pl_logit_norm_map.json"),
           auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null")
cat(sprintf("wrote v45_3pl_logit_norm_map.json LL=%.6f LP=%.6f iter=%d max|grad|=%.2e\n",
            res$loglik, res$log_prior, res$meta$iterations, max(abs(ev$grad))))
