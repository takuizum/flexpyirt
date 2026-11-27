# Robust standard errors (VALIDATION.md V35) from R mirt: SE.type = "crossprod" and "sandwich".
# Run from the repository root:  Rscript validation/r/m7_robust_se_reference.R [case ...]
# Writes tests/reference/v35_<case>.json. CI does not run R; the JSON files are committed.
#
# Every case is fitted three times with the same EM settings (optimizer = "NR", accelerate =
# "none", TOL = 1e-10; BFGS for the three-factor case and TOL = 1e-7 for the empirical histogram,
# where mirt needs about 11,000 cycles and does not reach 1e-9 in 100,000), once per SE.type in
# "crossprod", "sandwich" and "sandwich.Louis"; the
# estimates are identical and only vcov(m) differs. The file stores mirt's parameter table
# (mod2values) and the full covariance matrices, each row labelled by the first `parnum` of its
# (possibly constrained) parameter, so the tests can rebuild mirt's solution and compare whole
# matrices.
#
# Notes on mirt 1.48 (mirt:::SE.simple):
# - "crossprod":  info = Igrad + Iprior                  (cross-product of the pattern scores plus
#                                                          minus the Hessian of the log prior)
# - "sandwich":   info = solve(H^-1 (Igrad + Iprior) H^-1), H = minus mirt's Oakes Hessian
# - "sandwich.Louis": the same with H from Louis' formula. Louis' H is not available when latent
#   parameters are free ("Louis" itself returns no matrix there) and differs from the Oakes one in
#   those cases; where it is recorded, it is only for the comparison in VALIDATION.md.
# - The scores of 3PL lower asymptotes are with respect to logit(g) (mirt's internal scale); the
#   tests transform flexpyirt's natural-scale matrices with the Jacobian 1 / (g (1 - g)).
# - mirt's Hessian of lnorm and expbeta priors is wrong (m3_priors_reference.R, VALIDATION.md V7),
#   so the prior case uses normal priors, whose derivatives are exact.
# - Empirical histogram: mirt's Oakes Hessian for EH is not the Hessian with the histogram held
#   fixed (VALIDATION.md V6), so mirt's "sandwich" is not comparable there. The EH case stores
#   "crossprod" and "Richardson" (the Hessian with the histogram fixed); the test compares
#   flexpyirt's sandwich with V_Richardson %*% solve(V_crossprod) %*% V_Richardson.
# - Davidian curves and mixtures: mirt computes neither type ("No standard error method currently
#   supported for Davidian curves", "Only Oakes and complete SE.types current supported for
#   mixture models"); flexpyirt raises SpecificationError there.
source(file.path("validation", "r", "setup.R"))  # mirt from GitHub; mirt_version / mirt_sha
suppressMessages({
  library(mirt)
  library(jsonlite)
})

out_dir <- file.path("tests", "reference")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
common <- list(TOL = 1e-10, accelerate = "none", optimizer = "NR", verbose = FALSE)
tech <- function(quad) list(NCYCLES = 100000, theta_lim = quad$theta_lim)
quad61 <- list(quadpts = 61, theta_lim = c(-6, 6))
quad81 <- list(quadpts = 81, theta_lim = c(-8, 8))
quad31 <- list(quadpts = 31, theta_lim = c(-6, 6))
quad15 <- list(quadpts = 15, theta_lim = c(-6, 6))

vcov_block <- function(m) {
  v <- vcov(m)
  stopifnot(all(is.finite(v)))
  first <- as.integer(vapply(strsplit(rownames(v), ".", fixed = TRUE), function(x) x[2], ""))
  list(parnum = first, matrix = unname(v))
}

write_case <- function(case, description, dat, grp, fit_fun, quad,
                       types = c("crossprod", "sandwich", "sandwich.Louis"), extra = list(),
                       optimizer = "NR", tol = 1e-10) {
  dat <- as.matrix(dat)
  fits <- lapply(types, function(t) tryCatch(suppressMessages(fit_fun(t)), error = function(e) NULL))
  names(fits) <- types
  m <- fits[[1]]
  values <- mod2values(m)
  blocks <- list()
  for (t in types) {
    mt <- fits[[t]]
    ok <- !is.null(mt) && isTRUE(extract.mirt(mt, "converged")) && isTRUE(extract.mirt(mt, "secondordertest"))
    if (ok) ok <- !is.null(tryCatch(vcov_block(mt), error = function(e) NULL))
    if (!ok) {
      if (t == "sandwich.Louis") next
      stop("SE.type = ", t, " failed for ", case)
    }
    stopifnot(max(abs(mod2values(mt)$value - values$value)) < 1e-10)
    blocks[[sub(".", "_", t, fixed = TRUE)]] <- vcov_block(mt)
  }
  glabel <- if (is.null(grp)) rep("all", nrow(dat)) else as.character(grp)
  key <- paste(glabel, apply(dat, 1, paste, collapse = ","))
  first <- !duplicated(key)
  grid_check <- range(m@Model$Theta)
  stopifnot(isTRUE(extract.mirt(m, "converged")), all(grid_check == quad$theta_lim))
  res <- c(list(
    meta = list(case = case, description = description, r_version = R.version.string,
                mirt_version = mirt_version_string(), mirt_sha = mirt_commit_sha(), TOL = tol,
                quadpts = quad$quadpts, theta_lim = quad$theta_lim, grid_check = grid_check,
                accelerate = "none", optimizer = optimizer, converged = TRUE,
                iterations = extract.mirt(m, "iterations")),
    items = colnames(dat),
    groups = I(unique(glabel)),
    patterns = unname(dat[first, , drop = FALSE]),
    pattern_groups = glabel[first],
    counts = as.integer(table(key)[key[first]]),
    loglik = extract.mirt(m, "logLik"),
    values = lapply(seq_len(nrow(values)), function(i) list(
      group = as.character(values$group[i]), item = as.character(values$item[i]),
      name = as.character(values$name[i]), parnum = values$parnum[i],
      value = values$value[i], est = values$est[i])),
    vcov = blocks
  ), extra)
  path <- file.path(out_dir, paste0("v35_", case, ".json"))
  write_json(res, path, auto_unbox = TRUE, digits = NA, pretty = FALSE, na = "null")
  se <- sapply(blocks, function(b) sqrt(diag(b$matrix))[1:3])
  cat(sprintf("wrote %s loglik=%.6f iter=%d types=%s\n", path, res$loglik, res$meta$iterations,
              paste(names(blocks), collapse = ",")))
  print(se)
}

RUN <- commandArgs(trailingOnly = TRUE)
run <- function(case) length(RUN) == 0 || case %in% RUN
skewed <- function(n) (rchisq(n, df = 4) - 4) / sqrt(8)  # mean 0, var 1, skewness sqrt(2)

# ---- 2PL fitted to misspecified data (guessing, skewed trait), 5% MCAR -----------------------
if (run("2pl_misfit")) {
  set.seed(20261125)
  J <- 12; N <- 2000
  a <- rlnorm(J, 0.1, 0.3); d <- rnorm(J, 0, 0.9)
  dat <- simdata(matrix(a), matrix(d), N, itemtype = "3PL", guess = runif(J, 0.1, 0.3),
                 Theta = matrix(skewed(N)))
  dat[matrix(runif(N * J) < 0.05, N, J)] <- NA
  colnames(dat) <- sprintf("q%02d", 1:J)
  write_case("2pl_misfit", "2PL fitted to 3PL data with a skewed trait, 5% MCAR, J=12, N=2000",
             dat, NULL,
             function(t) do.call(mirt, c(list(dat, 1, itemtype = "2PL", SE = TRUE, SE.type = t,
                                              quadpts = 61, technical = tech(quad61)), common)),
             quad61)
}

# ---- mixed formats: 3PL, GRM, GPCM, nominal --------------------------------------------------
if (run("mixed")) {
  set.seed(20261126)
  N <- 5000
  types <- c(rep("3PL", 4), rep("graded", 3), rep("gpcm", 3), rep("nominal", 2))
  a <- matrix(c(runif(4, 1.2, 1.8), rlnorm(8, 0, 0.25)))
  d <- matrix(NA_real_, 12, 4)
  d[1:4, 1] <- runif(4, -0.3, 0.6)
  for (j in 5:7) d[j, 1:3] <- sort(rnorm(3, 0, 1.2), decreasing = TRUE)
  for (j in 8:10) d[j, 1:3] <- c(0, rnorm(2, 0, 0.7))
  for (j in 11:12) d[j, 1:3] <- c(0, rnorm(2, 0, 0.7))
  nom <- matrix(NA_real_, 12, 3)
  for (j in 8:10) nom[j, ] <- 0:2
  nom[11, ] <- c(0, 1.4, 2); nom[12, ] <- c(0, 0.5, 2)
  dat <- simdata(a, d, N, itemtype = c(rep("3PL", 4), rep("graded", 3), rep("gpcm", 3), rep("nominal", 2)),
                 guess = c(rep(0.2, 4), rep(0, 8)), gpcm_mats = NULL, nominal = nom)
  colnames(dat) <- c(sprintf("mc%d", 1:4), sprintf("gr%d", 1:3), sprintf("pc%d", 1:3), sprintf("nm%d", 1:2))
  write_case("mixed", "4 3PL + 3 GRM(4) + 3 GPCM(3) + 2 nominal(3), N=5000", dat, NULL,
             function(t) do.call(mirt, c(list(dat, 1, itemtype = types, SE = TRUE, SE.type = t,
                                              quadpts = 61, technical = tech(quad61)), common)),
             quad61, extra = list(itemtypes = types))
}

# ---- MAP with normal priors (small N so that the priors matter) ------------------------------
if (run("2pl_map")) {
  set.seed(20261127)
  J <- 8; N <- 300
  a <- rlnorm(J, 0.1, 0.3); d <- rnorm(J)
  dat <- simdata(matrix(a), matrix(d), N, itemtype = "2PL")
  colnames(dat) <- sprintf("q%02d", 1:J)
  mod <- mirt.model("F = 1-8\nPRIOR = (1-8, a1, norm, 1.2, 0.5), (1-8, d, norm, 0, 2)")
  write_case("2pl_map", "2PL MAP, a ~ N(1.2, 0.5), d ~ N(0, 2), J=8, N=300", dat, NULL,
             function(t) do.call(mirt, c(list(dat, mod, itemtype = "2PL", SE = TRUE, SE.type = t,
                                              quadpts = 61, technical = tech(quad61)), common)),
             quad61, extra = list(prior = list(a = c(1.2, 0.5), d = c(0, 2))))
}

# ---- two groups, anchors, focal mean and variance free ---------------------------------------
if (run("twogroup")) {
  set.seed(20261128)
  J <- 12; N <- 1500
  a <- matrix(rlnorm(J, 0, 0.3)); d <- matrix(rnorm(J))
  a_f <- a; d_f <- d
  d_f[7:J] <- d[7:J] + rnorm(6, 0, 0.4)
  ref <- simdata(a, d, N, itemtype = "2PL", Theta = matrix(skewed(N)))
  foc <- simdata(a_f, d_f, N, itemtype = "2PL", mu = 0.4, sigma = matrix(1.3))
  dat <- rbind(ref, foc); colnames(dat) <- sprintf("q%02d", 1:J)
  grp <- rep(c("ref", "focal"), each = N)
  anchors <- sprintf("q%02d", 1:6)
  write_case("twogroup", "2PL, ref/focal (N=1500 each), anchors q01-q06, focal mean and variance free",
             dat, grp,
             function(t) do.call(multipleGroup, c(list(dat, 1, group = factor(grp, levels = c("ref", "focal")),
                                                       itemtype = "2PL",
                                                       invariance = c(anchors, "free_means", "free_var"),
                                                       SE = TRUE, SE.type = t, quadpts = 81,
                                                       technical = tech(quad81)), common)),
             quad81, extra = list(anchors = anchors))
}

# ---- FIPC with an equality constraint across items -------------------------------------------
if (run("fipc_equal")) {
  set.seed(20261129)
  J <- 12; N <- 2000
  a <- rlnorm(J, 0, 0.3); d <- rnorm(J)
  a[9:12] <- 1.1
  dat <- simdata(matrix(a), matrix(d), N, itemtype = "2PL", mu = 0.3, sigma = matrix(1.2))
  colnames(dat) <- sprintf("q%02d", 1:J)
  bank <- data.frame(item = sprintf("q%02d", 1:5), a = a[1:5], d = d[1:5])
  equal_items <- sprintf("q%02d", 9:12)
  fit_fun <- function(t) {
    sv <- mirt(dat, 1, "2PL", pars = "values", quadpts = 81, technical = tech(quad81))
    for (k in seq_len(nrow(bank))) {
      sv[sv$item == bank$item[k] & sv$name == "a1", c("value", "est")] <- list(bank$a[k], FALSE)
      sv[sv$item == bank$item[k] & sv$name == "d", c("value", "est")] <- list(bank$d[k], FALSE)
    }
    sv[sv$name %in% c("MEAN_1", "COV_11"), "est"] <- TRUE
    con <- list(sv$parnum[sv$item %in% equal_items & sv$name == "a1"])
    do.call(mirt, c(list(dat, 1, itemtype = "2PL", pars = sv, constrain = con, SE = TRUE,
                         SE.type = t, quadpts = 81, technical = tech(quad81)), common))
  }
  write_case("fipc_equal",
             "FIPC 2PL: q01-q05 fixed, latent mean and variance free, equal slopes on q09-q12, N=2000",
             dat, NULL, fit_fun, quad81, extra = list(bank = bank, equal_items = equal_items))
}

# ---- two factors with a cross-loading, correlated --------------------------------------------
if (run("two_factor")) {
  set.seed(20261130)
  J <- 12; N <- 2000
  a <- matrix(0, J, 2); a[1:6, 1] <- runif(6, 0.9, 1.6); a[7:12, 2] <- runif(6, 0.9, 1.6); a[7, 1] <- 0.6
  d <- matrix(rnorm(J))
  dat <- simdata(a, d, N, itemtype = "2PL", sigma = matrix(c(1, 0.5, 0.5, 1), 2))
  colnames(dat) <- sprintf("q%02d", 1:J)
  mod <- mirt.model("F1 = 1-7\nF2 = 7-12\nCOV = F1*F2")
  write_case("two_factor", "2D 2PL, F1 = q01-q07, F2 = q07-q12, correlated, N=2000", dat, NULL,
             function(t) do.call(mirt, c(list(dat, mod, itemtype = "2PL", SE = TRUE, SE.type = t,
                                              quadpts = 31, technical = tech(quad31)), common)),
             quad31, extra = list(factors = list(F1 = sprintf("q%02d", 1:7), F2 = sprintf("q%02d", 7:12))))
}

# ---- three factors, simple structure, correlated ---------------------------------------------
if (run("three_factor")) {
  set.seed(20261131)
  J <- 12; N <- 2000
  a <- matrix(0, J, 3); a[1:4, 1] <- runif(4, 0.9, 1.6); a[5:8, 2] <- runif(4, 0.9, 1.6)
  a[9:12, 3] <- runif(4, 0.9, 1.6)
  d <- matrix(rnorm(J))
  sigma <- matrix(c(1, 0.5, 0.3, 0.5, 1, 0.4, 0.3, 0.4, 1), 3)
  dat <- simdata(a, d, N, itemtype = "2PL", sigma = sigma)
  colnames(dat) <- sprintf("q%02d", 1:J)
  mod <- mirt.model("F1 = 1-4\nF2 = 5-8\nF3 = 9-12\nCOV = F1*F2*F3")
  # As in m4_mirt_reference.R (V14d): the NR M-step is very slow on the 15^3 grid, so BFGS with a
  # tight tolerance is used here.
  bfgs <- list(TOL = 1e-10, accelerate = "none", optimizer = "BFGS", verbose = FALSE,
               control = list(reltol = 1e-13))
  write_case("three_factor", "3D 2PL, simple structure, correlated, N=2000", dat, NULL,
             function(t) do.call(mirt, c(list(dat, mod, itemtype = "2PL", SE = TRUE, SE.type = t,
                                              quadpts = 15, technical = tech(quad15)), bfgs)),
             quad15, optimizer = "BFGS (reltol 1e-13)", extra = list(factors = list(F1 = sprintf("q%02d", 1:4), F2 = sprintf("q%02d", 5:8),
                                                 F3 = sprintf("q%02d", 9:12))))
}

# ---- empirical histogram under FIPC ----------------------------------------------------------
if (run("fipc_eh")) {
  set.seed(20261132)
  J <- 16; N <- 3000
  a <- rlnorm(J, 0.1, 0.25); d <- rnorm(J, 0, 0.9)
  dat <- simdata(matrix(a), matrix(d), N, itemtype = "2PL", Theta = matrix(skewed(N)))
  colnames(dat) <- sprintf("i%02d", 1:J)
  bank <- data.frame(item = sprintf("i%02d", 1:8), a = a[1:8], d = d[1:8])
  eh_common <- list(TOL = 1e-7, accelerate = "none", optimizer = "NR", verbose = FALSE)
  fit_fun <- function(t) {
    sv <- mirt(dat, 1, "2PL", pars = "values", dentype = "EH", quadpts = 61,
               technical = list(theta_lim = quad61$theta_lim))
    for (k in seq_len(nrow(bank))) {
      sv[sv$item == bank$item[k] & sv$name == "a1", c("value", "est")] <- list(bank$a[k], FALSE)
      sv[sv$item == bank$item[k] & sv$name == "d", c("value", "est")] <- list(bank$d[k], FALSE)
    }
    do.call(mirt, c(list(dat, 1, itemtype = "2PL", pars = sv, dentype = "EH", SE = TRUE,
                         SE.type = t, quadpts = 61, technical = tech(quad61)), eh_common))
  }
  m <- fit_fun("crossprod")
  write_case("fipc_eh", "EH under FIPC (i01-i08 fixed), skewed trait, 2PL J=16, N=3000", dat, NULL,
             fit_fun, quad61, types = c("crossprod", "Richardson", "sandwich"), tol = 1e-7,
             extra = list(bank = bank, nodes = as.vector(m@Model$Theta),
                          eh_weights = as.vector(extract.mirt(m, "Prior")[[1]])))
}
