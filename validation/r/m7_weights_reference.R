# Sampling (survey) weights (VALIDATION.md V34) from R mirt's `survey.weights`.
# Run from the repository root:  Rscript validation/r/m7_weights_reference.R
# Writes tests/reference/v34_*.json. CI does not run R; the JSON files are committed.
#
# As in m2_reference.R: optimizer = "NR", accelerate = "none", TOL = 1e-10, SE.type = "Oakes".
# Cases with a free latent mean/variance use 81 points on [-8, 8] (VALIDATION.md, DESIGN.md §14.9).
# Each file stores the raw rows (`data`, NA = missing), the group label and the weight of every
# row, so the Python side exercises the weighted pattern compression itself.
suppressMessages({
  library(mirt)
  library(jsonlite)
})

out_dir <- file.path("tests", "reference")
common <- list(TOL = 1e-10, accelerate = "none", optimizer = "NR", verbose = FALSE)
default_quad <- list(quadpts = 61, theta_lim = c(-6, 6))
wide_quad <- list(quadpts = 81, theta_lim = c(-8, 8))
tech <- function(quad) list(NCYCLES = 50000, theta_lim = quad$theta_lim)

clean_item <- function(x) {
  nm <- colnames(x)
  keep <- nm[nm == "a1" | grepl("^d[0-9]*$", nm)]
  par <- x["par", keep]; se <- if ("SE" %in% rownames(x)) x["SE", keep] else rep(NA, length(keep))
  names(par) <- sub("^a1$", "a", keep); names(se) <- paste0(names(par), "_se")
  c(as.list(par), as.list(se))
}
clean_group <- function(cf) {
  items <- setdiff(names(cf), "GroupPars")
  params <- lapply(items, function(i) clean_item(cf[[i]])); names(params) <- items
  gp <- cf$GroupPars
  se <- if ("SE" %in% rownames(gp)) gp["SE", ] else c(NA, NA)
  list(params = params,
       latent = list(mean = gp["par", "MEAN_1"], var = gp["par", "COV_11"],
                     mean_se = unname(se[1]), var_se = unname(se[2])))
}
extract <- function(m, groups) {
  cf <- coef(m, printSE = TRUE)
  if (length(groups) == 1) cf <- list(cf)
  names(cf) <- groups
  lapply(cf, clean_group)
}

write_case <- function(case, description, dat, groups, group_labels, weights, fit_fun, quad,
                       extra = list()) {
  m <- fit_fun(weights)
  # The same model without weights: the weighted fit must differ from it (guards against a
  # reference in which the weights were silently ignored).
  m_unweighted <- fit_fun(NULL)
  res <- c(list(
    meta = list(case = case, description = description,
                r_version = R.version.string, mirt_version = as.character(packageVersion("mirt")),
                TOL = 1e-10, quadpts = quad$quadpts, theta_lim = quad$theta_lim,
                grid_check = range(m@Model$Theta),
                accelerate = "none", optimizer = "NR", se_type = "Oakes",
                converged = extract.mirt(m, "converged"),
                iterations = extract.mirt(m, "iterations"),
                n_free = extract.mirt(m, "nest")),
    items = colnames(dat),
    groups = I(groups),
    data = unname(as.matrix(dat)),
    group = group_labels,
    weights = weights,
    n_rows = nrow(dat),
    sum_weights = sum(weights),
    loglik = extract.mirt(m, "logLik"),
    aic = extract.mirt(m, "AIC"),
    bic = extract.mirt(m, "BIC"),
    loglik_unweighted = extract.mirt(m_unweighted, "logLik"),
    estimates = extract(m, groups)
  ), extra)
  stopifnot(
    isTRUE(res$meta$converged),
    all(res$meta$grid_check == quad$theta_lim),
    abs(res$loglik - res$loglik_unweighted) > 1e-3
  )
  path <- file.path(out_dir, paste0(case, ".json"))
  write_json(res, path, auto_unbox = TRUE, digits = NA, na = "null")
  cat("wrote", path, " loglik =", res$loglik, " iter =", res$meta$iterations,
      " nest =", res$meta$n_free, " sum(w) =", res$sum_weights, " BIC =", res$bic, "\n")
}

# ---- V34a: 2PL, non-integer weights that do not sum to N, five rows with weight 0 ---------------
set.seed(20261201)
J <- 10; N <- 1000
dat_a <- as.data.frame(simdata(matrix(rlnorm(J, 0, 0.3)), matrix(rnorm(J)), N, itemtype = "2PL"))
colnames(dat_a) <- sprintf("q%02d", 1:J)
# Weights related to the responses (informative), so that the weighted and unweighted estimates
# differ clearly.
w_a <- rlnorm(N, 0, 0.6) * (0.6 + 0.15 * rowSums(dat_a))
w_a[sample.int(N, 5)] <- 0
write_case("v34_weights_2pl",
           "V34a: 2PL, J=10, N=1000, lognormal weights (sum != N), five zero weights",
           dat_a, "all", rep("all", N), w_a,
           function(w) do.call(mirt, c(list(dat_a, 1, itemtype = "2PL", SE = TRUE,
                                            SE.type = "Oakes", survey.weights = w, quadpts = 61,
                                            technical = tech(default_quad)), common)),
           default_quad)

# ---- V34b: graded response model (Science), weights normalized to sum to N ---------------------
set.seed(20261202)
sci <- as.data.frame(Science - 1L)
w_b <- runif(nrow(sci), 0.3, 3) * (1 + 0.2 * (sci$Comfort >= 2))
w_b <- w_b / sum(w_b) * nrow(sci)
write_case("v34_weights_grm",
           "V34b: graded (Science, K=4), N=392, uniform weights normalized to sum to N",
           sci, "all", rep("all", nrow(sci)), w_b,
           function(w) do.call(mirt, c(list(sci, 1, itemtype = "graded", SE = TRUE,
                                            SE.type = "Oakes", survey.weights = w, quadpts = 61,
                                            technical = tech(default_quad)), common)),
           default_quad)

# ---- V34c: two groups, anchors, focal mean/variance free, 5% MCAR missing ----------------------
set.seed(20261203)
J <- 10; N <- 800
a <- matrix(rlnorm(J, 0, 0.3)); d <- matrix(rnorm(J))
d_f <- d; d_f[6:J] <- d[6:J] + rnorm(J - 5, 0, 0.4)
ref <- simdata(a, d, N, itemtype = "2PL")
foc <- simdata(a, d_f, N, itemtype = "2PL", mu = 0.5, sigma = matrix(1.2))
dat_c <- as.data.frame(rbind(ref, foc)); colnames(dat_c) <- sprintf("q%02d", 1:J)
dat_c[matrix(runif(2 * N * J) < 0.05, 2 * N, J)] <- NA
grp_c <- rep(c("ref", "focal"), each = N)
# Different weight distributions in the two groups (the groups' weight totals differ).
w_c <- c(rgamma(N, shape = 2, rate = 2), rgamma(N, shape = 3, rate = 1.5)) *
  (0.7 + 0.1 * rowSums(dat_c, na.rm = TRUE))
anchors <- sprintf("q%02d", 1:5)
write_case("v34_weights_twogroup",
           "V34c: 2PL, groups ref/focal (N=800 each), anchors q01-q05, focal latent free, 5% missing, gamma weights",
           dat_c, c("ref", "focal"), grp_c, w_c,
           function(w) do.call(multipleGroup,
                               c(list(dat_c, 1, group = factor(grp_c, levels = c("ref", "focal")),
                                      itemtype = "2PL",
                                      invariance = c(anchors, "free_means", "free_var"),
                                      SE = TRUE, SE.type = "Oakes", survey.weights = w,
                                      quadpts = 81, technical = tech(wide_quad)), common)),
           wide_quad, extra = list(anchors = anchors))

# ---- V34d: FIPC, items q01-q05 fixed at generating values, latent mean/variance free -----------
set.seed(20261204)
J <- 10; N <- 1000
a_d <- rlnorm(J, 0, 0.3); d_d <- rnorm(J)
dat_d <- as.data.frame(simdata(matrix(a_d), matrix(d_d), N, itemtype = "2PL", mu = 0.4,
                               sigma = matrix(1.3)))
colnames(dat_d) <- sprintf("q%02d", 1:J)
w_d <- rlnorm(N, 0, 0.8) * (1.5 - 0.08 * rowSums(dat_d))
w_d <- w_d / sum(w_d) * N
bank_items <- sprintf("q%02d", 1:5)
bank <- data.frame(item = bank_items, a = a_d[1:5], d = d_d[1:5])
fit_d <- function(w) {
  sv <- mirt(dat_d, 1, "2PL", pars = "values", quadpts = 81, technical = tech(wide_quad))
  for (k in seq_len(nrow(bank))) {
    sv[sv$item == bank$item[k] & sv$name == "a1", c("value", "est")] <- list(bank$a[k], FALSE)
    sv[sv$item == bank$item[k] & sv$name == "d", c("value", "est")] <- list(bank$d[k], FALSE)
  }
  sv[sv$name %in% c("MEAN_1", "COV_11"), "est"] <- TRUE
  do.call(mirt, c(list(dat_d, 1, itemtype = "2PL", pars = sv, SE = TRUE, SE.type = "Oakes",
                       survey.weights = w, quadpts = 81, technical = tech(wide_quad)), common))
}
write_case("v34_weights_fipc",
           "V34d: FIPC, 2PL, q01-q05 fixed at generating values, latent free, N=1000, lognormal weights",
           dat_d, "all", rep("all", N), w_d, fit_d, wide_quad, extra = list(bank = bank))

# ---- V34e: bifactor 2PL (bfactor, Gibbons-Hedeker dimension reduction) with weights -------------
# General factor plus three specific factors of three items; 21 points per dimension on [-6, 6].
set.seed(20261205)
J <- 9L; N <- 1000L
a_e <- matrix(0, nrow = J, ncol = 4L)
a_e[, 1] <- c(1.4, 1.2, 1.6, 1.0, 1.3, 1.5, 1.1, 1.4, 1.2)
a_e[1:3, 2] <- c(0.9, 1.1, 0.7)
a_e[4:6, 3] <- c(1.2, 0.8, 1.0)
a_e[7:9, 4] <- c(0.8, 1.0, 1.1)
dat_e <- as.data.frame(simdata(a_e, matrix(rnorm(J, 0, 0.8)), N, itemtype = "2PL",
                               sigma = diag(4)))
colnames(dat_e) <- sprintf("q%02d", 1:J)
w_e <- rlnorm(N, 0, 0.6) * (0.7 + 0.1 * rowSums(dat_e))
specific_e <- list(s1 = colnames(dat_e)[1:3], s2 = colnames(dat_e)[4:6], s3 = colnames(dat_e)[7:9])
fit_e <- function(w) {
  bfactor(dat_e, rep(1:3, each = 3), itemtype = "2PL", SE = TRUE, SE.type = "Oakes",
          survey.weights = w, quadpts = 21, TOL = 1e-10, accelerate = "none", optimizer = "NR",
          technical = list(NCYCLES = 20000L, theta_lim = c(-6, 6)), verbose = FALSE)
}
m_e <- fit_e(w_e)
m_e0 <- fit_e(NULL)
cf_e <- coef(m_e, printSE = TRUE)
params_e <- lapply(colnames(dat_e), function(i) {
  x <- cf_e[[i]]
  keep <- c(paste0("a", 1:4), "d")
  se <- x["SE", keep]; names(se) <- paste0(keep, "_se")
  c(as.list(x["par", keep]), as.list(se))
})
names(params_e) <- colnames(dat_e)
res_e <- list(
  meta = list(case = "v34_weights_bifactor",
              description = "V34e: bifactor 2PL, J=9, three specific factors, N=1000, lognormal weights (sum != N)",
              r_version = R.version.string, mirt_version = as.character(packageVersion("mirt")),
              TOL = 1e-10, quadpts = 21, theta_lim = c(-6, 6), grid_check = range(m_e@Model$Theta),
              accelerate = "none", optimizer = "NR", se_type = "Oakes",
              converged = extract.mirt(m_e, "converged"),
              iterations = extract.mirt(m_e, "iterations"), n_free = extract.mirt(m_e, "nest")),
  items = colnames(dat_e), specific = specific_e, general = "general",
  data = unname(as.matrix(dat_e)), weights = w_e, n_rows = nrow(dat_e), sum_weights = sum(w_e),
  loglik = extract.mirt(m_e, "logLik"), aic = extract.mirt(m_e, "AIC"),
  bic = extract.mirt(m_e, "BIC"), loglik_unweighted = extract.mirt(m_e0, "logLik"),
  params = params_e
)
stopifnot(isTRUE(res_e$meta$converged), all(res_e$meta$grid_check == c(-6, 6)),
          nrow(m_e@Model$Theta) == 21^2, abs(res_e$loglik - res_e$loglik_unweighted) > 1e-3)
write_json(res_e, file.path(out_dir, "v34_weights_bifactor.json"), auto_unbox = TRUE, digits = NA,
           na = "null")
cat("wrote v34_weights_bifactor.json  loglik =", res_e$loglik, " iter =", res_e$meta$iterations,
    " nest =", res_e$meta$n_free, " sum(w) =", res_e$sum_weights, "\n")
