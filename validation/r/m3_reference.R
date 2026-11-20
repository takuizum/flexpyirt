# M3 reference values (VALIDATION.md V11: 3PL and 4PL) from R mirt.
# Run from the repository root:  Rscript validation/r/m3_reference.R
# Writes tests/reference/v11_3pl.json and tests/reference/v11_4pl.json. CI does not run R.
#
# As in m1/m2_reference.R: optimizer = "NR", accelerate = "none", TOL = 1e-10, 61 points on [-6, 6]
# (theta_lim inside `technical`; the grid actually used is recorded in meta.grid_check).
#
# Scale of g and u: mirt estimates them on the logit scale, and coef(m, printSE = TRUE) reports
# columns "logit(g)" / "logit(u)" with logit-scale SEs (rawug makes no difference once printSE is
# TRUE). flexpyirt works on the natural scale, so this script converts both estimates and SEs:
#   g = plogis(logit_g),  SE_g = g (1 - g) SE_logit(g)   (delta method; same for u).
# At an interior ML solution the observed information transforms exactly with the Jacobian, so the
# converted SEs are the natural-scale SEs. The raw logit-scale values are kept under "raw_logit".
#
# Simulation design note (VALIDATION.md): unregularized ML for 3PL/4PL easily produces boundary
# solutions (g -> 0 or u -> 1, with singular information) when items are too easy/hard or slopes are
# too low for both asymptotes to be observed. V11a uses hard-to-moderate items (d ~ U(-1.5, 0.5))
# with moderate-to-high slopes; V11b uses moderate difficulties (d ~ U(-0.5, 0.5)), stronger slopes
# and N = 20000 (at most 2^12 = 4096 unique patterns). For V11b, mirt's default starting slope
# (a = 0.851) causes NR on the logit scale to overshoot logit(g) to ~-15 on early iterations and
# become trapped at a strictly lower log-likelihood; starting mirt at a = 2.0 avoids that trap and
# reaches the interior global MLE (verified against BFGS and flexpyirt from default starts).
suppressMessages({
  library(mirt)
  library(jsonlite)
})

out_dir <- file.path("tests", "reference")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
common <- list(TOL = 1e-10, accelerate = "none", optimizer = "NR", verbose = FALSE)
quad <- list(quadpts = 61, theta_lim = c(-6, 6))
tech <- list(NCYCLES = 50000, theta_lim = quad$theta_lim)

clean_item <- function(x, natural) {
  par <- x["par", ]; se <- x["SE", ]
  out <- list(a = par[["a1"]], d = par[["d"]], a_se = se[["a1"]], d_se = se[["d"]])
  raw <- list()
  for (p in c("g", "u")) {
    lp <- par[[sprintf("logit(%s)", p)]]; lse <- se[[sprintf("logit(%s)", p)]]
    v <- plogis(lp)
    stopifnot(abs(v - natural[[p]]) < 1e-12)  # coef() without printSE gives the natural scale
    out[[p]] <- v
    # A fixed parameter (u = 1 in the 3PL, logit 999) has SE NA.
    out[[paste0(p, "_se")]] <- if (is.na(lse)) NA else v * (1 - v) * lse
    raw[[paste0("logit_", p)]] <- lp
    raw[[paste0("logit_", p, "_se")]] <- lse
  }
  out <- out[c("a", "d", "g", "u", "a_se", "d_se", "g_se", "u_se")]
  c(out, list(raw_logit = raw))
}
extract <- function(m) {
  cf <- coef(m, printSE = TRUE)
  nat <- coef(m)
  items <- setdiff(names(cf), "GroupPars")
  params <- lapply(items, function(i) clean_item(cf[[i]], nat[[i]][1, ])); names(params) <- items
  list(all = list(params = params,
                  latent = list(mean = 0, var = 1, mean_se = NA, var_se = NA)))
}

write_case <- function(case, description, dat, itemtype, truth, pars = NULL) {
  fit_fun <- function(se) {
    args <- list(dat, 1, itemtype = itemtype, SE = TRUE, SE.type = se,
                 quadpts = quad$quadpts, technical = tech)
    if (!is.null(pars)) args$pars <- pars
    do.call(mirt, c(args, common))
  }
  m_oakes <- fit_fun("Oakes")
  m_rich <- fit_fun("Richardson")
  key <- apply(dat, 1, paste, collapse = ",")
  first <- !duplicated(key)
  patterns <- as.matrix(dat[first, , drop = FALSE])
  sc <- fscores(m_oakes, method = "EAP", response.pattern = patterns,
                quadpts = quad$quadpts, theta_lim = quad$theta_lim)
  res <- list(
    meta = list(case = case, description = description, itemtype = itemtype,
                r_version = R.version.string, mirt_version = as.character(packageVersion("mirt")),
                TOL = 1e-10, quadpts = quad$quadpts, theta_lim = quad$theta_lim,
                grid_check = range(m_oakes@Model$Theta),
                accelerate = "none", optimizer = "NR",
                converged = extract.mirt(m_oakes, "converged"),
                iterations = extract.mirt(m_oakes, "iterations"),
                converged_richardson = extract.mirt(m_rich, "converged"),
                se_scale = "natural (mirt's logit-scale SEs converted by the delta method)"),
    items = colnames(dat),
    groups = I("all"),
    patterns = unname(patterns),
    pattern_groups = rep("all", sum(first)),
    counts = as.integer(table(key)[key[first]]),
    loglik = extract.mirt(m_oakes, "logLik"),
    estimates = extract(m_oakes),
    se_richardson = extract(m_rich),
    eap = list(theta = unname(sc[, "F1"]), se = unname(sc[, "SE_F1"])),
    truth = truth
  )
  stopifnot(
    isTRUE(res$meta$converged),
    isTRUE(res$meta$converged_richardson),
    all(res$meta$grid_check == quad$theta_lim),
    all(range(m_rich@Model$Theta) == quad$theta_lim)
  )
  path <- file.path(out_dir, paste0(case, ".json"))
  write_json(res, path, auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null")
  est <- res$estimates$all$params
  rich <- res$se_richardson$all$params
  se_cols <- if (itemtype == "3PL") c("a_se", "d_se", "g_se") else c("a_se", "d_se", "g_se", "u_se")
  max_se_diff <- max(sapply(names(est), function(i) {
    max(sapply(se_cols, function(c) abs(est[[i]][[c]] - rich[[i]][[c]]) / abs(rich[[i]][[c]])))
  }))
  cat("wrote", path, " loglik =", sprintf("%.6f", res$loglik), " iter =", res$meta$iterations,
      " converged =", res$meta$converged, "/", res$meta$converged_richardson,
      " g in [", sprintf("%.4f", min(sapply(est, `[[`, "g"))), ",",
      sprintf("%.4f", max(sapply(est, `[[`, "g"))), "]",
      " u in [", sprintf("%.4f", min(sapply(est, `[[`, "u"))), ",",
      sprintf("%.4f", max(sapply(est, `[[`, "u"))), "]",
      " max |a| =", sprintf("%.3f", max(abs(sapply(est, `[[`, "a")))),
      " max Oakes/Rich SE rel =", sprintf("%.2e", max_se_diff), "\n")
}

# ---- V11a: 3PL, J = 15, N = 5000 ----------------------------------------------------------------
set.seed(20261124)
J <- 15; N <- 5000
a3 <- rlnorm(J, 0.4, 0.2); d3 <- runif(J, -1.5, 0.5); g3 <- runif(J, 0.12, 0.25)
dat3 <- as.data.frame(simdata(matrix(a3), matrix(d3), N, itemtype = "3PL", guess = g3))
colnames(dat3) <- sprintf("q%02d", 1:J)
write_case("v11_3pl",
           "V11: 3PL, J=15, N=5000, a~lognormal(0.4,0.2), d~U(-1.5,0.5), g~U(0.12,0.25)",
           dat3, "3PL", list(a = a3, d = d3, g = g3))

# ---- V11b: 4PL, J = 12, N = 20000 ---------------------------------------------------------------
set.seed(20261125)
J <- 12; N <- 20000
a4 <- rlnorm(J, 0.6, 0.15); d4 <- runif(J, -0.5, 0.5); g4 <- runif(J, 0.10, 0.22); u4 <- runif(J, 0.78, 0.90)
dat4 <- as.data.frame(simdata(matrix(a4), matrix(d4), N, itemtype = "4PL", guess = g4, upper = u4))
colnames(dat4) <- sprintf("q%02d", 1:J)
sv4 <- mirt(dat4, 1, "4PL", pars = "values", quadpts = quad$quadpts, technical = tech)
sv4[sv4$name == "a1", "value"] <- 2.0
write_case("v11_4pl",
           "V11: 4PL, J=12, N=20000, a~lognormal(0.6,0.15), d~U(-0.5,0.5), g~U(0.10,0.22), u~U(0.78,0.90)",
           dat4, "4PL", list(a = a4, d = d4, g = g4, u = u4), pars = sv4)
