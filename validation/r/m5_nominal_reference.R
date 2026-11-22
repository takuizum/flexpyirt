# M5 reference values (VALIDATION.md V23: nominal response model) from R mirt.
# Run from the repository root:  Rscript validation/r/m5_nominal_reference.R
# Writes tests/reference/v23_nominal_{science,sim,mixed,unordered}.json (unidimensional) and
# tests/reference/v23_nominal_2d.json (two-factor confirmatory). CI does not run R; the JSON files
# are committed.
#
# As in m1/m2/m3_gpcm_reference.R: optimizer = "NR", accelerate = "none", TOL = 1e-10
# (theta_lim inside `technical`; the grid actually used is recorded in meta.grid_check).
#
# Parameterization mapping from mirt to flexpyirt (DESIGN.md §16.8):
#   mirt's itemtype = "nominal" (Thissen, Cai & Bock, 2010) returns columns
#   a1, ak0..ak{K-1}, d0, d1..d{K-1}, with ak0 = 0, ak{K-1} = K - 1, and d0 = 0 fixed.
#   flexpyirt keeps those three identification constants in NominalKernel and estimates
#   a (= a1), ak1..ak{K-2}, and d1..d{K-1}.
#
# Starting values for mirt's optimizer = "NR":
#   mirt's default nominal starting values set ak1 = ... = ak{K-2} = 0.5 and d1 = ... = d{K-1} = 0,
#   giving all interior categories identical predictors at iteration 0; on skewed data such as
#   Science, undamped NR can step to an indefinite joint Hessian and trap a1 = 0. Starting NR at
#   the GPCM point (ak_k = k, d_k = log(p_k / p_0)) avoids the iteration-0 symmetry saddle.
#   (With mirt's default starts, NR on Science stops at logLik -1631.79 with Comfort a1 ~ 0,
#   versus the ML solution -1608.452.) Every unidimensional case also fits alternative mirt starts
#   (default-start NR, default-start BFGS with mirt's default acceleration, and three
#   GenRandomPars NR starts). If one of them reaches a higher log-likelihood than the GPCM-point NR
#   (V23e: the GPCM-point NR stops at a local maximum, -29561.14 vs -26107.48), NR is restarted
#   from that alternative's estimates; meta$reference_start records which start was used. All
#   starts are stored in `alt_starts` (log-likelihood, convergence flag, largest absolute item
#   parameter difference from the reference); the script asserts that none exceeds the reference.
#   flexpyirt always fits from its own default starting values.
#
# S-X2 (V15-style): itemfit(fit_stats = "S_X2") at the reference estimates on the fit's grid.
#
# ML scoring (V23d): every pattern mirt scores as +/-Inf by ML (or leaves at the max_theta = 20
# bound: V23e has one true +Inf pattern, not all-top-category, that mirt returns as 19.99989
# with SE NA) is also scanned independently
# (`ml_scan`): the pattern log-likelihood from probtrace() on a 0.01 grid over [-20, 20], refined
# with optimize(); a maximum at the grid boundary is recorded as +/-Inf, otherwise the interior
# maximizer and SE = (-d2 logL / d theta2)^(-1/2) (central differences).
# mirt 1.46.1 fscores(method = "ML") returns -Inf / +Inf for every
# all-category-0 / all-top-category pattern, regardless of the nominal scoring coefficients. The
# simulated and mixed cases include an all-0 pattern for which an item has an estimated ak1 < 0, so
# category 1 (not 0) has the lowest category slope and the true ML estimate is interior;
# flexpyirt returns that interior estimate (DESIGN.md §16.8, VALIDATION.md V23). Non-finite scores
# are written as "Inf" / "-Inf" (as in m4_scoring_reference.R).
suppressMessages({
  library(mirt)
  library(jsonlite)
})

out_dir <- file.path("tests", "reference")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
common <- list(TOL = 1e-10, accelerate = "none", optimizer = "NR", verbose = FALSE)
default_quad <- list(quadpts = 61, theta_lim = c(-6, 6))
tech <- function(quad) list(NCYCLES = 50000, theta_lim = quad$theta_lim)

nominal_start_pars <- function(dat, itemtype, quad, model = 1) {
  sv <- mirt(dat, model, itemtype = itemtype, pars = "values",
             quadpts = quad$quadpts, technical = tech(quad))
  nom_items <- unique(as.character(sv$item[sv$class == "nominal"]))
  for (itm in nom_items) {
    col <- dat[[itm]]
    col <- col[!is.na(col)]
    K <- max(col) + 1L
    for (k in seq_len(K - 2L)) {
      sv$value[sv$item == itm & sv$name == paste0("ak", k)] <- as.numeric(k)
    }
    tab <- table(factor(col, levels = 0:(K - 1L)))
    p <- pmax(as.numeric(tab) / sum(tab), 0.01)
    d_start <- log(p / p[1L])
    for (k in seq_len(K - 1L)) {
      sv$value[sv$item == itm & sv$name == paste0("d", k)] <- d_start[k + 1L]
    }
  }
  sv
}

clean_item <- function(x) {
  nm <- colnames(x)
  if ("ak0" %in% nm) {
    ak_cols <- nm[grepl("^ak[0-9]+$", nm)]
    K <- length(ak_cols)
    stopifnot(K >= 3L)
    stopifnot(x["par", "ak0"] == 0)
    stopifnot(x["par", paste0("ak", K - 1L)] == K - 1L)
    stopifnot("d0" %in% nm && x["par", "d0"] == 0)
    if ("SE" %in% rownames(x)) {
      stopifnot(is.na(x["SE", "ak0"]), is.na(x["SE", paste0("ak", K - 1L)]), is.na(x["SE", "d0"]))
    }
    keep <- c("a1", paste0("ak", 1:(K - 2L)), paste0("d", 1:(K - 1L)))
  } else {
    # Dichotomous 2PL item in mixed model
    stopifnot(all(c("a1", "d") %in% nm))
    keep <- c("a1", "d")
  }
  par <- x["par", keep]
  se <- if ("SE" %in% rownames(x)) x["SE", keep] else rep(NA, length(keep))
  names(par) <- sub("^a1$", "a", keep)
  names(se) <- paste0(names(par), "_se")
  c(as.list(par), as.list(se))
}

extract <- function(m) {
  cf <- coef(m, printSE = TRUE)
  items <- setdiff(names(cf), "GroupPars")
  params <- lapply(items, function(i) clean_item(cf[[i]]))
  names(params) <- items
  gp <- cf$GroupPars
  se <- if ("SE" %in% rownames(gp)) gp["SE", ] else c(NA, NA)
  list(all = list(
    params = params,
    latent = list(mean = gp["par", "MEAN_1"], var = gp["par", "COV_11"],
                  mean_se = unname(se[1]), var_se = unname(se[2]))
  ))
}

# Non-finite ML scores are written as the strings "Inf" / "-Inf" (as in m4_scoring_reference.R);
# NA / NaN are written as null.
encode_num <- function(x) {
  lapply(unname(as.numeric(x)), function(v) {
    if (is.na(v) || is.nan(v)) {
      NA_real_
    } else if (is.infinite(v)) {
      if (v > 0) "Inf" else "-Inf"
    } else {
      v
    }
  })
}

pattern_loglik <- function(m, pattern) {
  function(theta) {
    th <- matrix(theta)
    total <- numeric(length(theta))
    for (j in seq_along(pattern)) {
      pr <- probtrace(extract.item(m, j), th)
      total <- total + log(pr[, pattern[j] + 1L])
    }
    total
  }
}

ml_scan_one <- function(m, pattern) {
  f <- pattern_loglik(m, pattern)
  grid <- seq(-20, 20, by = 0.01)
  vals <- f(grid)
  i <- which.max(vals)
  if (i <= 2L) return(list(theta = -Inf, se = NA_real_))
  if (i >= length(grid) - 1L) return(list(theta = Inf, se = NA_real_))
  opt <- optimize(f, c(grid[i - 1L], grid[i + 1L]), maximum = TRUE, tol = 1e-12)
  t <- opt$maximum
  h <- 1e-4
  d2 <- (f(t + h) - 2 * f(t) + f(t - h)) / h^2
  list(theta = t, se = 1 / sqrt(-d2))
}

ml_scan <- function(m, pats, ml_theta) {
  # mirt's +/-Inf patterns and those where nlm stopped at the max_theta = 20 bound.
  idx <- which(is.infinite(ml_theta) | abs(ml_theta) > 19.9)
  out <- lapply(idx, function(i) ml_scan_one(m, as.integer(pats[i, ])))
  list(index = as.integer(idx - 1L),  # 0-based rows of fscores$patterns
       theta = encode_num(vapply(out, function(o) o$theta, numeric(1))),
       se = encode_num(vapply(out, function(o) o$se, numeric(1))))
}

item_param_vector <- function(m) {
  cf <- coef(m)
  items <- setdiff(names(cf), "GroupPars")
  unlist(lapply(items, function(i) {
    x <- cf[[i]]
    nm <- colnames(x)
    keep <- nm[grepl("^(a[0-9]+|ak[1-9][0-9]*|d[0-9]*)$", nm)]
    x[1L, keep]
  }))
}

alt_start_fits <- function(dat, itemtype, quad) {
  run <- function(label, ...) {
    m <- tryCatch(
      suppressWarnings(mirt(dat, 1, itemtype = itemtype, TOL = 1e-10, verbose = FALSE,
                            quadpts = quad$quadpts,
                            technical = list(NCYCLES = 20000, theta_lim = quad$theta_lim), ...)),
      error = function(e) NULL
    )
    list(label = label, model = m)
  }
  fits <- list(
    run("default_nr", optimizer = "NR", accelerate = "none"),
    run("default_bfgs", optimizer = "BFGS")
  )
  for (s in 1:3) {
    set.seed(s)
    fits[[length(fits) + 1L]] <- run(sprintf("random_nr_seed%d", s), optimizer = "NR",
                                     accelerate = "none", GenRandomPars = TRUE)
  }
  fits
}

summarize_alt <- function(label, m, ref_vec) {
  if (is.null(m)) {
    return(list(label = label, loglik = NA_real_, converged = FALSE,
                max_abs_param_diff = NA_real_))
  }
  out <- list(label = label, loglik = extract.mirt(m, "logLik"),
              converged = isTRUE(extract.mirt(m, "converged")),
              max_abs_param_diff = max(abs(item_param_vector(m) - ref_vec)))
  cat(sprintf("    alt %-18s loglik=%.6f converged=%s max|dpar|=%.2e\n",
              out$label, out$loglik, out$converged, out$max_abs_param_diff))
  out
}

score_subset <- function(m, patterns, quad, max_pats = 40L, extra_patterns = NULL) {
  n_p <- nrow(patterns)
  if (n_p <= max_pats) {
    sub_pats <- patterns
  } else {
    idx <- unique(round(seq(1L, n_p, length.out = max_pats)))
    sub_pats <- patterns[idx, , drop = FALSE]
  }
  if (!is.null(extra_patterns)) {
    colnames(extra_patterns) <- colnames(patterns)
    sub_pats <- rbind(extra_patterns, sub_pats)
    key_s <- apply(sub_pats, 1, paste, collapse = ",")
    sub_pats <- sub_pats[!duplicated(key_s), , drop = FALSE]
  }
  run_fs <- function(method) {
    sc <- suppressWarnings(fscores(
      m, method = method, response.pattern = sub_pats,
      quadpts = quad$quadpts, theta_lim = quad$theta_lim, full.scores.SE = TRUE
    ))
    list(theta = encode_num(sc[, "F1"]), se = encode_num(sc[, "SE_F1"]))
  }
  ml <- run_fs("ML")
  ml_theta <- vapply(ml$theta, function(v) if (is.character(v)) ifelse(v == "Inf", Inf, -Inf)
                     else as.numeric(v), numeric(1))
  list(
    patterns = unname(sub_pats),
    map = run_fs("MAP"),
    ml = ml,
    wle = run_fs("WLE"),
    ml_scan = ml_scan(m, sub_pats, ml_theta)
  )
}

write_case <- function(case, description, dat, itemtype, quad, n_cat,
                       extra_patterns = NULL, extra = list()) {
  # Reference start: the GPCM point. If an alternative start (alt_start_fits) reaches a higher
  # log-likelihood, NR is restarted from that alternative's estimates instead, so the reference
  # is the best solution found (recorded in meta$reference_start).
  sv <- nominal_start_pars(dat, itemtype, quad)
  fit_fun <- function(se, pars) {
    do.call(mirt, c(list(dat, 1, itemtype = itemtype, pars = pars, SE = TRUE, SE.type = se,
                         quadpts = quad$quadpts, technical = tech(quad)), common))
  }
  m_oakes <- fit_fun("Oakes", sv)
  gpcm_ll <- extract.mirt(m_oakes, "logLik")
  alt_models <- alt_start_fits(dat, itemtype, quad)
  alt_ll <- vapply(alt_models, function(f) if (is.null(f$model)) -Inf
                   else extract.mirt(f$model, "logLik"), numeric(1))
  reference_start <- "gpcm_point"
  if (max(alt_ll) > gpcm_ll + 1e-6 * abs(gpcm_ll)) {
    best <- alt_models[[which.max(alt_ll)]]
    reference_start <- best$label
    sv <- mod2values(best$model)
    # mirt's BFGS stores the fixed nominal constants with rounding error (ak{K-1} = K - 1 - 4e-15);
    # restore them exactly before the NR refit.
    nom <- sv$class == "nominal"
    for (itm in unique(sv$item[nom])) {
      sel <- nom & sv$item == itm
      ak_names <- grep("^ak[0-9]+$", sv$name[sel], value = TRUE)
      top <- length(ak_names) - 1L
      sv$value[sel & sv$name == "ak0"] <- 0
      sv$value[sel & sv$name == paste0("ak", top)] <- top
      sv$value[sel & sv$name == "d0"] <- 0
    }
    m_oakes <- fit_fun("Oakes", sv)
  }
  m_rich <- fit_fun("Richardson", sv)
  ref_vec <- item_param_vector(m_oakes)
  alt <- c(
    list(list(label = "gpcm_point_nr", loglik = gpcm_ll, converged = TRUE,
              max_abs_param_diff = NA_real_)),
    lapply(alt_models, function(f) summarize_alt(f$label, f$model, ref_vec))
  )
  cat(sprintf("    gpcm-point NR loglik=%.6f; reference start = %s\n", gpcm_ll, reference_start))
  key <- apply(dat, 1, paste, collapse = ",")
  first <- !duplicated(key)
  patterns <- as.matrix(dat[first, , drop = FALSE])
  sc <- fscores(m_oakes, method = "EAP", response.pattern = patterns,
                quadpts = quad$quadpts, theta_lim = quad$theta_lim)
  fsc <- score_subset(m_oakes, patterns, quad, extra_patterns = extra_patterns)
  sx2 <- itemfit(m_oakes, fit_stats = "S_X2", quadpts = quad$quadpts, theta_lim = quad$theta_lim)
  res <- c(list(
    meta = list(case = case, description = description, itemtype = itemtype, n_cat = n_cat,
                r_version = R.version.string, mirt_version = as.character(packageVersion("mirt")),
                TOL = 1e-10, quadpts = quad$quadpts, theta_lim = quad$theta_lim,
                grid_check = range(m_oakes@Model$Theta),
                accelerate = "none", optimizer = "NR", reference_start = reference_start,
                converged = extract.mirt(m_oakes, "converged"),
                iterations = extract.mirt(m_oakes, "iterations"),
                converged_richardson = extract.mirt(m_rich, "converged")),
    items = colnames(dat),
    groups = I("all"),
    patterns = unname(patterns),
    pattern_groups = rep("all", sum(first)),
    counts = as.integer(table(key)[key[first]]),
    loglik = extract.mirt(m_oakes, "logLik"),
    estimates = extract(m_oakes),
    se_richardson = extract(m_rich),
    eap = list(theta = unname(sc[, "F1"]), se = unname(sc[, "SE_F1"])),
    fscores = fsc,
    s_x2 = list(item = sx2$item, S_X2 = sx2$S_X2, df = sx2$df.S_X2, RMSEA = sx2$RMSEA.S_X2,
                p = sx2$p.S_X2),
    alt_starts = alt
  ), extra)
  stopifnot(
    isTRUE(res$meta$converged),
    isTRUE(res$meta$converged_richardson),
    all(res$meta$grid_check == quad$theta_lim),
    all(range(m_rich@Model$Theta) == quad$theta_lim),
    all(vapply(alt, function(f) is.na(f$loglik) || f$loglik <= res$loglik + 1e-6 * abs(res$loglik),
               logical(1)))
  )
  path <- file.path(out_dir, paste0(case, ".json"))
  write_json(res, path, auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null")
  cat("wrote", path, " loglik =", sprintf("%.6f", res$loglik),
      " iter =", res$meta$iterations,
      " converged =", res$meta$converged, "/", res$meta$converged_richardson,
      " grid =", paste(res$meta$grid_check, collapse = ".."), "\n")
}

# ---- V23a: Nominal on mirt::Science (recoded 1..4 -> 0..3) -------------------------------------
sci <- as.data.frame(as.matrix(Science) - 1L)
write_case("v23_nominal_science",
           "V23a: Nominal response model, mirt::Science recoded to 0..3",
           sci, "nominal", default_quad, n_cat = 4L)

# ---- V23b: Nominal, simulated J = 10, K = 4, N = 3000 with non-monotone categories -------------
set.seed(20261130)
J <- 10L; K <- 4L; N <- 3000L
a_nom <- rlnorm(J, 0.1, 0.25)
ak_nom <- cbind(0, runif(J, 0.6, 1.4), runif(J, 1.6, 2.4), 3)
# Item 9: swapped interior categories (ak1 = 2.2 > ak2 = 0.7)
ak_nom[9, ] <- c(0, 2.2, 0.7, 3)
# Item 10: category 1 has negative scoring coefficient (ak1 = -0.4 < ak0 = 0)
ak_nom[10, ] <- c(0, -0.4, 1.8, 3)
d_nom <- cbind(0, runif(J, -0.8, 0.8), runif(J, -0.8, 0.8), runif(J, -1.2, 0.4))
dat_nom <- as.data.frame(simdata(matrix(a_nom), d_nom, N, itemtype = "nominal", nominal = ak_nom))
colnames(dat_nom) <- sprintf("q%02d", 1:J)
# Include all-0 pattern (interior ML because q10 has ak1 < 0) and true extreme patterns
extra_pats_sim <- rbind(
  rep(0L, J),
  c(rep(0L, J - 1L), 1L),
  rep(3L, J)
)
write_case("v23_nominal_sim",
           "V23b: Nominal response model, simulated J=10, K=4, N=3000 with non-monotone items (seed 20261130)",
           dat_nom, "nominal", default_quad, n_cat = K,
           extra_patterns = extra_pats_sim,
           extra = list(truth = list(a = a_nom, ak = unname(ak_nom[, 2:3]), d = unname(d_nom[, -1]))))

# ---- V23c: Mixed model (5 nominal K = 4 + 5 2PL K = 2, N = 2500) -------------------------------
set.seed(20261201)
N_mix <- 2500L
theta_mix <- rnorm(N_mix)
ak_mix5 <- ak_nom[c(1, 2, 3, 9, 10), ]
a_mix_nom <- a_nom[c(1, 2, 3, 9, 10)]
d_mix_nom <- d_nom[c(1, 2, 3, 9, 10), ]
dat_nom5 <- simdata(matrix(a_mix_nom), d_mix_nom, N = N_mix, Theta = matrix(theta_mix),
                    itemtype = "nominal", nominal = ak_mix5)
a_2pl5 <- rlnorm(5, 0.1, 0.25)
d_2pl5 <- runif(5, -1.0, 1.0)
dat_2pl5 <- simdata(matrix(a_2pl5), matrix(d_2pl5), N = N_mix, Theta = matrix(theta_mix),
                    itemtype = "2PL")
dat_mix <- as.data.frame(cbind(dat_nom5, dat_2pl5))
colnames(dat_mix) <- sprintf("q%02d", 1:10)
itemtypes_mix <- c(rep("nominal", 5), rep("2PL", 5))
n_cats_mix <- c(rep(4L, 5), rep(2L, 5))
extra_pats_mix <- rbind(
  c(0L, 0L, 0L, 0L, 1L, rep(0L, 5)),
  c(rep(3L, 5), rep(1L, 5)),
  rep(0L, 10)
)
write_case("v23_nominal_mixed",
           "V23c: Mixed nominal (5 items, K=4) + 2PL (5 items, K=2), N=2500 (seed 20261201)",
           dat_mix, itemtypes_mix, default_quad, n_cat = n_cats_mix,
           extra_patterns = extra_pats_mix,
           extra = list(truth = list(a_nominal = a_mix_nom, ak_nominal = unname(ak_mix5[, 2:3]),
                                     d_nominal = unname(d_mix_nom[, -1]),
                                     a_2pl = a_2pl5, d_2pl = d_2pl5)))

# ---- V23e: Nominal, simulated J = 8, K = 4, N = 3000, mostly non-monotone scoring functions -----
# Six of eight items have unordered scoring coefficients, including ak > K - 1 (q03, q05) and
# negative ak (q02, q04, q06); flexpyirt must reach the ML solution from its own default starts.
set.seed(20261205)
J_u <- 8L; N_u <- 3000L
a_u <- rlnorm(J_u, 0.1, 0.2)
ak_u <- rbind(
  c(0, 2.4, 0.8, 3),
  c(0, -0.6, 1.5, 3),
  c(0, 3.6, 1.2, 3),
  c(0, 1.9, -0.3, 3),
  c(0, 0.5, 4.0, 3),
  c(0, -0.5, -0.8, 3),
  c(0, 1.0, 2.0, 3),
  c(0, 0.8, 2.2, 3)
)
d_u <- cbind(0, runif(J_u, -0.6, 0.6), runif(J_u, -0.6, 0.6), runif(J_u, -1.0, 0.2))
dat_u <- as.data.frame(simdata(matrix(a_u), d_u, N_u, itemtype = "nominal", nominal = ak_u))
colnames(dat_u) <- sprintf("q%02d", 1:J_u)
low_u <- apply(ak_u, 1, which.min) - 1L
high_u <- apply(ak_u, 1, which.max) - 1L
extra_pats_u <- rbind(rep(0L, J_u), rep(3L, J_u), low_u, high_u)
write_case("v23_nominal_unordered",
           "V23e: Nominal response model, simulated J=8, K=4, N=3000, six non-monotone items (seed 20261205)",
           dat_u, "nominal", default_quad, n_cat = 4L,
           extra_patterns = extra_pats_u,
           extra = list(truth = list(a = a_u, ak = unname(ak_u[, 2:3]), d = unname(d_u[, -1]))))

# ---- V23f: Two-factor confirmatory nominal (F1 = q01-q04, F2 = q05-q08, COV = F1*F2) -----------
# V14-style: 31 points per dimension on [-6, 6] (961 nodes), NR, Oakes SEs, latent means 0 and
# variances 1 fixed, correlation free. mirt's multidimensional nominal model multiplies the
# scoring coefficients by a1*theta1 + a2*theta2 (DESIGN.md §16.8); unloaded slopes are fixed at 0.
extract_latent_2d <- function(gp) {
  list(mean_1 = unname(gp["par", "MEAN_1"]), mean_2 = unname(gp["par", "MEAN_2"]),
       var_1 = unname(gp["par", "COV_11"]), var_2 = unname(gp["par", "COV_22"]),
       cor_12 = unname(gp["par", "COV_21"]),
       mean_1_se = unname(gp["SE", "MEAN_1"]), mean_2_se = unname(gp["SE", "MEAN_2"]),
       var_1_se = unname(gp["SE", "COV_11"]), var_2_se = unname(gp["SE", "COV_22"]),
       cor_12_se = unname(gp["SE", "COV_21"]))
}

set.seed(20261206)
J2 <- 8L; N2 <- 3000L
a_2d <- cbind(c(rlnorm(4, 0.1, 0.25), rep(0, 4)), c(rep(0, 4), rlnorm(4, 0.1, 0.25)))
ak_2d <- cbind(0, runif(J2, 0.6, 1.4), runif(J2, 1.6, 2.4), 3)
ak_2d[4, ] <- c(0, 2.2, 0.7, 3)
ak_2d[8, ] <- c(0, -0.4, 1.8, 3)
d_2d <- cbind(0, runif(J2, -0.8, 0.8), runif(J2, -0.8, 0.8), runif(J2, -1.2, 0.4))
cor_2d <- 0.4
dat_2d <- as.data.frame(simdata(a_2d, d_2d, N2, itemtype = "nominal", nominal = ak_2d,
                                sigma = matrix(c(1, cor_2d, cor_2d, 1), 2)))
colnames(dat_2d) <- sprintf("q%02d", 1:J2)
quad_2d <- list(quadpts = 31L, theta_lim = c(-6, 6))
mod_2d <- mirt.model("F1 = 1-4\nF2 = 5-8\nCOV = F1*F2")
sv_2d <- nominal_start_pars(dat_2d, "nominal", quad_2d, model = mod_2d)
m_2d <- do.call(mirt, c(list(dat_2d, mod_2d, itemtype = "nominal", pars = sv_2d, SE = TRUE,
                             SE.type = "Oakes", quadpts = quad_2d$quadpts,
                             technical = tech(quad_2d)), common))
m_2d_default <- suppressWarnings(do.call(mirt, c(list(dat_2d, mod_2d, itemtype = "nominal",
                                                      quadpts = quad_2d$quadpts,
                                                      technical = tech(quad_2d)), common)))
cf_2d <- coef(m_2d, printSE = TRUE)
params_2d <- lapply(colnames(dat_2d), function(i) {
  x <- cf_2d[[i]]
  keep <- c("a1", "a2", "ak1", "ak2", "d1", "d2", "d3")
  stopifnot(x["par", "ak0"] == 0, x["par", "ak3"] == 3, x["par", "d0"] == 0)
  par <- x["par", keep]
  se <- x["SE", keep]
  names(se) <- paste0(keep, "_se")
  c(as.list(par), as.list(se))
})
names(params_2d) <- colnames(dat_2d)
key_2d <- apply(dat_2d, 1, paste, collapse = ",")
first_2d <- !duplicated(key_2d)
pats_2d <- as.matrix(dat_2d[first_2d, , drop = FALSE])
eap_idx_2d <- unique(as.integer(round(seq(1L, nrow(pats_2d), length.out = 200L))))
sc_2d <- fscores(m_2d, method = "EAP", response.pattern = pats_2d[eap_idx_2d, , drop = FALSE],
                 quadpts = quad_2d$quadpts, theta_lim = quad_2d$theta_lim)
res_2d <- list(
  meta = list(case = "v23_nominal_2d",
              description = "V23f: Two-factor confirmatory nominal, J=8, K=4, N=3000, cor 0.4 (seed 20261206)",
              itemtype = "nominal", n_cat = 4L, n_factors = 2L,
              r_version = R.version.string, mirt_version = as.character(packageVersion("mirt")),
              TOL = 1e-10, quadpts = quad_2d$quadpts, theta_lim = quad_2d$theta_lim,
              grid_check = range(m_2d@Model$Theta), n_nodes = nrow(m_2d@Model$Theta),
              accelerate = "none", optimizer = "NR",
              converged = extract.mirt(m_2d, "converged"),
              iterations = extract.mirt(m_2d, "iterations")),
  items = colnames(dat_2d),
  factors = list(F1 = colnames(dat_2d)[1:4], F2 = colnames(dat_2d)[5:8]),
  patterns = unname(pats_2d),
  counts = as.integer(table(key_2d)[key_2d[first_2d]]),
  loglik = extract.mirt(m_2d, "logLik"),
  estimates = list(params = params_2d, latent = extract_latent_2d(cf_2d$GroupPars)),
  eap = list(pattern_indices = eap_idx_2d - 1L,
             theta = list(F1 = unname(sc_2d[, "F1"]), F2 = unname(sc_2d[, "F2"])),
             se = list(F1 = unname(sc_2d[, "SE_F1"]), F2 = unname(sc_2d[, "SE_F2"]))),
  alt_starts = list(list(label = "default_nr", loglik = extract.mirt(m_2d_default, "logLik"),
                         converged = isTRUE(extract.mirt(m_2d_default, "converged")))),
  truth = list(a = unname(a_2d), ak = unname(ak_2d[, 2:3]), d = unname(d_2d[, -1]), cor = cor_2d)
)
stopifnot(isTRUE(res_2d$meta$converged), all(res_2d$meta$grid_check == quad_2d$theta_lim),
          res_2d$meta$n_nodes == quad_2d$quadpts^2,
          res_2d$alt_starts[[1]]$loglik <= res_2d$loglik + 1e-6 * abs(res_2d$loglik))
path_2d <- file.path(out_dir, "v23_nominal_2d.json")
write_json(res_2d, path_2d, auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null")
cat("wrote", path_2d, " loglik =", sprintf("%.6f", res_2d$loglik),
    " iter =", res_2d$meta$iterations, " cor =", sprintf("%.5f", res_2d$estimates$latent$cor_12),
    " default-start loglik =", sprintf("%.6f", res_2d$alt_starts[[1]]$loglik), "\n")
