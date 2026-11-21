# M5 reference values (VALIDATION.md V23: nominal response model) from R mirt.
# Run from the repository root:  Rscript validation/r/m5_nominal_reference.R
# Writes tests/reference/v23_nominal_science.json, tests/reference/v23_nominal_sim.json, and
# tests/reference/v23_nominal_mixed.json. CI does not run R; the JSON files are committed.
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
#   the GPCM point (ak_k = k, d_k = log(p_k / p_0)) avoids the iteration-0 symmetry saddle and
#   converges to the exact global ML solution (verified against default-start BFGS to < 2e-5 on
#   Science and < 3e-7 on the simulated cases). flexpyirt fits from its own default starting values.
#   (With mirt's default starts, NR on Science stops at logLik -1631.79 with Comfort a1 ~ 0,
#   versus the ML solution -1608.452.)
#
# ML scoring (V23d): mirt 1.46.1 fscores(method = "ML") returns -Inf / +Inf for every
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

nominal_start_pars <- function(dat, itemtype, quad) {
  sv <- mirt(dat, 1, itemtype = itemtype, pars = "values",
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
  list(
    patterns = unname(sub_pats),
    map = run_fs("MAP"),
    ml = run_fs("ML"),
    wle = run_fs("WLE")
  )
}

write_case <- function(case, description, dat, itemtype, quad, n_cat,
                       extra_patterns = NULL, extra = list()) {
  sv <- nominal_start_pars(dat, itemtype, quad)
  fit_fun <- function(se) {
    do.call(mirt, c(list(dat, 1, itemtype = itemtype, pars = sv, SE = TRUE, SE.type = se,
                         quadpts = quad$quadpts, technical = tech(quad)), common))
  }
  m_oakes <- fit_fun("Oakes")
  m_rich <- fit_fun("Richardson")
  key <- apply(dat, 1, paste, collapse = ",")
  first <- !duplicated(key)
  patterns <- as.matrix(dat[first, , drop = FALSE])
  sc <- fscores(m_oakes, method = "EAP", response.pattern = patterns,
                quadpts = quad$quadpts, theta_lim = quad$theta_lim)
  fsc <- score_subset(m_oakes, patterns, quad, extra_patterns = extra_patterns)
  res <- c(list(
    meta = list(case = case, description = description, itemtype = itemtype, n_cat = n_cat,
                r_version = R.version.string, mirt_version = as.character(packageVersion("mirt")),
                TOL = 1e-10, quadpts = quad$quadpts, theta_lim = quad$theta_lim,
                grid_check = range(m_oakes@Model$Theta),
                accelerate = "none", optimizer = "NR",
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
    fscores = fsc
  ), extra)
  stopifnot(
    isTRUE(res$meta$converged),
    isTRUE(res$meta$converged_richardson),
    all(res$meta$grid_check == quad$theta_lim),
    all(range(m_rich@Model$Theta) == quad$theta_lim)
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
