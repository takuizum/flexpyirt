# M4 reference values (VALIDATION.md V13: MAP, ML, and WLE ability scoring) from R mirt.
# Run from the repository root:  Rscript validation/r/m4_scoring_reference.R
# Writes tests/reference/v13_fscores.json. CI does not run R; the JSON is committed.
#
# Settings match m1–m4 references: optimizer = "NR", accelerate = "none", TOL = 1e-10,
# theta_lim inside technical = list(...), convergence and grid_check asserted for every case.
#
# mirt 1.46.1 fscores() conventions recorded here (VALIDATION.md V13):
#   - Bounds: theta_lim is used for EAP quadrature (and EAP starting values for MAP/ML/WLE);
#     MAP, ML, and WLE optimize over [-max_theta, max_theta] with default max_theta = 20.
#   - SEs (default expected.info = FALSE): inverse square root of the Hessian returned by nlm()
#     at the optimum — i.e., observed posterior info for MAP, observed info for ML, and observed
#     curvature of -(log L + 0.5 * log I) for WLE.
#   - Extreme patterns for ML: all-lowest observed -> -Inf (SE NA), all-highest -> +Inf (SE NA),
#     all-missing -> NA (SE NA), and diverging non-concave 3PL/4PL tails where SE > 30 -> +/-Inf
#     (SE NA). (Note two mirt 1.46.1 bugs documented in VALIDATION.md V13: (a) in 3PL flat tails
#     where nlm's finite-difference Hessian turns <= 0, mirt leaves F1 ~ -17 with SE = NA instead
#     of setting -Inf; (b, observed; cause unconfirmed in mirt's R source) with missing data on the last item column, mirt's tabdata[, -ncol()]
#     undercounts NAs and leaves all-extreme rows at ~20 with SE = NA.)
source(file.path("validation", "r", "setup.R"))  # mirt from GitHub; mirt_version / mirt_sha
suppressMessages({
  library(mirt)
  library(jsonlite)
})

out_dir <- file.path("tests", "reference")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
common <- list(TOL = 1e-10, accelerate = "none", optimizer = "NR", verbose = FALSE)
default_quad <- list(quadpts = 61L, theta_lim = c(-6, 6))
wide_quad <- list(quadpts = 81L, theta_lim = c(-8, 8))
tech <- function(quad) list(NCYCLES = 50000L, theta_lim = quad$theta_lim)

extract_si_item <- function(x, nat, itemtype) {
  nm <- colnames(x)
  if (itemtype == "3PL") {
    par <- x["par", ]
    return(list(
      a = par[["a1"]],
      d = par[["d"]],
      g = nat[["g"]],
      u = nat[["u"]]
    ))
  } else if (itemtype == "gpcm") {
    ak_cols <- nm[grepl("^ak[0-9]+$", nm)]
    stopifnot(all(x["par", ak_cols] == seq(0L, length(ak_cols) - 1L)))
    stopifnot("d0" %in% nm && x["par", "d0"] == 0)
    keep <- c("a1", nm[grepl("^d[1-9][0-9]*$", nm)])
  } else if (itemtype == "graded") {
    keep <- c("a1", nm[grepl("^d[1-9][0-9]*$", nm)])
  } else {
    keep <- c("a1", "d")
  }
  par <- x["par", keep]
  names(par) <- sub("^a1$", "a", keep)
  as.list(par)
}

extract_si_group <- function(cf_si, cf_nat, itemtype) {
  items <- setdiff(names(cf_si), "GroupPars")
  params <- lapply(items, function(i) extract_si_item(cf_si[[i]], cf_nat[[i]][1, ], itemtype))
  names(params) <- items
  gp <- cf_si$GroupPars
  list(
    params = params,
    latent = list(mean = gp["par", "MEAN_1"], var = gp["par", "COV_11"])
  )
}

encode_num <- function(x) {
  lapply(as.numeric(x), function(v) {
    if (is.na(v) || is.nan(v)) {
      NA_real_
    } else if (is.infinite(v)) {
      if (v > 0) "Inf" else "-Inf"
    } else {
      v
    }
  })
}

score_all <- function(m, pats, quad) {
  out <- list()
  for (meth in c("EAP", "MAP", "ML", "WLE")) {
    sc <- fscores(
      m,
      method = meth,
      response.pattern = pats,
      full.scores.SE = TRUE,
      quadpts = quad$quadpts,
      theta_lim = quad$theta_lim
    )
    fail <- attr(sc, "failed2converge")
    stopifnot(is.null(fail) || length(fail) == 0L)
    out[[meth]] <- list(
      theta = encode_num(sc[, "F1"]),
      se = encode_num(sc[, "SE_F1"])
    )
  }
  out
}

build_single_case <- function(case, description, dat, itemtype, quad, n_cat, score_pats,
                              source_ref = NULL) {
  m <- do.call(
    mirt,
    c(list(dat, 1L, itemtype = itemtype, SE = FALSE,
           quadpts = quad$quadpts, technical = tech(quad)), common)
  )
  conv <- extract.mirt(m, "converged")
  grid_check <- range(m@Model$Theta)
  stopifnot(isTRUE(conv), all(grid_check == quad$theta_lim))

  cf_si <- coef(m, printSE = TRUE)
  cf_nat <- coef(m)
  si <- extract_si_group(cf_si, cf_nat, itemtype)
  scores <- score_all(m, score_pats, quad)

  meta <- list(
    case = case,
    description = description,
    itemtype = itemtype,
    n_cat = as.integer(n_cat),
    TOL = 1e-10,
    quadpts = quad$quadpts,
    theta_lim = quad$theta_lim,
    grid_check = grid_check,
    accelerate = "none",
    optimizer = "NR",
    converged = conv,
    iterations = extract.mirt(m, "iterations"),
    fscores_max_theta = 20,
    fscores_expected_info = FALSE
  )
  if (!is.null(source_ref)) meta$source_reference <- source_ref

  res <- list(
    meta = meta,
    items = colnames(dat),
    groups = I("all"),
    loglik = extract.mirt(m, "logLik"),
    slope_intercept = list(all = si),
    score_patterns = unname(as.matrix(score_pats)),
    score_groups = rep("all", nrow(score_pats)),
    fscores = scores
  )
  # For self-contained public datasets (LSAT7, Science), also store compressed training patterns
  # so tests can verify the exact log-likelihood at mirt's estimates.
  if (is.null(source_ref)) {
    key <- apply(dat, 1, paste, collapse = ",")
    first <- !duplicated(key)
    res$train_patterns <- unname(as.matrix(dat[first, , drop = FALSE]))
    res$train_counts <- as.integer(table(key)[key[first]])
  }
  res
}

# ---- 1. 2PL on LSAT7 (all 32 unique patterns) ---------------------------------------------------
lsat7 <- as.data.frame(expand.table(LSAT7))
lsat7_uniq <- unique(lsat7)
rownames(lsat7_uniq) <- NULL
c_2pl <- build_single_case(
  "lsat7_2pl", "V13a: 2PL on mirt::LSAT7 (all 32 unique patterns)",
  lsat7, "2PL", default_quad, n_cat = 2L, score_pats = lsat7_uniq
)

# ---- 2. GRM on Science 0..3 (all 90 unique patterns) --------------------------------------------
sci <- as.data.frame(as.matrix(Science) - 1L)
sci_uniq <- unique(sci)
rownames(sci_uniq) <- NULL
c_grm <- build_single_case(
  "science_grm", "V13b: GRM on mirt::Science recoded 0..3 (all 90 unique patterns)",
  sci, "graded", default_quad, n_cat = 4L, score_pats = sci_uniq
)

# ---- 3. GPCM on Science 0..3 (all 90 unique patterns) -------------------------------------------
c_gpcm <- build_single_case(
  "science_gpcm", "V13c: GPCM on mirt::Science recoded 0..3 (all 90 unique patterns)",
  sci, "gpcm", default_quad, n_cat = 4L, score_pats = sci_uniq
)

# ---- 4. 3PL on existing V11a dataset (v11_3pl.json) ---------------------------------------------
v11 <- fromJSON(file.path(out_dir, "v11_3pl.json"))
dat3 <- as.data.frame(v11$patterns[rep(seq_len(nrow(v11$patterns)), v11$counts), ])
colnames(dat3) <- v11$items
m3_tmp <- do.call(
  mirt,
  c(list(dat3, 1L, itemtype = "3PL", SE = FALSE,
         quadpts = default_quad$quadpts, technical = tech(default_quad)), common)
)
# Choose 120 unique patterns covering all-0, all-1, genuine -Inf diverging tails, and interior
# scores across the full sum-score range (excluding mirt's 31 finite-theta/NA-SE Hessian-bug rows,
# which are documented in VALIDATION.md V13).
all3 <- as.data.frame(v11$patterns); colnames(all3) <- v11$items
rs3 <- rowSums(all3)
set.seed(20261150)
cand3 <- sort(unique(c(
  which(rs3 == 0L),
  which(rs3 == ncol(all3)),
  unlist(lapply(1:(ncol(all3) - 1L), function(s) {
    idx <- which(rs3 == s)
    if (length(idx) <= 8L) idx else sample(idx, 8L)
  }))
)))
sub3 <- all3[cand3, , drop = FALSE]
ml_check3 <- suppressWarnings(fscores(
  m3_tmp, method = "ML", response.pattern = sub3, full.scores.SE = TRUE,
  quadpts = default_quad$quadpts, theta_lim = default_quad$theta_lim
))
# Drop any pattern hitting mirt 1.46.1's finite-theta / NA-SE bug on flat 3PL tails
mirt_bug3 <- is.finite(ml_check3[, "F1"]) & is.na(ml_check3[, "SE_F1"])
sub3 <- sub3[!mirt_bug3, , drop = FALSE]
rownames(sub3) <- NULL
c_3pl <- build_single_case(
  "v11_3pl",
  sprintf("V13d: 3PL on V11a dataset (%d unique patterns covering extremes and diverging tails)", nrow(sub3)),
  dat3, "3PL", default_quad, n_cat = 2L, score_pats = sub3, source_ref = "v11_3pl.json"
)

# ---- 5. Two-group 2PL on existing V4 dataset (v4_twogroup_2pl.json) -----------------------------
v4 <- fromJSON(file.path(out_dir, "v4_twogroup_2pl.json"))
dat4 <- as.data.frame(v4$patterns[rep(seq_len(nrow(v4$patterns)), v4$counts), ])
colnames(dat4) <- v4$items
grp4 <- factor(rep(v4$pattern_groups, v4$counts), levels = c("ref", "focal"))
m4 <- do.call(
  multipleGroup,
  c(list(dat4, 1L, group = grp4, itemtype = "2PL",
         invariance = c(v4$anchors, "free_means", "free_var"),
         SE = FALSE, quadpts = wide_quad$quadpts, technical = tech(wide_quad)), common)
)
conv4 <- extract.mirt(m4, "converged")
grid4 <- range(m4@Model$Theta)
stopifnot(isTRUE(conv4), all(grid4 == wide_quad$theta_lim))
cf4_si <- coef(m4, printSE = TRUE)
cf4_nat <- coef(m4)
si_ref <- extract_si_group(cf4_si$ref, cf4_nat$ref, "2PL")
si_foc <- extract_si_group(cf4_si$focal, cf4_nat$focal, "2PL")

set.seed(20261151)
mg_pats_list <- list()
mg_grps_list <- list()
mg_scores <- list(
  EAP = list(theta = list(), se = list()),
  MAP = list(theta = list(), se = list()),
  ML  = list(theta = list(), se = list()),
  WLE = list(theta = list(), se = list())
)
for (g in c("ref", "focal")) {
  gpats_all <- as.data.frame(v4$patterns[v4$pattern_groups == g, , drop = FALSE])
  colnames(gpats_all) <- v4$items
  # Ensure both all-0 and all-1 extreme patterns are included in each group
  ext_df <- as.data.frame(rbind(rep(0L, ncol(gpats_all)), rep(1L, ncol(gpats_all))))
  colnames(ext_df) <- v4$items
  pick_g <- sample(nrow(gpats_all), min(50L, nrow(gpats_all)))
  gpats <- unique(rbind(ext_df, gpats_all[pick_g, , drop = FALSE]))
  rownames(gpats) <- NULL
  mg_g <- extract.group(m4, g)
  sc_g <- score_all(mg_g, gpats, wide_quad)
  mg_pats_list[[g]] <- as.matrix(gpats)
  mg_grps_list[[g]] <- rep(g, nrow(gpats))
  for (meth in names(mg_scores)) {
    mg_scores[[meth]]$theta <- c(mg_scores[[meth]]$theta, sc_g[[meth]]$theta)
    mg_scores[[meth]]$se <- c(mg_scores[[meth]]$se, sc_g[[meth]]$se)
  }
}
c_mg <- list(
  meta = list(
    case = "twogroup_2pl",
    description = sprintf(
      "V13e: Two-group 2PL on V4 dataset (%d patterns across ref/focal, 81 pts on [-8, 8])",
      length(unlist(mg_grps_list))
    ),
    itemtype = "2PL",
    n_cat = 2L,
    source_reference = "v4_twogroup_2pl.json",
    TOL = 1e-10,
    quadpts = wide_quad$quadpts,
    theta_lim = wide_quad$theta_lim,
    grid_check = grid4,
    accelerate = "none",
    optimizer = "NR",
    converged = conv4,
    iterations = extract.mirt(m4, "iterations"),
    fscores_max_theta = 20,
    fscores_expected_info = FALSE
  ),
  items = v4$items,
  groups = c("ref", "focal"),
  anchors = v4$anchors,
  loglik = extract.mirt(m4, "logLik"),
  slope_intercept = list(ref = si_ref, focal = si_foc),
  score_patterns = unname(do.call(rbind, mg_pats_list)),
  score_groups = unname(unlist(mg_grps_list)),
  fscores = mg_scores
)

# ---- 6. Missing-data 2PL on existing V9 dataset (v9_missing_2pl.json) ---------------------------
v9 <- fromJSON(file.path(out_dir, "v9_missing_2pl.json"))
dat9 <- as.data.frame(v9$patterns[rep(seq_len(nrow(v9$patterns)), v9$counts), ])
colnames(dat9) <- v9$items
all9 <- as.data.frame(v9$patterns); colnames(all9) <- v9$items
n_na9 <- rowSums(is.na(all9))
n_obs9 <- ncol(all9) - n_na9
all_zero9 <- rowSums(all9 == 0, na.rm = TRUE) == n_obs9 & n_obs9 > 0L
all_one9 <- rowSums(all9 == 1, na.rm = TRUE) == n_obs9 & n_obs9 > 0L
last_obs9 <- !is.na(all9[, ncol(all9)])
set.seed(20261152)
# Include partially-missing all-0, partially-missing all-1 (with last column observed to avoid
# mirt 1.46.1's observed last-item NA behaviour; cause unconfirmed), and interior missing patterns with 1..5 NAs
pick9 <- sort(unique(c(
  which(all_zero9 & n_na9 > 0L & last_obs9)[1:min(3L, sum(all_zero9 & n_na9 > 0L & last_obs9))],
  which(all_one9 & n_na9 > 0L & last_obs9)[1:min(3L, sum(all_one9 & n_na9 > 0L & last_obs9))],
  sample(which(n_na9 > 0L & !all_zero9 & !all_one9), 70L),
  sample(which(n_na9 == 0L & !all_zero9 & !all_one9), 10L)
)))
sub9 <- all9[pick9, , drop = FALSE]
# Add one explicit partially-missing all-0 and all-1 row if none had missingness
sub9 <- unique(rbind(
  sub9,
  setNames(as.data.frame(matrix(c(0, NA, 0, 0, NA, 0, 0, 0, 0, NA, 0, 0), 1L)), v9$items),
  setNames(as.data.frame(matrix(c(1, 1, NA, 1, 1, NA, 1, 1, NA, 1, 1, 1), 1L)), v9$items)
))
rownames(sub9) <- NULL
c_miss <- build_single_case(
  "missing_2pl",
  sprintf("V13f: 2PL with missing responses on V9 dataset (%d unique patterns)", nrow(sub9)),
  dat9, "2PL", default_quad, n_cat = 2L, score_pats = sub9, source_ref = "v9_missing_2pl.json"
)

cases <- list(
  lsat7_2pl = c_2pl,
  science_grm = c_grm,
  science_gpcm = c_gpcm,
  v11_3pl = c_3pl,
  twogroup_2pl = c_mg,
  missing_2pl = c_miss
)

out <- list(
  meta = list(
    case = "v13_fscores",
    description = "V13: Ability scoring (EAP, MAP, ML, WLE) across 2PL, GRM, GPCM, 3PL, two-group, and missing data",
    r_version = R.version.string,
    mirt_version = mirt_version_string(), mirt_sha = mirt_commit_sha(),
    TOL = 1e-10,
    quadpts = default_quad$quadpts,
    theta_lim = default_quad$theta_lim,
    grid_check = default_quad$theta_lim,
    accelerate = "none",
    optimizer = "NR",
    converged = all(vapply(cases, function(x) isTRUE(x$meta$converged), logical(1))),
    fscores_settings = list(
      max_theta = 20,
      expected_info = FALSE,
      se_convention = "inverse sqrt of nlm Hessian at optimum (observed info for MAP/ML/WLE)",
      bounds_note = "theta_lim sets EAP quadrature and EAP starting values; MAP/ML/WLE optimize on [-max_theta, max_theta]"
    )
  ),
  cases = cases
)

path <- file.path(out_dir, "v13_fscores.json")
write_json(out, path, auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null")
for (nm in names(cases)) {
  x <- cases[[nm]]
  ml_th <- x$fscores$ML$theta
  n_inf <- sum(vapply(ml_th, function(v) identical(v, "Inf") || identical(v, "-Inf"), logical(1)))
  cat(sprintf(
    "  %-14s pats=%3d  loglik=%11.5f  iter=%4d  conv=%s  grid=[%g,%g]  ML_inf=%d\n",
    nm, nrow(x$score_patterns), x$loglik, x$meta$iterations, x$meta$converged,
    x$meta$grid_check[1], x$meta$grid_check[2], n_inf
  ))
}
cat("wrote", path, "\n")
