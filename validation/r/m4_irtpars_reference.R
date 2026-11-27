# M4 reference values (VALIDATION.md V12: delta-method SEs for display IRT parameterization)
# from R mirt.
# Run from the repository root:  Rscript validation/r/m4_irtpars_reference.R
# Writes tests/reference/v12_irtpars.json. CI does not run R; the JSON is committed.
#
# Settings match m1/m2/m3 references: optimizer = "NR", accelerate = "none", TOL = 1e-10,
# theta_lim inside technical = list(...), convergence and grid_check asserted for every case.
#
# Parameterization mapping (coef(m, IRTpars = TRUE, printSE = TRUE)):
#   - 2PL / Rasch:  b = -d / a  (Rasch has a = 1 fixed, so b = -d and SE(b) = SE(d)).
#   - GRM (graded): b_k = -d_k / a  for k = 1..K-1.
#   - GPCM / PCM:   b_k = -(d_k - d_{k-1}) / a  with d_0 = 0  for k = 1..K-1.
#                   This matches flexpyirt's step-difficulty definition identically.
#
# Note on mirt 1.46.1 bug in mirt:::mirt2traditional for GPCM b1_se (VALIDATION.md V12):
#   In mirt 1.46.1, cls == "gpcm" overwrites delta_index[[nfact + 1L]] with c(which.a, ncat+nfact+2L)
#   (the positions of a1 and d1 in opar, which includes d0). Inside fns, ret drops d0 so b1 is at
#   position ncat+nfact+1L and b2 is at ncat+nfact+2L; ret[index[2]] therefore returns b2 = -(d2-d1)/a
#   differentiated w.r.t. (a, d1) only (holding d2 fixed). Consequently:
#     - Point estimates b1..b{K-1} are exact.
#     - SEs for b2..b{K-1} are exact delta-method SEs (verified against vcov(m) below to < 1e-9).
#     - SE for b1 in GPCM (free a) is wrong in mirt 1.46.1; for PCM (a = 1 fixed), the bug cancels
#       because |db2/dd1| = |1| = |-1| = |db1/dd1|.
#   For GPCM we store both mirt's raw IRTpars output and exact_delta_se computed directly from
#   vcov(m) in R, and assert the exact bug formula matches mirt's raw b1_se to < 1e-9.
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

extract_si_item <- function(x, itemtype) {
  nm <- colnames(x)
  if (itemtype == "gpcm" || (itemtype == "Rasch" && "d0" %in% nm)) {
    ak_cols <- nm[grepl("^ak[0-9]+$", nm)]
    stopifnot(length(ak_cols) >= 2L)
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

extract_irt_item <- function(x, itemtype) {
  nm <- colnames(x)
  if (itemtype %in% c("2PL", "Rasch") && "b" %in% nm) {
    keep <- c("a", "b")
  } else {
    keep <- c("a", nm[grepl("^b[1-9][0-9]*$", nm)])
  }
  par <- x["par", keep]
  se <- x["SE", keep]
  names(se) <- paste0(keep, "_se")
  c(as.list(par), as.list(se))
}

extract_group <- function(cf_si, cf_irt, itemtype) {
  items <- setdiff(names(cf_si), "GroupPars")
  si_params <- lapply(items, function(i) extract_si_item(cf_si[[i]], itemtype))
  names(si_params) <- items
  irt_params <- lapply(items, function(i) extract_irt_item(cf_irt[[i]], itemtype))
  names(irt_params) <- items
  gp <- cf_si$GroupPars
  list(
    si = list(
      params = si_params,
      latent = list(mean = gp["par", "MEAN_1"], var = gp["par", "COV_11"])
    ),
    irt = irt_params
  )
}

pack_patterns <- function(dat, grp = NULL) {
  if (is.null(grp)) grp <- rep("all", nrow(dat))
  key <- apply(cbind(grp, dat), 1, paste, collapse = ",")
  first <- !duplicated(key)
  patterns <- as.matrix(dat[first, , drop = FALSE])
  list(
    patterns = unname(patterns),
    pattern_groups = unname(as.character(grp[first])),
    counts = as.integer(table(key)[key[first]])
  )
}

run_single <- function(case, description, dat, itemtype, quad, n_cat) {
  m <- do.call(
    mirt,
    c(list(dat, 1L, itemtype = itemtype, SE = TRUE, SE.type = "Oakes",
           quadpts = quad$quadpts, technical = tech(quad)), common)
  )
  conv <- extract.mirt(m, "converged")
  grid_check <- range(m@Model$Theta)
  stopifnot(isTRUE(conv), all(grid_check == quad$theta_lim))

  cf_si <- coef(m, printSE = TRUE)
  cf_irt <- coef(m, IRTpars = TRUE, printSE = TRUE)
  g_out <- extract_group(cf_si, cf_irt, itemtype)
  pat <- pack_patterns(dat)

  res <- list(
    meta = list(
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
      iterations = extract.mirt(m, "iterations")
    ),
    items = colnames(dat),
    groups = I("all"),
    patterns = pat$patterns,
    pattern_groups = pat$pattern_groups,
    counts = pat$counts,
    loglik = extract.mirt(m, "logLik"),
    slope_intercept = list(all = g_out$si),
    irtpars = list(all = g_out$irt)
  )
  list(res = res, model = m)
}

# ---- 1. 2PL on LSAT7 ----------------------------------------------------------------------------
lsat7 <- as.data.frame(expand.table(LSAT7))
c_2pl <- run_single(
  "lsat7_2pl", "V12a: 2PL on mirt::LSAT7", lsat7, "2PL", default_quad, n_cat = 2L
)

# ---- 2. Rasch on LSAT7 --------------------------------------------------------------------------
c_rasch <- run_single(
  "lsat7_rasch", "V12b: Rasch on mirt::LSAT7 (a=1 fixed, b_se = d_se)",
  lsat7, "Rasch", default_quad, n_cat = 2L
)

# ---- 3. GRM on Science (0..3) -------------------------------------------------------------------
sci <- as.data.frame(as.matrix(Science) - 1L)
c_grm <- run_single(
  "science_grm", "V12c: GRM (graded) on mirt::Science recoded 0..3",
  sci, "graded", default_quad, n_cat = 4L
)

# ---- 4. GPCM on Science (0..3) ------------------------------------------------------------------
c_gpcm <- run_single(
  "science_gpcm", "V12d: GPCM on mirt::Science recoded 0..3",
  sci, "gpcm", default_quad, n_cat = 4L
)
# Cross-check exact delta-method SEs directly from vcov(m) and verify mirt 1.46.1's b1_se bug.
m_gpcm <- c_gpcm$model
V_gpcm <- vcov(m_gpcm)
cf_gpcm_si <- coef(m_gpcm, printSE = TRUE)
cf_gpcm_irt <- coef(m_gpcm, IRTpars = TRUE, printSE = TRUE)
exact_gpcm_se <- list()
for (idx in seq_along(colnames(sci))) {
  item <- colnames(sci)[idx]
  par <- cf_gpcm_si[[item]]["par", ]
  a <- par[["a1"]]; d1 <- par[["d1"]]; d2 <- par[["d2"]]; d3 <- par[["d3"]]
  cols <- ((idx - 1L) * 4L + 1L):(idx * 4L)
  subV <- V_gpcm[cols, cols]
  J_mat <- rbind(
    c(d1 / a^2, -1 / a, 0, 0),
    c((d2 - d1) / a^2, 1 / a, -1 / a, 0),
    c((d3 - d2) / a^2, 0, 1 / a, -1 / a)
  )
  se_exact <- sqrt(diag(J_mat %*% subV %*% t(J_mat)))
  # Verify b2_se and b3_se match mirt's IRTpars SE to < 1e-9.
  stopifnot(abs(se_exact[2] - cf_gpcm_irt[[item]]["SE", "b2"]) < 1e-9)
  stopifnot(abs(se_exact[3] - cf_gpcm_irt[[item]]["SE", "b3"]) < 1e-9)
  # Verify mirt's raw b1_se equals the exact off-by-one bug formula: grad of b2 wrt (a, d1) only.
  g_bug <- c((d2 - d1) / a^2, 1 / a)
  se_bug <- sqrt(as.numeric(t(g_bug) %*% subV[1:2, 1:2] %*% g_bug))
  stopifnot(abs(se_bug - cf_gpcm_irt[[item]]["SE", "b1"]) < 1e-9)
  exact_gpcm_se[[item]] <- list(b1_se = se_exact[1], b2_se = se_exact[2], b3_se = se_exact[3])
}
c_gpcm$res$exact_delta_se <- list(all = exact_gpcm_se)
c_gpcm$res$meta$mirt_gpcm_b1_se_bug <- paste(
  "mirt 1.46.1 mirt:::mirt2traditional off-by-one bug computes b1_se as delta SE of b2 wrt (a, d1)",
  "only; verified to < 1e-9. exact_delta_se records exact delta-method SEs from vcov(m)."
)

# ---- 5. PCM on Science (0..3) -------------------------------------------------------------------
c_pcm <- run_single(
  "science_pcm", "V12e: PCM (itemtype='Rasch' polytomous) on mirt::Science recoded 0..3",
  sci, "Rasch", wide_quad, n_cat = 4L
)

# ---- 6. Multiple-group 2PL with anchor items ----------------------------------------------------
set.seed(20261140)
J_mg <- 8L; N_g <- 1000L
a_mg <- rlnorm(J_mg, 0.2, 0.25)
d_ref <- runif(J_mg, -1.2, 1.2)
d_foc <- d_ref
d_foc[5:J_mg] <- d_foc[5:J_mg] + c(0.4, -0.5, 0.3, -0.4)
dat_ref <- simdata(matrix(a_mg), matrix(d_ref), N_g, itemtype = "2PL")
dat_foc <- simdata(matrix(a_mg), matrix(d_foc), N_g, itemtype = "2PL",
                   mu = 0.5, sigma = matrix(1.2))
dat_mg <- as.data.frame(rbind(dat_ref, dat_foc))
colnames(dat_mg) <- sprintf("q%02d", 1:J_mg)
grp_mg <- factor(rep(c("ref", "focal"), each = N_g), levels = c("ref", "focal"))
anchors <- sprintf("q%02d", 1:4)

m_mg <- do.call(
  multipleGroup,
  c(list(dat_mg, 1L, group = grp_mg, itemtype = "2PL",
         invariance = c(anchors, "free_means", "free_var"),
         SE = TRUE, SE.type = "Oakes",
         quadpts = wide_quad$quadpts, technical = tech(wide_quad)), common)
)
conv_mg <- extract.mirt(m_mg, "converged")
grid_mg <- range(m_mg@Model$Theta)
stopifnot(isTRUE(conv_mg), all(grid_mg == wide_quad$theta_lim))
cf_mg_si <- coef(m_mg, printSE = TRUE)
cf_mg_irt <- coef(m_mg, IRTpars = TRUE, printSE = TRUE)
g_ref <- extract_group(cf_mg_si$ref, cf_mg_irt$ref, "2PL")
g_foc <- extract_group(cf_mg_si$focal, cf_mg_irt$focal, "2PL")
pat_mg <- pack_patterns(dat_mg, grp_mg)

c_mg_res <- list(
  meta = list(
    case = "mg_2pl",
    description = "V12f: Two-group 2PL with 4 anchor items and focal group free (81 pts on [-8, 8])",
    itemtype = "2PL",
    n_cat = 2L,
    TOL = 1e-10,
    quadpts = wide_quad$quadpts,
    theta_lim = wide_quad$theta_lim,
    grid_check = grid_mg,
    accelerate = "none",
    optimizer = "NR",
    converged = conv_mg,
    iterations = extract.mirt(m_mg, "iterations")
  ),
  items = colnames(dat_mg),
  groups = c("ref", "focal"),
  anchors = anchors,
  patterns = pat_mg$patterns,
  pattern_groups = pat_mg$pattern_groups,
  counts = pat_mg$counts,
  loglik = extract.mirt(m_mg, "logLik"),
  slope_intercept = list(ref = g_ref$si, focal = g_foc$si),
  irtpars = list(ref = g_ref$irt, focal = g_foc$irt)
)

cases <- list(
  lsat7_2pl = c_2pl$res,
  lsat7_rasch = c_rasch$res,
  science_grm = c_grm$res,
  science_gpcm = c_gpcm$res,
  science_pcm = c_pcm$res,
  mg_2pl = c_mg_res
)

out <- list(
  meta = list(
    case = "v12_irtpars",
    description = "V12: Delta-method SEs for display IRT parameterization (2PL, Rasch, GRM, GPCM, PCM, 2-group 2PL)",
    r_version = R.version.string,
    mirt_version = mirt_version_string(), mirt_sha = mirt_commit_sha(),
    TOL = 1e-10,
    quadpts = default_quad$quadpts,
    theta_lim = default_quad$theta_lim,
    grid_check = default_quad$theta_lim,
    accelerate = "none",
    optimizer = "NR",
    converged = all(vapply(cases, function(x) isTRUE(x$meta$converged), logical(1)))
  ),
  cases = cases
)

path <- file.path(out_dir, "v12_irtpars.json")
write_json(out, path, auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null")
for (nm in names(cases)) {
  x <- cases[[nm]]
  cat(sprintf("  %-14s loglik=%11.5f  iter=%4d  conv=%s  grid=[%g, %g]\n",
              nm, x$loglik, x$meta$iterations, x$meta$converged,
              x$meta$grid_check[1], x$meta$grid_check[2]))
}
cat("wrote", path, "\n")
