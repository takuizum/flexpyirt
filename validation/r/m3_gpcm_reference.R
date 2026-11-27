# M3 reference values (VALIDATION.md V8: GPCM and PCM) from R mirt.
# Run from the repository root:  Rscript validation/r/m3_gpcm_reference.R
# Writes tests/reference/v8_science_gpcm.json, tests/reference/v8_sim_gpcm.json, and
# tests/reference/v8_sim_pcm.json. CI does not run R; the JSON files are committed.
#
# As in m1/m2/m3_reference.R: optimizer = "NR", accelerate = "none", TOL = 1e-10
# (theta_lim inside `technical`; the grid actually used is recorded in meta.grid_check).
#
# Parameterization mapping from mirt to flexpyirt:
#   mirt's itemtype = "gpcm" (and itemtype = "Rasch" on polytomous data, i.e. PCM) returns columns
#   a1, ak0..ak{K-1}, d0, d1..d{K-1}, with ak_k = k and d0 = 0 fixed. flexpyirt omits the fixed
#   ak_k and d0, keeping a (= a1) and d1..d{K-1}. For itemtype = "Rasch" on polytomous data, mirt
#   fixes a1 = 1 and estimates COV_11 (latent variance) with MEAN_1 = 0 fixed — identical to
#   flexpyirt's PCM default.
#
# Grid: v8_sim_pcm has a free latent variance (generating var = 1.5), so it uses 81 points on
# [-8, 8] (same 0.2 spacing) to make grid truncation negligible (DESIGN.md §14.9, VALIDATION.md).
source(file.path("validation", "r", "setup.R"))  # mirt from GitHub; mirt_version / mirt_sha
suppressMessages({
  library(mirt)
  library(jsonlite)
})

out_dir <- file.path("tests", "reference")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
common <- list(TOL = 1e-10, accelerate = "none", optimizer = "NR", verbose = FALSE)
default_quad <- list(quadpts = 61, theta_lim = c(-6, 6))
wide_quad <- list(quadpts = 81, theta_lim = c(-8, 8))
tech <- function(quad) list(NCYCLES = 50000, theta_lim = quad$theta_lim)

clean_item <- function(x) {
  nm <- colnames(x)
  # Verify mirt's fixed gpcm scoring coefficients ak_k = k and d0 = 0.
  ak_cols <- nm[grepl("^ak[0-9]+$", nm)]
  stopifnot(length(ak_cols) >= 2)
  stopifnot(all(x["par", ak_cols] == seq(0, length(ak_cols) - 1)))
  stopifnot("d0" %in% nm && x["par", "d0"] == 0)

  keep <- c("a1", nm[grepl("^d[1-9][0-9]*$", nm)])
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

write_case <- function(case, description, dat, itemtype, quad, n_cat, extra = list()) {
  fit_fun <- function(se) {
    do.call(mirt, c(list(dat, 1, itemtype = itemtype, SE = TRUE, SE.type = se,
                         quadpts = quad$quadpts, technical = tech(quad)), common))
  }
  m_oakes <- fit_fun("Oakes")
  m_rich <- fit_fun("Richardson")
  key <- apply(dat, 1, paste, collapse = ",")
  first <- !duplicated(key)
  patterns <- as.matrix(dat[first, , drop = FALSE])
  sc <- fscores(m_oakes, method = "EAP", response.pattern = patterns,
                quadpts = quad$quadpts, theta_lim = quad$theta_lim)
  res <- c(list(
    meta = list(case = case, description = description, itemtype = itemtype, n_cat = n_cat,
                r_version = R.version.string, mirt_version = mirt_version_string(), mirt_sha = mirt_commit_sha(),
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
    eap = list(theta = unname(sc[, "F1"]), se = unname(sc[, "SE_F1"]))
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
      " var =", sprintf("%.5f", res$estimates$all$latent$var),
      " grid =", paste(res$meta$grid_check, collapse = ".."), "\n")
}

# ---- V8a: GPCM on mirt::Science (recoded 1..4 -> 0..3) -----------------------------------------
sci <- as.data.frame(as.matrix(Science) - 1L)
write_case("v8_science_gpcm",
           "V8: GPCM, mirt::Science recoded to 0..3",
           sci, "gpcm", default_quad, n_cat = 4L)

# ---- V8b: GPCM, simulated J = 10, K = 4, N = 3000 ----------------------------------------------
set.seed(20261126)
J <- 10; K <- 4; N <- 3000
a_gpcm <- rlnorm(J, 0.0, 0.3)
# Step difficulties b1, b2, b3 (with item 10 having reversed steps b2 < b1 to exercise unordered d)
b_gpcm <- cbind(runif(J, -1.6, -0.4), runif(J, -0.4, 0.4), runif(J, 0.4, 1.6))
b_gpcm[J, ] <- c(0.3, -0.5, 0.9)
d_gpcm <- cbind(0, -a_gpcm * t(apply(b_gpcm, 1, cumsum)))
dat_gpcm <- as.data.frame(simdata(matrix(a_gpcm), d_gpcm, N, itemtype = "gpcm"))
colnames(dat_gpcm) <- sprintf("q%02d", 1:J)
write_case("v8_sim_gpcm",
           "V8: GPCM, simulated J=10, K=4, N=3000 (seed 20261126)",
           dat_gpcm, "gpcm", default_quad, n_cat = K,
           extra = list(truth = list(a = a_gpcm, d = unname(d_gpcm[, -1]), b = unname(b_gpcm))))

# ---- V8c: PCM, simulated J = 10, K = 4, N = 3000, latent variance = 1.5 ------------------------
set.seed(20261127)
J <- 10; K <- 4; N <- 3000
a_pcm <- rep(1.0, J)
b_pcm <- cbind(runif(J, -1.6, -0.4), runif(J, -0.4, 0.4), runif(J, 0.4, 1.6))
b_pcm[J, ] <- c(0.4, -0.4, 1.0)
d_pcm <- cbind(0, -t(apply(b_pcm, 1, cumsum)))
dat_pcm <- as.data.frame(simdata(matrix(a_pcm), d_pcm, N, itemtype = "gpcm",
                                 mu = 0, sigma = matrix(1.5)))
colnames(dat_pcm) <- sprintf("q%02d", 1:J)
write_case("v8_sim_pcm",
           "V8: PCM (mirt itemtype='Rasch'), simulated J=10, K=4, N=3000, latent var 1.5 (seed 20261127)",
           dat_pcm, "Rasch", wide_quad, n_cat = K,
           extra = list(truth = list(a = a_pcm, d = unname(d_pcm[, -1]), b = unname(b_pcm),
                                     latent_var = 1.5)))
