# M4 reference values (VALIDATION.md V14: confirmatory 2-3 dimensional MIRT) from R mirt.
# Run from the repository root:  Rscript validation/r/m4_mirt_reference.R
# Writes tests/reference/v14_mirt2d.json. CI does not run R; the JSON is committed.
#
# Settings match m1/m2/m3/m4 references: optimizer = "NR" (2D cases; the 3D case uses "BFGS" with
# control = list(reltol = 1e-13)), accelerate = "none", TOL = 1e-10,
# SE.type = "Oakes", theta_lim inside technical = list(...), quadpts matching mirt's defaults
# (31 per dimension for 2D -> 961 nodes; 15 per dimension for 3D -> 3375 nodes).
# Convergence and grid_check are asserted for every case.
#
# Parameter mapping:
#   - Item slopes: a1..aD (same names as flexpyirt; unloaded slopes fixed at 0).
#   - Item intercepts: d (2PL) or d1..d{K-1} (GRM, categories 0..K-1).
#   - Latent parameters: MEAN_1..MEAN_D -> mean_1..mean_D (fixed at 0),
#     COV_11..COV_DD -> var_1..var_D (fixed at 1),
#     COV_21 / COV_31 / COV_32 -> cor_12 / cor_13 / cor_23 (free; since variances are fixed at 1,
#     covariance equals correlation identically, and SEs are on the same natural scale).
suppressMessages({
  library(mirt)
  library(jsonlite)
})

out_dir <- file.path("tests", "reference")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

extract_item <- function(x, itemtype, n_factors) {
  nm <- colnames(x)
  slopes <- paste0("a", seq_len(n_factors))
  if (itemtype == "graded") {
    intercepts <- nm[grepl("^d[1-9][0-9]*$", nm)]
  } else {
    intercepts <- "d"
  }
  keep <- c(slopes, intercepts)
  par <- x["par", keep]
  se <- x["SE", keep]
  names(se) <- paste0(keep, "_se")
  c(as.list(par), as.list(se))
}

extract_latent <- function(gp, n_factors) {
  out <- list()
  for (i in seq_len(n_factors)) {
    k <- sprintf("MEAN_%d", i)
    out[[sprintf("mean_%d", i)]] <- unname(gp["par", k])
  }
  for (i in seq_len(n_factors)) {
    k <- sprintf("COV_%d%d", i, i)
    out[[sprintf("var_%d", i)]] <- unname(gp["par", k])
  }
  for (i in seq_len(n_factors - 1L)) {
    for (j in seq(i + 1L, n_factors)) {
      k <- sprintf("COV_%d%d", j, i)
      out[[sprintf("cor_%d%d", i, j)]] <- unname(gp["par", k])
    }
  }
  for (i in seq_len(n_factors)) {
    k <- sprintf("MEAN_%d", i)
    out[[sprintf("mean_%d_se", i)]] <- unname(gp["SE", k])
  }
  for (i in seq_len(n_factors)) {
    k <- sprintf("COV_%d%d", i, i)
    out[[sprintf("var_%d_se", i)]] <- unname(gp["SE", k])
  }
  for (i in seq_len(n_factors - 1L)) {
    for (j in seq(i + 1L, n_factors)) {
      k <- sprintf("COV_%d%d", j, i)
      out[[sprintf("cor_%d%d_se", i, j)]] <- unname(gp["SE", k])
    }
  }
  out
}

pack_patterns <- function(dat) {
  key <- apply(dat, 1, paste, collapse = ",")
  first <- !duplicated(key)
  patterns <- as.matrix(dat[first, , drop = FALSE])
  list(
    patterns = unname(patterns),
    counts = as.integer(table(key)[key[first]])
  )
}

run_case <- function(case, description, dat, model_str, factors, itemtype, quadpts, theta_lim,
                     n_cat, truth, optimizer = "NR", control = NULL) {
  n_factors <- length(factors)
  mod <- mirt.model(model_str)
  tech <- list(NCYCLES = 20000L, theta_lim = theta_lim)
  t0 <- proc.time()[3L]
  args <- list(
    dat, mod,
    itemtype = itemtype,
    SE = TRUE,
    SE.type = "Oakes",
    quadpts = quadpts,
    TOL = 1e-10,
    accelerate = "none",
    optimizer = optimizer,
    technical = tech,
    verbose = FALSE
  )
  if (!is.null(control)) args$control <- control
  m <- do.call(mirt, args)
  elapsed <- proc.time()[3L] - t0

  conv <- isTRUE(extract.mirt(m, "converged"))
  grid_check <- range(m@Model$Theta)
  n_nodes_used <- nrow(m@Model$Theta)
  stopifnot(
    conv,
    all(grid_check == theta_lim),
    n_nodes_used == quadpts^n_factors
  )

  cf <- coef(m, printSE = TRUE)
  items <- colnames(dat)
  params <- lapply(items, function(i) extract_item(cf[[i]], itemtype, n_factors))
  names(params) <- items
  latent <- extract_latent(cf$GroupPars, n_factors)

  packed <- pack_patterns(dat)
  # Evaluate EAP on up to 200 unique patterns (all patterns if <= 200) to keep JSON compact.
  n_pat <- nrow(packed$patterns)
  eap_idx <- if (n_pat <= 200L) seq_len(n_pat) else seq(1L, n_pat, length.out = 200L)
  eap_idx <- unique(as.integer(round(eap_idx)))
  eap_pats <- packed$patterns[eap_idx, , drop = FALSE]
  sc <- fscores(
    m,
    method = "EAP",
    response.pattern = eap_pats,
    quadpts = quadpts,
    theta_lim = theta_lim
  )
  fnames <- names(factors)
  eap <- list(
    pattern_indices = eap_idx - 1L,  # 0-based indices into `patterns`
    theta = setNames(lapply(fnames, function(f) unname(sc[, f])), fnames),
    se = setNames(lapply(fnames, function(f) unname(sc[, paste0("SE_", f)])), fnames)
  )

  cat(sprintf(
    "%-20s loglik=%.6f iter=%d Q=%d n_pat=%d time=%.1fs cors=%s\n",
    case,
    extract.mirt(m, "logLik"),
    extract.mirt(m, "iterations"),
    n_nodes_used,
    n_pat,
    elapsed,
    paste(
      sprintf("%s=%.4f(SE=%.4f)",
              grep("^cor_[0-9]+$", names(latent), value = TRUE),
              unlist(latent[grep("^cor_[0-9]+$", names(latent), value = TRUE)]),
              unlist(latent[grep("^cor_[0-9]+_se$", names(latent), value = TRUE)])),
      collapse = ", "
    )
  ))

  list(
    meta = list(
      case = case,
      description = description,
      itemtype = itemtype,
      n_factors = n_factors,
      n_cat = n_cat,
      quadpts = quadpts,
      theta_lim = theta_lim,
      grid_check = grid_check,
      n_nodes = n_nodes_used,
      TOL = 1e-10,
      accelerate = "none",
      optimizer = optimizer,
      converged = conv,
      iterations = extract.mirt(m, "iterations")
    ),
    items = items,
    factors = factors,
    patterns = packed$patterns,
    counts = packed$counts,
    loglik = extract.mirt(m, "logLik"),
    estimates = list(params = params, latent = latent),
    eap = eap,
    truth = truth
  )
}

cases <- list()

# ---- V14a: 2D confirmatory 2PL, simple structure (F1 = 1-6, F2 = 7-12, COV = F1*F2) ------------
set.seed(20261130)
J <- 12L; N <- 2000L
a_s <- matrix(0, nrow = J, ncol = 2L)
a_s[1:6, 1] <- c(1.2, 1.5, 0.9, 1.3, 1.1, 1.4)
a_s[7:12, 2] <- c(1.1, 1.6, 1.0, 1.3, 0.9, 1.5)
d_s <- matrix(seq(-1.0, 1.0, length.out = J))
rho_s <- 0.5
sig_s <- matrix(c(1, rho_s, rho_s, 1), 2L, 2L)
dat_s <- as.data.frame(simdata(a_s, d_s, N, itemtype = "2PL", sigma = sig_s))
colnames(dat_s) <- sprintf("q%02d", seq_len(J))
cases$mirt2d_2pl_simple <- run_case(
  case = "mirt2d_2pl_simple",
  description = "V14a: 2D confirmatory 2PL, simple structure (F1=q01-q06, F2=q07-q12, rho=0.5)",
  dat = dat_s,
  model_str = "F1 = 1-6\nF2 = 7-12\nCOV = F1*F2",
  factors = list(F1 = sprintf("q%02d", 1:6), F2 = sprintf("q%02d", 7:12)),
  itemtype = "2PL",
  quadpts = 31L,
  theta_lim = c(-6, 6),
  n_cat = 2L,
  truth = list(a = a_s, d = as.numeric(d_s), cor_12 = rho_s)
)

# ---- V14b: 2D confirmatory 2PL with cross-loading item (q07 loads on F1 and F2) ----------------
set.seed(20261131)
J <- 12L; N <- 2500L
a_c <- matrix(0, nrow = J, ncol = 2L)
a_c[1:6, 1] <- c(1.3, 1.1, 1.5, 0.9, 1.2, 1.4)
a_c[7, ] <- c(0.8, 0.9)
a_c[8:12, 2] <- c(1.2, 1.5, 1.0, 1.3, 1.1)
d_c <- matrix(seq(-0.9, 0.9, length.out = J))
rho_c <- 0.45
sig_c <- matrix(c(1, rho_c, rho_c, 1), 2L, 2L)
dat_c <- as.data.frame(simdata(a_c, d_c, N, itemtype = "2PL", sigma = sig_c))
colnames(dat_c) <- sprintf("q%02d", seq_len(J))
cases$mirt2d_2pl_cross <- run_case(
  case = "mirt2d_2pl_cross",
  description = "V14b: 2D confirmatory 2PL with cross-loading q07 (F1=q01-q07, F2=q07-q12, rho=0.45)",
  dat = dat_c,
  model_str = "F1 = 1-7\nF2 = 7-12\nCOV = F1*F2",
  factors = list(F1 = sprintf("q%02d", 1:7), F2 = sprintf("q%02d", 7:12)),
  itemtype = "2PL",
  quadpts = 31L,
  theta_lim = c(-6, 6),
  n_cat = 2L,
  truth = list(a = a_c, d = as.numeric(d_c), cor_12 = rho_c)
)

# ---- V14c: 2D confirmatory GRM (K=4, categories 0..3, F1=1-5, F2=6-10, COV=F1*F2) --------------
set.seed(20261132)
J <- 10L; N <- 2000L; K <- 4L
a_g <- matrix(0, nrow = J, ncol = 2L)
a_g[1:5, 1] <- c(1.3, 1.1, 1.5, 1.0, 1.2)
a_g[6:10, 2] <- c(1.2, 1.4, 0.9, 1.3, 1.1)
d_g <- t(sapply(seq_len(J), function(j) c(1.2, 0.0, -1.2) + (j - 5.5) * 0.12))
rho_g <- 0.55
sig_g <- matrix(c(1, rho_g, rho_g, 1), 2L, 2L)
dat_g <- as.data.frame(simdata(a_g, d_g, N, itemtype = "graded", sigma = sig_g))
colnames(dat_g) <- sprintf("g%02d", seq_len(J))
stopifnot(all(apply(dat_g, 2, function(x) length(unique(x)) == K)))
cases$mirt2d_grm <- run_case(
  case = "mirt2d_grm",
  description = "V14c: 2D confirmatory GRM (K=4, F1=g01-g05, F2=g06-g10, rho=0.55)",
  dat = dat_g,
  model_str = "F1 = 1-5\nF2 = 6-10\nCOV = F1*F2",
  factors = list(F1 = sprintf("g%02d", 1:5), F2 = sprintf("g%02d", 6:10)),
  itemtype = "graded",
  quadpts = 31L,
  theta_lim = c(-6, 6),
  n_cat = K,
  truth = list(a = a_g, d = d_g, cor_12 = rho_g)
)

# ---- V14d: 3D confirmatory 2PL (F1=1-3, F2=4-6, F3=7-9, all COV free, 15^3=3375 nodes) ---------
set.seed(20261133)
J <- 9L; N <- 2000L
a_3 <- matrix(0, nrow = J, ncol = 3L)
a_3[1:3, 1] <- c(1.3, 1.1, 1.5)
a_3[4:6, 2] <- c(1.2, 1.4, 1.0)
a_3[7:9, 3] <- c(1.1, 1.5, 1.3)
d_3 <- matrix(seq(-0.8, 0.8, length.out = J))
sig_3 <- matrix(c(
  1.0,  0.45, 0.25,
  0.45, 1.0,  0.50,
  0.25, 0.50, 1.0
), 3L, 3L, byrow = TRUE)
dat_3 <- as.data.frame(simdata(a_3, d_3, N, itemtype = "2PL", sigma = sig_3))
colnames(dat_3) <- sprintf("t%02d", seq_len(J))
cases$mirt3d_2pl <- run_case(
  case = "mirt3d_2pl",
  description = "V14d: 3D confirmatory 2PL (F1=t01-t03, F2=t04-t06, F3=t07-t09, 15^3 nodes)",
  dat = dat_3,
  model_str = "F1 = 1-3\nF2 = 4-6\nF3 = 7-9\nCOV = F1*F2, F1*F3, F2*F3",
  factors = list(
    F1 = sprintf("t%02d", 1:3),
    F2 = sprintf("t%02d", 4:6),
    F3 = sprintf("t%02d", 7:9)
  ),
  itemtype = "2PL",
  quadpts = 15L,
  theta_lim = c(-6, 6),
  n_cat = 2L,
  truth = list(a = a_3, d = as.numeric(d_3), cor_12 = 0.45, cor_13 = 0.25, cor_23 = 0.50),
  optimizer = "BFGS",
  control = list(reltol = 1e-13, maxit = 500L)
)

bundle <- list(
  meta = list(
    case = "v14_mirt2d",
    description = "V14: Confirmatory 2-3 dimensional MIRT (2PL simple, 2PL cross-loading, GRM, 3D 2PL)",
    r_version = R.version.string,
    mirt_version = as.character(packageVersion("mirt")),
    TOL = 1e-10,
    accelerate = "none",
    optimizer = "per case (NR for 2D, BFGS reltol 1e-13 for 3D)",
    theta_lim = c(-6, 6),
    grid_check = c(-6, 6),
    converged = all(vapply(cases, function(c) isTRUE(c$meta$converged), logical(1)))
  ),
  cases = cases
)
stopifnot(isTRUE(bundle$meta$converged))

path <- file.path(out_dir, "v14_mirt2d.json")
write_json(bundle, path, auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null")
cat("wrote", path, "\n")
