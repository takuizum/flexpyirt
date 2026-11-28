# V58 reference values (VALIDATION.md V58: quasi-Monte Carlo quadrature for a 4-factor
# confirmatory model, DESIGN.md §28, #164) from R mirt.
# Run from the repository root:  Rscript validation/r/v58_qmc_reference.R
# Writes tests/reference/v58_qmc.json. CI does not run R; the JSON is committed.
#
# The reference is mirt's ordinary EM on a fine product grid: 13 points per dimension on
# [-4.8, 4.8] (13^4 = 28,561 nodes, spacing 0.8; technical MAXQUAD raised from mirt's 20,000),
# accelerate = "none", TOL = 1e-6, SE.type = "Oakes". flexpyirt's fixed-node QMC is compared with
# it within the QMC error of tests/test_mirt_v58_qmc.py.
#
# mirt's method = "QMCEM" is NOT the reference: on this case it drifts to slopes about 1.6 times
# too large and factor correlations of 0.91-0.97 (the data were simulated with 0.3-0.5), with its
# own log-likelihood decreasing over the last cycles, at both its default 5,000 points and at
# 20,000, while mirt's EM on product grids (9^4 and 13^4) and flexpyirt agree with the truth. Its
# default-point result is stored under `mirt_qmcem` to document the finding (VALIDATION.md V58).
#
# Parameter mapping as in m4_mirt_reference.R: a1..aD, d; MEAN_i fixed 0, COV_ii fixed 1,
# COV_ji (i < j) -> cor_ij (free; equals the correlation since the variances are 1).
source(file.path("validation", "r", "setup.R"))  # mirt from GitHub; mirt_version / mirt_sha
suppressMessages({
  library(mirt)
  library(jsonlite)
})

out_dir <- file.path("tests", "reference")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

pack_patterns <- function(dat) {
  key <- apply(dat, 1, paste, collapse = ",")
  first <- !duplicated(key)
  patterns <- as.matrix(dat[first, , drop = FALSE])
  counts <- as.integer(table(factor(key, levels = key[first])))
  list(patterns = unname(patterns), counts = counts)
}

extract <- function(m, items, n_factors, se) {
  cf <- coef(m, printSE = se)
  slopes <- paste0("a", seq_len(n_factors))
  keep <- c(slopes, "d")
  params <- lapply(items, function(i) {
    x <- cf[[i]]
    out <- as.list(x["par", keep])
    if (se) {
      s <- x["SE", keep]
      names(s) <- paste0(keep, "_se")
      out <- c(out, as.list(s))
    }
    out
  })
  names(params) <- items
  gp <- cf$GroupPars
  latent <- list()
  for (i in seq_len(n_factors - 1L)) {
    for (j in seq(i + 1L, n_factors)) {
      k <- sprintf("COV_%d%d", j, i)
      latent[[sprintf("cor_%d%d", i, j)]] <- unname(gp["par", k])
      if (se) latent[[sprintf("cor_%d%d_se", i, j)]] <- unname(gp["SE", k])
    }
  }
  list(params = params, latent = latent)
}

# ---- V58a: 4-factor confirmatory 2PL, simple structure, correlated factors ----------------------
set.seed(20261128)
J <- 20L; N <- 2000L; D <- 4L
a4 <- matrix(0, nrow = J, ncol = D)
for (m in seq_len(D)) a4[(5L * (m - 1L) + 1L):(5L * m), m] <- runif(5L, 1.0, 2.0)
d4 <- matrix(seq(-1.2, 1.2, length.out = J))
sig4 <- matrix(c(
  1.0, 0.5, 0.4, 0.3,
  0.5, 1.0, 0.5, 0.4,
  0.4, 0.5, 1.0, 0.5,
  0.3, 0.4, 0.5, 1.0
), D, D, byrow = TRUE)
dat <- as.data.frame(simdata(a4, d4, N, itemtype = "2PL", sigma = sig4))
items <- sprintf("q%02d", seq_len(J))
colnames(dat) <- items
factors <- list(
  F1 = sprintf("q%02d", 1:5), F2 = sprintf("q%02d", 6:10),
  F3 = sprintf("q%02d", 11:15), F4 = sprintf("q%02d", 16:20)
)
spec <- mirt.model(paste(
  "F1 = 1-5", "F2 = 6-10", "F3 = 11-15", "F4 = 16-20", "COV = F1*F2*F3*F4", sep = "\n"
))
theta_lim <- c(-4.8, 4.8)
quadpts <- 13L

t0 <- proc.time()[3L]
m_em <- mirt(
  dat, spec,
  itemtype = "2PL",
  quadpts = quadpts,
  SE = TRUE,
  SE.type = "Oakes",
  TOL = 1e-6,
  accelerate = "none",
  technical = list(theta_lim = theta_lim, MAXQUAD = 30000L, NCYCLES = 5000L),
  verbose = FALSE
)
em_seconds <- unname(proc.time()[3L] - t0)
conv <- isTRUE(extract.mirt(m_em, "converged"))
grid_check <- range(m_em@Model$Theta)
stopifnot(conv, all(grid_check == theta_lim), nrow(m_em@Model$Theta) == quadpts^D)

t0 <- proc.time()[3L]
m_qmc <- mirt(dat, spec, itemtype = "2PL", method = "QMCEM", SE = FALSE, verbose = FALSE)
qmc_seconds <- unname(proc.time()[3L] - t0)

packed <- pack_patterns(dat)
cat(sprintf("EM %d nodes: loglik=%.6f iter=%d time=%.0fs; QMCEM %d points: loglik=%.6f time=%.0fs\n",
            nrow(m_em@Model$Theta), extract.mirt(m_em, "logLik"),
            extract.mirt(m_em, "iterations"), em_seconds, nrow(m_qmc@Internals$Theta),
            extract.mirt(m_qmc, "logLik"), qmc_seconds))

case <- list(
  meta = list(
    case = "qmc4d_2pl",
    description = "V58a: 4-factor confirmatory 2PL (5 items per factor, correlations 0.3-0.5)",
    itemtype = "2PL",
    n_factors = D,
    quadpts = quadpts,
    theta_lim = theta_lim,
    grid_check = grid_check,
    n_nodes = nrow(m_em@Model$Theta),
    TOL = 1e-6,
    accelerate = "none",
    converged = conv,
    iterations = extract.mirt(m_em, "iterations"),
    seconds = em_seconds
  ),
  items = items,
  factors = factors,
  patterns = packed$patterns,
  counts = packed$counts,
  loglik = extract.mirt(m_em, "logLik"),
  estimates = extract(m_em, items, D, se = TRUE),
  mirt_qmcem = c(
    list(
      n_points = nrow(m_qmc@Internals$Theta),
      loglik = extract.mirt(m_qmc, "logLik"),
      converged = isTRUE(extract.mirt(m_qmc, "converged")),
      seconds = qmc_seconds
    ),
    extract(m_qmc, items, D, se = FALSE)
  ),
  truth = list(a = a4, d = as.numeric(d4), sigma = sig4)
)

bundle <- list(
  meta = list(
    case = "v58_qmc",
    description = "V58: QMC quadrature, 4-factor confirmatory 2PL against mirt EM on a 13^4 grid",
    r_version = R.version.string,
    mirt_version = mirt_version_string(), mirt_sha = mirt_commit_sha(),
    TOL = 1e-6,
    accelerate = "none",
    theta_lim = theta_lim,
    grid_check = grid_check,
    converged = conv
  ),
  cases = list(qmc4d_2pl = case)
)

path <- file.path(out_dir, "v58_qmc.json")
write_json(bundle, path, auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null")
cat("wrote", path, "\n")
