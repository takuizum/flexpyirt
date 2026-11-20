# M1 reference values (VALIDATION.md V1, V2, V10) from R mirt.
# Run from the repository root:  Rscript validation/r/m1_reference.R
# Writes tests/reference/<case>.json. CI does not run R; the JSON files are committed.
#
# optimizer = "NR": with mirt's default M-step optimizer (nlminb for Rasch, BFGS for 2PL) the
# M-step iteration cap shrinks as the log-likelihood stabilizes, the M-step stops moving, and the
# TOL test passes before EM has converged (Rasch on LSAT7: log-likelihood still rising ~2e-9 per
# cycle, latent variance off by ~2e-5, independent of TOL). Newton-Raphson converges properly.
suppressMessages({
  library(mirt)
  library(jsonlite)
})

out_dir <- file.path("tests", "reference")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

fit_mirt <- function(dat, itemtype, se_type) {
  mirt(dat, 1, itemtype = itemtype, SE = TRUE, SE.type = se_type,
       TOL = 1e-10, accelerate = "none", optimizer = "NR", quadpts = 61,
       technical = list(NCYCLES = 50000, theta_lim = c(-6, 6)), verbose = FALSE)
}

extract <- function(m) {
  cf <- coef(m, printSE = TRUE)
  items <- setdiff(names(cf), "GroupPars")
  params <- lapply(items, function(i) {
    x <- cf[[i]]
    list(a = x["par", "a1"], d = x["par", "d"],
         a_se = unname(x["SE", "a1"]), d_se = unname(x["SE", "d"]))
  })
  names(params) <- items
  gp <- cf$GroupPars
  list(params = params,
       latent = list(mean = gp["par", "MEAN_1"], var = gp["par", "COV_11"],
                     mean_se = unname(gp["SE", "MEAN_1"]), var_se = unname(gp["SE", "COV_11"])))
}

write_case <- function(case, dat, itemtype, description) {
  dat <- as.data.frame(dat)
  m_oakes <- fit_mirt(dat, itemtype, "Oakes")
  m_rich <- fit_mirt(dat, itemtype, "Richardson")
  key <- apply(dat, 1, paste, collapse = "")
  first <- !duplicated(key)
  patterns <- as.matrix(dat[first, , drop = FALSE])
  counts <- as.integer(table(key)[key[first]])
  eap <- fscores(m_oakes, method = "EAP", response.pattern = patterns, quadpts = 61,
                 theta_lim = c(-6, 6))
  ex <- extract(m_oakes)
  rich <- extract(m_rich)
  res <- list(
    meta = list(case = case, description = description,
                r_version = R.version.string, mirt_version = as.character(packageVersion("mirt")),
                itemtype = itemtype, TOL = 1e-10, quadpts = 61, theta_lim = c(-6, 6),
                accelerate = "none", optimizer = "NR", converged = extract.mirt(m_oakes, "converged"),
                iterations = extract.mirt(m_oakes, "iterations")),
    items = colnames(dat),
    patterns = unname(patterns),
    counts = counts,
    loglik = extract.mirt(m_oakes, "logLik"),
    params = ex$params,
    latent = ex$latent,
    se_richardson = list(params = lapply(rich$params, function(p) p[c("a_se", "d_se")]),
                         latent = rich$latent[c("mean_se", "var_se")]),
    eap = list(theta = unname(eap[, "F1"]), se = unname(eap[, "SE_F1"]))
  )
  path <- file.path(out_dir, paste0(case, ".json"))
  write_json(res, path, auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null")
  cat("wrote", path, " loglik =", res$loglik, " iter =", res$meta$iterations, "\n")
}

lsat7 <- expand.table(LSAT7)
write_case("v1_lsat7_2pl", lsat7, "2PL", "V1: 2PL, LSAT7")
write_case("v2_lsat7_rasch", lsat7, "Rasch", "V2: Rasch, LSAT7")

# Simulated data (fixed seed): 12 items, N = 2000.
set.seed(20261120)
J <- 12
a_true <- matrix(rlnorm(J, 0, 0.3))
d_true <- matrix(rnorm(J, 0, 1))
sim_2pl <- simdata(a_true, d_true, N = 2000, itemtype = "2PL")
colnames(sim_2pl) <- sprintf("q%02d", seq_len(J))
write_case("v1_sim_2pl", sim_2pl, "2PL", "V1: 2PL, simulated J=12 N=2000 (seed 20261120)")

sim_rasch <- simdata(matrix(1, J), d_true, N = 2000, itemtype = "2PL",
                     Theta = matrix(rnorm(2000, 0, sqrt(1.5))))
colnames(sim_rasch) <- sprintf("q%02d", seq_len(J))
write_case("v2_sim_rasch", sim_rasch, "Rasch",
           "V2: Rasch, simulated J=12 N=2000, latent var 1.5 (seed 20261120)")
