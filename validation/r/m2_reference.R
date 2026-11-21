# M2 reference values (VALIDATION.md V3, V4, V5, V9) from R mirt.
# Run from the repository root:  Rscript validation/r/m2_reference.R
# Writes tests/reference/<case>.json. CI does not run R; the JSON files are committed.
#
# As in m1_reference.R: optimizer = "NR", accelerate = "none", TOL = 1e-10.
# Cases with a free latent mean/variance (V4, V5) use 81 points on [-8, 8] so that grid
# truncation is negligible (VALIDATION.md, DESIGN.md §14.9).
suppressMessages({
  library(mirt)
  library(jsonlite)
})

out_dir <- file.path("tests", "reference")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
# NOTE: theta_lim must go inside `technical` for mirt()/multipleGroup(); as a top-level argument
# it is silently ignored (fscores() does take a top-level theta_lim).
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

write_case <- function(case, description, dat, groups, group_labels, fit_fun, quad, extra = list(),
                       eap = TRUE) {
  m_oakes <- fit_fun("Oakes")
  m_rich <- fit_fun("Richardson")
  key <- paste(group_labels, apply(dat, 1, paste, collapse = ","))
  first <- !duplicated(key)
  patterns <- as.matrix(dat[first, , drop = FALSE])
  ex <- extract(m_oakes, groups)
  rich <- extract(m_rich, groups)
  res <- c(list(
    meta = list(case = case, description = description,
                r_version = R.version.string, mirt_version = as.character(packageVersion("mirt")),
                TOL = 1e-10, quadpts = quad$quadpts, theta_lim = quad$theta_lim,
                grid_check = range(m_oakes@Model$Theta),
                accelerate = "none", optimizer = "NR",
                converged = extract.mirt(m_oakes, "converged"),
                iterations = extract.mirt(m_oakes, "iterations"),
                converged_richardson = extract.mirt(m_rich, "converged")),
    items = colnames(dat),
    groups = I(groups),
    patterns = unname(patterns),
    pattern_groups = group_labels[first],
    counts = as.integer(table(key)[key[first]]),
    loglik = extract.mirt(m_oakes, "logLik"),
    estimates = ex,
    se_richardson = rich
  ), extra)
  stopifnot(
    isTRUE(res$meta$converged),
    isTRUE(res$meta$converged_richardson),
    all(res$meta$grid_check == quad$theta_lim),
    all(range(m_rich@Model$Theta) == quad$theta_lim)
  )
  if (eap) {
    sc <- fscores(m_oakes, method = "EAP", response.pattern = patterns,
                  quadpts = quad$quadpts, theta_lim = quad$theta_lim)
    res$eap <- list(theta = unname(sc[, "F1"]), se = unname(sc[, "SE_F1"]))
  }
  path <- file.path(out_dir, paste0(case, ".json"))
  write_json(res, path, auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null")
  cat("wrote", path, " loglik =", res$loglik, " iter =", res$meta$iterations,
      " converged =", res$meta$converged, "\n")
}


# ---- V3: GRM, Science (recoded 1..4 -> 0..3) -----------------------------------------------
sci <- as.data.frame(as.matrix(Science) - 1L)
write_case("v3_science_grm", "V3: graded response model, mirt::Science recoded to 0..3",
           sci, "all", rep("all", nrow(sci)),
           function(se) do.call(mirt, c(list(sci, 1, itemtype = "graded", SE = TRUE, SE.type = se,
                                             quadpts = 61, technical = tech(default_quad)), common)),
           default_quad)

# ---- V4: 2PL, two groups, 5 anchors, focal mean/var free ------------------------------------
set.seed(20261121)
J <- 15; N <- 1500
a <- matrix(rlnorm(J, 0, 0.3)); d <- matrix(rnorm(J))
a_f <- a; d_f <- d
dif <- 6:J
d_f[dif] <- d[dif] + rnorm(length(dif), 0, 0.4)
a_f[dif] <- a[dif] * exp(rnorm(length(dif), 0, 0.15))
ref <- simdata(a, d, N, itemtype = "2PL")
foc <- simdata(a_f, d_f, N, itemtype = "2PL", mu = 0.5, sigma = matrix(1.2))
dat4 <- as.data.frame(rbind(ref, foc)); colnames(dat4) <- sprintf("q%02d", 1:J)
grp4 <- rep(c("ref", "focal"), each = N)
anchors <- sprintf("q%02d", 1:5)
write_case("v4_twogroup_2pl",
           "V4: 2PL, groups ref/focal (N=1500 each), anchors q01-q05, focal N(0.5, 1.2) free",
           dat4, c("ref", "focal"), grp4,
           function(se) do.call(multipleGroup, c(list(dat4, 1, group = factor(grp4, levels = c("ref", "focal")),
                                                      itemtype = "2PL",
                                                      invariance = c(anchors, "free_means", "free_var"),
                                                      SE = TRUE, SE.type = se,
                                                      quadpts = 81, technical = tech(wide_quad)), common)),
           wide_quad, extra = list(anchors = anchors), eap = FALSE)

# ---- V5: FIPC, items q01-q08 fixed at bank values, latent mean/var free -----------------------
set.seed(20261122)
J <- 15; N <- 2000
a5 <- rlnorm(J, 0, 0.3); d5 <- rnorm(J)
dat5 <- as.data.frame(simdata(matrix(a5), matrix(d5), N, itemtype = "2PL", mu = 0.4,
                              sigma = matrix(1.3)))
colnames(dat5) <- sprintf("q%02d", 1:J)
bank_items <- sprintf("q%02d", 1:8)
bank <- data.frame(item = bank_items, a = a5[1:8], d = d5[1:8])
fit5 <- function(se) {
  sv <- mirt(dat5, 1, "2PL", pars = "values", quadpts = 81, technical = tech(wide_quad))
  for (k in seq_len(nrow(bank))) {
    sv[sv$item == bank$item[k] & sv$name == "a1", c("value", "est")] <- list(bank$a[k], FALSE)
    sv[sv$item == bank$item[k] & sv$name == "d", c("value", "est")] <- list(bank$d[k], FALSE)
  }
  sv[sv$name %in% c("MEAN_1", "COV_11"), "est"] <- TRUE
  do.call(mirt, c(list(dat5, 1, itemtype = "2PL", pars = sv, SE = TRUE, SE.type = se,
                       quadpts = 81, technical = tech(wide_quad)), common))
}
write_case("v5_fipc_2pl", "V5: FIPC, 2PL, q01-q08 fixed at generating values, latent free, N=2000",
           dat5, "all", rep("all", N), fit5, wide_quad, extra = list(bank = bank))

# ---- V9: 2PL with 15% MCAR missing ---------------------------------------------------------------
set.seed(20261123)
J <- 12; N <- 2000
dat9 <- as.data.frame(simdata(matrix(rlnorm(J, 0, 0.3)), matrix(rnorm(J)), N, itemtype = "2PL"))
colnames(dat9) <- sprintf("q%02d", 1:J)
dat9[matrix(runif(N * J) < 0.15, N, J)] <- NA
write_case("v9_missing_2pl", "V9: 2PL, J=12, N=2000, 15% MCAR missing",
           dat9, "all", rep("all", N),
           function(se) do.call(mirt, c(list(dat9, 1, itemtype = "2PL", SE = TRUE, SE.type = se,
                                             quadpts = 61, technical = tech(default_quad)), common)),
           default_quad)
