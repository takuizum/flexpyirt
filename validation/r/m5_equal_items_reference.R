# Equality constraints across items (VALIDATION.md V16) from R mirt's CONSTRAIN.
# Run from the repository root:  Rscript validation/r/m5_equal_items_reference.R
# Writes tests/reference/v16_*.json. CI does not run R; the JSON files are committed.
#
# As in m2_reference.R: optimizer = "NR", accelerate = "none", TOL = 1e-10, SE.type = "Oakes"
# (and "Richardson" for a second set of SEs). The two-group case frees the focal mean and
# variance and uses 81 points on [-8, 8] (VALIDATION.md, DESIGN.md §14.9).
suppressMessages({
  library(mirt)
  library(jsonlite)
})

out_dir <- file.path("tests", "reference")
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

write_case <- function(case, description, dat, groups, group_labels, fit_fun, quad, extra = list()) {
  m_oakes <- fit_fun("Oakes")
  m_rich <- fit_fun("Richardson")
  key <- paste(group_labels, apply(dat, 1, paste, collapse = ","))
  first <- !duplicated(key)
  patterns <- as.matrix(dat[first, , drop = FALSE])
  res <- c(list(
    meta = list(case = case, description = description,
                r_version = R.version.string, mirt_version = as.character(packageVersion("mirt")),
                TOL = 1e-10, quadpts = quad$quadpts, theta_lim = quad$theta_lim,
                grid_check = range(m_oakes@Model$Theta),
                accelerate = "none", optimizer = "NR",
                converged = extract.mirt(m_oakes, "converged"),
                iterations = extract.mirt(m_oakes, "iterations"),
                converged_richardson = extract.mirt(m_rich, "converged"),
                n_free = extract.mirt(m_oakes, "nest")),
    items = colnames(dat),
    groups = I(groups),
    patterns = unname(patterns),
    pattern_groups = group_labels[first],
    counts = as.integer(table(key)[key[first]]),
    loglik = extract.mirt(m_oakes, "logLik"),
    estimates = extract(m_oakes, groups),
    se_richardson = extract(m_rich, groups)
  ), extra)
  stopifnot(
    isTRUE(res$meta$converged),
    isTRUE(res$meta$converged_richardson),
    all(res$meta$grid_check == quad$theta_lim)
  )
  path <- file.path(out_dir, paste0(case, ".json"))
  write_json(res, path, auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null")
  cat("wrote", path, " loglik =", res$loglik, " iter =", res$meta$iterations,
      " nest =", res$meta$n_free, "\n")
}

# ---- V16a: LSAT7, one common slope for all five items ("1PL" with N(0, 1)) --------------------
lsat <- expand.table(LSAT7)
colnames(lsat) <- sprintf("i%d", 1:5)
mod16a <- mirt.model("F = 1-5\nCONSTRAIN = (1-5, a1)")
write_case("v16_lsat7_common_slope", "V16a: 2PL with one common slope (LSAT7), N(0,1)",
           lsat, "all", rep("all", nrow(lsat)),
           function(se) do.call(mirt, c(list(lsat, mod16a, itemtype = "2PL", SE = TRUE,
                                             SE.type = se, quadpts = 61,
                                             technical = tech(default_quad)), common)),
           default_quad, extra = list(constraints = list(list(items = colnames(lsat), params = "a"))))

# ---- V16b: mixed 2PL / graded items, constraints across item models -----------------------------
# Items i1-i6 are 2PL, i7-i10 graded with 3 categories. Constraints:
#   a1 equal for (i1, i2, i7)  (2PL and graded items share a slope),
#   d  equal for (i3, i4),
#   a1 equal for (i8, i9, i10), and d1 equal for (i9, i10).
set.seed(20261124)
N <- 2500
a_true <- c(1.2, 1.2, 0.8, 1.5, 1.0, 0.7, 1.2, 1.4, 1.4, 1.4)
d2 <- c(0.5, -0.3, 0.2, 0.2, 1.0, -0.8)
dg <- rbind(c(1.5, -0.5), c(1.0, -1.0), c(0.8, -0.4), c(0.8, -1.2))
dmat <- rbind(cbind(d2, NA), dg)
dat16b <- as.data.frame(simdata(matrix(a_true), dmat, N,
                                itemtype = c(rep("2PL", 6), rep("graded", 4))))
colnames(dat16b) <- sprintf("i%d", 1:10)
mod16b <- mirt.model("F = 1-10\nCONSTRAIN = (1, 2, 7, a1), (3, 4, d), (8, 9, 10, a1), (9, 10, d1)")
write_case("v16_mixed_constraints",
           "V16b: 2PL (i1-i6) and graded K=3 (i7-i10), constraints across items and item models",
           dat16b, "all", rep("all", N),
           function(se) do.call(mirt, c(list(dat16b, mod16b,
                                             itemtype = c(rep("2PL", 6), rep("graded", 4)),
                                             SE = TRUE, SE.type = se, quadpts = 61,
                                             technical = tech(default_quad)), common)),
           default_quad,
           extra = list(constraints = list(
             list(items = c("i1", "i2", "i7"), params = "a"),
             list(items = c("i3", "i4"), params = "d"),
             list(items = c("i8", "i9", "i10"), params = "a"),
             list(items = c("i9", "i10"), params = "d1"))))

# ---- V16c: two groups, anchors equal across groups, and a common slope across items ----------
# Items q01-q05 are anchors (all parameters equal across groups); within each group the slopes
# of q06-q10 are one common slope (CONSTRAIN in each group, not across groups). Focal mean and
# variance free.
set.seed(20261125)
J <- 10; N <- 1500
a <- rlnorm(J, 0, 0.25); d <- rnorm(J)
a[6:10] <- 1.1
a_f <- a; a_f[6:10] <- 0.9
d_f <- d; d_f[6:10] <- d[6:10] + rnorm(5, 0, 0.3)
ref <- simdata(matrix(a), matrix(d), N, itemtype = "2PL")
foc <- simdata(matrix(a_f), matrix(d_f), N, itemtype = "2PL", mu = 0.4, sigma = matrix(1.3))
dat16c <- as.data.frame(rbind(ref, foc)); colnames(dat16c) <- sprintf("q%02d", 1:J)
grp16c <- rep(c("ref", "focal"), each = N)
anchors <- sprintf("q%02d", 1:5)
mod16c <- mirt.model("F = 1-10\nCONSTRAIN = (6-10, a1)")
write_case("v16_twogroup_common_slope",
           "V16c: 2PL two groups, anchors q01-q05, common slope of q06-q10 within each group",
           dat16c, c("ref", "focal"), grp16c,
           function(se) do.call(multipleGroup, c(list(dat16c, mod16c,
                                                      group = factor(grp16c, levels = c("ref", "focal")),
                                                      itemtype = "2PL",
                                                      invariance = c(anchors, "free_means", "free_var"),
                                                      SE = TRUE, SE.type = se, quadpts = 81,
                                                      technical = tech(wide_quad)), common)),
           wide_quad,
           extra = list(anchors = anchors,
                        constraints = list(list(items = sprintf("q%02d", 6:10), params = "a"))))
