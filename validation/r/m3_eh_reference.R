# M3 empirical histogram reference values (VALIDATION.md V6) from R mirt.
# Run from the repository root:  Rscript validation/r/m3_eh_reference.R [v6a] [v6b]
# Writes tests/reference/v6_*.json. CI does not run R; the JSON files are committed.
#
# mirt's dentype = "EH" re-estimates the histogram from the posterior counts at every cycle
# without standardizing it, so with all items free the scale is not identified and EM drifts
# (a free 2PL calibration did not converge in 50,000 cycles; the histogram variance drifted to
# 1.5). flexpyirt standardizes the reference group instead (DESIGN.md §15.34), which mirt does not
# offer. Both programs leave the histogram unconstrained under FIPC, where fixed items set the
# scale, so V6 compares FIPC fits: one group, and two groups with the free items equal across
# groups. Settings as in m2_reference.R: optimizer = "NR", accelerate = "none", 61 points on
# [-6, 6] (theta_lim inside `technical`).
suppressMessages({
  library(mirt)
  library(jsonlite)
})

out_dir <- file.path("tests", "reference")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
TOL <- 1e-9
quad <- list(quadpts = 61, theta_lim = c(-6, 6))
common <- list(TOL = TOL, accelerate = "none", optimizer = "NR", verbose = FALSE,
               dentype = "EH", quadpts = quad$quadpts,
               technical = list(NCYCLES = 100000, theta_lim = quad$theta_lim))

set.seed(20261123)
J <- 20; N <- 3000
a <- rlnorm(J, 0.1, 0.25); d <- rnorm(J, 0, 0.9)
skewed <- function(n) (rchisq(n, df = 4) - 4) / sqrt(8)  # mean 0, var 1, skewness sqrt(2)
items <- sprintf("i%02d", 1:J)
bank_items <- items[1:10]
free_items <- items[11:J]

clean_items <- function(cf, which) {
  out <- lapply(which, function(i) {
    x <- cf[[i]]
    se <- if ("SE" %in% rownames(x)) x["SE", c("a1", "d")] else c(NA, NA)
    list(a = unname(x["par", "a1"]), d = unname(x["par", "d"]),
         a_se = unname(se[1]), d_se = unname(se[2]))
  })
  names(out) <- which
  out
}

fix_bank <- function(sv) {
  for (k in seq_along(bank_items)) {
    sel <- sv$item == bank_items[k]
    sv[sel & sv$name == "a1", c("value", "est")] <- list(a[k], FALSE)
    sv[sel & sv$name == "d", c("value", "est")] <- list(d[k], FALSE)
  }
  sv
}

write_case <- function(case, description, dat, group, fit_fun, richardson = TRUE) {
  m_oakes <- fit_fun("Oakes")
  m_rich <- if (richardson) fit_fun("Richardson") else NULL
  groups <- if (is.null(group)) "all" else levels(group)
  labels <- if (is.null(group)) rep("all", nrow(dat)) else as.character(group)
  key <- paste(labels, apply(dat, 1, paste, collapse = ","))
  first <- !duplicated(key)
  cf_o <- coef(m_oakes, printSE = TRUE)
  cf_r <- if (richardson) coef(m_rich, printSE = TRUE) else NULL
  if (is.null(group)) { cf_o <- list(cf_o); cf_r <- list(cf_r) }
  names(cf_o) <- groups
  if (richardson) names(cf_r) <- groups
  prior <- extract.mirt(m_oakes, "Prior")
  res <- list(
    meta = list(case = case, description = description,
                r_version = R.version.string, mirt_version = as.character(packageVersion("mirt")),
                TOL = TOL, quadpts = quad$quadpts, theta_lim = quad$theta_lim,
                grid_check = range(m_oakes@Model$Theta),
                accelerate = "none", optimizer = "NR", dentype = "EH",
                converged = extract.mirt(m_oakes, "converged"),
                iterations = extract.mirt(m_oakes, "iterations")),
    items = colnames(dat),
    groups = I(groups),
    bank = list(item = bank_items, a = a[1:10], d = d[1:10]),
    free_items = free_items,
    patterns = unname(as.matrix(dat[first, , drop = FALSE])),
    pattern_groups = labels[first],
    counts = as.integer(table(key)[key[first]]),
    nodes = as.numeric(m_oakes@Model$Theta[, 1]),
    loglik = extract.mirt(m_oakes, "logLik"),
    estimates = lapply(cf_o, clean_items, which = free_items),
    eh_weights = lapply(prior, as.numeric)
  )
  names(res$eh_weights) <- groups
  if (richardson) {  # assigning NULL drops the element, so the keys are absent otherwise
    res$meta$converged_richardson <- extract.mirt(m_rich, "converged")
    res$se_richardson <- lapply(cf_r, clean_items, which = free_items)
  }
  stopifnot(
    isTRUE(res$meta$converged),
    !richardson || isTRUE(res$meta$converged_richardson),
    all(res$meta$grid_check == quad$theta_lim),
    !richardson || all(range(m_rich@Model$Theta) == quad$theta_lim)
  )
  path <- file.path(out_dir, paste0(case, ".json"))
  write_json(res, path, auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null")
  cat("wrote", path, " loglik =", res$loglik, " iter =", res$meta$iterations, "\n")
}

RUN <- commandArgs(trailingOnly = TRUE)
if (length(RUN) == 0) RUN <- c("v6a", "v6b")

# ---- V6a: one group, skewed latent, i01-i10 fixed at generating values ------------------------
dat1 <- as.data.frame(simdata(matrix(a), d, N, itemtype = "2PL", Theta = matrix(skewed(N))))
colnames(dat1) <- items
fit1 <- function(se) {
  sv <- fix_bank(mirt(dat1, 1, "2PL", pars = "values", dentype = "EH", quadpts = quad$quadpts,
                      technical = list(theta_lim = quad$theta_lim)))
  do.call(mirt, c(list(dat1, 1, itemtype = "2PL", pars = sv, SE = TRUE, SE.type = se), common))
}
if ("v6a" %in% RUN) write_case("v6_fipc_eh", "V6a: EH, FIPC (i01-i10 fixed), skewed latent, 2PL J=20, N=3000",
           dat1, NULL, fit1)

# ---- V6b: two groups, same bank fixed in both, free items equal across groups ------------------
theta_ref <- skewed(N)
theta_foc <- 0.5 - 1.2 * skewed(N)  # shifted, wider, negatively skewed
dat2 <- as.data.frame(rbind(
  simdata(matrix(a), d, N, itemtype = "2PL", Theta = matrix(theta_ref)),
  simdata(matrix(a), d, N, itemtype = "2PL", Theta = matrix(theta_foc))
))
colnames(dat2) <- items
grp2 <- factor(rep(c("ref", "focal"), each = N), levels = c("ref", "focal"))
fit2 <- function(se) {
  sv <- fix_bank(multipleGroup(dat2, 1, group = grp2, itemtype = "2PL", pars = "values",
                               invariance = free_items, dentype = "EH", quadpts = quad$quadpts,
                               technical = list(theta_lim = quad$theta_lim)))
  do.call(multipleGroup, c(list(dat2, 1, group = grp2, itemtype = "2PL", pars = sv,
                                invariance = free_items, SE = TRUE, SE.type = se), common))
}
if ("v6b" %in% RUN) write_case("v6_twogroup_fipc_eh",
           "V6b: EH, two groups, i01-i10 fixed in both, i11-i20 equal across groups, N=3000 each",
           dat2, grp2, fit2, richardson = FALSE)  # mirt 1.46.1: Richardson SEs fail here
