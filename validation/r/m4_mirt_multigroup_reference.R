# Multiple-group confirmatory 2D MIRT reference (VALIDATION.md V16) from R mirt.
# Run from the repository root:  Rscript validation/r/m4_mirt_multigroup_reference.R
# Writes tests/reference/v16_mirt_multigroup.json. CI does not run R; the JSON is committed.
#
# Two groups, F1 = items 1-6, F2 = items 7-12 (case b: item 7 also loads on F1), COV = F1*F2,
# anchors (equal across groups) on both factors, invariance "free_means" and "free_var" for the
# focal group. Settings as for V14: optimizer = "NR", accelerate = "none", TOL = 1e-10,
# SE.type = "Oakes", 81 points per dimension on [-6, 6] (theta_lim inside technical). The grid is
# finer than V14's 31 because the focal variances are free: mirt updates latent variances with
# closed-form moments while flexpyirt maximizes the discretized likelihood exactly, and on a coarse
# grid that difference (VALIDATION.md, M1 notes) moves anchor slopes by ~3e-4.
# mirt reports the focal covariance COV_21; the correlation cor_12 = COV_21 / sqrt(COV_11 COV_22)
# and its SE (delta method with mirt's vcov) are written alongside.
suppressMessages({
  library(mirt)
  library(jsonlite)
})

quad <- list(quadpts = 81, theta_lim = c(-6, 6))

run_case <- function(case, description, dat, grp, spec, anchors, factors) {
  mod <- mirt.model(spec)
  m <- multipleGroup(dat, mod, group = grp, itemtype = "2PL",
                     invariance = c(anchors, "free_means", "free_var"),
                     SE = TRUE, SE.type = "Oakes", optimizer = "NR", accelerate = "none",
                     TOL = 1e-10, verbose = FALSE, quadpts = quad$quadpts,
                     technical = list(NCYCLES = 20000, theta_lim = quad$theta_lim))
  stopifnot(isTRUE(extract.mirt(m, "converged")))
  grid_check <- range(m@Model$Theta)
  stopifnot(all(grid_check == quad$theta_lim))
  cf <- coef(m, printSE = TRUE)
  V <- vcov(m)
  groups <- levels(grp)
  est <- list()
  for (g in groups) {
    items <- colnames(dat)
    params <- lapply(items, function(i) {
      x <- cf[[g]][[i]]
      out <- list(a1 = x["par", "a1"], a2 = x["par", "a2"], d = x["par", "d"])
      se <- if ("SE" %in% rownames(x)) x["SE", c("a1", "a2", "d")] else c(NA, NA, NA)
      c(out, list(a1_se = unname(se[1]), a2_se = unname(se[2]), d_se = unname(se[3])))
    })
    names(params) <- items
    gp <- cf[[g]]$GroupPars
    v1 <- gp["par", "COV_11"]; v2 <- gp["par", "COV_22"]; c12 <- gp["par", "COV_21"]
    se_of <- function(nm) if ("SE" %in% rownames(gp)) unname(gp["SE", nm]) else NA
    # Delta-method SE of the correlation from mirt's vcov (free group parameters only).
    obj <- extract.group(m, g)@ParObjects$pars[[ncol(dat) + 1L]]
    pn <- setNames(obj@parnum, names(obj@est))
    free <- obj@est[c("COV_11", "COV_22", "COV_21")]
    keys <- paste0(c("COV_11", "COV_22", "COV_21"), ".", pn[c("COV_11", "COV_22", "COV_21")])
    grad <- c(-0.5 * c12 / (v1^1.5 * sqrt(v2)), -0.5 * c12 / (sqrt(v1) * v2^1.5), 1 / sqrt(v1 * v2))
    keep <- free & keys %in% rownames(V)
    cor_se <- if (any(keep)) sqrt(drop(t(grad[keep]) %*% V[keys[keep], keys[keep], drop = FALSE] %*% grad[keep])) else NA
    est[[g]] <- list(
      params = params,
      latent = list(mean_1 = gp["par", "MEAN_1"], mean_2 = gp["par", "MEAN_2"], var_1 = v1, var_2 = v2,
                    cor_12 = c12 / sqrt(v1 * v2), cov_12 = c12, cov_12_se = se_of("COV_21"),
                    mean_1_se = se_of("MEAN_1"), mean_2_se = se_of("MEAN_2"),
                    var_1_se = se_of("COV_11"), var_2_se = se_of("COV_22"), cor_12_se = cor_se)
    )
  }
  key <- paste(as.character(grp), apply(dat, 1, paste, collapse = ","))
  first <- !duplicated(key)
  cat(sprintf("%-22s loglik=%.6f iter=%d\n", case, extract.mirt(m, "logLik"), extract.mirt(m, "iterations")))
  list(
    meta = list(case = case, description = description, mirt_version = as.character(packageVersion("mirt")),
                r_version = R.version.string, quadpts = quad$quadpts, theta_lim = quad$theta_lim,
                grid_check = grid_check, TOL = 1e-10, optimizer = "NR", accelerate = "none",
                converged = TRUE, iterations = extract.mirt(m, "iterations")),
    items = colnames(dat), groups = I(groups), anchors = anchors, factors = factors,
    patterns = unname(as.matrix(dat[first, ])), pattern_groups = as.character(grp)[first],
    counts = as.integer(table(key)[key[first]]),
    loglik = extract.mirt(m, "logLik"), estimates = est
  )
}

simulate <- function(a, d, n, mu, sigma) {
  as.data.frame(simdata(a, d, n, itemtype = "2PL", mu = mu, sigma = sigma))
}

set.seed(20261201)
J <- 12L; N <- 1500L
a <- matrix(0, J, 2L); a[1:6, 1] <- runif(6, 0.9, 1.6); a[7:12, 2] <- runif(6, 0.9, 1.6)
d <- matrix(rnorm(J))
s_ref <- matrix(c(1, 0.5, 0.5, 1), 2); s_foc <- matrix(c(1.3, 0.4, 0.4, 0.8), 2)
dat <- rbind(simulate(a, d, N, c(0, 0), s_ref), simulate(a, d, N, c(0.4, -0.3), s_foc))
colnames(dat) <- sprintf("q%02d", 1:J)
grp <- factor(rep(c("ref", "focal"), each = N), levels = c("ref", "focal"))
anchors <- sprintf("q%02d", c(1:3, 7:9))
cases <- list()
cases$mg2d_simple <- run_case("mg2d_simple", "2 groups, 2D 2PL simple structure, anchors q01-q03 and q07-q09",
                              dat, grp, "F1 = 1-6\nF2 = 7-12\nCOV = F1*F2", anchors,
                              list(F1 = sprintf("q%02d", 1:6), F2 = sprintf("q%02d", 7:12)))

set.seed(20261202)
a2 <- a; a2[7, 1] <- 0.6
dat2 <- rbind(simulate(a2, d, N, c(0, 0), s_ref), simulate(a2, d, N, c(-0.3, 0.2), matrix(c(0.8, 0.2, 0.2, 1.2), 2)))
colnames(dat2) <- sprintf("q%02d", 1:J)
cases$mg2d_cross <- run_case("mg2d_cross", "2 groups, 2D 2PL with cross-loading q07, anchors q01-q03 and q08-q10",
                             dat2, grp, "F1 = 1-7\nF2 = 7-12\nCOV = F1*F2", sprintf("q%02d", c(1:3, 8:10)),
                             list(F1 = sprintf("q%02d", 1:7), F2 = sprintf("q%02d", 7:12)))

res <- list(meta = list(case = "v16_mirt_multigroup", mirt_version = as.character(packageVersion("mirt")),
                        converged = TRUE), cases = cases)
write_json(res, file.path("tests", "reference", "v16_mirt_multigroup.json"), auto_unbox = TRUE,
           digits = NA, pretty = TRUE, na = "null")
cat("wrote tests/reference/v16_mirt_multigroup.json\n")
