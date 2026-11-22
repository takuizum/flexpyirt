# S-X2 item fit for multiple-group models (VALIDATION.md V15, DESIGN.md §15.45) from R mirt.
# Run from the repository root:  Rscript validation/r/m5_itemfit_multigroup_reference.R
# Writes tests/reference/v15_sx2_multigroup.json. CI does not run R; the JSON is committed.
#
# mirt's itemfit() on a MultipleGroupClass evaluates S-X2 group by group (MGC2SC): each group's
# responses, that group's item parameters and latent distribution, and degrees of freedom net of
# that group's free item and latent parameters (shared anchors count in every group).
#
# twogroup_2pl: V4 data and mirt's V4 estimates, rebuilt with TOL = NaN (no EM step) as in V15.
# science_grm_2g: Science (GRM, 4 items) split into two groups by row parity, all items invariant,
#   focal mean/variance free; fitted here, the data and estimates are stored for the Python fit.
suppressMessages({
  library(mirt)
  library(jsonlite)
})

sx2_by_group <- function(m, quadpts, theta_lim, mincell = 1) {
  fits <- itemfit(m, fit_stats = "S_X2", mincell = mincell, quadpts = quadpts,
                  theta_lim = theta_lim)
  lapply(fits, function(f) list(item = f$item, S_X2 = f$S_X2, df = f$df.S_X2,
                                RMSEA = f$RMSEA.S_X2, p = f$p.S_X2))
}

# --- twogroup_2pl ---------------------------------------------------------------------------
r <- fromJSON(file.path("tests", "reference", "v4_twogroup_2pl.json"), simplifyVector = FALSE)
pats <- do.call(rbind, lapply(r$patterns, unlist))
colnames(pats) <- unlist(r$items)
reps <- rep(seq_len(nrow(pats)), unlist(r$counts))
dat <- as.data.frame(pats[reps, , drop = FALSE])
grp <- factor(unlist(r$pattern_groups)[reps], levels = unlist(r$groups))
quadpts <- r$meta$quadpts
theta_lim <- unlist(r$meta$theta_lim)
inv <- c(unlist(r$anchors), "free_means", "free_var")
sv <- multipleGroup(dat, 1, group = grp, itemtype = "2PL", invariance = inv, pars = "values",
                    quadpts = quadpts, technical = list(theta_lim = theta_lim))
for (g in names(r$estimates)) {
  est <- r$estimates[[g]]
  for (i in unlist(r$items)) {
    p <- est$params[[i]]
    sv$value[sv$group == g & sv$item == i & sv$name == "a1"] <- p$a
    sv$value[sv$group == g & sv$item == i & sv$name == "d"] <- p$d
  }
  sv$value[sv$group == g & sv$name == "MEAN_1"] <- est$latent$mean
  sv$value[sv$group == g & sv$name == "COV_11"] <- est$latent$var
}
# multipleGroup() with TOL = NaN still takes one EM step, so fit with every parameter fixed
# (equality constraints dropped, since they need free parameters), then restore each parameter's
# `est` flag from the free specification: itemfit() reads only the flags for its degrees of freedom.
sv_fixed <- sv
sv_fixed$est <- FALSE
sv_fixed$const <- "none"
m <- multipleGroup(dat, 1, group = grp, itemtype = "2PL", invariance = "", pars = sv_fixed,
                   verbose = FALSE, quadpts = quadpts, technical = list(theta_lim = theta_lim))
stopifnot(abs(extract.mirt(m, "logLik") - r$loglik) < 1e-6 * abs(r$loglik))
for (gi in seq_along(levels(grp))) {
  g <- levels(grp)[gi]
  obj <- m@ParObjects$pars[[gi]]
  for (k in seq_along(obj@ParObjects$pars)) {
    nm <- names(obj@ParObjects$pars[[k]]@est)
    item <- if (k <= ncol(dat)) colnames(dat)[k] else "GROUP"
    flags <- sv$est[sv$group == g & sv$item == item]
    names(flags) <- sv$name[sv$group == g & sv$item == item]
    obj@ParObjects$pars[[k]]@est[] <- unname(flags[nm])
  }
  m@ParObjects$pars[[gi]] <- obj
}
twogroup <- list(source = "v4_twogroup_2pl", mincell = 1,
                 groups = sx2_by_group(m, quadpts, theta_lim))

# --- science_grm_2g -------------------------------------------------------------------------
sci <- Science
items <- colnames(sci)
sci[] <- lapply(sci, function(v) as.integer(v) - 1L)
sgrp <- factor(rep(c("ref", "focal"), length.out = nrow(sci)), levels = c("ref", "focal"))
quadpts2 <- 81
theta_lim2 <- c(-6, 6)
m2 <- multipleGroup(sci, 1, group = sgrp, itemtype = "graded",
                    invariance = c(items, "free_means", "free_var"), quadpts = quadpts2,
                    TOL = 1e-10, verbose = FALSE, technical = list(theta_lim = theta_lim2,
                                                                   NCYCLES = 20000))
stopifnot(extract.mirt(m2, "converged"))
key <- apply(cbind(sci, sgrp), 1, paste, collapse = ",")
first <- !duplicated(key)
counts <- as.integer(table(factor(key, levels = key[first])))
est2 <- list()
for (g in c("ref", "focal")) {
  cf <- coef(m2, simplify = TRUE)[[g]]
  est2[[g]] <- list(items = lapply(setNames(items, items), function(i) as.list(cf$items[i, ])),
                    mean = unname(cf$means[1]), var = unname(cf$cov[1, 1]))
}
science <- list(
  items = items, patterns = unname(as.matrix(sci[first, ])), pattern_groups = as.character(sgrp[first]),
  counts = counts, loglik = extract.mirt(m2, "logLik"), estimates = est2,
  quadpts = quadpts2, theta_lim = theta_lim2, mincell = 1,
  groups = sx2_by_group(m2, quadpts2, theta_lim2)
)

res <- list(
  meta = list(case = "v15_sx2_multigroup", r_version = R.version.string,
              mirt_version = as.character(packageVersion("mirt")), converged = TRUE),
  cases = list(twogroup_2pl = twogroup, science_grm_2g = science)
)
write_json(res, file.path("tests", "reference", "v15_sx2_multigroup.json"), auto_unbox = TRUE,
           digits = NA, pretty = TRUE, na = "null")
cat("wrote tests/reference/v15_sx2_multigroup.json\n")
