# Multiple-group likelihood-ratio DIF tests (VALIDATION.md V17) from R mirt's DIF().
# Run from the repository root:  Rscript validation/r/m5_dif_reference.R
# Writes tests/reference/v17_dif.json. CI does not run R; the JSON is committed.
#
# Reuses the V4 data (2PL, groups ref/focal, anchors q01-q05, focal mean and variance free) and the
# V4 grid (81 points on [-8, 8]). Every fit, including DIF()'s refits, uses optimizer = "NR",
# accelerate = "none" and TOL = 1e-10.
source(file.path("validation", "r", "setup.R"))  # mirt from GitHub; mirt_version / mirt_sha
suppressMessages({
  library(mirt)
  library(jsonlite)
})

r <- fromJSON(file.path("tests", "reference", "v4_twogroup_2pl.json"), simplifyVector = FALSE)
pats <- do.call(rbind, lapply(r$patterns, unlist))
colnames(pats) <- unlist(r$items)
reps <- rep(seq_len(nrow(pats)), unlist(r$counts))
dat <- as.data.frame(pats[reps, , drop = FALSE])
grp <- factor(unlist(r$pattern_groups)[reps], levels = c("ref", "focal"))
items <- colnames(dat)
anchors <- unlist(r$anchors)

settings <- list(TOL = 1e-10, accelerate = "none", optimizer = "NR", verbose = FALSE,
                 quadpts = 81, technical = list(NCYCLES = 50000, theta_lim = c(-8, 8)))
mg <- function(invariance) {
  do.call(multipleGroup, c(list(dat, 1, group = grp, itemtype = "2PL", invariance = invariance),
                           settings))
}
run_dif <- function(base, which_par, items2test, scheme) {
  out <- do.call(DIF, c(list(base, which.par = which_par, items2test = items2test, scheme = scheme,
                             return_models = TRUE), settings))
  tab <- do.call(DIF, c(list(base, which.par = which_par, items2test = items2test, scheme = scheme),
                        settings))
  stopifnot(all(tab$converged))
  list(which_par = I(sub("^a1$", "a", which_par)), scheme = scheme,
       base_loglik = extract.mirt(base, "logLik"), base_nest = extract.mirt(base, "nest"),
       base_converged = extract.mirt(base, "converged"),
       item = rownames(tab), aic = tab$AIC, sabic = tab$SABIC, hq = tab$HQ, bic = tab$BIC,
       x2 = tab$X2, df = tab$df, p = tab$p, adj_p = tab$adj_p,
       loglik = unname(sapply(out, function(m) extract.mirt(m, "logLik"))))
}

base_add <- mg(c(anchors, "free_means", "free_var"))
base_drop <- mg(c(items, "free_means", "free_var"))
stopifnot(extract.mirt(base_add, "converged"), extract.mirt(base_drop, "converged"))
cases <- list(
  add_ad = run_dif(base_add, c("a1", "d"), items[6:15], "add"),
  add_d = run_dif(base_add, "d", items[6:15], "add"),
  drop_ad = run_dif(base_drop, c("a1", "d"), items, "drop")
)
res <- list(
  meta = list(case = "v17_dif", source = "v4_twogroup_2pl", r_version = R.version.string,
              mirt_version = mirt_version_string(), mirt_sha = mirt_commit_sha(), TOL = 1e-10, quadpts = 81,
              theta_lim = c(-8, 8), grid_check = range(base_add@Model$Theta),
              accelerate = "none", optimizer = "NR", converged = TRUE),
  anchors = anchors,
  cases = cases
)
path <- file.path("tests", "reference", "v17_dif.json")
write_json(res, path, auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null")
for (nm in names(cases)) {
  cat(sprintf("%-8s x2 = %s\n", nm, paste(round(cases[[nm]]$x2, 3), collapse = ", ")))
}
cat("wrote", path, "\n")
