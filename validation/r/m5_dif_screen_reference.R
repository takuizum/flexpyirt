# DIF screening with anchor purification, Wald DIF tests and DIF effect sizes (VALIDATION.md V49)
# from R mirt's DIF(), DRF(), empirical_ES() and RMSD_DIF().
# Run from the repository root:  Rscript validation/r/m5_dif_screen_reference.R
# Writes tests/reference/v49_dif_screen.json. CI does not run R; the JSON is committed.
#
# Data "screen": two groups of 1200, 8 2PL items and 4 graded items (K = 3), DIF in q02 (slope),
# q04 and q07 (intercept) and q09 (graded intercepts); focal N(0.4, 1.3).
#   - all_others            DIF(<fully invariant>, scheme = "drop")
#   - sequential_free       DIF(<fully invariant>, scheme = "drop_sequential")
#   - sequential_constrain  DIF(<anchors q10-q12>, scheme = "add_sequential", items2test = 1:9)
#   (seq_stat = 0.05, p.adjust = "BH"; max_run = 2, 3 and Inf give the flags after 1, 2 and all
#   rounds; items2test is a prefix 1:k because mirt's sequential loop indexes item names by
#   position in items2test.)
#   - wald                  DIF(<anchors q10-q12>, Wald = TRUE) with Oakes standard errors
# Data "three": three groups of 800, 8 2PL items, DIF in q03 for g3 (omnibus tests, scheme drop).
# Effect sizes on "screen": mirt's estimates of the partially invariant model (q02, q04, q07, q09
# free) and of the fully invariant model are stored; the Python tests put them into fixed
# FitResults on the same 201-point grid on [-6, 6], so only the statistics are compared.
#   - DRF's point estimates with best_fitting = TRUE, den.type = "focal" / "reference", from
#     mirt:::calc_DRFs() (the function DRF() calls): DRF() itself stops in mirt 1.48 for
#     den.type != "marginal" (add_std.DRF() refers to an undefined VAR)
#   - empirical_ES(DIF = TRUE / FALSE, Theta.focal = EAP on the 201-point grid)
#   - RMSD_DIF(probfun = FALSE) of both models
#
# Every fit, including DIF()'s refits, uses optimizer = "NR", accelerate = "none", TOL = 1e-10 and
# 81 points on [-8, 8].
# The DIF() runs take long; set V49_CACHE=<file.rds> to keep them between runs.
suppressMessages({
  library(mirt)
  library(jsonlite)
})

set.seed(20261126)
settings <- list(TOL = 1e-10, accelerate = "none", optimizer = "NR", verbose = FALSE,
                 quadpts = 81, technical = list(NCYCLES = 50000, theta_lim = c(-8, 8)))

# ---- data "screen" ----------------------------------------------------------------------------
ng <- 1200L
n_dich <- 8L
a <- matrix(c(runif(n_dich, 0.9, 1.7), runif(4, 1.0, 1.6)))
d <- matrix(NA_real_, 12, 2)
d[1:n_dich, 1] <- seq(-1.2, 1.2, length.out = n_dich)
d[9:12, ] <- cbind(c(1.2, 0.8, 1.5, 0.6), c(-0.6, -1.0, -0.3, -1.2))
a_f <- a
d_f <- d
a_f[2] <- a[2] * 0.55
d_f[4, 1] <- d[4, 1] - 0.6
d_f[7, 1] <- d[7, 1] + 0.45
d_f[9, ] <- d[9, ] - 0.5
sim_types <- c(rep("dich", n_dich), rep("graded", 4))
dat <- rbind(simdata(a, d, ng, itemtype = sim_types),
             simdata(a_f, d_f, ng, itemtype = sim_types, mu = 0.4, sigma = matrix(1.3)))
colnames(dat) <- sprintf("q%02d", 1:12)
dat <- as.data.frame(dat)
grp <- factor(rep(c("ref", "focal"), each = ng), levels = c("ref", "focal"))
items <- colnames(dat)
itemtype <- c(rep("2PL", n_dich), rep("graded", 4))
anchors <- items[10:12]
dif_items <- items[c(2, 4, 7, 9)]
which_all <- c("a1", "d", "d1", "d2")

mg <- function(data, group, itemtype, invariance, SE = FALSE, ...) {
  do.call(multipleGroup, c(list(data, 1, group = group, itemtype = itemtype,
                                invariance = invariance, SE = SE, ...), settings))
}

base_inv <- mg(dat, grp, itemtype, c(items, "free_means", "free_var"))
base_anchor <- mg(dat, grp, itemtype, c(anchors, "free_means", "free_var"))
base_anchor_se <- mg(dat, grp, itemtype, c(anchors, "free_means", "free_var"), SE = TRUE,
                     SE.type = "Oakes")
for (m in list(base_inv, base_anchor, base_anchor_se)) stopifnot(extract.mirt(m, "converged"))

dif_table <- function(tab) {
  if (!nrow(tab)) {
    return(list(item = I(character(0)), x2 = I(numeric(0)), df = I(integer(0)),
                p = I(numeric(0)), adj_p = I(numeric(0)), converged = I(logical(0))))
  }
  list(item = I(rownames(tab)), x2 = I(tab$X2), df = I(tab$df), p = I(tab$p),
       adj_p = I(tab$adj_p), converged = I(tab$converged))
}
run_dif <- function(base, scheme, items2test, which_par = which_all, ...) {
  tab <- do.call(DIF, c(list(base, which.par = which_par, items2test = items2test,
                             scheme = scheme, p.adjust = "BH", ...), settings))
  dif_table(tab)
}
loglik_of <- function(m) extract.mirt(m, "logLik")

cache <- Sys.getenv("V49_CACHE", "")
cached <- nzchar(cache) && file.exists(cache)
if (cached) {
  saved <- readRDS(cache)
  screen <- saved$screen
} else {
screen <- list(
  all_others = run_dif(base_inv, "drop", 1:12),
  all_others_intercepts = run_dif(base_inv, "drop", 1:12, which_par = c("d", "d1", "d2"))
)
for (scheme in c("drop_sequential", "add_sequential")) {
  base <- if (scheme == "drop_sequential") base_inv else base_anchor
  items2test <- if (scheme == "drop_sequential") 1:12 else 1:9
  runs <- list()
  for (max_run in c(2, 3, Inf)) {
    runs[[length(runs) + 1L]] <- c(
      list(max_run = if (is.finite(max_run)) max_run else "Inf"),
      run_dif(base, scheme, items2test, seq_stat = 0.05, max_run = max_run)
    )
  }
  seq_model <- do.call(DIF, c(list(base, which.par = which_all, items2test = items2test,
                                   scheme = scheme, p.adjust = "BH", seq_stat = 0.05,
                                   return_seq_model = TRUE), settings))
  name <- if (scheme == "drop_sequential") "sequential_free" else "sequential_constrain"
  screen[[name]] <- list(runs = runs, final_loglik = loglik_of(seq_model),
                         final_nest = extract.mirt(seq_model, "nest"))
}
wald <- do.call(DIF, list(base_anchor_se, which.par = which_all, items2test = 1:9,
                          Wald = TRUE, p.adjust = "holm"))
screen$wald <- list(item = I(rownames(wald)), x2 = I(wald$W), df = I(wald$df), p = I(wald$p),
                    adj_p = I(wald$adj_p))
screen$base_loglik <- list(invariant = loglik_of(base_inv), anchor = loglik_of(base_anchor),
                           anchor_se = loglik_of(base_anchor_se))
}

# ---- data "three" -----------------------------------------------------------------------------
n3 <- 800L
a3 <- matrix(runif(8, 0.9, 1.7))
d3 <- matrix(seq(-1, 1, length.out = 8))
d3_g3 <- d3
d3_g3[3] <- d3[3] + 0.8
dat3 <- rbind(simdata(a3, d3, n3, itemtype = "dich"),
              simdata(a3, d3, n3, itemtype = "dich", mu = 0.3),
              simdata(a3, d3_g3, n3, itemtype = "dich", mu = -0.2, sigma = matrix(0.8)))
colnames(dat3) <- sprintf("q%02d", 1:8)
dat3 <- as.data.frame(dat3)
grp3 <- factor(rep(c("g1", "g2", "g3"), each = n3), levels = c("g1", "g2", "g3"))
if (cached) {
  three <- saved$three
} else {
  inv3 <- mg(dat3, grp3, "2PL", c(colnames(dat3), "free_means", "free_var"))
  stopifnot(extract.mirt(inv3, "converged"))
  three <- list(
    base_loglik = loglik_of(inv3),
    all_others = run_dif(inv3, "drop", 1:8, which_par = c("a1", "d")),
    all_others_intercepts = run_dif(inv3, "drop", 1:8, which_par = "d")
  )
  if (nzchar(cache)) saveRDS(list(screen = screen, three = three), cache)
}

# ---- effect sizes -----------------------------------------------------------------------------
es_settings <- list(quadpts = 201, theta_lim = c(-6, 6))
partial <- mg(dat, grp, itemtype, c(setdiff(items, dif_items), "free_means", "free_var"))
stopifnot(extract.mirt(partial, "converged"))

group_estimates <- function(m) {
  out <- list()
  for (g in levels(grp)) {
    cf <- coef(extract.group(m, g), simplify = TRUE)
    pars <- lapply(items, function(i) {
      v <- cf$items[i, ]
      v <- v[!is.na(v)]
      names(v)[names(v) == "a1"] <- "a"
      as.list(v[!names(v) %in% c("g", "u")])
    })
    out[[g]] <- list(params = setNames(pars, items),
                     latent = list(mean = unname(cf$means[1]), var = unname(cf$cov[1, 1])))
  }
  out
}
drf <- function(den, dif) {
  # DRF()'s grid: quadpts points on theta_lim.
  theta <- matrix(seq(-6, 6, length.out = 201))
  r <- mirt:::calc_DRFs(partial, theta, DIF = dif, DIF.cats = FALSE, plot = FALSE,
                        max_score = NULL, focal_items = seq_along(items), details = NULL,
                        den.type = den, best_fitting = TRUE, whc_grp = 1:2)
  if (dif) {
    r <- matrix(r, length(items))
    return(list(item = I(items), signed = I(r[, 1]), unsigned = I(r[, 2])))
  }
  list(signed = unname(r["sDRF"]), unsigned = unname(r["uDRF"]))
}
drf_item <- function(den) drf(den, TRUE)
drf_test <- function(den) drf(den, FALSE)
eap <- fscores(partial, method = "EAP", quadpts = 201, theta_lim = c(-6, 6))
theta_focal <- eap[grp == "focal", , drop = FALSE]
es_item <- empirical_ES(partial, Theta.focal = theta_focal, DIF = TRUE)
es_test <- empirical_ES(partial, Theta.focal = theta_focal, DIF = FALSE)
rmsd_of <- function(m) {
  r <- RMSD_DIF(m, probfun = FALSE)
  lapply(r, function(x) unname(x[items, 1]))
}
effect_sizes <- list(
  partial = group_estimates(partial),
  invariant = group_estimates(base_inv),
  free_items = dif_items,
  eap_focal = unname(theta_focal[, 1]),
  drf_focal = drf_item("focal"),
  drf_reference = drf_item("reference"),
  dtf_focal = drf_test("focal"),
  dtf_reference = drf_test("reference"),
  essd = I(unname(es_item$ESSD)),
  etssd = es_test$Value[es_test$`Effect Size` == "ETSSD"],
  rmsd_partial = rmsd_of(partial),
  rmsd_invariant = rmsd_of(base_inv)
)

pack <- function(data, group) {
  key <- apply(cbind(as.character(group), as.matrix(data)), 1, paste, collapse = " ")
  first <- !duplicated(key)
  counts <- as.vector(table(factor(key, levels = key[first])))
  list(patterns = unname(as.matrix(data)[first, , drop = FALSE]),
       pattern_groups = as.character(group)[first], counts = counts)
}

res <- list(
  meta = list(case = "v49_dif_screen", r_version = R.version.string,
              mirt_version = as.character(packageVersion("mirt")), TOL = 1e-10, quadpts = 81,
              theta_lim = c(-8, 8), grid_check = range(base_inv@Model$Theta),
              accelerate = "none", optimizer = "NR", seq_stat = 0.05, p_adjust = "BH",
              effect_size_quadpts = 201, effect_size_theta_lim = c(-6, 6), converged = TRUE),
  screen_data = c(list(items = items, itemtype = itemtype, n_categories = unname(
    extract.mirt(base_inv, "K")), anchors = anchors, group_names = levels(grp)),
    pack(dat, grp)),
  three_data = c(list(items = colnames(dat3), group_names = levels(grp3)), pack(dat3, grp3)),
  screen = screen,
  three = three,
  effect_sizes = effect_sizes
)
path <- file.path("tests", "reference", "v49_dif_screen.json")
write_json(res, path, auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null")
cat("all_others flagged:", screen$all_others$item[screen$all_others$adj_p < 0.05], "\n")
for (nm in c("sequential_free", "sequential_constrain")) {
  for (r in screen[[nm]]$runs) cat(nm, "max_run", r$max_run, ":", r$item, "\n")
}
cat("wald x2:", round(screen$wald$x2, 3), "\n")
cat("wrote", path, "\n")
