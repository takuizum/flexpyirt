# M7 reference values (VALIDATION.md V37: information, expected score, reliability and
# summed-score EAP; DESIGN.md §18.2) from R mirt.
# Run from the repository root:  Rscript validation/r/m7_information_reference.R
# Writes tests/reference/v37_information.json. CI does not run R; the JSON is committed.
#
# Per case (and per group for the two-group model) the file stores mirt's estimates, the data
# as unique patterns with counts, and
#   - iteminfo() per item, testinfo(), expected.item() per item, expected.test() on a theta grid
#   - marginal_rxx() with R's default integrate() tolerance  -> marginal_rxx
#     and recomputed with rel.tol = 1e-12                    -> marginal_rxx_tight
#   - the scale-invariant form  int s2*I/(s2*I + 1) dnorm(theta; mu, s) dtheta  with testinfo(),
#     rel.tol = 1e-12                                         -> marginal_invariant
#   - (two-group model, per group) marginal_rxx() of the standardized equivalent single-group
#     model (a* = a*sigma, d* = d + a*mu, N(0, 1))            -> marginal_rxx_standardized(_tight)
#   - empirical_rxx(fscores(method = "EAP", full.scores.SE = TRUE))
#   - fscores(method = "EAPsum", full.scores = FALSE): Sum.Scores, F1, SE_F1, expected
#   - fscores(method = "EAPsum", full.scores = TRUE, full.scores.SE = TRUE) per unique pattern
# The case twopl_missing has missing responses: mirt refuses the EAPsum table there, so it
# stores only the per-pattern EAPsum scores (NA for incomplete patterns) and empirical_rxx().
source(file.path("validation", "r", "setup.R"))  # mirt from GitHub; mirt_version / mirt_sha
suppressMessages({
  library(mirt)
  library(jsonlite)
})

quadpts <- 61L
theta_lim <- c(-6, 6)
theta_grid <- c(-6, seq(-4, 4, by = 0.5), 6)
Theta <- matrix(theta_grid)

fit_single <- function(dat, itemtype) {
  m <- mirt(dat, 1, itemtype = itemtype, TOL = 1e-8, verbose = FALSE, quadpts = quadpts,
            technical = list(NCYCLES = 20000L, theta_lim = theta_lim))
  stopifnot(isTRUE(extract.mirt(m, "converged")))
  m
}

# Item parameters in flexpyirt's names: a, d / g / u, d1.., ak1..ak{K-2} (nominal only).
item_params <- function(m_single, itemtype) {
  cf <- coef(m_single, simplify = TRUE)$items
  items <- rownames(cf)
  out <- lapply(seq_along(items), function(j) {
    v <- cf[j, ]
    v <- v[!is.na(v)]
    names(v)[names(v) == "a1"] <- "a"
    k <- extract.mirt(m_single, "K")[j]
    drop <- c("d0", "ak0", paste0("ak", k - 1L))
    if (itemtype[j] == "gpcm") drop <- c(drop, grep("^ak", names(v), value = TRUE))
    as.list(v[!names(v) %in% drop])
  })
  setNames(out, items)
}

tight <- function(fn) {
  integrate(fn, lower = -Inf, upper = Inf, rel.tol = 1e-12, abs.tol = 0,
            subdivisions = 2000L)$value
}

# Everything that needs only the (single-group) model object.
curves_and_marginal <- function(m_single, mu, s2) {
  n_items <- extract.mirt(m_single, "nitems")
  info <- sapply(seq_len(n_items), function(j) iteminfo(extract.item(m_single, j), Theta))
  expected <- sapply(seq_len(n_items), function(j) expected.item(extract.item(m_single, j), Theta))
  ti <- function(theta) testinfo(m_single, matrix(theta))
  list(
    theta = theta_grid,
    iteminfo = unname(info),
    testinfo = testinfo(m_single, Theta),
    expected_item = unname(expected),
    expected_test = expected.test(m_single, Theta),
    marginal_rxx = marginal_rxx(m_single),
    marginal_rxx_tight = tight(function(theta) ti(theta) / (ti(theta) + s2) * dnorm(theta)),
    marginal_invariant = tight(function(theta) {
      s2 * ti(theta) / (s2 * ti(theta) + 1) * dnorm(theta, mu, sqrt(s2))
    })
  )
}

eapsum_table <- function(tab) {
  list(sum_scores = tab[, "Sum.Scores"], eap = tab[, "F1"], eap_sd = tab[, "SE_F1"],
       expected = tab[, "expected"], observed = tab[, "observed"])
}

compress <- function(dat, grp = NULL) {
  key <- apply(cbind(dat, grp), 1, paste, collapse = ",")
  first <- !duplicated(key)
  list(first = first, counts = as.integer(table(factor(key, levels = key[first]))))
}

single_case <- function(description, dat, itemtype) {
  itemtype <- rep(itemtype, length.out = ncol(dat))
  m <- fit_single(dat, itemtype)
  cf <- coef(m, simplify = TRUE)
  cp <- compress(dat)
  out <- c(
    list(description = description, n = nrow(dat), items = colnames(dat), itemtype = itemtype,
         n_categories = unname(extract.mirt(m, "K")),
         patterns = unname(as.matrix(dat)[cp$first, , drop = FALSE]), counts = cp$counts,
         params = item_params(m, itemtype),
         latent = list(mean = unname(cf$means[1]), var = unname(cf$cov[1, 1]))),
    curves_and_marginal(m, 0, 1)
  )
  eap <- fscores(m, method = "EAP", full.scores.SE = TRUE, quadpts = quadpts,
                 theta_lim = theta_lim)
  out$empirical_rxx <- unname(empirical_rxx(eap))
  full <- fscores(m, method = "EAPsum", full.scores = TRUE, full.scores.SE = TRUE,
                  quadpts = quadpts, theta_lim = theta_lim, verbose = FALSE)
  out$eapsum_full <- list(eap = full[cp$first, "F1"], eap_sd = full[cp$first, "SE_F1"])
  if (!anyNA(dat)) {
    out$eapsum <- eapsum_table(fscores(m, method = "EAPsum", full.scores = FALSE,
                                       quadpts = quadpts, theta_lim = theta_lim,
                                       verbose = FALSE))
  }
  out
}

# --- data -----------------------------------------------------------------------------------
set.seed(20261125)
N <- 2000L

J <- 12L
a2 <- matrix(runif(J, 0.8, 2.0)); d2 <- matrix(seq(-1.8, 1.8, length.out = J))
dat_2pl <- simdata(a2, d2, N, itemtype = "dich")
dat_2pl_missing <- dat_2pl[1:800, ]
dat_2pl_missing[matrix(runif(length(dat_2pl_missing)) < 0.08, nrow(dat_2pl_missing))] <- NA

a3 <- matrix(runif(J, 1.0, 2.2)); d3 <- matrix(seq(-1.0, 2.0, length.out = J))
dat_3pl <- simdata(a3, d3, 5000L, itemtype = "dich", guess = rep(c(0.15, 0.2, 0.25), 4))

Jp <- 8L
ap <- matrix(runif(Jp, 0.9, 1.9))
dp <- t(sapply(seq_len(Jp), function(j) c(1.4, 0.1, -1.2) + (j - 4.5) * 0.15))
dat_grm <- simdata(ap, dp, N, itemtype = "graded")
dat_gpcm <- simdata(ap, cbind(0, dp), N, itemtype = "gpcm")

Jn <- 6L
an <- matrix(runif(Jn, 0.8, 1.5))
akn <- t(sapply(seq_len(Jn), function(j) c(0, runif(2, 0.4, 2.6), 3)))
dn <- cbind(0, matrix(rnorm(Jn * 3, 0, 0.7), Jn))
dat_nominal <- simdata(an, dn, 3000L, itemtype = "nominal", nominal = akn)

# Mixed format: 4 x 2PL, 3 x 3PL, 3 x GRM (K = 4), 2 x GPCM (K = 3), 2 x nominal (K = 3).
theta_mix <- matrix(rnorm(5000L))
mix_parts <- list(
  simdata(matrix(runif(4, 0.9, 1.8)), matrix(c(-1, -0.3, 0.4, 1.1)), Theta = theta_mix,
          itemtype = "dich"),
  simdata(matrix(runif(3, 1.2, 2.0)), matrix(c(0.2, 0.8, 1.4)), Theta = theta_mix,
          itemtype = "dich", guess = c(0.2, 0.2, 0.25)),
  simdata(matrix(runif(3, 0.9, 1.8)), dp[1:3, ], Theta = theta_mix, itemtype = "graded"),
  simdata(matrix(runif(2, 0.9, 1.6)), cbind(0, c(0.6, 0.2), c(-0.4, 0.5)), Theta = theta_mix,
          itemtype = "gpcm"),
  simdata(matrix(runif(2, 0.9, 1.4)), cbind(0, c(0.5, -0.3), c(0.2, 0.6)), Theta = theta_mix,
          itemtype = "nominal", nominal = rbind(c(0, 0.7, 2), c(0, 1.4, 2)))
)
dat_mixed <- do.call(cbind, mix_parts)
colnames(dat_mixed) <- sprintf("q%02d", seq_len(ncol(dat_mixed)))
types_mixed <- c(rep("2PL", 4), rep("3PL", 3), rep("graded", 3), rep("gpcm", 2),
                 rep("nominal", 2))

cases <- list(
  twopl = single_case("2PL, J = 12, N = 2000", dat_2pl, "2PL"),
  twopl_missing = single_case("2PL, J = 12, N = 800, 8% missing", dat_2pl_missing, "2PL"),
  threepl = single_case("3PL, J = 12, N = 5000", dat_3pl, "3PL"),
  grm = single_case("GRM, K = 4, J = 8, N = 2000", dat_grm, "graded"),
  gpcm = single_case("GPCM, K = 4, J = 8, N = 2000", dat_gpcm, "gpcm"),
  nominal = single_case("nominal, K = 4, J = 6, N = 3000", dat_nominal, "nominal"),
  mixed = single_case("mixed: 4 2PL, 3 3PL, 3 GRM (K = 4), 2 GPCM (K = 3), 2 nominal (K = 3)",
                      dat_mixed, types_mixed)
)

# --- two groups: 6 x 2PL + 4 x GRM (K = 4), all items invariant, focal mean / variance free ---
Ng <- 1500L
a_g <- matrix(runif(10, 0.9, 1.8))
d_g <- matrix(NA_real_, 10, 3)
d_g[1:6, 1] <- seq(-1.2, 1.2, length.out = 6)
d_g[7:10, ] <- dp[1:4, ]
types_g <- c(rep("2PL", 6), rep("graded", 4))
sim_types <- c(rep("dich", 6), rep("graded", 4))
dat_g <- rbind(simdata(a_g, d_g, Ng, itemtype = sim_types),
               simdata(a_g, d_g, Ng, itemtype = sim_types, mu = 0.5, sigma = matrix(1.6)))
grp <- factor(rep(c("ref", "focal"), each = Ng), levels = c("ref", "focal"))
mg <- multipleGroup(dat_g, 1, group = grp, itemtype = types_g,
                    invariance = c(colnames(dat_g), "free_means", "free_var"), TOL = 1e-8,
                    quadpts = quadpts, verbose = FALSE,
                    technical = list(NCYCLES = 20000L, theta_lim = theta_lim))
stopifnot(isTRUE(extract.mirt(mg, "converged")))
cp <- compress(dat_g, as.character(grp))
eap_g <- fscores(mg, method = "EAP", full.scores.SE = TRUE, quadpts = quadpts,
                 theta_lim = theta_lim)
full_g <- fscores(mg, method = "EAPsum", full.scores = TRUE, full.scores.SE = TRUE,
                  quadpts = quadpts, theta_lim = theta_lim, verbose = FALSE)
tab_g <- fscores(mg, method = "EAPsum", full.scores = FALSE, quadpts = quadpts,
                 theta_lim = theta_lim, verbose = FALSE)
groups <- list()
for (g in levels(grp)) {
  sg <- extract.group(mg, g)
  cf <- coef(sg, simplify = TRUE)
  mu <- unname(cf$means[1]); s2 <- unname(cf$cov[1, 1])
  # Standardized equivalent single-group model: theta* = (theta - mu) / sigma.
  dat_one <- dat_g[grp == g, ]
  sv <- mirt(dat_one, 1, itemtype = types_g, pars = "values")
  for (i in colnames(dat_g)) {
    a <- cf$items[i, "a1"]
    sv$value[sv$item == i & sv$name == "a1"] <- a * sqrt(s2)
    for (nm in c("d", "d1", "d2", "d3")) {
      if (nm %in% colnames(cf$items) && !is.na(cf$items[i, nm])) {
        sv$value[sv$item == i & sv$name == nm] <- cf$items[i, nm] + a * mu
      }
    }
  }
  sv$est <- FALSE
  m_std <- mirt(dat_one, 1, itemtype = types_g, pars = sv, verbose = FALSE, quadpts = quadpts,
                technical = list(theta_lim = theta_lim))
  ti_std <- function(theta) testinfo(m_std, matrix(theta))
  groups[[g]] <- c(
    list(n = sum(grp == g), params = item_params(sg, types_g),
         latent = list(mean = mu, var = s2)),
    curves_and_marginal(sg, mu, s2),
    list(marginal_rxx_standardized = marginal_rxx(m_std),
         marginal_rxx_standardized_tight = tight(function(theta) {
           ti_std(theta) / (ti_std(theta) + 1) * dnorm(theta)
         }),
         empirical_rxx = unname(empirical_rxx(eap_g[grp == g, , drop = FALSE])),
         eapsum = eapsum_table(tab_g[[g]]))
  )
}
twogroup <- list(
  description = "two groups: 6 2PL + 4 GRM (K = 4), all items invariant, focal N(mu, s2) free",
  items = colnames(dat_g), itemtype = types_g, n_categories = unname(extract.mirt(mg, "K")),
  patterns = unname(as.matrix(dat_g)[cp$first, , drop = FALSE]),
  pattern_groups = as.character(grp)[cp$first], counts = cp$counts,
  group_names = levels(grp), groups = groups,
  empirical_rxx_pooled = unname(empirical_rxx(eap_g)),
  eapsum_full = list(eap = full_g[cp$first, "F1"], eap_sd = full_g[cp$first, "SE_F1"])
)

bundle <- list(
  meta = list(
    case = "v37_information",
    description = paste("V37: information, expected score, reliability and summed-score EAP",
                        "vs mirt iteminfo / testinfo / expected.item / expected.test /",
                        "marginal_rxx / empirical_rxx / fscores(method = 'EAPsum')"),
    r_version = R.version.string,
    mirt_version = mirt_version_string(), mirt_sha = mirt_commit_sha(),
    quadpts = quadpts, theta_lim = theta_lim,
    # Every mirt fit above is asserted to have converged (fit_single(), multipleGroup()).
    converged = TRUE
  ),
  cases = cases,
  twogroup = twogroup
)
path <- file.path("tests", "reference", "v37_information.json")
write_json(bundle, path, auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null")
cat("wrote", path, "\n")
