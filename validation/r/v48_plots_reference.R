# V48 reference values (VALIDATION.md V48: the numbers behind the plots; DESIGN.md §20) from R
# mirt and TAM.
# Run from the repository root:  Rscript validation/r/v48_plots_reference.R
# Writes tests/reference/v48_plots.json. CI does not run R; the JSON is committed.
#
# Per case (and per group for the two-group model) the file stores mirt's estimates, the data
# (unique patterns plus the pattern of every row, so the row order is kept) and
#   - fscores(method = "EAP") of every row                                    -> eap
#   - itemfit(fit_stats = "X2", return.tables = TRUE, group.bins = n_bins, mincell.X2 = 0,
#             Theta = eap, which.items = j) for every item j: per theta bin the bin label
#     "theta = <mean EAP rounded to 4 digits>", the observed counts and N * P(mean EAP) of every
#     category (mincell.X2 = 0 turns mirt's cell collapsing off)          -> empirical
#   - probtrace() of every item on a theta grid                               -> probtrace
#   - Thurstonian thresholds: for every item and k = 1 .. K - 1 the root of
#     sum(probtrace(item, theta)[k + 1 .. K]) = 0.5 by uniroot(tol = 1e-13), NA without a
#     sign change on [-20, 20]                                                -> thresholds
#   - coef(IRTpars = TRUE) b / b1.. of the 2PL and graded items              -> irt_b
# The case gpcm_tam is a TAM fit (tam.mml.2pl(irtmodel = "GPCM")): its slopes and intercepts
# (from B and AXsi) and tam.threshold() (bisection to 1e-4).
suppressMessages({
  library(mirt)
  library(TAM)
  library(jsonlite)
})

quadpts <- 61L
theta_lim <- c(-6, 6)
theta_grid <- c(-6, seq(-4, 4, by = 0.5), 6)
Theta_grid <- matrix(theta_grid)
n_bins <- 10L

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

compress <- function(dat, grp = NULL) {
  key <- apply(cbind(dat, grp), 1, paste, collapse = ",")
  first <- !duplicated(key)
  list(first = first, row_pattern = match(key, key[first]) - 1L)
}

thurstonian <- function(item, n_cat) {
  sapply(seq_len(n_cat - 1L), function(k) {
    f <- function(t) sum(probtrace(item, matrix(t))[1, (k + 1L):n_cat]) - 0.5
    lo <- f(-20); hi <- f(20)
    if (sign(lo) == sign(hi)) return(NA_real_)
    uniroot(f, c(-20, 20), tol = 1e-13, maxiter = 10000L)$root
  })
}

irt_b <- function(m_single, itemtype) {
  cf <- coef(m_single, IRTpars = TRUE, simplify = TRUE)$items
  out <- list()
  for (j in seq_along(itemtype)) {
    if (itemtype[j] %in% c("2PL", "graded")) {
      v <- cf[j, grep("^b", colnames(cf))]
      out[[rownames(cf)[j]]] <- unname(v[!is.na(v)])
    }
  }
  out
}

# itemfit(return.tables = TRUE) returns the table of the first item in which.items only.
empirical_tables <- function(m, eap, n_items) {
  lapply(seq_len(n_items), function(j) {
    tabs <- itemfit(m, fit_stats = "X2", return.tables = TRUE, group.bins = n_bins,
                    mincell.X2 = 0, Theta = eap, which.items = j)
    tabs <- Filter(Negate(is.null), tabs)
    list(label = names(tabs),
         observed = unname(lapply(tabs, function(t) t$Observed)),
         expected = unname(lapply(tabs, function(t) t$Expected)))
  })
}

curves <- function(m_single, itemtype) {
  n_items <- extract.mirt(m_single, "nitems")
  K <- extract.mirt(m_single, "K")
  list(
    theta = theta_grid,
    probtrace = lapply(seq_len(n_items), function(j) {
      unname(probtrace(extract.item(m_single, j), Theta_grid))
    }),
    thresholds = lapply(seq_len(n_items), function(j) {
      thurstonian(extract.item(m_single, j), K[j])
    }),
    irt_b = irt_b(m_single, itemtype)
  )
}

single_case <- function(description, dat, itemtype) {
  itemtype <- rep(itemtype, length.out = ncol(dat))
  m <- fit_single(dat, itemtype)
  cf <- coef(m, simplify = TRUE)
  cp <- compress(dat)
  eap <- fscores(m, method = "EAP", quadpts = quadpts, theta_lim = theta_lim)
  c(
    list(description = description, n = nrow(dat), items = colnames(dat), itemtype = itemtype,
         n_categories = unname(extract.mirt(m, "K")),
         patterns = unname(as.matrix(dat)[cp$first, , drop = FALSE]),
         row_pattern = cp$row_pattern,
         params = item_params(m, itemtype),
         latent = list(mean = unname(cf$means[1]), var = unname(cf$cov[1, 1])),
         eap = unname(eap[, 1]),
         empirical = empirical_tables(m, eap, ncol(dat))),
    curves(m, itemtype)
  )
}

# --- data -----------------------------------------------------------------------------------
set.seed(20261126)

# N = 1003: 1003 %% 10 = 3, so mirt puts 1 extra respondent in the first bin and 2 in the last.
J <- 10L
a2 <- matrix(runif(J, 0.8, 2.0)); d2 <- matrix(seq(-1.8, 1.8, length.out = J))
dat_2pl <- simdata(a2, d2, 1003L, itemtype = "dich")
dat_2pl_missing <- simdata(a2, d2, 1200L, itemtype = "dich")
dat_2pl_missing[matrix(runif(length(dat_2pl_missing)) < 0.1, nrow(dat_2pl_missing))] <- NA

Jp <- 6L
ap <- matrix(runif(Jp, 0.9, 1.9))
dp <- t(sapply(seq_len(Jp), function(j) c(1.4, 0.1, -1.2) + (j - 3.5) * 0.2))
dat_grm <- simdata(ap, dp, 1500L, itemtype = "graded")

# Mixed format: 3 x 2PL, 3 x 3PL, 2 x GPCM (K = 4), 2 x nominal (K = 3).
theta_mix <- matrix(rnorm(3000L))
mix_parts <- list(
  simdata(matrix(runif(3, 0.9, 1.8)), matrix(c(-1, 0, 1)), Theta = theta_mix, itemtype = "dich"),
  simdata(matrix(runif(3, 1.2, 2.0)), matrix(c(0.2, 0.8, 1.4)), Theta = theta_mix,
          itemtype = "dich", guess = c(0.15, 0.2, 0.25)),
  simdata(matrix(runif(2, 0.9, 1.6)), cbind(0, dp[1:2, ]), Theta = theta_mix,
          itemtype = "gpcm"),
  simdata(matrix(runif(2, 0.9, 1.4)), cbind(0, c(0.5, -0.3), c(0.2, 0.6)), Theta = theta_mix,
          itemtype = "nominal", nominal = rbind(c(0, 0.7, 2), c(0, 1.4, 2)))
)
dat_mixed <- do.call(cbind, mix_parts)
colnames(dat_mixed) <- sprintf("q%02d", seq_len(ncol(dat_mixed)))
types_mixed <- c(rep("2PL", 3), rep("3PL", 3), rep("gpcm", 2), rep("nominal", 2))

cases <- list(
  twopl = single_case("2PL, J = 10, N = 1003", dat_2pl, "2PL"),
  twopl_missing = single_case("2PL, J = 10, N = 1200, 10% missing", dat_2pl_missing, "2PL"),
  grm = single_case("GRM, K = 4, J = 6, N = 1500", dat_grm, "graded"),
  mixed = single_case("mixed: 3 2PL, 3 3PL, 2 GPCM (K = 4), 2 nominal (K = 3), N = 3000",
                      dat_mixed, types_mixed)
)

# --- two groups: 5 x 2PL + 3 x GRM (K = 4), all items invariant, focal mean / variance free ---
Ng <- 805L
a_g <- matrix(runif(8, 0.9, 1.8))
d_g <- matrix(NA_real_, 8, 3)
d_g[1:5, 1] <- seq(-1.2, 1.2, length.out = 5)
d_g[6:8, ] <- dp[1:3, ]
types_g <- c(rep("2PL", 5), rep("graded", 3))
sim_types <- c(rep("dich", 5), rep("graded", 3))
dat_g <- rbind(simdata(a_g, d_g, Ng, itemtype = sim_types),
               simdata(a_g, d_g, Ng, itemtype = sim_types, mu = 0.5, sigma = matrix(1.6)))
grp <- factor(rep(c("ref", "focal"), each = Ng), levels = c("ref", "focal"))
mg <- multipleGroup(dat_g, 1, group = grp, itemtype = types_g,
                    invariance = c(colnames(dat_g), "free_means", "free_var"), TOL = 1e-8,
                    quadpts = quadpts, verbose = FALSE,
                    technical = list(NCYCLES = 20000L, theta_lim = theta_lim))
stopifnot(isTRUE(extract.mirt(mg, "converged")))
cp <- compress(dat_g, as.character(grp))
eap_g <- fscores(mg, method = "EAP", quadpts = quadpts, theta_lim = theta_lim)
groups <- list()
for (g in levels(grp)) {
  sg <- extract.group(mg, g)
  cf <- coef(sg, simplify = TRUE)
  groups[[g]] <- c(
    list(n = sum(grp == g), params = item_params(sg, types_g),
         latent = list(mean = unname(cf$means[1]), var = unname(cf$cov[1, 1]))),
    curves(sg, types_g)
  )
}
# itemfit(return.tables = TRUE) refuses a multipleGroup fit; its per-group itemfit() runs on
# extract.group() with that group's rows of Theta, which is what is done here.
emp_g <- lapply(seq_len(ncol(dat_g)), function(j) {
  setNames(lapply(levels(grp), function(g) {
    tabs <- itemfit(extract.group(mg, g), fit_stats = "X2", return.tables = TRUE,
                    group.bins = n_bins, mincell.X2 = 0,
                    Theta = eap_g[grp == g, , drop = FALSE], which.items = j)
    t <- Filter(Negate(is.null), tabs)
    list(label = names(t),
         observed = unname(lapply(t, function(x) x$Observed)),
         expected = unname(lapply(t, function(x) x$Expected)))
  }), levels(grp))
})
twogroup <- list(
  description = "two groups: 5 2PL + 3 GRM (K = 4), all items invariant, focal N(mu, s2) free",
  n = nrow(dat_g), items = colnames(dat_g), itemtype = types_g,
  n_categories = unname(extract.mirt(mg, "K")),
  patterns = unname(as.matrix(dat_g)[cp$first, , drop = FALSE]),
  pattern_groups = as.character(grp)[cp$first], row_pattern = cp$row_pattern,
  group_names = levels(grp), groups = groups, eap = unname(eap_g[, 1]), empirical = emp_g
)

# --- TAM: GPCM (K = 4), its own estimates and tam.threshold() ------------------------------
dat_tam <- simdata(ap, cbind(0, dp), 1500L, itemtype = "gpcm")
colnames(dat_tam) <- sprintf("p%d", seq_len(ncol(dat_tam)))
tam <- tam.mml.2pl(dat_tam, irtmodel = "GPCM", verbose = FALSE)
K_tam <- 4L
# TAM: P_k proportional to exp(B[i, k + 1, 1] theta + AXsi[i, k + 1]), B[i, k + 1, 1] = k a_i.
tam_params <- setNames(lapply(seq_len(ncol(dat_tam)), function(i) {
  c(list(a = tam$B[i, 2, 1]), setNames(as.list(tam$AXsi[i, 2:K_tam]), paste0("d", 1:(K_tam - 1))))
}), colnames(dat_tam))
stopifnot(all(abs(tam$B[, 3, 1] - 2 * tam$B[, 2, 1]) < 1e-12))
gpcm_tam <- list(
  description = "GPCM (K = 4), J = 6, N = 1500, TAM tam.mml.2pl(irtmodel = 'GPCM')",
  items = colnames(dat_tam), n_categories = rep(K_tam, ncol(dat_tam)), params = tam_params,
  latent = list(mean = 0, var = unname(tam$variance[1, 1])),
  tam_threshold = unname(lapply(seq_len(ncol(dat_tam)), function(i) {
    unname(tam.threshold(tam)[i, ])
  }))
)

bundle <- list(
  meta = list(
    case = "v48_plots",
    description = paste("V48: the numbers behind the plots vs mirt itemfit(return.tables = TRUE)",
                        "/ probtrace / Thurstonian thresholds by uniroot / coef(IRTpars = TRUE)",
                        "and TAM tam.threshold"),
    r_version = R.version.string,
    mirt_version = as.character(packageVersion("mirt")),
    tam_version = as.character(packageVersion("TAM")),
    quadpts = quadpts, theta_lim = theta_lim, n_bins = n_bins,
    # Every mirt fit above is asserted to have converged (fit_single(), multipleGroup()).
    converged = TRUE
  ),
  cases = cases,
  twogroup = twogroup,
  gpcm_tam = gpcm_tam
)
path <- file.path("tests", "reference", "v48_plots.json")
write_json(bundle, path, auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null")
cat("wrote", path, "\n")
