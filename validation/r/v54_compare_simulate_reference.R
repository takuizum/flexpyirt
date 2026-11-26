# Model comparison and the deterministic part of simulation (VALIDATION.md V54) from R mirt.
# Run from the repository root:  Rscript validation/r/v54_compare_simulate_reference.R
# Writes tests/reference/v54_compare_fits.json and tests/reference/v54_simulate.json. CI does not
# run R; the JSON is committed.
#
# compare_fits(): mirt's anova(mod1, mod2, ...) for
#   - dichotomous: Rasch vs 2PL vs 3PL on the V11a data (3PL, J = 15, N = 5000),
#   - polytomous:  PCM (itemtype = "Rasch") vs GPCM on the V8b data (J = 10, K = 4, N = 3000),
#   - two groups:  fully invariant 2PL vs anchors q01-q05 only, and vs the last item free across
#                  groups; focal mean and variance free; V4 data (81 points on [-8, 8], as V4 / V17).
# Every fit uses optimizer = "NR", accelerate = "none" and TOL = 1e-10, as the source cases.
#
# simulate_responses(): random streams cannot match, so the reference is deterministic. For given
# item parameters, theta and uniform numbers u, the response is the number of cumulative
# category probabilities below u,  x = sum_{k < K-1} [u > F_k(theta)],  with F from mirt's
# probtrace(). Stored per case: the item parameters (flexpyirt names), theta, u, probtrace() and
# the responses.
suppressMessages({
  library(mirt)
  library(jsonlite)
})

common <- list(TOL = 1e-10, accelerate = "none", optimizer = "NR", verbose = FALSE)

read_case <- function(name) {
  r <- fromJSON(file.path("tests", "reference", paste0(name, ".json")), simplifyVector = FALSE)
  pats <- do.call(rbind, lapply(r$patterns, unlist))
  colnames(pats) <- unlist(r$items)
  reps <- rep(seq_len(nrow(pats)), unlist(r$counts))
  list(raw = r, dat = as.data.frame(pats[reps, , drop = FALSE]),
       group = unlist(r$pattern_groups)[reps])
}

single <- function(dat, itemtype, quadpts = 61, theta_lim = c(-6, 6)) {
  m <- do.call(mirt, c(list(dat, 1, itemtype = itemtype, quadpts = quadpts,
                            technical = list(NCYCLES = 50000, theta_lim = theta_lim)), common))
  stopifnot(isTRUE(extract.mirt(m, "converged")))
  m
}

anova_table <- function(models, labels) {
  # Call anova() with symbols: mirt names the rows by deparsing the call.
  tab <- if (length(models) == 2) {
    anova(models[[1]], models[[2]], verbose = FALSE)
  } else {
    anova(models[[1]], models[[2]], models[[3]], verbose = FALSE)
  }
  list(model = labels, n_params = unname(sapply(models, extract.mirt, "nest")),
       loglik = tab$logLik, aic = tab$AIC, sabic = tab$SABIC, hq = tab$HQ, bic = tab$BIC,
       x2 = tab$X2, df = tab$df, p = tab$p,
       n_obs = nrow(extract.mirt(models[[1]], "data")))
}

# ---- compare_fits ------------------------------------------------------------------------------
v11 <- read_case("v11_3pl")
dich <- anova_table(list(single(v11$dat, "Rasch"), single(v11$dat, "2PL"),
                         single(v11$dat, "3PL")), c("Rasch", "2PL", "3PL"))

v8 <- read_case("v8_sim_gpcm")
poly <- anova_table(list(single(v8$dat, "Rasch"), single(v8$dat, "gpcm")), c("PCM", "GPCM"))

v4 <- read_case("v4_twogroup_2pl")
items4 <- colnames(v4$dat)
anchors <- unlist(v4$raw$anchors)
grp <- factor(v4$group, levels = c("ref", "focal"))
mg <- function(invariance) {
  m <- do.call(multipleGroup, c(list(v4$dat, 1, group = grp, itemtype = "2PL",
                                     invariance = invariance, quadpts = 81,
                                     technical = list(NCYCLES = 50000, theta_lim = c(-8, 8))),
                                common))
  stopifnot(isTRUE(extract.mirt(m, "converged")))
  m
}
invariant <- mg(c(items4, "free_means", "free_var"))
twogroup <- anova_table(list(invariant, mg(c(anchors, "free_means", "free_var"))),
                        c("invariant", "anchored"))
# One item (the last) free across groups: a test without DIF, so p is not 0.
last_free <- anova_table(list(invariant, mg(c(head(items4, -1), "free_means", "free_var"))),
                         c("invariant", "last_free"))

compare <- list(
  meta = list(case = "v54_compare_fits", r_version = R.version.string,
              mirt_version = as.character(packageVersion("mirt")), TOL = 1e-10,
              accelerate = "none", optimizer = "NR", converged = TRUE,
              sources = list(dichotomous = "v11_3pl", polytomous = "v8_sim_gpcm",
                             twogroup = "v4_twogroup_2pl")),
  anchors = anchors,
  cases = list(dichotomous = dich, polytomous = poly, twogroup = twogroup,
               twogroup_last_free = last_free)
)
path <- file.path("tests", "reference", "v54_compare_fits.json")
write_json(compare, path, auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null")
cat("wrote", path, "\n")

# ---- simulate_responses: deterministic inversion ----------------------------------------------
set.seed(20261126)

# A mirt object holding the given parameters (no estimation: TOL = NaN).
fixed_model <- function(model, itemtype, values, n_cat) {
  n_items <- length(itemtype)
  dat <- as.data.frame(sapply(n_cat, function(k) rep(seq_len(k) - 1L, length.out = 40)))
  colnames(dat) <- names(values)
  sv <- mirt(dat, model, itemtype = itemtype, pars = "values")
  for (j in seq_len(n_items)) {
    for (p in names(values[[j]])) {
      sv$value[sv$item == names(values)[j] & sv$name == p] <- values[[j]][[p]]
    }
  }
  mirt(dat, model, itemtype = itemtype, pars = sv, TOL = NaN, verbose = FALSE)
}

# Item parameters in flexpyirt's names (a / a1, a2; d / d1..; g; u; ak1..ak{K-2}).
flexpyirt_params <- function(m, itemtype) {
  cf <- coef(m, simplify = TRUE)$items
  lapply(seq_len(nrow(cf)), function(j) {
    v <- cf[j, ]
    v <- v[!is.na(v)]
    k <- extract.mirt(m, "K")[j]
    if (extract.mirt(m, "nfact") == 1) names(v)[names(v) == "a1"] <- "a"
    drop <- c("d0", "ak0", paste0("ak", k - 1L))
    if (itemtype[j] %in% c("gpcm", "graded", "2PL", "3PL", "4PL", "Rasch")) {
      drop <- c(drop, grep("^ak", names(v), value = TRUE))
    }
    if (itemtype[j] %in% c("2PL", "Rasch", "graded", "gpcm", "nominal")) drop <- c(drop, "g", "u")
    if (itemtype[j] == "3PL") drop <- c(drop, "u")
    as.list(v[!names(v) %in% drop])
  }) |> setNames(rownames(cf))
}

deterministic <- function(model, itemtype, values, n_cat, theta) {
  m <- fixed_model(model, itemtype, values, n_cat)
  n <- nrow(theta)
  u <- matrix(runif(n * length(itemtype)), n)
  traces <- list()
  resp <- matrix(0L, n, length(itemtype))
  for (j in seq_along(itemtype)) {
    pr <- probtrace(extract.item(m, j), theta)
    cum <- t(apply(pr, 1, cumsum))
    resp[, j] <- as.integer(rowSums(u[, j] > cum[, -ncol(cum), drop = FALSE]))
    traces[[j]] <- unname(pr)
  }
  colnames(resp) <- names(values)
  list(items = names(values), itemtype = itemtype, n_categories = n_cat,
       params = flexpyirt_params(m, itemtype), theta = unname(theta), u = unname(u),
       probtrace = setNames(traces, names(values)), responses = unname(resp))
}

uni_items <- c("Rasch", "2PL", "3PL", "4PL", "graded", "gpcm", "nominal", "2PL", "graded")
uni_k <- c(2L, 2L, 2L, 2L, 4L, 4L, 4L, 2L, 3L)
uni_values <- list(
  i1 = list(d = 0.3),
  i2 = list(a1 = 1.4, d = -0.5),
  i3 = list(a1 = 1.8, d = 0.2, g = 0.2),
  i4 = list(a1 = 1.2, d = -0.4, g = 0.15, u = 0.9),
  i5 = list(a1 = 1.1, d1 = 2.0, d2 = 0.3, d3 = -1.5),
  i6 = list(a1 = 0.9, d1 = 0.8, d2 = 1.1, d3 = -0.2),
  i7 = list(a1 = 1.0, ak1 = 0.7, ak2 = 2.4, d1 = 0.5, d2 = 0.9, d3 = -0.3),
  i8 = list(a1 = 0.6, d = 1.7),
  i9 = list(a1 = 2.2, d1 = 1.0, d2 = -1.2)
)
uni <- deterministic(1, uni_items, uni_values, uni_k, matrix(c(-6, -3, 0, 3, 6, rnorm(395))))

two_items <- c("2PL", "2PL", "2PL", "graded", "graded", "3PL")
two_k <- c(2L, 2L, 2L, 3L, 3L, 2L)
two_values <- list(
  j1 = list(a1 = 1.3, a2 = 0, d = 0.1),
  j2 = list(a1 = 0.8, a2 = 0, d = -0.6),
  j3 = list(a1 = 1.0, a2 = 0.7, d = 0.4),
  j4 = list(a1 = 0, a2 = 1.5, d1 = 1.2, d2 = -0.8),
  j5 = list(a1 = 0, a2 = 0.9, d1 = 0.2, d2 = -1.4),
  j6 = list(a1 = 0, a2 = 1.6, d = -0.2, g = 0.1)
)
two_model <- "F1 = 1-3\nF2 = 3-6\nCOV = F1*F2"
two <- deterministic(mirt.model(two_model), two_items, two_values, two_k,
                     matrix(rnorm(600), 300, 2))

simulate <- list(
  meta = list(case = "v54_simulate", r_version = R.version.string,
              mirt_version = as.character(packageVersion("mirt")), seed = 20261126,
              converged = TRUE),
  unidimensional = uni,
  two_factor = c(two, list(factors = list(F1 = c("j1", "j2", "j3"),
                                          F2 = c("j3", "j4", "j5", "j6"))))
)
path <- file.path("tests", "reference", "v54_simulate.json")
write_json(simulate, path, auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null")
cat("wrote", path, "\n")
print(as.data.frame(dich))
print(as.data.frame(poly))
print(as.data.frame(twogroup))
print(as.data.frame(last_free))
