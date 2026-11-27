# Local dependence beyond single-group unidimensional Normal models (VALIDATION.md V26, M7,
# issue #81).
# Run from the repository root:  Rscript validation/r/m7_local_dependence_reference.R
# Writes tests/reference/v26_local_dependence_extensions.json. CI does not run R; the JSON is
# committed.
#
# mirt residuals(type = "LD" / "LDG2" / "Q3") on
#   - multipleGroup() fits (1 and 2 factors): one set of matrices per group;
#   - dentype = "EH" and "Davidian-2" fits, with the fitted density:
#       residuals(m, type = "LD" / "LDG2", use_dentype_estimate = TRUE)
#       residuals(m, type = "Q3", Theta = fscores(m, use_dentype_estimate = TRUE))
#     (the default takes a N(0, 1) prior; type = "Q3" with use_dentype_estimate = TRUE evaluates
#     the residuals at the grid nodes in mirt 1.48 and is not usable);
#   - a normal-mixture density: mirt has no residuals() for its mixture class, so the mixture
#     estimates of V22a (tests/reference/v22_mixture_2pl_k2.json, flexpyirt's scale) are put into
#     a single-group mirt object (items fixed, Internals$Prior set to the mixture weights computed
#     here) and the same residuals() calls as for EH are made. This checks the statistics given
#     the density, not a mirt mixture fit;
#   - confirmatory mirt.model() fits with 2 and 3 factors (31 and 15 points per dimension).
# Grids: LD uses the fit's grid (Options$quadpts). Q3 uses fscores(), whose default grid is 121 /
# 61 / 31 points per dimension for 1 / 2 / 3 factors whatever the fit used. flexpyirt computes the
# EAP on the fitted grid, so
#   - single-group cases pass Theta = fscores(m, quadpts = <the fit's>) ("q3"), and also store
#     the default call ("q3_default") to show the size of the grid effect;
#   - multipleGroup cases, where Theta cannot be passed per group, are fitted on fscores()'s
#     default grid (121 points for one factor, 61 per dimension for two).
# The Python test puts mirt's estimates into a flexpyirt FitResult (every parameter fixed) and
# compares, so the comparison does not depend on how tightly either program converged.
source(file.path("validation", "r", "setup.R"))  # mirt from GitHub; mirt_version / mirt_sha
suppressMessages({
  library(mirt)
  library(jsonlite)
})

LIM <- c(-6, 6)

rows_of <- function(mat) {
  lapply(seq_len(nrow(mat)), function(i) lapply(unname(mat[i, ]),
                                                function(v) if (is.na(v)) NULL else v))
}

# Lower triangle of residuals(type = "LD"): X2 (row > col); upper: signed Cramer's V.
pair_list <- function(ld, ldg2, q3, q3_default = NULL) {
  ld <- unclass(ld); ldg2 <- unclass(ldg2); q3 <- unclass(q3)
  if (!is.null(q3_default)) q3_default <- unclass(q3_default)
  J <- ncol(ld)
  lower <- which(lower.tri(matrix(0, J, J)), arr.ind = TRUE)
  lapply(seq_len(nrow(lower)), function(k) {
    r <- lower[k, 1]; cl <- lower[k, 2]
    list(i = cl - 1L, j = r - 1L, x2 = ld[r, cl], cramer_v = ld[cl, r], g2 = ldg2[r, cl],
         q3 = q3[cl, r], q3_default = if (is.null(q3_default)) NULL else q3_default[cl, r])
  })
}

item_coefs <- function(cf) {
  keep <- setdiff(colnames(cf), c("g", "u"))
  out <- lapply(seq_len(nrow(cf)), function(k) as.list(cf[k, keep]))
  names(out) <- rownames(cf)
  out
}

latent_pars <- function(gp, nfact) {
  if (nfact == 1L) return(list(mean = gp$means[[1]], var = gp$cov[1, 1]))
  out <- list()
  cr <- cov2cor(gp$cov)
  for (m in seq_len(nfact)) {
    out[[paste0("mean_", m)]] <- gp$means[[m]]
    out[[paste0("var_", m)]] <- gp$cov[m, m]
  }
  for (m in seq_len(nfact - 1L)) for (l in (m + 1L):nfact) {
    out[[paste0("cor_", m, l)]] <- cr[l, m]
  }
  out
}

sim_2pl <- function(eta) {
  x <- (matrix(runif(length(eta)), nrow(eta)) < plogis(eta)) * 1L
  colnames(x) <- sprintf("q%02d", seq_len(ncol(x)))
  x
}

sim_grm <- function(theta, a, b) {
  x <- sapply(seq_along(a), function(j) {
    cum <- plogis(outer(theta, b[j, ], function(t, bb) a[j] * (t - bb)))
    rowSums(runif(length(theta)) < cum)
  })
  colnames(x) <- sprintf("g%02d", seq_along(a))
  x
}

add_missing <- function(x, rate) {
  x[matrix(runif(length(x)) < rate, nrow(x))] <- NA
  x
}

# ---- multiple groups ---------------------------------------------------------------------------
group_case <- function(name, dat, grp, model, itemtype, anchors, quadpts, factors = NULL) {
  m <- multipleGroup(dat, model, group = grp, itemtype = itemtype,
                     invariance = c(anchors, "free_means", "free_var"), quadpts = quadpts,
                     TOL = 1e-6, verbose = FALSE, technical = list(NCYCLES = 5000))
  stopifnot(extract.mirt(m, "converged"))
  nfact <- extract.mirt(m, "nfact")
  # The multipleGroup method passes verbose = FALSE to each group itself.
  ld <- residuals(m, type = "LD")
  ldg2 <- residuals(m, type = "LDG2")
  q3 <- residuals(m, type = "Q3")
  cf <- coef(m, simplify = TRUE)
  groups <- lapply(names(ld), function(g) {
    list(group = g, items = item_coefs(cf[[g]]$items), latent = latent_pars(cf[[g]], nfact),
         pairs = pair_list(ld[[g]], ldg2[[g]], q3[[g]]))
  })
  list(name = name, kind = "groups", itemtype = itemtype, items = colnames(dat), n = nrow(dat),
       quadpts = quadpts, factors = factors, anchors = anchors, data = rows_of(dat),
       group = as.character(grp), loglik = extract.mirt(m, "logLik"), groups = groups)
}

# ---- single group: dentype or several factors -------------------------------------------------
single_case <- function(name, kind, m, dat, itemtype, quadpts, factors = NULL, extra = list(),
                        use_density = FALSE, theta = NULL) {
  if (use_density) {
    ld <- residuals(m, type = "LD", verbose = FALSE, use_dentype_estimate = TRUE)
    ldg2 <- residuals(m, type = "LDG2", verbose = FALSE, use_dentype_estimate = TRUE)
    if (is.null(theta)) theta <- fscores(m, use_dentype_estimate = TRUE, full.scores = TRUE)
    q3 <- residuals(m, type = "Q3", verbose = FALSE, Theta = theta)
    q3_default <- NULL
  } else {
    ld <- residuals(m, type = "LD", verbose = FALSE)
    ldg2 <- residuals(m, type = "LDG2", verbose = FALSE)
    theta <- fscores(m, full.scores = TRUE, quadpts = quadpts)
    q3 <- residuals(m, type = "Q3", verbose = FALSE, Theta = theta)
    q3_default <- residuals(m, type = "Q3", verbose = FALSE)
  }
  nfact <- extract.mirt(m, "nfact")
  cf <- coef(m, simplify = TRUE)
  g <- c(list(group = NULL, items = item_coefs(cf$items),
              latent = if (kind == "factors") latent_pars(cf, nfact) else NULL,
              pairs = pair_list(ld, ldg2, q3, q3_default)), extra)
  list(name = name, kind = kind, itemtype = itemtype, items = colnames(dat), n = nrow(dat),
       quadpts = quadpts, factors = factors, data = rows_of(dat), group = NULL,
       loglik = extract.mirt(m, "logLik"), groups = list(g))
}

# ---- Davidian helpers (as validation/r/m5_davidian_reference.R) --------------------------------
# dcurver sine-first angles -> unit Hermite vector c; c -> flexpyirt's cosine-first angles;
# closed-form mean and variance of P_k(z)^2 dnorm(z).
phi_dc_to_c <- function(phi) {
  k <- length(phi)
  s <- sin(phi); cc <- cos(phi)
  cvec <- numeric(k + 1)
  for (j in 0:(k - 1)) cvec[j + 1] <- (if (j > 0) prod(cc[1:j]) else 1) * s[j + 1]
  cvec[k + 1] <- prod(cc)
  cvec
}

c_to_phi_cos <- function(cvec) {
  if (cvec[1] < 0) cvec <- -cvec
  k <- length(cvec) - 1L
  phi <- numeric(k)
  rem <- 1.0
  for (j in seq(k, 1L, by = -1L)) {
    val <- max(-1.0, min(1.0, cvec[j + 1L] / rem))
    phi[j] <- asin(val)
    rem <- rem * cos(phi[j])
    if (abs(rem) < 1e-15) break
  }
  phi
}

dc_moments_c <- function(cvec) {
  k <- length(cvec) - 1L
  mu <- 2.0 * sum(sqrt(seq_len(k)) * cvec[seq_len(k)] * cvec[seq_len(k) + 1L])
  e2 <- sum((2.0 * (0:k) + 1.0) * cvec^2)
  if (k >= 2L) {
    e2 <- e2 + 2.0 * sum(sqrt(seq_len(k - 1L) * (seq_len(k - 1L) + 1L)) *
                           cvec[seq_len(k - 1L)] * cvec[seq_len(k - 1L) + 2L])
  }
  c(mean = mu, var = e2 - mu^2)
}

# ---- mixture helper (as validation/r/m5_mixture_reference.R) -----------------------------------
# flexpyirt's standardized mixture on the grid: shape = (rmean_c, rvar_c, logit_c) for c >= 2,
# raw class 1 = N(0, 1); components normalized on the grid, then mixed.
mixture_weights <- function(nodes, rmean, rvar, logit) {
  rm <- c(0, rmean); rv <- c(1, rvar); lg <- c(0, logit)
  p <- exp(lg - max(lg)); p <- p / sum(p)
  M <- sum(p * rm)
  V <- sum(p * (rv + rm^2)) - M^2
  w <- numeric(length(nodes))
  for (c in seq_along(rm)) {
    lk <- dnorm(nodes, (rm[c] - M) / sqrt(V), sqrt(rv[c] / V), log = TRUE)
    w <- w + p[c] * exp(lk - max(lk)) / sum(exp(lk - max(lk)))
  }
  w
}

set.seed(8101)
cases <- list()

# (1) Two groups, 2PL, 10 items: q01-q05 equal across groups, focal mean and variance free,
#     q07 and q08 locally dependent in both groups, 4% MCAR.
a <- c(1.4, 1.2, 1.0, 0.8, 1.6, 1.1, 0.9, 1.3, 1.5, 0.7)
d <- c(0.2, -0.4, 0.8, -1.0, 0.0, 1.2, -0.6, 0.5, -0.2, 0.9)
gen_group <- function(n, mean, sd, shift) {
  theta <- rnorm(n, mean, sd)
  eta <- outer(theta, a) + matrix(d + shift, n, length(a), byrow = TRUE)
  eta[, 7:8] <- eta[, 7:8] + 1.2 * rnorm(n)
  sim_2pl(eta)
}
dat <- rbind(gen_group(700, 0, 1, 0), gen_group(600, 0.5, 1.2, c(rep(0, 5), 0.4, -0.3, 0, 0.3, 0)))
dat <- add_missing(dat, 0.04)
grp <- rep(c("ref", "focal"), c(700, 600))
cases[[1]] <- group_case("groups_2pl", dat, grp, 1, "2PL", colnames(dat)[1:5], 121)

# (2) Two groups, GRM with 4 categories, 6 items: g01-g03 equal across groups.
a <- c(1.3, 1.0, 1.6, 0.9, 1.2, 1.4)
b <- rbind(c(-1, 0, 1), c(-0.5, 0.3, 1.2), c(-1.2, -0.2, 0.6), c(-0.8, 0.4, 1.0),
           c(-1.5, -0.5, 0.5), c(-0.3, 0.5, 1.4))
dat <- rbind(sim_grm(rnorm(600), a, b), sim_grm(rnorm(600, -0.4, 0.9), a, b))
grp <- rep(c("ref", "focal"), c(600, 600))
cases[[2]] <- group_case("groups_grm", dat, grp, 1, "graded", colnames(dat)[1:3], 121)

# (3) Empirical histogram, 2PL, 12 items, skewed trait, q03 and q04 locally dependent, 3% MCAR.
a <- c(1.4, 1.2, 1.0, 0.8, 1.6, 1.1, 0.9, 1.3, 1.5, 0.7, 1.0, 1.2)
d <- c(0.2, -0.4, 0.8, -1.0, 0.0, 1.2, -0.6, 0.5, -0.2, 0.9, 0.3, -0.8)
n <- 1500
theta <- (rchisq(n, df = 4) - 4) / sqrt(8)
eta <- outer(theta, a) + matrix(d, n, length(a), byrow = TRUE)
eta[, 3:4] <- eta[, 3:4] + 1.2 * rnorm(n)
dat_skew <- add_missing(sim_2pl(eta), 0.03)
m <- mirt(dat_skew, 1, "2PL", dentype = "EH", quadpts = 61, verbose = FALSE,
          technical = list(NCYCLES = 2000))
stopifnot(extract.mirt(m, "converged"))
cases[[3]] <- single_case("eh_2pl", "eh", m, dat_skew, "2PL", 61, use_density = TRUE,
                          extra = list(prior = as.numeric(m@Internals$Prior[[1L]])))

# (4) Davidian curve of degree 2 on the same data.
m <- mirt(dat_skew, 1, "2PL", dentype = "Davidian-2", quadpts = 61, verbose = FALSE,
          technical = list(NCYCLES = 2000))
stopifnot(extract.mirt(m, "converged"))
gp <- coef(m)$GroupPars
cvec <- phi_dc_to_c(as.numeric(gp[1, c("PHI_1", "PHI_2")]))
mom <- dc_moments_c(cvec)
cases[[4]] <- single_case(
  "davidian_2pl", "davidian", m, dat_skew, "2PL", 61, use_density = TRUE,
  extra = list(prior = as.numeric(m@Internals$Prior[[1L]]), degree = 2L,
               davidian = list(mean = unname(mom["mean"]), var = unname(mom["var"]),
                               phi = as.list(c_to_phi_cos(cvec))))
)

# (5) Normal mixture: V22a's estimates in a single-group mirt object with the mixture weights
#     as its prior (see the header).
v22 <- fromJSON(file.path("tests", "reference", "v22_mixture_2pl_k2.json"),
                simplifyVector = FALSE)
items <- unlist(v22$items)
pats <- do.call(rbind, lapply(v22$patterns, function(r) as.integer(unlist(r))))
dat_mix <- pats[rep(seq_len(nrow(pats)), unlist(v22$counts)), , drop = FALSE]
colnames(dat_mix) <- items
lat <- v22$exact$latent
sv <- mirt(dat_mix, 1, "2PL", pars = "values", quadpts = 61)
for (it in items) {
  sv$value[sv$item == it & sv$name == "a1"] <- v22$exact$items[[it]]$a
  sv$value[sv$item == it & sv$name == "d"] <- v22$exact$items[[it]]$d
}
sv$est <- FALSE
m <- mirt(dat_mix, 1, "2PL", pars = sv, quadpts = 61, verbose = FALSE)
nodes <- as.numeric(m@Model$Theta)
stopifnot(length(nodes) == 61, all.equal(range(nodes), LIM))
w <- mixture_weights(nodes, lat$rmean_2, lat$rvar_2, lat$logit_2)
m@Internals$Prior[[1L]] <- w
# EAP under the mixture: posterior mean on the grid with the mixture weights.
theta_mix <- fscores(m, use_dentype_estimate = TRUE, full.scores = TRUE)
cases[[5]] <- single_case("mixture_2pl", "mixture", m, dat_mix, "2PL", 61, use_density = TRUE,
                          theta = theta_mix,
                          extra = list(prior = w, n_components = 2L, mixture = lat))
cases[[5]]$data <- NULL  # the data are V22a's patterns and counts
cases[[5]]$data_from <- "v22_mixture_2pl_k2"
cases[[5]]$loglik <- v22$exact$loglik

# (6) Two factors, 2PL, 10 items (q05 loads on both), correlation 0.5, q01 and q02 locally
#     dependent, 3% MCAR; 31 points per dimension.
n <- 1200
z <- matrix(rnorm(2 * n), n) %*% chol(matrix(c(1, 0.5, 0.5, 1), 2))
a1 <- c(1.3, 1.1, 0.9, 1.5, 0.8, 0, 0, 0, 0, 0)
a2 <- c(0, 0, 0, 0, 0.7, 1.2, 1.0, 1.4, 0.9, 1.1)
d <- c(0.3, -0.5, 0.7, -0.9, 0.1, 1.0, -0.4, 0.4, -0.1, 0.6)
eta <- z[, 1] %o% a1 + z[, 2] %o% a2 + matrix(d, n, 10, byrow = TRUE)
eta[, 1:2] <- eta[, 1:2] + 1.2 * rnorm(n)
dat <- add_missing(sim_2pl(eta), 0.03)
spec <- mirt.model("F1 = 1-5\nF2 = 5-10\nCOV = F1*F2")
m <- mirt(dat, spec, "2PL", quadpts = 31, TOL = 1e-6, verbose = FALSE,
          technical = list(NCYCLES = 5000))
stopifnot(extract.mirt(m, "converged"))
factors2 <- list(F1 = colnames(dat)[1:5], F2 = colnames(dat)[5:10])
cases[[6]] <- single_case("factors2_2pl", "factors", m, dat, "2PL", 31, factors = factors2)

# (7) Two factors, GRM with 3 categories, 8 items, simple structure, correlation 0.4.
n <- 1000
z <- matrix(rnorm(2 * n), n) %*% chol(matrix(c(1, 0.4, 0.4, 1), 2))
a <- c(1.3, 1.0, 1.6, 0.9, 1.2, 1.4, 1.1, 0.8)
b <- cbind(c(-1, -0.5, -1.2, -0.8, -1.5, -0.3, -0.9, -0.6), c(0.5, 0.8, 0.2, 0.9, 0.1, 1.2, 0.4, 0.7))
dat <- cbind(sim_grm(z[, 1], a[1:4], b[1:4, ]), sim_grm(z[, 2], a[5:8], b[5:8, ]))
colnames(dat) <- sprintf("g%02d", 1:8)
spec <- mirt.model("F1 = 1-4\nF2 = 5-8\nCOV = F1*F2")
m <- mirt(dat, spec, "graded", quadpts = 31, TOL = 1e-6, verbose = FALSE,
          technical = list(NCYCLES = 5000))
stopifnot(extract.mirt(m, "converged"))
factors <- list(F1 = colnames(dat)[1:4], F2 = colnames(dat)[5:8])
cases[[7]] <- single_case("factors2_grm", "factors", m, dat, "graded", 31, factors = factors)

# (8) Three factors, 2PL, 9 items, simple structure; 15 points per dimension.
n <- 1000
S <- matrix(c(1, 0.5, 0.3, 0.5, 1, 0.4, 0.3, 0.4, 1), 3)
z <- matrix(rnorm(3 * n), n) %*% chol(S)
a <- c(1.3, 1.1, 0.9, 1.5, 0.8, 1.2, 1.0, 1.4, 0.9)
d <- c(0.3, -0.5, 0.7, -0.9, 0.1, 1.0, -0.4, 0.4, -0.1)
eta <- sapply(1:9, function(j) a[j] * z[, (j - 1) %/% 3 + 1] + d[j])
dat <- sim_2pl(eta)
spec <- mirt.model("F1 = 1-3\nF2 = 4-6\nF3 = 7-9\nCOV = F1*F2*F3")
m <- mirt(dat, spec, "2PL", quadpts = 15, TOL = 1e-6, verbose = FALSE,
          technical = list(NCYCLES = 5000))
stopifnot(extract.mirt(m, "converged"))
factors <- list(F1 = colnames(dat)[1:3], F2 = colnames(dat)[4:6], F3 = colnames(dat)[7:9])
cases[[8]] <- single_case("factors3_2pl", "factors", m, dat, "2PL", 15, factors = factors)

# (9) Two groups and two factors, 2PL: the first two items of each factor equal across groups,
#     focal means and variances free; 61 points per dimension.
gen2 <- function(n, mu, sd, rho) {
  z <- matrix(rnorm(2 * n), n) %*% chol(matrix(c(1, rho, rho, 1), 2))
  z <- sweep(sweep(z, 2, sd, `*`), 2, mu, `+`)
  sim_2pl(z[, 1] %o% a1 + z[, 2] %o% a2 + matrix(d, n, 10, byrow = TRUE))
}
a1[5] <- 0  # simple structure
d <- c(0.3, -0.5, 0.7, -0.9, 0.1, 1.0, -0.4, 0.4, -0.1, 0.6)
dat <- rbind(gen2(800, c(0, 0), c(1, 1), 0.5), gen2(800, c(0.4, -0.3), c(1.2, 0.9), 0.3))
grp <- rep(c("ref", "focal"), c(800, 800))
spec <- mirt.model("F1 = 1-5\nF2 = 6-10\nCOV = F1*F2")
anchors <- colnames(dat)[c(1, 2, 6, 7)]
factors <- list(F1 = colnames(dat)[1:5], F2 = colnames(dat)[6:10])
cases[[9]] <- group_case("groups_factors2_2pl", dat, grp, spec, "2PL", anchors, 61,
                         factors = factors)

res <- list(
  meta = list(case = "v26_local_dependence_extensions", r_version = R.version.string,
              mirt_version = mirt_version_string(), mirt_sha = mirt_commit_sha(), converged = TRUE,
              theta_lim = LIM, grid_check = LIM),
  cases = cases
)
write_json(res, file.path("tests", "reference", "v26_local_dependence_extensions.json"),
           auto_unbox = TRUE, digits = NA, null = "null", pretty = FALSE)
cat("wrote tests/reference/v26_local_dependence_extensions.json\n")
