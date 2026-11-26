# M2* for multiple groups, missing data and MAP fits (VALIDATION.md V42, DESIGN.md §16.20).
# Run from the repository root:  Rscript validation/r/m6_m2_groups_missing_map_reference.R
# Writes tests/reference/v42_m2_groups_missing_map.json. CI does not run R; the JSON is committed.
#
# Every case is simulated here, fitted with mirt (mirt() or multipleGroup()), and M2() is called on
# the fit's grid (quadpts / theta_lim of the fit). The JSON records the data, mirt's estimates (in
# flexpyirt's parameter names), mirt's raw M2() output, and `exact`: M2* recomputed in R from the
# formulas of DESIGN.md §16.20 with
#   - Delta from central differences of expected.item() and of the normalized latent weights over
#     every free parameter (one column per parameter, equality constraints across groups merged,
#     free latent means / variances / correlations included), the rows of group g scaled by
#     sqrt(N_g / N);
#   - Xi_2 per group from mirt's own buildXi2els, rescaled to N_g N_ab / (N_a N_b) Xi_ab when
#     responses are missing (N_ab = respondents who answered every item of margins a and b);
#   - the independence model (intercepts at the available-case proportions in every group) for
#     TLI / CFI.
# The same R function with `mirt_convention = TRUE` (no latent columns, unscaled group rows, mirt's
# N.ratio weighting of the residuals) must reproduce mirt's raw M2 (asserted below), which checks
# the R implementation against mirt itself.
suppressMessages({
  library(mirt)
  library(jsonlite)
})

QUADPTS_1D <- 61L
QUADPTS_2D <- 31L
THETA_LIM <- c(-6, 6)
TOL <- 1e-10
EPS <- 1e-6

pairs_of <- function(J) {
  ii <- integer(0); jj <- integer(0)
  # mirt's column-major lower.tri order of (row j, col i) with j > i, i.e. pairs (i, j), i < j,
  # ordered by i then j.
  for (i in seq_len(J)) for (j in seq_len(J)) if (i < j) { ii <- c(ii, i); jj <- c(jj, j) }
  list(i = ii, j = jj)
}

sample_margins <- function(dat) {
  J <- ncol(dat)
  pr <- pairs_of(J)
  seen <- !is.na(dat)
  yz <- dat; yz[!seen] <- 0
  ind <- cbind(seen, seen[, pr$i, drop = FALSE] & seen[, pr$j, drop = FALSE]) * 1
  vals <- cbind(yz, yz[, pr$i, drop = FALSE] * yz[, pr$j, drop = FALSE])
  cnt <- colSums(ind)
  list(obs = colSums(vals) / cnt, count = cnt, joint = crossprod(ind), n = nrow(dat))
}

# Margins of one group from item objects and a GroupPars object on the grid Theta.
group_moments <- function(items, gp_obj, Theta) {
  J <- length(items)
  pr <- pairs_of(J)
  gp <- mirt:::ExtractGroupPars(gp_obj)
  Prior <- mirt:::mirt_dmvnorm(Theta, gp$gmeans, gp$gcov)
  Prior <- Prior / sum(Prior)
  EIs <- EIs2 <- matrix(0, nrow(Theta), J)
  for (i in seq_len(J)) {
    x <- items[[i]]
    EIs[, i] <- expected.item(x, Theta, min = 0L)
    prob <- mirt:::ProbTrace(x, Theta)
    EIs2[, i] <- as.vector(prob %*% ((0:(x@ncat - 1L))^2))
  }
  e <- c(colSums(EIs * Prior), colSums(EIs[, pr$i, drop = FALSE] * EIs[, pr$j, drop = FALSE] * Prior))
  list(e = e, EIs = EIs, EIs2 = EIs2, Prior = Prior)
}

# Item and latent objects of every group of a mirt fit (single-group fits give one group).
group_objects <- function(m) {
  ng <- extract.mirt(m, "ngroups")
  J <- extract.mirt(m, "nitems")
  lapply(seq_len(ng), function(g) {
    pars <- if (ng > 1L) m@ParObjects$pars[[g]]@ParObjects$pars else m@ParObjects$pars
    list(items = pars[seq_len(J)], gp = pars[[J + 1L]])
  })
}

# Free parameters: a list of parnum sets (constraints merged), each flagged latent or not.
free_parameters <- function(m) {
  objs <- group_objects(m)
  J <- extract.mirt(m, "nitems")
  rows <- list()
  for (g in seq_along(objs)) {
    for (i in seq_len(J + 1L)) {
      obj <- if (i <= J) objs[[g]]$items[[i]] else objs[[g]]$gp
      for (k in which(obj@est)) rows[[length(rows) + 1L]] <- c(obj@parnum[k], i > J)
    }
  }
  pn <- vapply(rows, `[`, numeric(1), 1L)
  lat <- vapply(rows, `[`, numeric(1), 2L) == 1
  sets <- lapply(pn, function(p) p)
  for (cs in extract.mirt(m, "constrain")) {
    keep <- which(pn == cs[1L])
    drop <- which(pn %in% cs[-1L])
    sets[[keep]] <- cs
    sets[drop] <- list(NULL)
  }
  alive <- !vapply(sets, is.null, logical(1))
  list(sets = sets[alive], latent = lat[alive])
}

perturb <- function(objs, set, h) {
  J <- length(objs[[1L]]$items)
  for (g in seq_along(objs)) {
    for (i in seq_len(J)) {
      hit <- which(objs[[g]]$items[[i]]@parnum %in% set)
      if (length(hit)) objs[[g]]$items[[i]]@par[hit] <- objs[[g]]$items[[i]]@par[hit] + h
    }
    hit <- which(objs[[g]]$gp@parnum %in% set)
    if (length(hit)) objs[[g]]$gp@par[hit] <- objs[[g]]$gp@par[hit] + h
  }
  objs
}

all_margins <- function(objs, Theta) {
  unlist(lapply(objs, function(o) group_moments(o$items, o$gp, Theta)$e))
}

# Quadratic form r' Dc (Dc' Gamma Dc)^{-1} Dc' r with Dc the SVD complement of col(delta).
quad_form <- function(r, delta, Gamma) {
  sv <- svd(delta, nu = nrow(delta))
  tol <- max(sv$d) * max(dim(delta)) * .Machine$double.eps
  rk <- sum(sv$d > tol)
  stopifnot(rk == ncol(delta), rk < nrow(delta))
  dc <- sv$u[, (rk + 1L):nrow(delta), drop = FALSE]
  proj <- crossprod(dc, r)
  as.numeric(abs(crossprod(proj, solve(crossprod(dc, Gamma %*% dc), proj))))
}

block_diag <- function(blocks) {
  n <- sum(vapply(blocks, nrow, integer(1)))
  out <- matrix(0, n, n)
  pos <- 0L
  for (b in blocks) {
    k <- nrow(b)
    out[pos + seq_len(k), pos + seq_len(k)] <- b
    pos <- pos + k
  }
  out
}

resid_gamma <- function(smp, e, Xi2, mirt_convention) {
  d <- smp$obs - e
  if (mirt_convention) return(list(r = smp$count * d / sqrt(smp$n), Gamma = Xi2))
  Gamma <- Xi2 * (smp$n * smp$joint / outer(smp$count, smp$count))
  list(r = sqrt(smp$n) * d, Gamma = Gamma)
}

null_m2 <- function(samples, J, mirt_convention) {
  pr <- pairs_of(J)
  s <- J + length(pr$i)
  G <- length(samples)
  Ntot <- sum(vapply(samples, function(x) x$n, numeric(1)))
  rs <- list(); Gs <- list()
  delta <- matrix(0, G * s, G * J)
  for (g in seq_len(G)) {
    smp <- samples[[g]]
    p <- smp$obs[seq_len(J)]
    v <- p * (1 - p)
    e <- c(p, p[pr$i] * p[pr$j])
    X <- .Call("buildXi2els", J, length(pr$i), J, matrix(p, 1L), matrix(p, 1L), 1, PACKAGE = "mirt")
    Xi2 <- rbind(cbind(X$Xi11, X$Xi12), cbind(t(X$Xi12), X$Xi22))
    rg <- resid_gamma(smp, e, Xi2, mirt_convention)
    rs[[g]] <- rg$r; Gs[[g]] <- rg$Gamma
    blk <- matrix(0, s, J)
    blk[cbind(seq_len(J), seq_len(J))] <- v
    rows <- J + seq_along(pr$i)
    blk[cbind(rows, pr$i)] <- v[pr$i] * p[pr$j]
    blk[cbind(rows, pr$j)] <- v[pr$j] * p[pr$i]
    if (!mirt_convention) blk <- blk * sqrt(smp$n / Ntot)
    delta[(g - 1L) * s + seq_len(s), (g - 1L) * J + seq_len(J)] <- blk
  }
  list(M2 = quad_form(unlist(rs), delta, block_diag(Gs)), df = G * length(pr$i))
}

exact_m2 <- function(m, dat, group, Theta, mirt_convention = FALSE, ci = 0.90) {
  dat <- as.matrix(dat)
  objs <- group_objects(m)
  G <- length(objs)
  J <- ncol(dat)
  gnames <- if (G > 1L) extract.mirt(m, "groupNames") else "all"
  rows <- if (G > 1L) lapply(gnames, function(g) which(group == g)) else list(seq_len(nrow(dat)))
  N <- nrow(dat)
  samples <- lapply(rows, function(r) sample_margins(dat[r, , drop = FALSE]))
  fp <- free_parameters(m)
  sets <- fp$sets
  if (mirt_convention) sets <- sets[!fp$latent]
  e_all <- all_margins(objs, Theta)
  s <- length(e_all) / G
  delta <- matrix(0, length(e_all), length(sets))
  for (k in seq_along(sets)) {
    delta[, k] <- (all_margins(perturb(objs, sets[[k]], EPS), Theta) -
      all_margins(perturb(objs, sets[[k]], -EPS), Theta)) / (2 * EPS)
  }
  rs <- list(); Gs <- list()
  for (g in seq_len(G)) {
    mom <- group_moments(objs[[g]]$items, objs[[g]]$gp, Theta)
    X <- .Call("buildXi2els", J, s - J, J, mom$EIs, mom$EIs2, mom$Prior, PACKAGE = "mirt")
    Xi2 <- rbind(cbind(X$Xi11, X$Xi12), cbind(t(X$Xi12), X$Xi22))
    rg <- resid_gamma(samples[[g]], mom$e, Xi2, mirt_convention)
    rs[[g]] <- rg$r; Gs[[g]] <- rg$Gamma
    idx <- (g - 1L) * s + seq_len(s)
    if (!mirt_convention) delta[idx, ] <- delta[idx, ] * sqrt(samples[[g]]$n / N)
  }
  M2v <- quad_form(unlist(rs), delta, block_diag(Gs))
  dfv <- G * s - extract.mirt(m, "nestpars")
  alpha <- (1 - ci) / 2
  civ <- mirt:::RMSEA.CI(M2v, dfv, N, ci.lower = alpha, ci.upper = 1 - alpha)
  out <- list(M2 = M2v, df = dfv, p = pchisq(M2v, dfv, lower.tail = FALSE),
              RMSEA = mirt:::rmsea(X2 = M2v, df = dfv, N = N), RMSEA_5 = civ[1L], RMSEA_95 = civ[2L])
  if (all(extract.mirt(m, "K") == 2L)) {
    nl <- null_m2(samples, J, mirt_convention)
    out$M2_null <- nl$M2
    out$df_null <- nl$df
    out$TLI <- mirt:::tli(X2 = M2v, X2.null = nl$M2, df = dfv, df.null = nl$df)
    out$CFI <- mirt:::cfi(X2 = M2v, X2.null = nl$M2, df = dfv, df.null = nl$df)
  }
  out
}

# Estimates of every group in flexpyirt's names.
estimates <- function(m, factors = NULL) {
  objs <- group_objects(m)
  gnames <- if (length(objs) > 1L) extract.mirt(m, "groupNames") else "all"
  nfact <- extract.mirt(m, "nfact")
  items <- colnames(extract.mirt(m, "data"))
  out <- list()
  for (g in seq_along(objs)) {
    it <- list()
    for (i in seq_along(items)) {
      x <- objs[[g]]$items[[i]]
      nm <- names(x@est)
      v <- as.list(x@par)
      names(v) <- if (nfact == 1L) sub("^a1$", "a", nm) else nm
      it[[items[i]]] <- v
    }
    gp <- mirt:::ExtractGroupPars(objs[[g]]$gp)
    lat <- list()
    if (nfact == 1L) {
      lat$mean <- gp$gmeans[1L]; lat$var <- gp$gcov[1L, 1L]
    } else {
      for (a in seq_len(nfact)) {
        lat[[paste0("mean_", a)]] <- gp$gmeans[a]
        lat[[paste0("var_", a)]] <- gp$gcov[a, a]
      }
      for (a in seq_len(nfact)) for (b in seq_len(nfact)) if (a < b) {
        lat[[sprintf("cor_%d%d", a, b)]] <- gp$gcov[a, b] / sqrt(gp$gcov[a, a] * gp$gcov[b, b])
      }
    }
    out[[gnames[g]]] <- list(items = it, latent = lat)
  }
  out
}

mirt_m2 <- function(m, quadpts) {
  r <- M2(m, quadpts = quadpts, theta_lim = THETA_LIM)
  out <- as.list(r)
  lapply(out, function(v) as.numeric(v))
}

sim_2pl <- function(n, a, d, mean = 0, sd = 1) {
  theta <- rnorm(n, mean, sd)
  p <- plogis(outer(theta, a) + matrix(d, n, length(d), byrow = TRUE))
  (matrix(runif(length(p)), n) < p) * 1
}

sim_grm <- function(n, a, d, mean = 0, sd = 1) {
  theta <- rnorm(n, mean, sd)
  y <- matrix(0L, n, length(a))
  for (j in seq_along(a)) {
    pge <- plogis(outer(theta, rep(a[j], ncol(d))) + matrix(d[j, ], n, ncol(d), byrow = TRUE))
    y[, j] <- rowSums(matrix(runif(n), n, ncol(d)) < pge)
  }
  y
}

sim_2d <- function(n, A, d, mu, Sigma) {
  L <- chol(Sigma)
  theta <- matrix(rnorm(n * 2), n) %*% L + matrix(mu, n, 2, byrow = TRUE)
  p <- plogis(theta %*% t(A) + matrix(d, n, length(d), byrow = TRUE))
  (matrix(runif(length(p)), n) < p) * 1
}

mcar <- function(y, rate) {
  y[matrix(runif(length(y)), nrow(y)) < rate] <- NA
  y
}

as_case <- function(dat, group, extra) {
  key <- apply(cbind(dat, if (is.null(group)) NULL else as.character(group)), 1, paste, collapse = "|")
  first <- !duplicated(key)
  counts <- as.integer(table(factor(key, levels = key[first])))
  pats <- dat[first, , drop = FALSE]
  base <- list(
    items = colnames(dat),
    patterns = lapply(seq_len(nrow(pats)), function(r) as.list(unname(pats[r, ]))),
    counts = counts
  )
  if (!is.null(group)) base$pattern_groups <- as.character(group[first])
  c(base, extra)
}

record <- function(name, m, dat, group, Theta, quadpts, extra = list()) {
  stopifnot(extract.mirt(m, "converged"))
  raw <- mirt_m2(m, quadpts)
  mc <- exact_m2(m, dat, group, Theta, mirt_convention = TRUE)
  ex <- exact_m2(m, dat, group, Theta)
  # The R implementation with mirt's conventions reproduces mirt's own M2().
  stopifnot(abs(mc$M2 - raw$M2) <= 1e-6 * raw$M2, mc$df == raw$df)
  cat(sprintf("%-22s mirt M2 %.6f  exact %.6f  df %d  (rel diff %.2e)\n",
              name, raw$M2, ex$M2, ex$df, (raw$M2 - ex$M2) / ex$M2))
  as_case(dat, group, c(list(
    quadpts = quadpts, theta_lim = THETA_LIM, nest = extract.mirt(m, "nestpars"),
    loglik = extract.mirt(m, "logLik"), estimates = estimates(m), mirt = raw, exact = ex
  ), extra))
}

set.seed(20261126)
cases <- list()
th1 <- matrix(seq(THETA_LIM[1L], THETA_LIM[2L], length.out = QUADPTS_1D))
th2 <- mirt:::thetaComb(matrix(seq(THETA_LIM[1L], THETA_LIM[2L], length.out = QUADPTS_2D)), 2L)

# 1. Two groups of unequal size, 2PL, anchors q1-q3 equal across groups, focal mean/var free.
a <- c(1.2, 0.8, 1.5, 1.0, 0.7, 1.3); d <- c(-0.5, 0.3, 0.0, 0.8, -1.0, 0.4)
y <- rbind(sim_2pl(700, a, d), sim_2pl(300, a * c(1, 1, 1, 0.7, 1.3, 1), d + c(0, 0, 0, 0.5, -0.4, 0), 0.4, sqrt(1.3)))
colnames(y) <- paste0("q", 1:6)
grp <- rep(c("ref", "focal"), c(700, 300))
fit_tg <- function(dat) {
  multipleGroup(as.data.frame(dat), 1, group = factor(grp, levels = c("ref", "focal")),
                itemtype = "2PL", invariance = c("free_means", "free_var", "q1", "q2", "q3"),
                quadpts = QUADPTS_1D, TOL = TOL, verbose = FALSE, technical = list(NCYCLES = 5000))
}
m <- fit_tg(y)
cases$twogroup_2pl <- record("twogroup_2pl", m, y, grp, th1, QUADPTS_1D,
                             list(anchors = paste0("q", 1:3), groups = c("ref", "focal")))

# 2. Same design with 15% MCAR missing responses.
ym <- mcar(y, 0.15)
m <- fit_tg(ym)
cases$twogroup_2pl_missing <- record("twogroup_2pl_missing", m, ym, grp, th1, QUADPTS_1D,
                                     list(anchors = paste0("q", 1:3), groups = c("ref", "focal")))

# 3. Two equal groups, GRM K = 3, all items invariant, focal mean/var free.
ag <- c(1.1, 1.4, 0.9, 1.2, 1.6)
dg <- cbind(c(1.0, 1.5, 0.5, 1.2, 0.8), c(-0.8, -0.2, -1.2, -0.5, -1.0))
y <- rbind(sim_grm(500, ag, dg), sim_grm(500, ag, dg, -0.3, sqrt(0.8)))
colnames(y) <- paste0("g", 1:5)
grp_g <- rep(c("ref", "focal"), c(500, 500))
m <- multipleGroup(as.data.frame(y), 1, group = factor(grp_g, levels = c("ref", "focal")),
                   itemtype = "graded", invariance = c("free_means", "free_var", colnames(y)),
                   quadpts = QUADPTS_1D, TOL = TOL, verbose = FALSE, technical = list(NCYCLES = 5000))
cases$twogroup_grm <- record("twogroup_grm", m, y, grp_g, th1, QUADPTS_1D, list(groups = c("ref", "focal")))

# 4. Two groups, two-factor 2PL (simple structure), all items invariant, focal means, variances
#    and correlation free.
A <- cbind(c(1.2, 0.9, 1.4, 1.0, 0, 0, 0, 0), c(0, 0, 0, 0, 1.1, 1.3, 0.8, 1.0))
d2 <- c(0.2, -0.4, 0.6, 0.0, -0.3, 0.5, 0.1, -0.6)
y <- rbind(sim_2d(600, A, d2, c(0, 0), matrix(c(1, 0.4, 0.4, 1), 2)),
           sim_2d(400, A, d2, c(0.3, -0.2), matrix(c(1.2, 0.3, 0.3, 0.9), 2)))
colnames(y) <- paste0("i", 1:8)
grp_2d <- rep(c("ref", "focal"), c(600, 400))
spec2d <- mirt.model("F1 = 1-4\nF2 = 5-8\nCOV = F1*F2")
m <- multipleGroup(as.data.frame(y), spec2d, group = factor(grp_2d, levels = c("ref", "focal")),
                   itemtype = "2PL", invariance = c("free_means", "free_var", colnames(y)),
                   quadpts = QUADPTS_2D, TOL = 1e-8, verbose = FALSE, technical = list(NCYCLES = 5000))
cases$twogroup_2d <- record("twogroup_2d", m, y, grp_2d, th2, QUADPTS_2D,
                            list(groups = c("ref", "focal"),
                                 factors = list(F1 = paste0("i", 1:4), F2 = paste0("i", 5:8))))

# 5. Single group, 2PL, 20% MCAR missing responses.
a5 <- c(1.0, 1.4, 0.8, 1.2, 0.6, 1.5, 1.1); d5 <- c(0.5, -0.2, 1.0, -0.8, 0.0, 0.3, -1.2)
y <- mcar(sim_2pl(800, a5, d5), 0.20)
colnames(y) <- paste0("q", 1:7)
m <- mirt(as.data.frame(y), 1, itemtype = "2PL", quadpts = QUADPTS_1D, TOL = TOL, verbose = FALSE,
          technical = list(NCYCLES = 5000))
cases$missing_2pl <- record("missing_2pl", m, y, NULL, th1, QUADPTS_1D)

# 6. MAP: 2PL with a lognormal prior on a and a normal prior on d (small N).
y <- sim_2pl(250, a, d)
colnames(y) <- paste0("q", 1:6)
spec_map <- mirt.model("F = 1-6\nPRIOR = (1-6, a1, lnorm, 0.0, 0.5), (1-6, d, norm, 0.0, 2.0)")
m <- mirt(as.data.frame(y), spec_map, itemtype = "2PL", quadpts = QUADPTS_1D, TOL = TOL,
          verbose = FALSE, technical = list(NCYCLES = 5000))
cases$map_2pl <- record("map_2pl", m, y, NULL, th1, QUADPTS_1D,
                        list(prior = list(a = c(0.0, 0.5), d = c(0.0, 2.0)),
                             log_posterior = m@Fit$logPrior + extract.mirt(m, "logLik")))

meta <- list(
  case = "v42_m2_groups_missing_map", converged = TRUE,
  r_version = R.version.string, mirt_version = as.character(packageVersion("mirt"))
)
write_json(list(meta = meta, cases = cases),
           file.path("tests", "reference", "v42_m2_groups_missing_map.json"),
           digits = NA, auto_unbox = TRUE, pretty = FALSE, na = "null")
cat("wrote tests/reference/v42_m2_groups_missing_map.json\n")
