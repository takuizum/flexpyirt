# M2 / M2* and S-X2 for confirmatory multidimensional models (VALIDATION.md V39) from R mirt.
# Run from the repository root:  Rscript validation/r/m6_fit_multidim_reference.R
# Writes tests/reference/v39_fit_multidim.json. CI does not run R; the JSON is committed.
#
# Each case reuses a V14 reference (tests/reference/v14_mirt2d.json: data and mirt's estimates).
# The model is rebuilt with every parameter at those estimates (est flags kept, so the free
# correlations count against the degrees of freedom) and fitted with TOL = NaN, so M2() and
# itemfit() see exactly the reference estimates. Both are evaluated on the fit's product grid
# (quadpts per dimension and theta_lim of the V14 fit; these are also mirt's defaults,
# select_quadpts(2) = 31 and select_quadpts(3) = 15 on [-6, 6]).
#
# mirt's M2() forces the GroupPars `est` flags to FALSE when it builds Delta, so the free factor
# correlations have no column in Delta although they are subtracted in df (the same convention as
# V18's rasch_free_var). This script therefore records
#   - mirt's raw M2() output,
#   - `exact`: M2* recomputed in R from mirt's analytic dP (checked against central differences),
#     mirt's own buildXi2els and no latent column (must reproduce the raw output: asserted), and
#   - `exact_with_latent`: the same with one analytic Delta column per free correlation
#     (Maydeu-Olivares & Joe, 2005, 2006), which is flexpyirt's default.
source(file.path("validation", "r", "setup.R"))  # mirt from GitHub; mirt_version / mirt_sha
suppressMessages({
  library(mirt)
  library(jsonlite)
})

bundle <- fromJSON(file.path("tests", "reference", "v14_mirt2d.json"), simplifyVector = FALSE)

load_case <- function(name) {
  r <- bundle$cases[[name]]
  pats <- do.call(rbind, lapply(r$patterns, function(x) unlist(x)))
  colnames(pats) <- unlist(r$items)
  list(raw = r, data = as.data.frame(pats[rep(seq_len(nrow(pats)), unlist(r$counts)), , drop = FALSE]))
}

model_string <- function(r) {
  items <- unlist(r$items)
  fnames <- names(r$factors)
  lines <- vapply(fnames, function(f) {
    sprintf("%s = %s", f, paste(match(unlist(r$factors[[f]]), items), collapse = ","))
  }, character(1))
  paste(c(lines, sprintf("COV = %s", paste(fnames, collapse = "*"))), collapse = "\n")
}

product_grid <- function(m, quad) {
  nitems <- extract.mirt(m, "nitems")
  gp_obj <- m@ParObjects$pars[[nitems + 1L]]
  theta <- as.matrix(seq(quad$theta_lim[1L], quad$theta_lim[2L], length.out = quad$quadpts))
  Theta <- mirt:::thetaComb(theta, extract.mirt(m, "nfact"))
  prior_of <- function(obj) {
    gp <- mirt:::ExtractGroupPars(obj)
    pr <- mirt:::mirt_dmvnorm(Theta, gp$gmeans, gp$gcov)
    pr / sum(pr)
  }
  list(Theta = Theta, gp_obj = gp_obj, prior_of = prior_of)
}

exact_m2_star <- function(m, quad, include_latent, ci = 0.90) {
  dat <- t(t(m@Data$data) - m@Data$mins)
  N <- nrow(dat)
  nitems <- ncol(dat)
  cross <- crossprod(dat, dat)
  p <- c(colMeans(dat), cross[lower.tri(cross)] / N)

  grid <- product_grid(m, quad)
  Theta <- grid$Theta
  Prior <- grid$prior_of(grid$gp_obj)

  margins <- function(EIs, Prior) {
    E1 <- colSums(EIs * Prior)
    E2 <- crossprod(EIs * Prior, EIs)
    c(E1, E2[lower.tri(E2)])
  }

  EIs <- EIs2 <- matrix(0, nrow(Theta), nitems)
  eps <- 1e-6
  cols <- list()
  for (i in seq_len(nitems)) {
    x <- extract.item(m, i)
    scs <- 0:(x@ncat - 1L)
    EIs[, i] <- expected.item(x, Theta, min = 0L)
    EIs2[, i] <- as.vector(mirt:::ProbTrace(x, Theta) %*% (scs^2))
  }
  e <- margins(EIs, Prior)
  # Item columns: mirt's analytic dP (correct for 2PL and graded items; checked here against
  # central differences). Central differences alone are not accurate enough for the graded
  # case, whose Delta is ill-conditioned under item-score margins (the error of the difference
  # quotient, ~1e-10, is amplified to ~4e-6 relative in M2).
  for (i in seq_len(nitems)) {
    x <- extract.item(m, i)
    dp <- mirt:::dP(x, Theta)
    for (k in which(x@est)) {
      xp <- x; xm <- x
      xp@par[k] <- xp@par[k] + eps
      xm@par[k] <- xm@par[k] - eps
      num <- (expected.item(xp, Theta, min = 0L) - expected.item(xm, Theta, min = 0L)) / (2 * eps)
      stopifnot(max(abs(num - dp[, k])) < 1e-8)
      col <- numeric(length(p))
      col[i] <- sum(dp[, k] * Prior)
      ind <- nitems + 1L
      for (b in seq_len(nitems)) for (a in seq_len(nitems)) if (a > b) {
        if (b == i) col[ind] <- sum(dp[, k] * EIs[, a] * Prior)
        if (a == i) col[ind] <- sum(dp[, k] * EIs[, b] * Prior)
        ind <- ind + 1L
      }
      cols[[length(cols) + 1L]] <- col
    }
  }
  if (include_latent) {
    # Latent columns, analytic: with unit variances and Sigma = R, for a < b
    #   d log phi / d rho_ab = [R^-1 theta theta' R^-1 - R^-1]_ab,
    # and for the normalized weights w = phi / sum(phi): dw = w * (g - sum(w * g)).
    gp <- mirt:::ExtractGroupPars(grid$gp_obj)
    stopifnot(all(gp$gmeans == 0), all(diag(gp$gcov) == 1))
    Rinv <- solve(gp$gcov)
    Z <- Theta %*% Rinv
    nfact <- ncol(Theta)
    n_lat <- 0L
    for (a in seq_len(nfact - 1L)) for (b in seq(a + 1L, nfact)) {
      g <- Z[, a] * Z[, b] - Rinv[a, b]
      dPrior <- Prior * (g - sum(Prior * g))
      # check against a central difference of the normalized prior
      k <- which(names(grid$gp_obj@est) == sprintf("COV_%d%d", b, a))
      if (!length(k)) k <- which(grid$gp_obj@parnames == sprintf("COV_%d%d", b, a))
      stopifnot(length(k) == 1L, grid$gp_obj@est[k])
      gp_p <- grid$gp_obj; gp_m <- grid$gp_obj
      gp_p@par[k] <- gp_p@par[k] + eps
      gp_m@par[k] <- gp_m@par[k] - eps
      num <- (grid$prior_of(gp_p) - grid$prior_of(gp_m)) / (2 * eps)
      stopifnot(max(abs(num - dPrior)) < 1e-8)
      cols[[length(cols) + 1L]] <- margins(EIs, dPrior)
      n_lat <- n_lat + 1L
    }
    stopifnot(n_lat == sum(grid$gp_obj@est))
  }
  delta <- do.call(cbind, cols)

  Xi2els <- .Call("buildXi2els", nitems, length(p) - nitems, nitems, EIs, EIs2, Prior,
                  PACKAGE = "mirt")
  Xi2 <- rbind(cbind(Xi2els$Xi11, Xi2els$Xi12), cbind(t(Xi2els$Xi12), Xi2els$Xi22))
  deltac <- qr.Q(qr(delta), complete = TRUE)[, (ncol(delta) + 1L):nrow(delta), drop = FALSE]
  resid <- sqrt(N) * (p - e)
  C2 <- deltac %*% solve(t(deltac) %*% Xi2 %*% deltac, t(deltac))
  M2_val <- as.numeric(t(resid) %*% C2 %*% resid)
  df_val <- length(p) - extract.mirt(m, "nest")
  out <- list(M2 = M2_val, df = df_val, p = 1 - pchisq(M2_val, df_val),
              RMSEA = mirt:::rmsea(X2 = M2_val, df = df_val, N = N), n_delta_cols = ncol(delta))
  rci <- mirt:::RMSEA.CI(M2_val, df_val, N, ci.lower = (1 - ci) / 2, ci.upper = ci + (1 - ci) / 2)
  out$RMSEA_5 <- rci[1L]
  out$RMSEA_95 <- rci[2L]
  if (all(extract.mirt(m, "K") == 2L)) {
    null.mod <- mirt:::computeNullModel(data = m@Data$data, group = m@Data$group, key = m@Internals$key)
    null.fit <- M2(null.mod, calcNull = FALSE, type = "M2*", quadpts = 2)
    out$TLI <- mirt:::tli(X2 = M2_val, X2.null = null.fit$M2, df = df_val, df.null = null.fit$df)
    out$CFI <- mirt:::cfi(X2 = M2_val, X2.null = null.fit$M2, df = df_val, df.null = null.fit$df)
  }
  out
}

sx2_of <- function(m, quad, mincell) {
  fit <- itemfit(m, fit_stats = "S_X2", mincell = mincell, quadpts = quad$quadpts,
                 theta_lim = quad$theta_lim)
  list(mincell = mincell, item = fit$item, S_X2 = fit$S_X2, df = fit$df.S_X2,
       RMSEA = fit$RMSEA.S_X2, p = fit$p.S_X2)
}

run_case <- function(name, mincells = 1) {
  cs <- load_case(name); r <- cs$raw; dat <- cs$data
  quad <- list(quadpts = r$meta$quadpts, theta_lim = unlist(r$meta$theta_lim))
  mod <- mirt.model(model_string(r))
  n_factors <- length(r$factors)
  sv <- mirt(dat, mod, itemtype = r$meta$itemtype, pars = "values", quadpts = quad$quadpts,
             technical = list(theta_lim = quad$theta_lim))
  est_flags <- sv$est
  for (i in unlist(r$items)) {
    p <- r$estimates$params[[i]]
    for (nm in names(p)) {
      if (grepl("_se$", nm) || is.null(p[[nm]])) next
      sel <- sv$item == i & sv$name == nm
      stopifnot(sum(sel) == 1L)
      sv$value[sel] <- p[[nm]]
    }
  }
  lat <- r$estimates$latent
  for (a in seq_len(n_factors - 1L)) for (b in seq(a + 1L, n_factors)) {
    sel <- sv$name == sprintf("COV_%d%d", b, a)
    stopifnot(sum(sel) == 1L, sv$est[sel])
    sv$value[sel] <- lat[[sprintf("cor_%d%d", a, b)]]
  }
  stopifnot(identical(sv$est, est_flags))
  m <- mirt(dat, mod, itemtype = r$meta$itemtype, pars = sv, TOL = NaN, verbose = FALSE,
            quadpts = quad$quadpts, technical = list(theta_lim = quad$theta_lim))
  stopifnot(abs(extract.mirt(m, "logLik") - r$loglik) < 1e-8 * abs(r$loglik),
            nrow(m@Model$Theta) == quad$quadpts^n_factors)

  raw <- M2(m, quadpts = quad$quadpts, theta_lim = quad$theta_lim)
  ex_no <- exact_m2_star(m, quad, include_latent = FALSE)
  ex_lat <- exact_m2_star(m, quad, include_latent = TRUE)
  # The only difference between mirt's raw M2 and the exact statistic is the missing latent column.
  stopifnot(abs(ex_no$M2 - raw$M2) < 1e-10 * raw$M2, ex_no$df == raw$df)
  cat(sprintf("%-20s M2 raw = %.6f  exact(no latent) = %.6f  exact(latent) = %.6f  df = %d\n",
              name, raw$M2, ex_no$M2, ex_lat$M2, raw$df))
  sx2 <- lapply(mincells, function(mc) sx2_of(m, quad, mc))
  names(sx2) <- sprintf("mincell_%g", mincells)
  for (k in names(sx2)) cat(sprintf("  S-X2 %-10s df = %s\n", k, paste(sx2[[k]]$df, collapse = ",")))
  list(
    source = "v14_mirt2d", case = name, itemtype = r$meta$itemtype, n_factors = n_factors,
    quadpts = quad$quadpts, theta_lim = quad$theta_lim, n_nodes = nrow(m@Model$Theta),
    nest = extract.mirt(m, "nest"), n_free_latent = sum(sv$est[sv$item == "GROUP"]),
    loglik = extract.mirt(m, "logLik"),
    m2 = list(mirt = as.list(raw[1, ]), exact = ex_no, exact_with_latent = ex_lat),
    sx2 = sx2
  )
}

cases <- list(
  mirt2d_2pl_simple = run_case("mirt2d_2pl_simple"),
  mirt2d_2pl_cross = run_case("mirt2d_2pl_cross"),
  mirt2d_grm = run_case("mirt2d_grm", mincells = c(1, 3)),
  mirt3d_2pl = run_case("mirt3d_2pl")
)
res <- list(
  meta = list(case = "v39_fit_multidim", r_version = R.version.string,
              mirt_version = mirt_version_string(), mirt_sha = mirt_commit_sha(), converged = TRUE,
              description = paste("V39: M2 / M2* and S-X2 for confirmatory 2-3D MIRT at mirt's",
                                  "V14 estimates (TOL = NaN), on the fit's product grid")),
  cases = cases
)
write_json(res, file.path("tests", "reference", "v39_fit_multidim.json"), auto_unbox = TRUE,
           digits = NA, pretty = TRUE, na = "null")
cat("wrote tests/reference/v39_fit_multidim.json\n")
