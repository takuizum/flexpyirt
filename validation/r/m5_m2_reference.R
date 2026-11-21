# Limited-information goodness of fit: M2 / M2* (VALIDATION.md V18) from R mirt's M2().
# Run from the repository root:  Rscript validation/r/m5_m2_reference.R
# Writes tests/reference/v18_m2.json. CI does not run R; the JSON is committed.
#
# Each case reuses an existing reference (data and mirt's estimates). The model is rebuilt with
# every parameter at those estimates (est flags kept, so the parameter count is unchanged) and
# fitted with TOL = NaN, so M2() sees exactly the reference estimates. M2() is evaluated on the
# fit's quadrature grid. New simulated cases (grm_sim, gpcm_sim) are fitted here with the usual
# settings.
#
# Confirmed mirt 1.46.1 M2() issues cross-checked in R (VALIDATION.md V18):
#   1. dP("dich") uses the 2PL derivative P_0 * P_1 for (a1, d) instead of (u - g) * sigma * (1 - sigma)
#      and puts raw probabilities (P_0, P_1) in columns 3-4 for (logit(g), logit(u)) instead of
#      ((1 - sigma) * g * (1 - g), sigma * u * (1 - u)), shifting threepl's M2 by ~4.8% (relative to exact).
#   2. dP("gpcm") omits division by den = rowSums(num) in the a1 slope derivative
#      (rowSums(Theta[, i] * aknum[[i]]) instead of / den), shifting gpcm_sim's M2 by ~10.8% (relative to exact).
#   3. GroupPars (free latent mean/variance) are forced to est = FALSE in M2()'s Delta while still
#      subtracted in df, affecting rasch_free_var's M2 by ~4.7e-4.
#   4. For polytomous null models (grm_sim, gpcm_sim), Delta has rank J < sum(K_j - 1) = ncol(Delta);
#      mirt slices qr.Q(qr(delta), complete = TRUE) after ncol(delta), discarding arbitrary directions
#      that vary with item ordering.
# For threepl, gpcm_sim, and rasch_free_var, this script keeps mirt's raw M2() output untouched and
# also records exact R M2* values computed with central-difference derivatives of expected.item()
# and mirt's own buildXi2els C++ routine.
suppressMessages({
  library(mirt)
  library(jsonlite)
})

load_case <- function(name) {
  r <- fromJSON(file.path("tests", "reference", paste0(name, ".json")), simplifyVector = FALSE)
  pats <- do.call(rbind, lapply(r$patterns, function(x) unlist(lapply(x, function(v) if (is.null(v)) NA else v))))
  colnames(pats) <- unlist(r$items)
  reps <- rep(seq_len(nrow(pats)), unlist(r$counts))
  grp <- if (!is.null(r$pattern_groups)) unlist(r$pattern_groups)[reps] else NULL
  list(raw = r, data = as.data.frame(pats[reps, , drop = FALSE]), group = grp)
}

exact_m2_star <- function(m, quad, include_latent = FALSE, ci = 0.90) {
  dat <- t(t(m@Data$data) - m@Data$mins)
  N <- nrow(dat)
  nitems <- ncol(dat)
  p <- colMeans(dat)
  cross <- crossprod(dat, dat)
  p <- c(p, cross[lower.tri(cross)] / N)

  pars <- m@ParObjects$pars
  gp <- mirt:::ExtractGroupPars(pars[[nitems + 1L]])
  Theta <- matrix(seq(quad$theta_lim[1L], quad$theta_lim[2L], length.out = quad$quadpts))
  Prior <- mirt:::mirt_dmvnorm(Theta, gp$gmeans, gp$gcov)
  Prior <- Prior / sum(Prior)

  EIs <- EIs2 <- matrix(0, nrow(Theta), nitems)
  dp_list <- list()
  eps <- 1e-6
  for (i in seq_len(nitems)) {
    x <- extract.item(m, i)
    scs <- 0:(x@ncat - 1L)
    EIs[, i] <- expected.item(x, Theta, min = 0L)
    prob <- mirt:::ProbTrace(x, Theta)
    EIs2[, i] <- as.vector(prob %*% (scs^2))
    free_idx <- which(x@est)
    dpi <- matrix(0, nrow(Theta), length(free_idx))
    for (k in seq_along(free_idx)) {
      xp <- x; xm <- x
      xp@par[free_idx[k]] <- xp@par[free_idx[k]] + eps
      xm@par[free_idx[k]] <- xm@par[free_idx[k]] - eps
      dpi[, k] <- (expected.item(xp, Theta, min = 0L) - expected.item(xm, Theta, min = 0L)) / (2 * eps)
    }
    dp_list[[i]] <- dpi
  }

  E1 <- colSums(EIs * Prior)
  E2 <- matrix(0, nitems, nitems)
  for (i in seq_len(nitems)) for (j in seq_len(nitems)) if (i > j) E2[i, j] <- sum(EIs[, i] * EIs[, j] * Prior)
  e <- c(E1, E2[lower.tri(E2)])

  n_free_item <- sum(vapply(dp_list, ncol, integer(1)))
  delta <- matrix(0, length(p), n_free_item)
  col_off <- 0L
  for (i in seq_len(nitems)) {
    nc <- ncol(dp_list[[i]])
    if (nc > 0L) {
      cols <- (col_off + 1L):(col_off + nc)
      delta[i, cols] <- colSums(dp_list[[i]] * Prior)
      ind <- nitems + 1L
      for (ii in seq_len(nitems)) {
        for (jj in seq_len(nitems)) {
          if (ii < jj) {
            if (ii == i) delta[ind, cols] <- colSums(dp_list[[i]] * EIs[, jj] * Prior)
            if (jj == i) delta[ind, cols] <- colSums(dp_list[[i]] * EIs[, ii] * Prior)
            ind <- ind + 1L
          }
        }
      }
      col_off <- col_off + nc
    }
  }

  if (include_latent) {
    gp_obj <- pars[[nitems + 1L]]
    lat_free <- which(gp_obj@est)
    if (length(lat_free) > 0L) {
      dlat <- matrix(0, length(p), length(lat_free))
      for (k in seq_along(lat_free)) {
        gp_p <- gp_obj; gp_m <- gp_obj
        gp_p@par[lat_free[k]] <- gp_p@par[lat_free[k]] + eps
        gp_m@par[lat_free[k]] <- gp_m@par[lat_free[k]] - eps
        ep <- mirt:::ExtractGroupPars(gp_p); em <- mirt:::ExtractGroupPars(gp_m)
        Pp <- mirt:::mirt_dmvnorm(Theta, ep$gmeans, ep$gcov); Pp <- Pp / sum(Pp)
        Pm <- mirt:::mirt_dmvnorm(Theta, em$gmeans, em$gcov); Pm <- Pm / sum(Pm)
        dPrior <- (Pp - Pm) / (2 * eps)
        dE1 <- colSums(EIs * dPrior)
        dE2 <- matrix(0, nitems, nitems)
        for (i in seq_len(nitems)) for (j in seq_len(nitems)) if (i > j) dE2[i, j] <- sum(EIs[, i] * EIs[, j] * dPrior)
        dlat[, k] <- c(dE1, dE2[lower.tri(dE2)])
      }
      delta <- cbind(delta, dlat)
    }
  }

  Xi2els <- .Call("buildXi2els", nitems, length(p) - nitems, nitems, EIs, EIs2, Prior, PACKAGE = "mirt")
  Xi2 <- rbind(cbind(Xi2els$Xi11, Xi2els$Xi12), cbind(t(Xi2els$Xi12), Xi2els$Xi22))
  sv <- svd(delta, nu = nrow(delta))
  tol <- max(sv$d) * max(dim(delta)) * .Machine$double.eps
  rk <- sum(sv$d > tol)
  deltac <- sv$u[, (rk + 1L):nrow(delta), drop = FALSE]
  resid <- sqrt(N) * (p - e)
  proj <- crossprod(deltac, resid)
  M2_val <- as.numeric(abs(crossprod(proj, solve(crossprod(deltac, Xi2 %*% deltac), proj))))
  df_val <- length(p) - extract.mirt(m, "nestpars")
  alpha <- (1 - ci) / 2
  ci_vals <- mirt:::RMSEA.CI(M2_val, df_val, N, ci.lower = alpha, ci.upper = 1 - alpha)
  out <- list(
    M2 = M2_val,
    df = df_val,
    p = pchisq(M2_val, df_val, lower.tail = FALSE),
    RMSEA = mirt:::rmsea(X2 = M2_val, df = df_val, N = N),
    RMSEA_5 = ci_vals[1L],
    RMSEA_95 = ci_vals[2L]
  )
  if (all(m@Data$K == 2L)) {
    null.mod <- mirt:::computeNullModel(data = m@Data$data, group = m@Data$group, key = m@Internals$key)
    null.fit <- M2(null.mod, calcNull = FALSE, type = "M2*", quadpts = 2)
    out$TLI <- mirt:::tli(X2 = M2_val, X2.null = null.fit$M2, df = df_val, df.null = null.fit$df)
    out$CFI <- mirt:::cfi(X2 = M2_val, X2.null = null.fit$M2, df = df_val, df.null = null.fit$df)
  }
  out
}

stats_of <- function(m, quad, label, record_exact = FALSE, record_exact_latent = FALSE) {
  out <- M2(m, quadpts = quad$quadpts, theta_lim = quad$theta_lim)
  cat(sprintf("%-22s M2 = %.4f  df = %d\n", label, out$M2, out$df))
  res <- as.list(out[1, ])
  res$nest <- extract.mirt(m, "nest")
  res$loglik <- extract.mirt(m, "logLik")
  if (record_exact) {
    ex <- exact_m2_star(m, quad, include_latent = FALSE)
    res$exact <- ex
    cat(sprintf("  exact R M2*          M2 = %.4f  (mirt diff = %.2f%%)\n",
                ex$M2, 100 * abs(ex$M2 - out$M2) / ex$M2))
  }
  if (record_exact_latent) {
    ex_no <- exact_m2_star(m, quad, include_latent = FALSE)
    stopifnot(abs(ex_no$M2 - out$M2) < 1e-9 * out$M2)
    ex_lat <- exact_m2_star(m, quad, include_latent = TRUE)
    res$exact_with_latent <- ex_lat
    cat(sprintf("  exact R (lat in d)   M2 = %.4f  (mirt nolat = %.4f)\n", ex_lat$M2, out$M2))
  }
  res
}

refit_at <- function(name, itemtype, map_name = function(nm) if (nm == "a") "a1" else nm,
                     record_exact = FALSE, record_exact_latent = FALSE) {
  cs <- load_case(name); r <- cs$raw; dat <- cs$data
  quad <- list(quadpts = r$meta$quadpts, theta_lim = unlist(r$meta$theta_lim))
  est <- if (!is.null(r$estimates)) r$estimates[[1]] else list(params = r$params, latent = r$latent)
  sv <- mirt(dat, 1, itemtype = itemtype, pars = "values", quadpts = quad$quadpts,
             technical = list(theta_lim = quad$theta_lim))
  for (i in unlist(r$items)) {
    p <- est$params[[i]]
    for (nm in names(p)) {
      if (grepl("_se$", nm) || is.null(p[[nm]])) next
      sel <- sv$item == i & sv$name == map_name(nm)
      if (any(sel)) sv$value[sel] <- p[[nm]]
    }
  }
  lat <- est$latent
  if (!is.null(lat$var)) sv$value[sv$name == "COV_11"] <- lat$var
  if (!is.null(lat$mean)) sv$value[sv$name == "MEAN_1"] <- lat$mean
  sv$est[sv$name == "COV_11"] <- !is.null(lat$var_se)
  sv$est[sv$name == "MEAN_1"] <- !is.null(lat$mean_se)
  m <- mirt(dat, 1, itemtype = itemtype, pars = sv, TOL = NaN, verbose = FALSE,
            quadpts = quad$quadpts, technical = list(theta_lim = quad$theta_lim))
  stopifnot(abs(extract.mirt(m, "logLik") - r$loglik) < 1e-6 * abs(r$loglik))
  c(list(source = name, itemtype = itemtype),
    stats_of(m, quad, name, record_exact = record_exact, record_exact_latent = record_exact_latent))
}

common <- list(TOL = 1e-10, accelerate = "none", optimizer = "NR", verbose = FALSE)
default_quad <- list(quadpts = 61, theta_lim = c(-6, 6))

fit_new <- function(dat, itemtype, label, record_exact = FALSE) {
  m <- do.call(mirt, c(list(dat, 1, itemtype = itemtype, quadpts = 61,
                            technical = list(NCYCLES = 50000, theta_lim = c(-6, 6))), common))
  stopifnot(extract.mirt(m, "converged"))
  cf <- coef(m, simplify = TRUE)$items
  cf <- cf[, colnames(cf) == "a1" | grepl("^d[0-9]*$", colnames(cf)), drop = FALSE]
  colnames(cf) <- sub("^a1$", "a", colnames(cf))
  params <- lapply(seq_len(nrow(cf)), function(k) as.list(cf[k, !is.na(cf[k, ])]))
  names(params) <- rownames(cf)
  key <- apply(dat, 1, paste, collapse = ",")
  first <- !duplicated(key)
  c(list(source = NULL, itemtype = itemtype, items = colnames(dat),
         patterns = unname(as.matrix(dat[first, ])), counts = as.integer(table(key)[key[first]]),
         params = params),
    stats_of(m, default_quad, label, record_exact = record_exact))
}

# V18b: graded, J = 10, K = 3 (M2* on item-score moments).
set.seed(20261126)
a <- matrix(rlnorm(10, 0.1, 0.3)); d <- t(apply(matrix(rnorm(20), 10), 1, sort, decreasing = TRUE))
grm <- as.data.frame(simdata(a, d, 2000, itemtype = "graded")); colnames(grm) <- sprintf("g%02d", 1:10)
# V18c: GPCM, J = 8, K = 4.
set.seed(20261127)
a <- matrix(rlnorm(8, 0, 0.3)); d <- cbind(0, matrix(rnorm(24, 0, 0.8), 8))
d <- t(apply(d, 1, cumsum))
gpcm <- as.data.frame(simdata(a, d, 2000, itemtype = "gpcm")); colnames(gpcm) <- sprintf("p%02d", 1:8)

cases <- list(
  lsat7_2pl = refit_at("v1_lsat7_2pl", "2PL"),
  sim_2pl = refit_at("v1_sim_2pl", "2PL"),
  threepl = refit_at("v11_3pl", "3PL", function(nm) if (nm == "a") "a1" else nm, record_exact = TRUE),
  rasch_free_var = refit_at("v2_sim_rasch", "Rasch", record_exact_latent = TRUE),
  grm_sim = fit_new(grm, "graded", "grm_sim"),
  gpcm_sim = fit_new(gpcm, "gpcm", "gpcm_sim", record_exact = TRUE)
)
res <- list(
  meta = list(case = "v18_m2", r_version = R.version.string,
              mirt_version = as.character(packageVersion("mirt")), converged = TRUE,
              TOL = 1e-10, optimizer = "NR", accelerate = "none"),
  cases = cases
)
write_json(res, file.path("tests", "reference", "v18_m2.json"), auto_unbox = TRUE, digits = NA,
           pretty = TRUE, na = "null")
cat("wrote tests/reference/v18_m2.json\n")
