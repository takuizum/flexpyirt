# Limited-information goodness of fit: M2 / M2* (VALIDATION.md V18) from R mirt's M2().
# Run from the repository root:  Rscript validation/r/m5_m2_reference.R
# Writes tests/reference/v18_m2.json. CI does not run R; the JSON is committed.
#
# Each case reuses an existing reference (data and mirt's estimates). The model is rebuilt with
# every parameter at those estimates (est flags kept, so the parameter count is unchanged) and
# fitted with TOL = NaN, so M2() sees exactly the reference estimates. M2() is evaluated on the
# fit's quadrature grid. New simulated cases (V18b, V18c) are fitted here with the usual settings.
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

stats_of <- function(m, quad, label) {
  out <- M2(m, quadpts = quad$quadpts, theta_lim = quad$theta_lim)
  cat(sprintf("%-22s M2 = %.4f  df = %d\n", label, out$M2, out$df))
  res <- as.list(out[1, ])
  res$nest <- extract.mirt(m, "nest")
  res$loglik <- extract.mirt(m, "logLik")
  res
}

refit_at <- function(name, itemtype, map_name = function(nm) if (nm == "a") "a1" else nm) {
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
  c(list(source = name, itemtype = itemtype), stats_of(m, quad, name))
}

common <- list(TOL = 1e-10, accelerate = "none", optimizer = "NR", verbose = FALSE)
default_quad <- list(quadpts = 61, theta_lim = c(-6, 6))

fit_new <- function(dat, itemtype, label) {
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
    stats_of(m, default_quad, label))
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
  threepl = refit_at("v11_3pl", "3PL", function(nm) if (nm == "a") "a1" else nm),
  rasch_free_var = refit_at("v2_sim_rasch", "Rasch"),
  grm_sim = fit_new(grm, "graded", "grm_sim"),
  gpcm_sim = fit_new(gpcm, "gpcm", "gpcm_sim")
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
