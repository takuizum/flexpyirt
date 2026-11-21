# S-X2 item fit reference values (VALIDATION.md V15) from R mirt's itemfit().
# Run from the repository root:  Rscript validation/r/m5_itemfit_reference.R
# Writes tests/reference/v15_sx2.json. CI does not run R; the JSON is committed.
#
# Each case reuses an existing reference (data and mirt's estimates). The model is rebuilt with
# every parameter set to those estimates, keeping each parameter's `est` flag (free items and
# latent parameters stay free so they count against the S-X2 degrees of freedom, fixed FIPC bank
# items stay fixed), and fitted with TOL = NaN so no EM step moves anything. itemfit() is then
# evaluated on the same quadrature grid as the fit (its default would be 61 points on [-6, 6]).
suppressMessages({
  library(mirt)
  library(jsonlite)
})

load_case <- function(name) {
  r <- fromJSON(file.path("tests", "reference", paste0(name, ".json")), simplifyVector = FALSE)
  pats <- do.call(rbind, lapply(r$patterns, function(x) unlist(lapply(x, function(v) if (is.null(v)) NA else v))))
  colnames(pats) <- unlist(r$items)
  list(raw = r, data = as.data.frame(pats[rep(seq_len(nrow(pats)), unlist(r$counts)), , drop = FALSE]))
}

run_case <- function(name, itemtype, mincell = 1, label = name) {
  cs <- load_case(name); r <- cs$raw; dat <- cs$data
  quad <- list(quadpts = r$meta$quadpts, theta_lim = unlist(r$meta$theta_lim))
  est <- r$estimates[[1]]
  sv <- mirt(dat, 1, itemtype = itemtype, pars = "values", quadpts = quad$quadpts,
             technical = list(theta_lim = quad$theta_lim))
  for (i in unlist(r$items)) {
    p <- est$params[[i]]
    for (nm in names(p)) {
      if (grepl("_se$", nm) || is.null(p[[nm]])) next
      mname <- if (nm == "a") "a1" else nm
      sel <- sv$item == i & sv$name == mname
      if (any(sel)) sv$value[sel] <- p[[nm]]
    }
  }
  if (!is.null(r$bank)) for (b in r$bank) sv$est[sv$item == b$item & sv$name %in% c("a1", "d")] <- FALSE
  lat <- est$latent
  sv$value[sv$name == "MEAN_1"] <- lat$mean
  sv$value[sv$name == "COV_11"] <- lat$var
  sv$est[sv$name == "MEAN_1"] <- !is.null(lat$mean_se)
  sv$est[sv$name == "COV_11"] <- !is.null(lat$var_se)
  m <- mirt(dat, 1, itemtype = itemtype, pars = sv, TOL = NaN, verbose = FALSE,
            quadpts = quad$quadpts, technical = list(theta_lim = quad$theta_lim))
  stopifnot(abs(extract.mirt(m, "logLik") - r$loglik) < 1e-6 * abs(r$loglik))
  fit <- itemfit(m, fit_stats = "S_X2", mincell = mincell, quadpts = quad$quadpts,
                 theta_lim = quad$theta_lim)
  cat(sprintf("%-26s items=%d  df=%s\n", label, nrow(fit), paste(fit$df.S_X2, collapse = ",")))
  list(source = name, itemtype = itemtype, mincell = mincell, n_free_latent = sum(sv$est[sv$item == "GROUP"]),
       item = fit$item, S_X2 = fit$S_X2, df = fit$df.S_X2, RMSEA = fit$RMSEA.S_X2, p = fit$p.S_X2)
}

cases <- list(
  grm_science = run_case("v3_science_grm", "graded"),
  grm_science_mincell3 = run_case("v3_science_grm", "graded", mincell = 3, label = "grm_science_mincell3"),
  gpcm_science = run_case("v8_science_gpcm", "gpcm"),
  pcm_sim = run_case("v8_sim_pcm", "Rasch"),
  threepl_sim = run_case("v11_3pl", "3PL"),
  fipc_2pl = run_case("v5_fipc_2pl", "2PL")
)
res <- list(
  meta = list(case = "v15_sx2", r_version = R.version.string,
              mirt_version = as.character(packageVersion("mirt")), converged = TRUE),
  cases = cases
)
write_json(res, file.path("tests", "reference", "v15_sx2.json"), auto_unbox = TRUE, digits = NA,
           pretty = TRUE, na = "null")
cat("wrote tests/reference/v15_sx2.json\n")
