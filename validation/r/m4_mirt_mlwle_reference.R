# ML and WLE scoring reference for confirmatory 2-3 dimensional MIRT (VALIDATION.md V14) from R mirt.
# Run from the repository root:  Rscript validation/r/m4_mirt_mlwle_reference.R
# Reads tests/reference/v14_mirt2d.json and writes tests/reference/v14_mirt_mlwle.json.
#
# For each V14 case the model is rebuilt with every parameter fixed at mirt's estimates in the V14
# reference (est = FALSE, TOL = NaN, so no EM step moves anything), and fscores(method = "MAP")
# fscores(method = "ML" / "WLE") scores the same unique patterns used for EAP plus copies with
# some responses set to missing, all-lowest / all-highest patterns, and patterns that are extreme
# on one factor only. SEs are mirt's defaults (inverse nlm Hessian).
source(file.path("validation", "r", "setup.R"))  # mirt from GitHub; mirt_version / mirt_sha
suppressMessages({
  library(mirt)
  library(jsonlite)
})

v14 <- fromJSON(file.path("tests", "reference", "v14_mirt2d.json"), simplifyVector = FALSE)
out <- list()
for (case in names(v14$cases)) {
  cs <- v14$cases[[case]]
  items <- unlist(cs$items)
  fnames <- names(cs$factors)
  D <- length(fnames)
  pats <- do.call(rbind, lapply(cs$patterns, function(r) unlist(lapply(r, function(v) if (is.null(v)) NA else v))))
  colnames(pats) <- items
  dat <- as.data.frame(pats[rep(seq_len(nrow(pats)), unlist(cs$counts)), , drop = FALSE])
  spec <- paste(c(
    vapply(fnames, function(f) paste0(f, " = ", paste(unlist(cs$factors[[f]]), collapse = ", ")), ""),
    paste0("COV = ", paste(fnames, collapse = "*"))
  ), collapse = "\n")
  itemtype <- if (cs$meta$itemtype == "graded") "graded" else "2PL"
  quad <- list(quadpts = cs$meta$quadpts, theta_lim = unlist(cs$meta$theta_lim))
  sv <- mirt(dat, mirt.model(spec), itemtype = itemtype, pars = "values",
             quadpts = quad$quadpts, technical = list(theta_lim = quad$theta_lim))
  for (i in items) {
    p <- cs$estimates$params[[i]]
    for (nm in names(p)) {
      if (grepl("_se$", nm)) next
      sel <- sv$item == i & sv$name == nm
      if (any(sel)) sv$value[sel] <- p[[nm]]
    }
  }
  lat <- cs$estimates$latent
  for (a in seq_len(D)) for (b in seq_len(a)) {
    nm <- sprintf("COV_%d%d", a, b)
    val <- if (a == b) lat[[sprintf("var_%d", a)]] else lat[[sprintf("cor_%d%d", b, a)]]
    sv$value[sv$item == "GROUP" & sv$name == nm] <- val
  }
  sv$est <- FALSE
  m <- mirt(dat, mirt.model(spec), itemtype = itemtype, pars = sv, TOL = NaN, verbose = FALSE,
            quadpts = quad$quadpts, technical = list(theta_lim = quad$theta_lim))
  stopifnot(abs(extract.mirt(m, "logLik") - cs$loglik) < 1e-6 * abs(cs$loglik))

  idx <- unlist(cs$eap$pattern_indices) + 1L
  score_pats <- pats[idx, , drop = FALSE]
  set.seed(7)
  miss <- score_pats[seq_len(min(20L, nrow(score_pats))), , drop = FALSE]
  for (r in seq_len(nrow(miss))) miss[r, sample(ncol(miss), 3L)] <- NA
  score_pats <- rbind(score_pats, miss)
  ext <- rbind(rep(0, ncol(pats)), apply(pats, 2, max))
  f1 <- unlist(cs$factors[[1]])
  one <- score_pats[1, ]; one[items %in% f1] <- 0
  score_pats <- rbind(score_pats, ext, one)
  res_m <- list()
  for (meth in c("ML", "WLE")) {
    sc <- suppressWarnings(fscores(m, method = meth, response.pattern = score_pats))
    enc <- function(v) lapply(v, function(x) if (is.infinite(x)) if (x > 0) "Infinity" else "-Infinity" else x)
    res_m[[meth]] <- list(
      # +/-Inf would be written as null by jsonlite; encode them as strings.
      theta = setNames(lapply(fnames, function(f) enc(unname(sc[, f]))), fnames),
      se = setNames(lapply(fnames, function(f) unname(sc[, paste0("SE_", f)])), fnames)
    )
  }
  out[[case]] <- c(list(items = items, patterns = unname(score_pats)), res_m)
  cat(sprintf("%-20s %d patterns scored (loglik check passed)\n", case, nrow(score_pats)))
}
res <- list(
  meta = list(case = "v14_mirt_mlwle", r_version = R.version.string,
              mirt_version = mirt_version_string(), mirt_sha = mirt_commit_sha(),
              source = "tests/reference/v14_mirt2d.json (all parameters fixed at mirt's estimates)",
              converged = TRUE),
  cases = out
)
write_json(res, file.path("tests", "reference", "v14_mirt_mlwle.json"), auto_unbox = TRUE,
           digits = NA, pretty = TRUE, na = "null")
cat("wrote tests/reference/v14_mirt_mlwle.json\n")
