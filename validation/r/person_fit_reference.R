# Person fit and residual item fit reference values (VALIDATION.md V53, DESIGN.md §25) from R.
# Run from the repository root:  Rscript validation/r/person_fit_reference.R
# Writes tests/reference/v53_person_fit.json. CI does not run R; the JSON is committed.
#
# Data and mirt's estimates come from V37 (tests/reference/v37_information.json: 2PL, 2PL with 8%
# missing, 3PL, GRM, GPCM, nominal, a mixed-format test and a two-group 2PL + GRM model). Each
# model is rebuilt with every parameter fixed at those estimates (est = FALSE), on the V37 grid
# (61 points on [-6, 6]). Per case and per ability estimator (EAP, MAP, ML, WLE):
#   - theta = fscores(m, method = , full.scores = TRUE) on that grid
#   - mirt::personfit(m, Theta = theta): Zh, infit, z.infit, outfit, z.outfit
#   - z_outfit with q^2 = |sum(C / W^2) / n^2 - 1 / n| over the n answered items (Wright & Masters,
#     1982), from mirt's own residual moments (attr "inoutfitreturn"). mirt's personfit() divides
#     by J^2 (all items) instead of n^2 under the square root, which differs only for rows with
#     missing responses (DESIGN.md §25).
#   - l_z* (Snijders, 2001; Sinharay, 2016) recomputed here from the formulas: category
#     probabilities from mirt's probtrace(), their theta derivatives by central differences
#     (h = 1e-5), item information from mirt's iteminfo() and its derivative (WLE) by central
#     differences (h = 1e-4). There is no installed reference (PerFit is not installable here).
#   - mirt::itemfit(m, fit_stats = "infit", Theta = theta) for EAP, MAP and WLE (ML gives
#     infinite estimates, which mirt replaces by the finite extremes +/- 0.1), and item z_outfit
#     recomputed from mirt's residual moments with the per-item respondent count (mirt's
#     itemfit() recycles that count over rows; this matters only with missing responses).
# Person statistics are stored for the first N_STORE unique patterns of each case (all of them
# are computed on the full data); item statistics for every item.
#
# Rasch case (rasch_sirt): the V37 2PL data refitted as a Rasch model (free variance); WLE
# person and item fit from mirt, and sirt::personfit.stat(dat, abil = theta, b = -d) as an
# independent implementation of lz, infit and outfit.
suppressMessages({
  library(mirt)
  library(jsonlite)
  library(sirt)
})

N_STORE <- 300L
METHODS <- c("EAP", "MAP", "ML", "WLE")
v37 <- fromJSON(file.path("tests", "reference", "v37_information.json"), simplifyVector = FALSE)
quadpts <- v37$meta$quadpts
theta_lim <- unlist(v37$meta$theta_lim)

pat_matrix <- function(rows, items) {
  m <- do.call(rbind, lapply(rows, function(x) {
    vapply(x, function(v) if (is.null(v)) NA_real_ else as.numeric(v), numeric(1))
  }))
  colnames(m) <- items
  m
}

encode_num <- function(x) {
  lapply(as.numeric(x), function(v) {
    if (is.na(v)) NA_real_ else if (is.infinite(v)) (if (v > 0) "Inf" else "-Inf") else v
  })
}

# Put flexpyirt-named estimates (a, d, g, u, d1.., ak1..) into a mirt start-value table.
set_values <- function(sv, items, params, latent, group = NULL) {
  in_group <- if (is.null(group)) rep(TRUE, nrow(sv)) else sv$group == group
  for (i in items) {
    p <- params[[i]]
    for (nm in names(p)) {
      mname <- if (nm == "a") "a1" else nm
      sel <- in_group & sv$item == i & sv$name == mname
      stopifnot(sum(sel) == 1L)
      sv$value[sel] <- p[[nm]]
    }
  }
  sv$value[in_group & sv$name == "MEAN_1"] <- latent$mean
  sv$value[in_group & sv$name == "COV_11"] <- latent$var
  sv
}

scores <- function(m, method) {
  fscores(m, method = method, full.scores = TRUE, quadpts = quadpts, theta_lim = theta_lim,
          verbose = FALSE)[, 1]
}

# z_outfit with n (answered items) in q^2, from mirt's residual moments (single group).
z_outfit_answered <- function(m, theta) {
  attr(m, "inoutfitreturn") <- TRUE
  pf <- personfit(m, Theta = matrix(theta))
  n <- rowSums(!is.na(pf$resid))
  outfit <- rowSums(pf$resid^2 / pf$W, na.rm = TRUE) / n
  # abs() as in mirt's personfit(): with W floored at sqrt(1e-5) the difference can be < 0.
  q <- sqrt(abs(rowSums(pf$C / pf$W^2, na.rm = TRUE) / n^2 - 1 / n))
  q[q > 1.4142] <- 1.4142
  (outfit^(1 / 3) - 1) * (3 / q) + q / 3
}

# l_z* from the formulas (single-group model object, its prior mean / variance).
lz_star <- function(m, dat, theta, method, mu, s2) {
  h <- 1e-5
  h_info <- 1e-4
  n_items <- ncol(dat)
  items <- lapply(seq_len(n_items), function(j) extract.item(m, j))
  out <- rep(NA_real_, nrow(dat))
  for (i in seq_len(nrow(dat))) {
    t <- theta[i]
    if (!is.finite(t)) next
    answered <- which(!is.na(dat[i, ]))
    l0 <- e0 <- v0 <- cov0 <- info0 <- 0
    for (j in answered) {
      p <- probtrace(items[[j]], matrix(t))[1, ]
      dp <- (probtrace(items[[j]], matrix(t + h))[1, ] -
               probtrace(items[[j]], matrix(t - h))[1, ]) / (2 * h)
      lp <- log(p)
      mu_j <- sum(p * lp)
      l0 <- l0 + lp[dat[i, j] + 1]
      e0 <- e0 + mu_j
      v0 <- v0 + sum(p * lp^2) - mu_j^2
      cov0 <- cov0 + sum(dp * lp)
      info0 <- info0 + sum(dp^2 / p)
    }
    test_info <- function(s) sum(vapply(answered, function(j) iteminfo(items[[j]], matrix(s)),
                                        numeric(1)))
    r0 <- switch(method,
                 ML = 0,
                 MAP = -(t - mu) / s2,
                 WLE = (test_info(t + h_info) - test_info(t - h_info)) / (2 * h_info) /
                   (2 * test_info(t)))
    cc <- cov0 / info0
    tau2 <- v0 - 2 * cc * cov0 + cc^2 * info0
    out[i] <- (l0 - e0 + cc * r0) / sqrt(tau2)
  }
  out
}

person_block <- function(theta, pf, z_out_n, lzs, store) {
  list(theta = encode_num(theta[store]), lz = encode_num(pf$Zh[store]),
       infit = encode_num(pf$infit[store]), z_infit = encode_num(pf$z.infit[store]),
       outfit = encode_num(pf$outfit[store]), z_outfit_mirt = encode_num(pf$z.outfit[store]),
       z_outfit = encode_num(z_out_n[store]),
       lz_star = if (is.null(lzs)) NULL else encode_num(lzs[store]))
}

# Item z_outfit from mirt's residual moments with q^2 = sum_i(C / W^2) / N_j^2 - 1 / N_j per item.
# mirt's itemfit() writes colSums((C / W^2) / N^2) with N a length-J vector, which R recycles
# down the rows instead of across the columns; with missing responses (N_j unequal) its
# z.outfit is off. Without missing responses the two agree.
item_z_outfit <- function(m, theta) {
  attr(m, "inoutfitreturn") <- TRUE
  pf <- personfit(m, Theta = matrix(theta))
  n_j <- colSums(!is.na(pf$resid))
  outfit <- colSums(pf$resid^2 / pf$W, na.rm = TRUE) / n_j
  q <- sqrt(colSums(pf$C / pf$W^2, na.rm = TRUE) / n_j^2 - 1 / n_j)
  q[q > 1.4142] <- 1.4142
  (outfit^(1 / 3) - 1) * (3 / q) + q / 3
}

item_block <- function(f, z_outfit = f$z.outfit) {
  list(item = f$item, infit = f$infit, z_infit = f$z.infit, outfit = f$outfit,
       z_outfit = z_outfit, z_outfit_mirt = f$z.outfit)
}

# --- single-group cases ---------------------------------------------------------------------
single_case <- function(name) {
  ref <- v37$cases[[name]]
  items <- unlist(ref$items)
  itemtype <- unlist(ref$itemtype)
  pats <- pat_matrix(ref$patterns, items)
  counts <- unlist(ref$counts)
  dat <- as.data.frame(pats[rep(seq_len(nrow(pats)), counts), , drop = FALSE])
  sv <- mirt(dat, 1, itemtype = itemtype, pars = "values")
  sv <- set_values(sv, items, ref$params, ref$latent)
  sv$est <- FALSE
  m <- mirt(dat, 1, itemtype = itemtype, pars = sv, verbose = FALSE, quadpts = quadpts,
            technical = list(theta_lim = theta_lim))
  first <- cumsum(c(1L, counts))[seq_along(counts)]
  store <- first[seq_len(min(N_STORE, length(first)))]
  persons <- list()
  item_fit <- list()
  for (method in METHODS) {
    theta <- scores(m, method)
    pf <- personfit(m, Theta = matrix(theta))
    lzs <- if (method == "EAP") NULL else
      lz_star(m, as.matrix(dat[store, ]), theta[store], method, ref$latent$mean, ref$latent$var)
    lzs_full <- rep(NA_real_, nrow(dat))
    if (!is.null(lzs)) lzs_full[store] <- lzs
    persons[[method]] <- person_block(theta, pf, z_outfit_answered(m, theta),
                                      if (is.null(lzs)) NULL else lzs_full, store)
    if (method != "ML") {
      item_fit[[method]] <- item_block(itemfit(m, fit_stats = "infit", Theta = matrix(theta)),
                                       item_z_outfit(m, theta))
    }
  }
  cat(sprintf("%-14s N=%d patterns=%d stored=%d\n", name, nrow(dat), nrow(pats), length(store)))
  list(source = name, n_store = length(store), persons = persons, item_fit = item_fit)
}

cases <- lapply(setNames(names(v37$cases), names(v37$cases)), single_case)

# --- two groups -----------------------------------------------------------------------------
tg <- v37$twogroup
items <- unlist(tg$items)
types_g <- unlist(tg$itemtype)
pats <- pat_matrix(tg$patterns, items)
counts <- unlist(tg$counts)
reps <- rep(seq_len(nrow(pats)), counts)
dat_g <- as.data.frame(pats[reps, , drop = FALSE])
grp <- factor(unlist(tg$pattern_groups)[reps], levels = unlist(tg$group_names))
sv <- multipleGroup(dat_g, 1, group = grp, itemtype = types_g, pars = "values",
                    invariance = c(items, "free_means", "free_var"))
for (g in levels(grp)) sv <- set_values(sv, items, tg$groups[[g]]$params, tg$groups[[g]]$latent, g)
sv$est <- FALSE
sv$const <- "none"
mg <- multipleGroup(dat_g, 1, group = grp, itemtype = types_g, pars = sv, invariance = "",
                    verbose = FALSE, quadpts = quadpts, technical = list(theta_lim = theta_lim))
first <- cumsum(c(1L, counts))[seq_along(counts)]
store <- first[seq_len(min(N_STORE, length(first)))]
persons <- list()
item_fit <- list()
for (method in METHODS) {
  theta <- scores(mg, method)
  pf <- personfit(mg, Theta = matrix(theta))
  lzs <- rep(NA_real_, nrow(dat_g))
  if (method != "EAP") {
    for (g in levels(grp)) {
      rows <- store[grp[store] == g]
      lat <- tg$groups[[g]]$latent
      lzs[rows] <- lz_star(extract.group(mg, g), as.matrix(dat_g[rows, ]), theta[rows], method,
                           lat$mean, lat$var)
    }
  }
  # No missing responses here, so mirt's z.outfit already uses n = J.
  persons[[method]] <- person_block(theta, pf, pf$z.outfit, if (method == "EAP") NULL else lzs,
                                    store)
  if (method != "ML") {
    fits <- itemfit(mg, fit_stats = "infit", Theta = matrix(theta))
    item_fit[[method]] <- lapply(setNames(levels(grp), levels(grp)),
                                 function(g) item_block(fits[[g]]))
  }
}
cat(sprintf("%-14s N=%d patterns=%d stored=%d\n", "twogroup", nrow(dat_g), nrow(pats),
            length(store)))
twogroup <- list(source = "twogroup", n_store = length(store), persons = persons,
                 item_fit = item_fit)

# --- Rasch: mirt and sirt -------------------------------------------------------------------
ref <- v37$cases$twopl
items <- unlist(ref$items)
pats <- pat_matrix(ref$patterns, items)
counts <- unlist(ref$counts)
dat <- as.data.frame(pats[rep(seq_len(nrow(pats)), counts), , drop = FALSE])
mr <- mirt(dat, 1, itemtype = "Rasch", TOL = 1e-10, verbose = FALSE, quadpts = quadpts,
           technical = list(NCYCLES = 20000L, theta_lim = theta_lim))
stopifnot(isTRUE(extract.mirt(mr, "converged")))
cf <- coef(mr, simplify = TRUE)
d <- unname(cf$items[, "d"])
theta <- scores(mr, "WLE")
pf <- personfit(mr, Theta = matrix(theta))
ps <- personfit.stat(dat = as.matrix(dat), abil = theta, b = -d)
first <- cumsum(c(1L, counts))[seq_along(counts)]
store <- first[seq_len(min(N_STORE, length(first)))]
rasch <- list(
  source = "twopl", d = d, var = unname(cf$cov[1, 1]), n_store = length(store),
  persons = list(WLE = person_block(theta, pf, pf$z.outfit, NULL, store)),
  sirt = list(lz = ps$lz[store], infit = ps$infit[store], outfit = ps$outfit[store]),
  item_fit = list(WLE = item_block(itemfit(mr, fit_stats = "infit", Theta = matrix(theta))))
)
cat(sprintf("%-14s max |lz mirt - sirt| = %.2e, |infit| = %.2e, |outfit| = %.2e\n", "rasch_sirt",
            max(abs(pf$Zh - ps$lz)), max(abs(pf$infit - ps$infit)),
            max(abs(pf$outfit - ps$outfit))))

bundle <- list(
  meta = list(
    case = "v53_person_fit",
    description = paste("V53: person fit (lz, lz*, infit / outfit) and item infit / outfit vs",
                        "mirt personfit() / itemfit(fit_stats = 'infit'), sirt personfit.stat()",
                        "and an R recomputation of lz*"),
    r_version = R.version.string,
    mirt_version = as.character(packageVersion("mirt")),
    sirt_version = as.character(packageVersion("sirt")),
    quadpts = quadpts, theta_lim = theta_lim,
    # Every model is fixed at V37's estimates (converged there); the Rasch fit is asserted.
    converged = TRUE
  ),
  cases = cases,
  twogroup = twogroup,
  rasch_sirt = rasch
)
path <- file.path("tests", "reference", "v53_person_fit.json")
write_json(bundle, path, auto_unbox = TRUE, digits = NA, pretty = FALSE, na = "null")
cat("wrote", path, "\n")
