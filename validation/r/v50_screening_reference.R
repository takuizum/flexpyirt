# Common-item screening for linking and item parameter drift: reference values (VALIDATION.md V50).
# Run from the repository root:  Rscript validation/r/v50_screening_reference.R
# Writes tests/reference/v50_screening.json. CI does not run R; the JSON is committed.
#
# No R package implements the robust z of Huynh & Meyer (2010), the displacement or the RMSD
# between expected item score curves together with purification on characteristic-curve linking,
# so the rules of DESIGN.md §20 are recomputed here in base R:
#   - each form is calibrated separately with mirt (optimizer = "NR", accelerate = "none",
#     TOL = 1e-10, 61 equispaced quadrature points on [-6, 6], latent N(0, 1));
#   - the Stocking-Lord / Haebara linking on the current anchor set is the V20 base-R
#     implementation (probtrace() / expected.item(), optim(BFGS) + nlm() + Newton polish);
#   - locations, robust z (stats::median, stats::quantile(type = 7)), displacement and the RMSD of
#     expected.item() curves on the target grid follow the formulas, and the purification loop
#     removes items by the same rule.
# The simulated drift (planted in the source form) gives a known answer for a sanity check, and
# equateIRT::dif.test() (Lord's Wald test after linking, with purification) is run on the same
# mirt fits as an independent cross-check of which items drifted.
suppressMessages({
  library(mirt)
  library(jsonlite)
  library(equateIRT)
})

quadpts <- 61L
theta_lim <- c(-6, 6)
common_mirt <- list(
  TOL = 1e-10,
  accelerate = "none",
  optimizer = "NR",
  SE = FALSE,
  quadpts = quadpts,
  verbose = FALSE,
  technical = list(NCYCLES = 50000L, theta_lim = theta_lim)
)
robust_scale <- 0.74
iqr_floor <- 1e-12

# ---- Form extraction (as in V20 / V40) ----------------------------------------------------------

clean_item <- function(x) {
  nm <- colnames(x)
  keep <- nm[nm == "a1" | grepl("^d[1-9][0-9]*$|^d$", nm)]
  par <- x["par", keep]
  names(par) <- sub("^a1$", "a", keep)
  as.list(par)
}

extract_form <- function(m, dat, itemtypes, n_cats) {
  cf <- coef(m)
  items <- setdiff(names(cf), "GroupPars")
  params <- lapply(items, function(i) clean_item(cf[[i]]))
  names(params) <- items
  key <- apply(dat, 1, paste, collapse = ",")
  first <- !duplicated(key)
  list(
    items = items,
    n_obs = nrow(dat),
    patterns = unname(as.matrix(dat[first, , drop = FALSE])),
    counts = as.integer(table(key)[key[first]]),
    converged = extract.mirt(m, "converged"),
    itemtypes = as.list(itemtypes[items]),
    n_cats = as.list(n_cats[items]),
    latent = list(mean = 0, var = 1),
    params = params
  )
}

# ---- Base-R linking criteria (V20) ---------------------------------------------------------------

make_criterion <- function(m_src, m_tgt, common_items, theta, w, method, symmetric) {
  src_ext <- lapply(common_items, function(nm) extract.item(m_src, nm))
  tgt_ext <- lapply(common_items, function(nm) extract.item(m_tgt, nm))
  n_common <- length(common_items)
  if (method == "haebara") {
    p_tgt <- lapply(tgt_ext, function(it) probtrace(it, matrix(theta)))
    p_src <- lapply(src_ext, function(it) probtrace(it, matrix(theta)))
    return(function(par) {
      A <- par[1]; B <- par[2]
      if (!is.finite(A) || !is.finite(B) || A <= 0) return(1e30)
      val <- 0
      for (j in seq_len(n_common)) {
        d <- p_tgt[[j]] - probtrace(src_ext[[j]], matrix((theta - B) / A))
        val <- val + sum(w * rowSums(d * d))
        if (symmetric) {
          d2 <- p_src[[j]] - probtrace(tgt_ext[[j]], matrix(A * theta + B))
          val <- val + sum(w * rowSums(d2 * d2))
        }
      }
      val
    })
  }
  tcc_tgt <- Reduce(`+`, lapply(tgt_ext, function(it) expected.item(it, matrix(theta))))
  tcc_src <- Reduce(`+`, lapply(src_ext, function(it) expected.item(it, matrix(theta))))
  function(par) {
    A <- par[1]; B <- par[2]
    if (!is.finite(A) || !is.finite(B) || A <= 0) return(1e30)
    moved <- Reduce(`+`, lapply(src_ext, function(it) expected.item(it, matrix((theta - B) / A))))
    val <- sum(w * (tcc_tgt - moved)^2)
    if (symmetric) {
      moved_t <- Reduce(`+`, lapply(tgt_ext, function(it) expected.item(it, matrix(A * theta + B))))
      val <- val + sum(w * (tcc_src - moved_t)^2)
    }
    val
  }
}

solve_base_r <- function(fn, start) {
  grad_fd <- function(x, eps = 1e-6) {
    c(
      (fn(x + c(eps, 0)) - fn(x - c(eps, 0))) / (2 * eps),
      (fn(x + c(0, eps)) - fn(x - c(0, eps))) / (2 * eps)
    )
  }
  opt <- optim(
    start, fn, gr = grad_fd, method = "BFGS",
    control = list(reltol = 1e-15, abstol = 1e-15, maxit = 1000)
  )
  pol <- nlm(fn, opt$par, ndigit = 15, gradtol = 1e-13, steptol = 1e-13, iterlim = 1000)
  ab <- if (fn(pol$estimate) <= fn(opt$par)) pol$estimate else opt$par
  for (iter in seq_len(10L)) {
    g <- grad_fd(ab)
    if (max(abs(g)) < 1e-12) break
    e <- 1e-5
    H <- cbind(
      (grad_fd(ab + c(e, 0)) - grad_fd(ab - c(e, 0))) / (2 * e),
      (grad_fd(ab + c(0, e)) - grad_fd(ab - c(0, e))) / (2 * e)
    )
    H <- 0.5 * (H + t(H))
    step <- tryCatch(-solve(H, g), error = function(err) rep(0, 2))
    cand <- ab + step
    if (all(is.finite(cand)) && cand[1] > 0 && max(abs(grad_fd(cand))) < max(abs(g))) {
      ab <- cand
    } else {
      break
    }
  }
  grad <- grad_fd(ab)
  stopifnot(all(is.finite(ab)), ab[1] > 0, max(abs(grad)) < 1e-9)
  list(A = ab[1], B = ab[2], criterion = fn(ab), grad_norm = max(abs(grad)))
}

# ---- Screening quantities (DESIGN.md §20) --------------------------------------------------------

# Item location from mirt's slope-intercept parameters: -d / a for 2PL, the mean of the graded
# thresholds -d_k / a for GRM, and the mean of the step difficulties for GPCM.
item_location <- function(par, itemtype) {
  a <- par[["a"]]
  if (itemtype == "2PL") return(-par[["d"]] / a)
  ds <- unlist(par[grepl("^d[0-9]+$", names(par))])
  if (itemtype == "graded") return(mean(-ds / a))
  if (itemtype == "gpcm") {
    # mirt's gpcm intercepts d_k = -a * sum_{c <= k} b_c (d_0 = 0 dropped), so b_k = -(d_k - d_{k-1}) / a.
    return(mean(-diff(c(0, ds)) / a))
  }
  stop("unsupported itemtype ", itemtype)
}

# Median and IQR over all common items (DESIGN.md §20), not only the kept ones.
robust_z <- function(x) {
  ref <- x[is.finite(x)]
  q <- unname(quantile(ref, c(0.25, 0.75), type = 7))
  iqr <- q[2] - q[1]
  stopifnot(abs(iqr - IQR(ref)) < 1e-15)
  if (!(iqr > iqr_floor)) return(rep(NA_real_, length(x)))
  (x - median(ref)) / (robust_scale * iqr)
}

screen_table <- function(m_src, m_tgt, form_src, form_tgt, common_items, A, B, theta, w, kept,
                         statistic, threshold) {
  rows <- lapply(common_items, function(it) {
    type <- form_tgt$itemtypes[[it]]
    ps <- form_src$params[[it]]
    pt <- form_tgt$params[[it]]
    a_src <- ps[["a"]] / A
    b_src <- A * item_location(ps, type) + B
    a_tgt <- pt[["a"]]
    b_tgt <- item_location(pt, type)
    e_src <- expected.item(extract.item(m_src, it), matrix((theta - B) / A))
    e_tgt <- expected.item(extract.item(m_tgt, it), matrix(theta))
    data.frame(
      item = it, a_source = a_src, b_source = b_src, a_target = a_tgt, b_target = b_tgt,
      log_ratio_a = log(abs(a_src)) - log(abs(a_tgt)), difference_b = b_src - b_tgt,
      displacement = abs(b_src - b_tgt), rmsd = sqrt(sum(w * (e_src - e_tgt)^2))
    )
  })
  tab <- do.call(rbind, rows)
  tab$robust_z_a <- robust_z(tab$log_ratio_a)
  tab$robust_z_b <- robust_z(tab$difference_b)
  tab$statistic <- switch(
    statistic,
    robust_z = pmax(abs(tab$robust_z_a), abs(tab$robust_z_b), na.rm = TRUE),
    displacement = tab$displacement,
    rmsd = tab$rmsd
  )
  tab$flagged <- !is.na(tab$statistic) & tab$statistic > threshold
  tab$kept <- kept
  tab
}

run_screening <- function(m_src, m_tgt, form_src, form_tgt, common_items, theta, w, method,
                          symmetric, statistic, threshold, removal, min_items, max_iter) {
  kept <- rep(TRUE, length(common_items))
  removed_at <- rep(NA_integer_, length(common_items))
  removed_order <- character(0)
  history <- list()
  iteration <- 0L
  repeat {
    fn <- make_criterion(m_src, m_tgt, common_items[kept], theta, w, method, symmetric)
    sol <- solve_base_r(fn, start = c(1.2, 0.5))
    tab <- screen_table(
      m_src, m_tgt, form_src, form_tgt, common_items, sol$A, sol$B, theta, w, kept,
      statistic, threshold
    )
    step <- list(
      iteration = iteration, n_items = sum(kept), A = sol$A, B = sol$B,
      criterion = sol$criterion, grad_norm = sol$grad_norm, removed = character(0)
    )
    flagged_kept <- which(tab$flagged & kept)
    stop_now <- length(flagged_kept) == 0L || iteration >= max_iter ||
      sum(kept) - min_items <= 0L
    if (!stop_now) {
      ord <- flagged_kept[order(-tab$statistic[flagged_kept])]  # order() is stable on ties
      cand <- if (removal == "largest") ord[1] else ord
      cand <- cand[seq_len(min(length(cand), sum(kept) - min_items))]
      iteration <- iteration + 1L
      kept[cand] <- FALSE
      removed_at[cand] <- iteration
      removed_order <- c(removed_order, common_items[cand])
      step$removed <- common_items[cand]
    }
    history[[length(history) + 1L]] <- step
    if (stop_now) break
  }
  tab$iteration_removed <- removed_at
  cat(sprintf(
    "    %-12s %-8s removed: %s; final A=%.6f B=%.6f\n", statistic, removal,
    paste(removed_order, collapse = ","), sol$A, sol$B
  ))
  list(
    method = method, symmetric = symmetric, statistic = statistic, threshold = threshold,
    removal = removal, min_items = min_items, max_iter = max_iter,
    removed_items = removed_order, history = history, table = tab
  )
}

# ---- Cases ----------------------------------------------------------------------------------------

fit_form <- function(dat, itemtypes) {
  m <- do.call(mirt, c(list(dat, 1, itemtype = unname(itemtypes[colnames(dat)])), common_mirt))
  stopifnot(extract.mirt(m, "converged"))
  m
}

run_case <- function(case_name, description, dat_src, dat_tgt, itemtypes, n_cats, common_items,
                     variants, drifted) {
  cat(case_name, "\n")
  m_tgt <- fit_form(dat_tgt, itemtypes)
  m_src <- fit_form(dat_src, itemtypes)
  form_tgt <- extract_form(m_tgt, dat_tgt, itemtypes, n_cats)
  form_src <- extract_form(m_src, dat_src, itemtypes, n_cats)
  theta <- as.numeric(m_tgt@Model$Theta[, 1])
  w <- dnorm(theta)
  w <- w / sum(w)
  stopifnot(
    all(range(theta) == theta_lim), length(theta) == quadpts,
    max(abs(w - as.numeric(m_tgt@Internals$Prior[[1]]))) < 1e-13
  )
  out <- lapply(variants, function(v) {
    run_screening(
      m_src, m_tgt, form_src, form_tgt, common_items, theta, w, v$method, v$symmetric,
      v$statistic, v$threshold, v$removal, v$min_items, v$max_iter
    )
  })

  # Independent cross-check: equateIRT's Lord Wald test after Stocking-Lord linking, with
  # purification, on the same mirt fits (dichotomous common items only).
  cross <- NULL
  if (all(unlist(form_tgt$itemtypes[common_items]) == "2PL")) {
    m_src_se <- do.call(mirt, c(
      list(dat_src, 1, itemtype = unname(itemtypes[colnames(dat_src)])),
      modifyList(common_mirt, list(SE = TRUE, SE.type = "Oakes"))
    ))
    m_tgt_se <- do.call(mirt, c(
      list(dat_tgt, 1, itemtype = unname(itemtypes[colnames(dat_tgt)])),
      modifyList(common_mirt, list(SE = TRUE, SE.type = "Oakes"))
    ))
    dt <- dif.test(
      est.mods = list(target = m_tgt_se, source = m_src_se), method = "Stocking-Lord",
      purification = TRUE, signif.level = 0.05
    )
    tab <- dt$test
    flagged <- rownames(tab)[tab[, "p.value"] < 0.05]
    cross <- list(method = "equateIRT::dif.test Stocking-Lord, purification, alpha = 0.05",
                  flagged = flagged)
    cat("    equateIRT::dif.test flagged:", paste(flagged, collapse = ","), "\n")
  }

  list(
    meta = list(
      case = case_name, description = description, quadpts = quadpts, theta_lim = theta_lim,
      grid_check = range(m_tgt@Model$Theta), converged = TRUE
    ),
    common_items = common_items,
    drifted_items = drifted,
    target = form_tgt,
    source = form_src,
    screening = out,
    cross_check = cross
  )
}

variant <- function(statistic, threshold, removal = "largest", method = "stocking_lord",
                    symmetric = FALSE, min_items = 2L, max_iter = 10L) {
  list(statistic = statistic, threshold = threshold, removal = removal, method = method,
       symmetric = symmetric, min_items = min_items, max_iter = max_iter)
}

# True populations: target N(0, 1), source N(0.4, 1.2^2) on the target scale, so the linking of a
# source calibration on N(0, 1) has A = 1.2, B = 0.4 for undrifted items.
cases <- list()

# V50a: 2PL, 12 common + 8 unique items per form, N = 2000 per form. Drift planted in the source
# form: c01 location +0.8, c02 slope x 1.8 (same location), c03 location -0.6.
set.seed(20261126)
N <- 2000L
J_c <- 12L; J_u <- 8L
a_all <- rlnorm(J_c + 2L * J_u, meanlog = 0.1, sdlog = 0.25)
b_all <- rnorm(J_c + 2L * J_u, 0, 0.9)
idx_c <- seq_len(J_c); idx_t <- J_c + seq_len(J_u); idx_s <- J_c + J_u + seq_len(J_u)
common <- sprintf("c%02d", idx_c)
a_src <- a_all; b_src <- b_all
b_src[1] <- b_src[1] + 0.8
a_src[2] <- a_src[2] * 1.8
b_src[3] <- b_src[3] - 0.6
sim_2pl <- function(a, b, mu, sd, names) {
  th <- matrix(rnorm(N, mu, sd))
  out <- as.data.frame(simdata(matrix(a), matrix(-a * b), N, itemtype = "2PL", Theta = th))
  colnames(out) <- names
  out
}
dat_tgt <- sim_2pl(a_all[c(idx_c, idx_t)], b_all[c(idx_c, idx_t)], 0, 1,
                   c(common, sprintf("t%02d", seq_len(J_u))))
dat_src <- sim_2pl(a_src[c(idx_c, idx_s)], b_src[c(idx_c, idx_s)], 0.4, 1.2,
                   c(common, sprintf("s%02d", seq_len(J_u))))
all_names <- union(colnames(dat_tgt), colnames(dat_src))
cases[["v50a_2pl"]] <- run_case(
  "v50a_2pl",
  "V50a: 2PL, 12 common + 8 unique per form, N=2000/form, drift in c01 (b+0.8), c02 (a*1.8), c03 (b-0.6)",
  dat_src, dat_tgt,
  setNames(rep("2PL", length(all_names)), all_names),
  setNames(rep(2L, length(all_names)), all_names),
  common,
  list(
    robust_z = variant("robust_z", 1.645),
    robust_z_all = variant("robust_z", 1.645, removal = "all"),
    displacement = variant("displacement", 0.3),
    rmsd = variant("rmsd", 0.1),
    displacement_haebara_sym = variant("displacement", 0.3, method = "haebara", symmetric = TRUE),
    robust_z_max_iter_1 = variant("robust_z", 1.645, max_iter = 1L),
    rmsd_min_items_11 = variant("rmsd", 0.02, min_items = 11L)
  ),
  c("c01", "c02", "c03")
)

# V50b: mixed 2PL + GRM (K = 4) + GPCM (K = 4): 6 + 3 + 3 common, 4 + 2 + 2 unique per form,
# N = 2500 per form. Drift: d01 location +0.7, g01 thresholds +0.6, p02 slope x 0.5.
set.seed(20261127)
N <- 2500L
n_d <- 6L + 2L * 4L; n_g <- 3L + 2L * 2L; n_p <- 3L + 2L * 2L
a_d <- rlnorm(n_d, 0.1, 0.25); b_d <- rnorm(n_d, 0, 0.8)
a_g <- rlnorm(n_g, 0.1, 0.2); bm_g <- rnorm(n_g, 0, 0.5)
a_p <- rlnorm(n_p, 0.0, 0.2)
st_p <- cbind(rnorm(n_p, -0.9, 0.3), rnorm(n_p, 0.0, 0.3), rnorm(n_p, 0.9, 0.3))
grm_d <- function(a, bm) cbind(-a * (bm - 1.0), -a * bm, -a * (bm + 1.0))
gpcm_d <- function(a, st) cbind(0, -a * st[, 1], -a * (st[, 1] + st[, 2]), -a * rowSums(st))
mixed_form <- function(id, ig, ip, mu, sd, drift) {
  th <- matrix(rnorm(N, mu, sd))
  ad <- a_d[id]; bd <- b_d[id]; ag <- a_g[ig]; bg <- bm_g[ig]; ap <- a_p[ip]; sp <- st_p[ip, ]
  if (drift) {
    bd[1] <- bd[1] + 0.7
    bg[1] <- bg[1] + 0.6
    ap[2] <- ap[2] * 0.5
  }
  dd <- simdata(matrix(ad), matrix(-ad * bd), N, itemtype = "2PL", Theta = th)
  dg <- simdata(matrix(ag), grm_d(ag, bg), N, itemtype = "graded", Theta = th)
  dp <- simdata(matrix(ap), gpcm_d(ap, sp), N, itemtype = "gpcm", Theta = th)
  as.data.frame(cbind(dd, dg, dp))
}
common_d <- sprintf("d%02d", 1:6); common_g <- sprintf("g%02d", 1:3); common_p <- sprintf("p%02d", 1:3)
dat_tgt_m <- mixed_form(c(1:6, 7:10), c(1:3, 4:5), c(1:3, 4:5), 0, 1, FALSE)
colnames(dat_tgt_m) <- c(common_d, sprintf("td%02d", 1:4), common_g, sprintf("tg%02d", 1:2),
                         common_p, sprintf("tp%02d", 1:2))
dat_src_m <- mixed_form(c(1:6, 11:14), c(1:3, 6:7), c(1:3, 6:7), 0.4, 1.2, TRUE)
colnames(dat_src_m) <- c(common_d, sprintf("sd%02d", 1:4), common_g, sprintf("sg%02d", 1:2),
                         common_p, sprintf("sp%02d", 1:2))
all_m <- union(colnames(dat_tgt_m), colnames(dat_src_m))
types_m <- setNames(
  ifelse(grepl("^.?d", all_m), "2PL", ifelse(grepl("^.?g", all_m), "graded", "gpcm")), all_m
)
cats_m <- setNames(ifelse(types_m == "2PL", 2L, 4L), all_m)
cases[["v50b_mixed"]] <- run_case(
  "v50b_mixed",
  "V50b: mixed 2PL + GRM(K=4) + GPCM(K=4), 6+3+3 common, N=2500/form, drift in d01 (b+0.7), g01 (b+0.6), p02 (a*0.5)",
  dat_src_m, dat_tgt_m, types_m, cats_m, c(common_d, common_g, common_p),
  list(
    robust_z = variant("robust_z", 1.645),
    displacement = variant("displacement", 0.3),
    rmsd = variant("rmsd", 0.1),
    rmsd_haebara_all = variant("rmsd", 0.1, removal = "all", method = "haebara")
  ),
  c("d01", "g01", "p02")
)

res <- list(
  meta = list(
    case = "v50_screening",
    description = paste(
      "V50: common-item screening for linking (robust z, displacement, RMSD of expected item",
      "score curves) with iterative purification, on separate mirt calibrations."
    ),
    note = paste(
      "No R package implements these statistics with characteristic-curve linking; the rules of",
      "DESIGN.md section 20 are recomputed in base R (median, quantile(type = 7), expected.item())",
      "with the V20 base-R Stocking-Lord / Haebara linking."
    ),
    r_version = R.version.string,
    mirt_version = as.character(packageVersion("mirt")),
    equateIRT_version = as.character(packageVersion("equateIRT")),
    TOL = 1e-10,
    quadpts = quadpts,
    theta_lim = theta_lim,
    grid_check = theta_lim,
    optimizer = "NR",
    converged = TRUE
  ),
  cases = cases
)

path <- file.path("tests", "reference", "v50_screening.json")
write_json(res, path, auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null")
cat("wrote", path, "\n")
