# Observed-score DIF (VALIDATION.md V47): Mantel-Haenszel, generalized MH, Mantel and logistic
# regression, with purification of the matching score.
# Run from the repository root:  Rscript packages/flexpyctt/validation/r/ctt_dif_reference.R
# Writes packages/flexpyctt/tests/reference/v47_dif.json. CI does not run R; the JSON is committed.
#
# References: difR (difMH / mantelHaenszel, difGMH / genMantelHaenszel, difLogistic / Logistik,
# difMantel.poly, liu_agresti_ccor) and stats::mantelhaen.test. difMantel.poly rounds its output
# (statistic to 3 decimals) and purifies with a different matching rule (the studied item leaves
# the matching score when it is flagged), so the Mantel statistic and its purification path are
# recomputed here by a direct implementation of Mantel (1963) with difR's rule for the other
# methods (the studied item is always part of the matching score); difMantel.poly is kept as a
# rounded cross-check. liu_agresti_ccor gives the cumulative odds ratio; its "liu_agresti" SE is
# not used: its gradient with respect to the focal cumulative count has the wrong sign for the
# a/N term and it divides by sum(b c / N) instead of sum(a d / N), so it is neither SE(log psi)
# nor SE(psi). la_eq3() below is Liu & Agresti's (1996) eq. 3, the variance estimator that is
# consistent with large and with sparse strata, checked against the paper's worked example.
suppressMessages({
  library(difR)
  library(jsonlite)
})
# difLogistic() calls glm() with its default control (epsilon = 1e-8 on the relative deviance
# change), which leaves the coefficients, SEs and statistics up to ~2e-4 (relative) from the
# MLE on these data. Tighten glm's default so the references are the MLE to ~1e-10.
tight_glm_control <- function(epsilon = 1e-14, maxit = 100, trace = FALSE) {
  list(epsilon = epsilon, maxit = maxit, trace = trace)
}
assignInNamespace("glm.control", tight_glm_control, ns = "stats")

rows_of <- function(m) lapply(seq_len(nrow(m)), function(i) unname(as.list(m[i, ])))
na_null <- function(x) {
  x <- unname(as.numeric(x))
  lapply(x, function(v) if (is.na(v)) NULL else v)
}
flags_of <- function(difitems, n) {
  out <- rep(FALSE, n)
  if (!is.character(difitems)) out[difitems] <- TRUE
  out
}

# ---------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------
data(verbal)
verbal_items <- as.matrix(verbal[, 1:24])
verbal_group <- verbal$Gender  # focal = 1 (73 of 316)

simulate_2pl <- function(n_by_group, mu, a, b, a_shift, b_shift, seed) {
  # a_shift / b_shift: list (one per group) of length-J vectors added to a / b.
  set.seed(seed)
  out <- NULL
  grp <- NULL
  for (g in seq_along(n_by_group)) {
    n <- n_by_group[g]
    theta <- rnorm(n, mu[g], 1)
    ag <- a + a_shift[[g]]
    bg <- b + b_shift[[g]]
    p <- plogis(outer(theta, bg, "-") * matrix(ag, n, length(a), byrow = TRUE))
    out <- rbind(out, (matrix(runif(n * length(a)), n) < p) * 1L)
    grp <- c(grp, rep(g - 1L, n))
  }
  colnames(out) <- paste0("i", seq_along(a))
  list(y = out, group = grp)
}

J <- 20
a <- rep(c(1.0, 1.4, 0.8, 1.2, 1.6), 4)
b <- seq(-1.6, 1.6, length.out = J)
zero <- rep(0, J)
b_dif <- zero
b_dif[c(3, 7)] <- c(0.7, -0.6)   # uniform DIF
a_dif <- zero
a_dif[12] <- -0.7                # non-uniform DIF
sim2 <- simulate_2pl(c(900, 600), c(0, -0.5), a, b, list(zero, a_dif), list(zero, b_dif), 4701)

J3 <- 15
a3 <- rep(c(1.1, 1.5, 0.9), 5)
b3 <- seq(-1.4, 1.4, length.out = J3)
z3 <- rep(0, J3)
b3_g2 <- z3
b3_g2[4] <- 0.8
b3_g3 <- z3
b3_g3[9] <- -0.8
sim3 <- simulate_2pl(c(600, 600, 600), c(0, -0.3, 0.3), a3, b3, list(z3, z3, z3),
                     list(z3, b3_g2, b3_g3), 4702)

# Graded responses, 4 categories, three groups; DIF in items 2 (group 1) and 6 (groups 1, 2).
simulate_grm <- function(n_by_group, mu, seed) {
  set.seed(seed)
  Jp <- 10
  ap <- rep(c(1.2, 1.6, 0.9, 1.4, 1.1), 2)
  bp <- cbind(seq(-1.8, -0.4, length.out = Jp), seq(-0.6, 0.6, length.out = Jp),
              seq(0.4, 1.8, length.out = Jp))
  out <- NULL
  grp <- NULL
  for (g in seq_along(n_by_group)) {
    n <- n_by_group[g]
    theta <- rnorm(n, mu[g], 1)
    bg <- bp
    if (g >= 2) bg[2, ] <- bg[2, ] + 0.6
    if (g >= 2) bg[6, ] <- bg[6, ] - 0.5 * (g - 1)
    y <- matrix(0L, n, Jp)
    for (j in seq_len(Jp)) {
      u <- runif(n)
      for (k in 1:3) y[, j] <- y[, j] + (u < plogis(ap[j] * (theta - bg[j, k])))
    }
    out <- rbind(out, y)
    grp <- c(grp, rep(g - 1L, n))
  }
  colnames(out) <- paste0("p", seq_len(Jp))
  list(y = out, group = grp)
}
simp <- simulate_grm(c(800, 700, 700), c(0, -0.4, 0.2), 4703)

# An external matching variable: a coarsened total score of the first ten verbal items.
verbal_matching <- as.integer(rowSums(verbal_items[, 1:10]) %/% 2)

# ---------------------------------------------------------------------------------------------
# Mantel-Haenszel (difMH, mantelHaenszel, mantelhaen.test)
# ---------------------------------------------------------------------------------------------
mh_tables <- function(y, member, item, anchor) {
  cols <- union(anchor, item)
  score <- rowSums(y[, cols, drop = FALSE])
  tabs <- NULL
  for (s in sort(unique(score))) {
    rows <- score == s
    if (sum(rows) < 2) next
    A <- sum(rows & member == 0 & y[, item] == 1)
    B <- sum(rows & member == 0 & y[, item] == 0)
    C <- sum(rows & member == 1 & y[, item] == 1)
    D <- sum(rows & member == 1 & y[, item] == 0)
    tabs <- c(tabs, A, C, B, D)
  }
  array(tabs, c(2, 2, length(tabs) / 4))
}

mh_case <- function(name, y, group, focal) {
  member <- as.integer(group == focal)
  n_items <- ncol(y)
  plain <- lapply(c(TRUE, FALSE), function(correct) {
    r <- difMH(y, member, focal.name = 1, correct = correct)
    list(correct = correct, chi2 = na_null(r$MH), p = na_null(r$p.value),
         alpha_mh = na_null(r$alphaMH), var_lambda = na_null(r$varLambda),
         flagged = flags_of(r$DIFitems, n_items))
  })
  stats_mh <- lapply(seq_len(n_items), function(j) {
    tab <- mh_tables(y, member, j, seq_len(n_items))
    t1 <- mantelhaen.test(tab, correct = TRUE)
    t0 <- mantelhaen.test(tab, correct = FALSE)
    list(chi2_correct = unname(t1$statistic), chi2 = unname(t0$statistic),
         odds_ratio = unname(t0$estimate))
  })
  pur <- difMH(y, member, focal.name = 1, purify = TRUE, nrIter = 10, p.adjust.method = "holm")
  purified <- list(chi2 = na_null(pur$MH), p = na_null(pur$p.value),
                   adj_p = na_null(pur$adjusted.p), alpha_mh = na_null(pur$alphaMH),
                   var_lambda = na_null(pur$varLambda), n_iter = pur$nrPur,
                   converged = pur$convergence,
                   steps = if (is.null(pur$difPur)) list() else rows_of(rbind(pur$difPur) == 1),
                   flagged = flags_of(pur$DIFitems, n_items))
  bh <- difMH(y, member, focal.name = 1, p.adjust.method = "BH")
  list(name = name, items = colnames(y), data = rows_of(y), group = unname(group), focal = focal,
       plain = plain, mantelhaen = stats_mh, purified = purified,
       bh = list(adj_p = na_null(bh$adjusted.p), flagged = flags_of(bh$DIFitems, n_items)))
}

mh_cases <- list(
  mh_case("verbal", verbal_items, verbal_group, 1),
  mh_case("sim_2pl", sim2$y, sim2$group, 1)
)
# Anchor items given by the user (no purification): difR tests the other items only.
anchor <- c(1, 2, 4, 5, 6, 8, 9, 10)
ra <- difMH(sim2$y, sim2$group, focal.name = 1, anchor = anchor)
mh_anchor <- list(case = "sim_2pl", anchor_items = anchor - 1L, chi2 = na_null(ra$MH),
                  alpha_mh = na_null(ra$alphaMH), var_lambda = na_null(ra$varLambda))
rm <- difMH(verbal_items, verbal_group, focal.name = 1, match = verbal_matching)
mh_matching <- list(case = "verbal", matching = verbal_matching, chi2 = na_null(rm$MH),
                    alpha_mh = na_null(rm$alphaMH), var_lambda = na_null(rm$varLambda))

# ---------------------------------------------------------------------------------------------
# Generalized MH: several groups (difGMH) and nominal categories (mantelhaen.test, I x J x K)
# ---------------------------------------------------------------------------------------------
gmh_plain <- difGMH(sim3$y, sim3$group, focal.names = c(1, 2))
gmh_pur <- difGMH(sim3$y, sim3$group, focal.names = c(1, 2), purify = TRUE)
gmh_holm <- difGMH(sim3$y, sim3$group, focal.names = c(1, 2), p.adjust.method = "holm")
gmh_anchor <- difGMH(sim3$y, sim3$group, focal.names = c(1, 2), anchor = c(1, 2, 3, 5, 6))
gmh_case <- list(
  name = "sim_3groups", items = colnames(sim3$y), data = rows_of(sim3$y),
  group = sim3$group, focal = c(1, 2),
  plain = list(chi2 = na_null(gmh_plain$GMH), p = na_null(gmh_plain$p.value),
               flagged = flags_of(gmh_plain$DIFitems, J3)),
  purified = list(chi2 = na_null(gmh_pur$GMH), n_iter = gmh_pur$nrPur,
                  converged = gmh_pur$convergence,
                  steps = if (is.null(gmh_pur$difPur)) list() else rows_of(rbind(gmh_pur$difPur) == 1),
                  flagged = flags_of(gmh_pur$DIFitems, J3)),
  holm = list(adj_p = na_null(gmh_holm$adjusted.p), flagged = flags_of(gmh_holm$DIFitems, J3)),
  anchor = list(anchor_items = c(1, 2, 3, 5, 6) - 1L, chi2 = na_null(gmh_anchor$GMH))
)

nominal_stats <- function(y, group) {
  score <- rowSums(y)
  lapply(seq_len(ncol(y)), function(j) {
    keep <- score %in% as.numeric(names(which(table(score) >= 2)))
    tab <- table(factor(group[keep]), factor(y[keep, j]), factor(score[keep]))
    t <- mantelhaen.test(tab)
    list(chi2 = unname(t$statistic), df = unname(t$parameter), p = unname(t$p.value))
  })
}
two <- simp$group %in% c(0, 1)
nominal_cases <- list(
  list(name = "grm_3groups", items = colnames(simp$y), data = rows_of(simp$y),
       group = simp$group, focal = c(1, 2), stats = nominal_stats(simp$y, simp$group)),
  list(name = "grm_2groups", items = colnames(simp$y), data = rows_of(simp$y[two, ]),
       group = simp$group[two], focal = c(1), stats = nominal_stats(simp$y[two, ], simp$group[two]))
)

# ---------------------------------------------------------------------------------------------
# Mantel (1963) for ordered items and the Liu-Agresti cumulative common odds ratio
# ---------------------------------------------------------------------------------------------
mantel_stat <- function(y, member, item, anchor) {
  score <- rowSums(y[, union(anchor, item), drop = FALSE])
  s_f <- e_f <- v_f <- 0
  for (s in sort(unique(score))) {
    rows <- score == s
    x <- y[rows, item]
    f <- member[rows] == 1
    n_f <- sum(f)
    n_r <- sum(!f)
    n <- n_f + n_r
    if (n_f < 1 || n_r < 1 || n < 2) next
    s_f <- s_f + sum(x[f])
    e_f <- e_f + n_f * mean(x)
    v_f <- v_f + n_f * n_r / (n^2 * (n - 1)) * (n * sum(x^2) - sum(x)^2)
  }
  (s_f - e_f)^2 / v_f
}

# Liu & Agresti (1996), eq. 2 and eq. 3, in the paper's notation: row 1 is the focal group,
# row 2 the reference group, X*_ijk the cumulative counts (categories in increasing order).
# Eq. 3 is invariant to swapping the rows, so SE(log theta) is also SE(log psi) of difR's
# orientation (psi = 1 / theta).
la_eq3 <- function(x, member, score) {
  cats <- sort(unique(x))
  L <- length(cats)
  tabs <- list()
  for (s in sort(unique(score))) {
    r <- score == s
    n1 <- sum(r & member == 1)
    n2 <- sum(r & member == 0)
    if (n1 == 0 || n2 == 0) next
    X1 <- cumsum(sapply(cats, function(v) sum(r & member == 1 & x == v)))[-L]
    X2 <- cumsum(sapply(cats, function(v) sum(r & member == 0 & x == v)))[-L]
    tabs[[length(tabs) + 1]] <- list(n1 = n1, n2 = n2, N = n1 + n2, X1 = X1, X2 = X2)
  }
  R <- sum(sapply(tabs, function(t) sum(t$X1 * (t$n2 - t$X2) / t$N)))
  S <- sum(sapply(tabs, function(t) sum((t$n1 - t$X1) * t$X2 / t$N)))
  theta <- R / S
  phi <- function(t, j, s) {
    t$n1 * t$n2 / t$N^2 * (
      theta * (t$n1 - t$X1[s]) * t$X2[j] / t$n1 * (1 + (theta - 1) * t$X2[s] / t$n2) +
        t$X1[j] * (t$n2 - t$X2[s]) / t$n2 * (theta - (theta - 1) * t$X1[s] / t$n1))
  }
  xi <- 0
  for (t in tabs) {
    for (j in seq_len(L - 1)) {
      xi <- xi + phi(t, j, j)
      if (j < L - 1) for (s in (j + 1):(L - 1)) xi <- xi + 2 * phi(t, j, s)
    }
  }
  c(theta = theta, se_log = sqrt(xi / (theta^2 * S^2)))
}

# Check against the worked example of Liu & Agresti (1996, Table 1 and p. 1226): 28 centres,
# placebo (row 1) / active, responses better < unchanged < worse; log(theta) = -1.153 with
# SE .571, Mantel's statistic 4.84.
asthma <- matrix(c(
  0,2,1, 1,1,0,  0,1,0, 1,1,0,  1,1,0, 0,1,0,  1,0,0, 1,1,0,  1,0,0, 1,0,0,  1,0,0, 2,1,0,
  0,1,0, 2,1,0,  0,0,1, 0,1,0,  1,1,0, 1,1,0,  0,2,0, 1,0,0,  2,0,0, 1,0,1,  0,1,0, 1,0,0,
  1,0,0, 1,0,0,  0,1,0, 2,0,0,  1,0,0, 1,0,0,  0,1,0, 1,0,0,  0,2,0, 1,1,0,  0,1,0, 1,0,0,
  1,0,0, 1,0,0,  1,0,0, 1,0,0,  0,3,0, 0,1,0,  0,2,0, 1,0,0,  1,0,0, 1,0,0,  1,1,0, 1,0,0,
  1,0,0, 1,0,0,  0,1,1, 1,0,0,  0,1,0, 0,2,0,  1,0,0, 1,1,0), ncol = 6, byrow = TRUE)
asthma_long <- do.call(rbind, lapply(seq_len(nrow(asthma)), function(k) {
  do.call(rbind, lapply(1:2, function(i) {
    cnt <- asthma[k, (3 * (i - 1) + 1):(3 * i)]
    cbind(x = rep(0:2, cnt), member = rep(as.integer(i == 1), sum(cnt)), score = rep(k, sum(cnt)))
  }))
}))
asthma_la <- la_eq3(asthma_long[, "x"], asthma_long[, "member"], asthma_long[, "score"])
stopifnot(abs(log(asthma_la[["theta"]]) + 1.153) < 5e-4, abs(asthma_la[["se_log"]] - 0.571) < 5e-4)

mantel_run <- function(y, member, anchor) {
  out <- lapply(seq_len(ncol(y)), function(j) {
    chi2 <- mantel_stat(y, member, j, anchor)
    score <- rowSums(y[, union(anchor, j), drop = FALSE])
    la <- liu_agresti_ccor(y[, j], factor(member, levels = c(0, 1)), strata = score,
                           se = "naive")
    own <- la_eq3(y[, j], member, score)
    stopifnot(abs(own[["theta"]] * la[["Psi_hat"]] - 1) < 1e-12)
    list(chi2 = chi2, p = pchisq(chi2, 1, lower.tail = FALSE), psi = unname(la["Psi_hat"]),
         se_log_psi = own[["se_log"]])
  })
  out
}

mantel_case <- function(name, y, group, focal) {
  member <- as.integer(group == focal)
  n_items <- ncol(y)
  plain <- mantel_run(y, member, seq_len(n_items))
  # Purification with difR's rule for difMH (studied item always in the matching score).
  flagged <- which(sapply(plain, `[[`, "p") < 0.05)
  steps <- list(seq_len(n_items) %in% flagged)
  n_iter <- 0L
  converged <- length(flagged) == 0
  last <- plain
  while (!converged && n_iter < 10) {
    n_iter <- n_iter + 1L
    last <- mantel_run(y, member, setdiff(seq_len(n_items), flagged))
    new <- which(sapply(last, `[[`, "p") < 0.05)
    steps[[length(steps) + 1]] <- seq_len(n_items) %in% new
    if (setequal(new, flagged)) converged <- TRUE
    flagged <- new
  }
  dp <- suppressWarnings(difMantel.poly(as.data.frame(y), member, focal.name = 1, ref.name = 0))
  list(name = name, items = colnames(y), data = rows_of(y), group = unname(group), focal = focal,
       plain = list(chi2 = sapply(plain, `[[`, "chi2"), psi = sapply(plain, `[[`, "psi"),
                    se_log_psi = sapply(plain, `[[`, "se_log_psi")),
       difmantel_rounded = list(chi2 = dp$Stat, psi = dp$Psi_hat),
       purified = list(chi2 = sapply(last, `[[`, "chi2"), psi = sapply(last, `[[`, "psi"),
                       se_log_psi = sapply(last, `[[`, "se_log_psi"), n_iter = n_iter,
                       converged = converged, steps = steps,
                       flagged = seq_len(n_items) %in% flagged))
}
mantel_cases <- list(
  mantel_case("grm", simp$y[two, ], simp$group[two], 1),
  mantel_case("verbal", verbal_items, verbal_group, 1)
)

# ---------------------------------------------------------------------------------------------
# Logistic regression (difLogistic / Logistik)
# ---------------------------------------------------------------------------------------------
logistic_case <- function(name, y, group, focal) {
  member <- as.integer(group == focal)
  n_items <- ncol(y)
  runs <- list()
  for (type in c("both", "udif", "nudif")) {
    for (criterion in c("LRT", "Wald")) {
      r <- difLogistic(y, member, focal.name = 1, type = type, criterion = criterion)
      runs[[length(runs) + 1]] <- list(
        type = type, criterion = criterion, chi2 = na_null(r$Logistik), p = na_null(r$p.value),
        delta_r2 = na_null(r$deltaR2), par_full = rows_of(r$parM0), se_full = rows_of(r$seM0),
        flagged = flags_of(r$DIFitems, n_items))
    }
  }
  pur <- difLogistic(y, member, focal.name = 1, type = "both", purify = TRUE,
                     p.adjust.method = "BH")
  purified <- list(chi2 = na_null(pur$Logistik), p = na_null(pur$p.value),
                   adj_p = na_null(pur$adjusted.p), delta_r2 = na_null(pur$deltaR2),
                   par_full = rows_of(pur$parM0), n_iter = pur$nrPur,
                   converged = pur$convergence,
                   steps = if (is.null(pur$difPur)) list() else rows_of(rbind(pur$difPur) == 1),
                   flagged = flags_of(pur$DIFitems, n_items))
  list(name = name, items = colnames(y), data = rows_of(y), group = unname(group), focal = focal,
       runs = runs, purified = purified)
}
logistic_cases <- list(
  logistic_case("verbal", verbal_items, verbal_group, 1),
  logistic_case("sim_2pl", sim2$y, sim2$group, 1)
)
rl <- difLogistic(verbal_items, verbal_group, focal.name = 1, match = verbal_matching)
logistic_matching <- list(case = "verbal", matching = verbal_matching, chi2 = na_null(rl$Logistik),
                          delta_r2 = na_null(rl$deltaR2))
rla <- difLogistic(sim2$y, sim2$group, focal.name = 1, anchor = anchor, type = "udif")
logistic_anchor <- list(case = "sim_2pl", anchor_items = anchor - 1L, type = "udif",
                        chi2 = na_null(rla$Logistik), delta_r2 = na_null(rla$deltaR2))

res <- list(
  meta = list(case = "v47_dif", r_version = R.version.string,
              difR_version = as.character(packageVersion("difR")), converged = TRUE),
  mantel_haenszel = list(cases = mh_cases, anchor = mh_anchor, matching = mh_matching),
  generalized = list(groups = gmh_case, nominal = nominal_cases),
  mantel = mantel_cases,
  logistic = list(cases = logistic_cases, matching = logistic_matching, anchor = logistic_anchor)
)
write_json(res, file.path("packages", "flexpyctt", "tests", "reference", "v47_dif.json"),
           auto_unbox = TRUE, digits = NA, null = "null", pretty = FALSE)
cat("wrote packages/flexpyctt/tests/reference/v47_dif.json\n")
