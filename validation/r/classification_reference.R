# Reference values for classification accuracy and consistency (VALIDATION.md V52;
# DESIGN.md §18.4) from R: cacIRT (Lathrop, 2015) and an independent recomputation from mirt.
# Run from the repository root:  Rscript validation/r/classification_reference.R
# Writes tests/reference/v52_classification.json. CI does not run R; the JSON is committed.
#
# For each case (and group) mirt's estimates are stored, with the data as unique patterns and
# counts and mirt's EAP scores per pattern. Each run (method x cut scale x cuts x D / P) stores
#   - cacIRT, where it has the method:
#       class.Lee()  dichotomous items (2PL / 3PL), D (quadrature) and P (ability = mirt EAP)
#       Lee.poly.D() / Lee.poly.P()  polytomous items (GRM); cacIRT clips the category
#                    probabilities to [1e-4, 1 - 1e-4] (gen.rec.raw), so it is not exact
#       Rud.D() / Rud.P()  every item family, with SE = 1 / sqrt(mirt's testinfo()). class.Rud()
#                    is not used: its SEM() takes the 3PL information from cacIRT's iif(),
#                    which has (1 - c^2) where the information has (1 - c)^2 (Birnbaum, 1968),
#                    so its SE is wrong whenever c > 0
#     "Marginal" (per cut) and "Simultaneous" (all cuts) accuracy and consistency and the
#     conditional values;
#   - a recomputation from mirt's probtrace() / expected.test() / testinfo() with a
#     Lord-Wingersky recursion written here: the tables A (true x observed) and B (first x
#     second), overall accuracy / consistency / kappa, per-category and per-cut indices
#     (including the false positive / negative rates, which cacIRT does not have) and the
#     conditional values. The script asserts that it agrees with cacIRT's exact functions.
# cacIRT's logistic is 1 / (1 + exp(-D a (theta - b))) with D = 1.7 (class.Rud's D approach
# ignores its D argument), so mirt's slope-intercept parameters are passed as
# a / 1.7, b = -d / a, c = g.
suppressMessages({
  library(mirt)
  library(cacIRT)
  library(jsonlite)
})

quadpts <- 61L
theta_lim <- c(-6, 6)
nodes <- seq(theta_lim[1], theta_lim[2], length.out = quadpts)
Theta <- matrix(nodes)

compress <- function(dat, grp = NULL) {
  key <- apply(cbind(dat, grp), 1, paste, collapse = ",")
  first <- !duplicated(key)
  list(first = first, counts = as.integer(table(factor(key, levels = key[first]))))
}

# Item parameters in flexpyirt's names: a, d / g, d1.. (as in m7_information_reference.R).
item_params <- function(m_single) {
  cf <- coef(m_single, simplify = TRUE)$items
  out <- lapply(seq_len(nrow(cf)), function(j) {
    v <- cf[j, ]
    v <- v[!is.na(v)]
    names(v)[names(v) == "a1"] <- "a"
    as.list(v[!names(v) %in% c("u")])
  })
  setNames(out, rownames(cf))
}

# --- independent recomputation --------------------------------------------------------------
traces_at <- function(m_single, theta) {
  lapply(seq_len(extract.mirt(m_single, "nitems")),
         function(j) unname(probtrace(extract.item(m_single, j), matrix(theta))))
}

lord_wingersky <- function(traces) {
  lik <- matrix(1, nrow(traces[[1]]), 1)
  for (p in traces) {
    k <- ncol(p)
    new <- matrix(0, nrow(lik), ncol(lik) + k - 1)
    for (cat in seq_len(k)) {
      cols <- seq_len(ncol(lik)) + cat - 1
      new[, cols] <- new[, cols] + lik * p[, cat]
    }
    lik <- new
  }
  lik
}

category_of <- function(values, cuts) vapply(values, function(v) sum(v >= cuts), 0)

lee_probs <- function(lik, cuts) {
  h <- category_of(seq_len(ncol(lik)) - 1, cuts)
  sapply(0:length(cuts), function(cat) rowSums(lik[, h == cat, drop = FALSE]))
}

rudner_probs <- function(theta, se, cuts) {
  bounds <- c(-Inf, cuts, Inf)
  sapply(seq_len(length(cuts) + 1),
         function(cat) pnorm(bounds[cat + 1], theta, se) - pnorm(bounds[cat], theta, se))
}

kappa <- function(agree, marg) (agree - sum(marg^2)) / (1 - sum(marg^2))

indices <- function(P, true, w) {
  H <- ncol(P)
  A <- matrix(0, H, H)
  for (t in 0:(H - 1)) A[t + 1, ] <- colSums(P[true == t, , drop = FALSE] * w[true == t])
  B <- t(P) %*% (P * w)
  marg <- rowSums(B)
  truep <- rowSums(A)
  by_cut <- lapply(seq_len(H - 1), function(k) {
    lo <- seq_len(k); hi <- (k + 1):H
    cons <- sum(B[lo, lo]) + sum(B[hi, hi])
    list(accuracy = sum(A[lo, lo]) + sum(A[hi, hi]), consistency = cons,
         kappa = kappa(cons, c(sum(marg[lo]), sum(marg[hi]))),
         false_positive = sum(A[lo, hi]), false_negative = sum(A[hi, lo]))
  })
  list(
    accuracy = sum(diag(A)), consistency = sum(diag(B)), kappa = kappa(sum(diag(B)), marg),
    accuracy_table = unname(A), consistency_table = unname(B),
    by_category = list(true_proportion = truep, proportion = marg,
                       accuracy = diag(A) / truep, consistency = diag(B) / marg),
    by_cut = list(
      accuracy = sapply(by_cut, `[[`, "accuracy"),
      consistency = sapply(by_cut, `[[`, "consistency"),
      kappa = sapply(by_cut, `[[`, "kappa"),
      false_positive = sapply(by_cut, `[[`, "false_positive"),
      false_negative = sapply(by_cut, `[[`, "false_negative")
    ),
    conditional_accuracy = P[cbind(seq_len(nrow(P)), true + 1)],
    conditional_consistency = rowSums(P^2)
  )
}

inverse_tcc <- function(m_single, cuts) {
  vapply(cuts, function(c) {
    uniroot(function(t) expected.test(m_single, matrix(t)) - c, c(-40, 40), tol = 1e-14)$root
  }, 0)
}

# One run: the recomputation, plus cacIRT where it has the method.
run <- function(m_single, method, scale, cuts, theta, w, ip = NULL, keep_conditional = TRUE) {
  method_cuts <- cuts
  if (method == "lee" && scale == "theta") method_cuts <- expected.test(m_single, matrix(cuts))
  if (method == "rudner" && scale == "summed_score") method_cuts <- inverse_tcc(m_single, cuts)
  traces <- traces_at(m_single, theta)
  if (method == "lee") {
    true_score <- expected.test(m_single, matrix(theta))
    P <- lee_probs(lord_wingersky(traces), method_cuts)
    true <- category_of(true_score, method_cuts)
  } else {
    se <- 1 / sqrt(testinfo(m_single, matrix(theta)))
    P <- rudner_probs(theta, se, method_cuts)
    true <- category_of(theta, method_cuts)
  }
  rec <- indices(P, true, w)
  out <- list(method = method, scale = scale, cuts = cuts, method_cuts = method_cuts,
              recompute = rec)
  if (method == "lee") out$true_score <- true_score else out$se <- se
  is_d <- length(w) == quadpts && isTRUE(all.equal(theta, nodes))
  quad <- if (is_d) list(theta, w) else NULL
  ability <- if (is_d) NULL else theta
  cac <- NULL
  if (!is.null(ip) || method == "rudner") {
    cac <- if (method == "lee") {
      class.Lee(method_cuts, ip, ability = ability, quadrature = quad)
    } else if (is_d) {
      Rud.D(method_cuts, quad, se)
    } else {
      Rud.P(method_cuts, theta, se)
    }
  } else if (method == "lee") {
    k <- ncol(traces[[1]])
    Pij <- array(unlist(traces), dim = c(length(theta), k, length(traces)))
    cac <- if (is_d) Lee.poly.D(method_cuts, Pij, quad) else Lee.poly.P(method_cuts, Pij, theta)
  }
  if (!is.null(cac)) {
    marginal <- cac$Marginal
    n_cut <- length(cuts)
    simultaneous <- if (n_cut > 1) marginal["Simultaneous", ] else marginal[1, ]
    cond_acc <- as.matrix(cac$Conditional$Accuracy)
    cond_con <- as.matrix(cac$Conditional$Consistency)
    out$cacirt <- list(
      exact = !is.null(ip) || method == "rudner",
      per_cut_accuracy = unname(marginal[seq_len(n_cut), "Accuracy"]),
      per_cut_consistency = unname(marginal[seq_len(n_cut), "Consistency"]),
      accuracy = unname(simultaneous["Accuracy"]),
      consistency = unname(simultaneous["Consistency"]),
      conditional_accuracy = unname(cond_acc[, ncol(cond_acc)]),
      conditional_consistency = unname(cond_con[, ncol(cond_con)])
    )
    # The recomputation must reproduce cacIRT's exact (dichotomous) functions.
    if (out$cacirt$exact) {
      stopifnot(abs(out$cacirt$accuracy - rec$accuracy) < 1e-12,
                abs(out$cacirt$consistency - rec$consistency) < 1e-12,
                max(abs(out$cacirt$per_cut_accuracy - rec$by_cut$accuracy)) < 1e-12,
                max(abs(out$cacirt$per_cut_consistency - rec$by_cut$consistency)) < 1e-12,
                max(abs(out$cacirt$conditional_accuracy - rec$conditional_accuracy)) < 1e-12)
    }
    out$cacirt_max_abs_diff <- max(abs(c(
      out$cacirt$accuracy - rec$accuracy, out$cacirt$consistency - rec$consistency,
      out$cacirt$per_cut_accuracy - rec$by_cut$accuracy,
      out$cacirt$per_cut_consistency - rec$by_cut$consistency)))
  }
  if (!keep_conditional) out$recompute$conditional_accuracy <- NULL
  if (!keep_conditional) out$recompute$conditional_consistency <- NULL
  if (!keep_conditional) { out$true_score <- NULL; out$se <- NULL }
  out
}

cacirt_ip <- function(m_single) {
  cf <- coef(m_single, simplify = TRUE)$items
  g <- if ("g" %in% colnames(cf)) cf[, "g"] else rep(0, nrow(cf))
  cbind(cf[, "a1"] / 1.7, -cf[, "d"] / cf[, "a1"], g)
}

all_runs <- function(m_single, cut_sets, theta_hat, mu = 0, s2 = 1, dichotomous = TRUE) {
  ip <- if (dichotomous) cacirt_ip(m_single) else NULL
  w <- dnorm(nodes, mu, sqrt(s2)); w <- w / sum(w)
  p_w <- rep(1 / length(theta_hat), length(theta_hat))
  runs <- list()
  for (cs in cut_sets) {
    runs[[length(runs) + 1]] <- c(list(approach = "D"),
                                  run(m_single, cs$method, cs$scale, cs$cuts, nodes, w, ip))
    runs[[length(runs) + 1]] <- c(list(approach = "P"),
                                  run(m_single, cs$method, cs$scale, cs$cuts, theta_hat, p_w, ip,
                                      keep_conditional = FALSE))
  }
  runs
}

fit_single <- function(dat, itemtype) {
  m <- mirt(dat, 1, itemtype = itemtype, TOL = 1e-8, verbose = FALSE, quadpts = quadpts,
            technical = list(NCYCLES = 20000L, theta_lim = theta_lim))
  stopifnot(isTRUE(extract.mirt(m, "converged")))
  m
}

# A mirt model with every parameter fixed at the generating values (no estimation).
fixed_model <- function(dat, itemtype, a, d, g) {
  sv <- mirt(dat, 1, itemtype = itemtype, pars = "values")
  for (j in seq_len(ncol(dat))) {
    item <- colnames(dat)[j]
    sv$value[sv$item == item & sv$name == "a1"] <- a[j]
    sv$value[sv$item == item & sv$name == "d"] <- d[j]
    sv$value[sv$item == item & sv$name == "g"] <- g[j]
  }
  sv$est <- FALSE
  mirt(dat, 1, itemtype = itemtype, pars = sv, verbose = FALSE, quadpts = quadpts,
       technical = list(theta_lim = theta_lim))
}

eap_of <- function(m) {
  unname(fscores(m, method = "EAP", quadpts = quadpts, theta_lim = theta_lim)[, 1])
}

single_case <- function(description, dat, itemtype, cut_sets, dichotomous, m = NULL) {
  if (is.null(m)) m <- fit_single(dat, itemtype)
  cp <- compress(dat)
  theta_hat <- eap_of(m)
  list(description = description, n = nrow(dat), items = colnames(dat),
       itemtype = rep(itemtype, ncol(dat)), n_categories = unname(extract.mirt(m, "K")),
       patterns = unname(as.matrix(dat)[cp$first, , drop = FALSE]), counts = cp$counts,
       params = item_params(m), eap = theta_hat[cp$first],
       runs = all_runs(m, cut_sets, theta_hat, dichotomous = dichotomous))
}

cut_set <- function(method, scale, cuts) list(method = method, scale = scale, cuts = cuts)

# --- data -----------------------------------------------------------------------------------
set.seed(20261126)
J <- 20L
a3 <- matrix(runif(J, 0.9, 2.0)); d3 <- matrix(seq(-1.5, 2.0, length.out = J))
g3 <- rep(c(0.15, 0.2), J / 2)
dat_3pl <- simdata(a3, d3, 3000L, itemtype = "dich", guess = g3)
colnames(dat_3pl) <- sprintf("i%02d", seq_len(J))
# A 3PL fit of this size need not converge; the indices only need fixed parameters, so the
# 3PL case uses the generating parameters (mirt's EAP scores are computed with them).
m_3pl <- fixed_model(dat_3pl, "3PL", a3[, 1], d3[, 1], g3)
dich_sets <- list(
  cut_set("lee", "summed_score", 12),
  cut_set("lee", "summed_score", c(8, 14)),
  cut_set("lee", "summed_score", c(7.5, 13.2)),
  cut_set("lee", "theta", c(-0.5, 0.8)),
  cut_set("rudner", "theta", 0.3),
  cut_set("rudner", "theta", c(-0.5, 0.8)),
  cut_set("rudner", "summed_score", c(8, 14))
)

Jp <- 10L
ap <- matrix(runif(Jp, 0.9, 1.9))
dp <- t(sapply(seq_len(Jp), function(j) c(1.4, 0.1, -1.2) + (j - 5.5) * 0.12))
dat_grm <- simdata(ap, dp, 2000L, itemtype = "graded")
colnames(dat_grm) <- sprintf("g%02d", seq_len(Jp))
poly_sets <- list(
  cut_set("lee", "summed_score", 15),
  cut_set("lee", "summed_score", c(10, 20)),
  cut_set("rudner", "theta", c(-0.5, 0.8))
)

cases <- list(
  threepl = single_case("3PL (generating parameters, fixed), J = 20, N = 3000", dat_3pl, "3PL",
                        dich_sets, TRUE, m = m_3pl),
  grm = single_case("GRM, K = 4, J = 10, N = 2000", dat_grm, "graded", poly_sets, FALSE)
)

# --- two groups: 15 x 2PL, all items invariant, focal mean / variance free ------------------
Ng <- 1500L
Jg <- 15L
a_g <- matrix(runif(Jg, 0.9, 1.8)); d_g <- matrix(seq(-1.4, 1.4, length.out = Jg))
dat_g <- rbind(simdata(a_g, d_g, Ng, itemtype = "dich"),
               simdata(a_g, d_g, Ng, itemtype = "dich", mu = 0.5, sigma = matrix(1.6)))
colnames(dat_g) <- sprintf("t%02d", seq_len(Jg))
grp <- factor(rep(c("ref", "focal"), each = Ng), levels = c("ref", "focal"))
mg <- multipleGroup(dat_g, 1, group = grp, itemtype = "2PL",
                    invariance = c(colnames(dat_g), "free_means", "free_var"), TOL = 1e-8,
                    quadpts = quadpts, verbose = FALSE,
                    technical = list(NCYCLES = 20000L, theta_lim = theta_lim))
stopifnot(isTRUE(extract.mirt(mg, "converged")))
cp <- compress(dat_g, as.character(grp))
eap_g <- unname(fscores(mg, method = "EAP", quadpts = quadpts, theta_lim = theta_lim)[, 1])
group_sets <- list(cut_set("lee", "summed_score", c(5, 10)),
                   cut_set("rudner", "theta", c(-0.5, 0.8)))
groups <- list()
for (g in levels(grp)) {
  sg <- extract.group(mg, g)
  cf <- coef(sg, simplify = TRUE)
  mu <- unname(cf$means[1]); s2 <- unname(cf$cov[1, 1])
  groups[[g]] <- list(n = sum(grp == g), params = item_params(sg),
                      latent = list(mean = mu, var = s2),
                      runs = all_runs(sg, group_sets, eap_g[grp == g], mu, s2))
}
twogroup <- list(
  description = "two groups: 15 2PL, all items invariant, focal N(mu, s2) free",
  items = colnames(dat_g), itemtype = rep("2PL", Jg), n_categories = rep(2L, Jg),
  patterns = unname(as.matrix(dat_g)[cp$first, , drop = FALSE]),
  pattern_groups = as.character(grp)[cp$first], counts = cp$counts,
  eap = eap_g[cp$first], group_names = levels(grp), groups = groups
)

bundle <- list(
  meta = list(
    case = "v52_classification",
    description = paste("V52: classification accuracy and consistency (Lee, Rudner) vs",
                        "cacIRT class.Lee / class.Rud / Lee.poly.D / Lee.poly.P and an",
                        "independent recomputation from mirt"),
    r_version = R.version.string,
    mirt_version = as.character(packageVersion("mirt")),
    cacirt_version = as.character(packageVersion("cacIRT")),
    quadpts = quadpts, theta_lim = theta_lim,
    # Every mirt fit above is asserted to have converged (fit_single(), multipleGroup());
    # the 3PL case is not estimated (fixed_model()).
    converged = TRUE
  ),
  cases = cases,
  twogroup = twogroup
)
path <- file.path("tests", "reference", "v52_classification.json")
write_json(bundle, path, auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null")
cat("wrote", path, "\n")
