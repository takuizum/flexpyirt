# Classical item analysis (VALIDATION.md V46): keyed scoring, coefficient alpha, item statistics,
# point-biserial / biserial / polyserial correlations, omega total and option analysis.
# Run from the repository root:
#   Rscript packages/flexpyctt/validation/r/ctt_item_analysis_reference.R
# Writes packages/flexpyctt/tests/reference/v46_item_analysis.json. CI does not run R; the JSON is
# committed.
#
# References: psych (alpha, score.multiple.choice, responseFrequency, polyserial, biserial, fa,
# omega), ltm (cronbach.alpha), polycor (polyserial, two-step), lavaan (lavCor for the polychoric
# matrix behind omega). Option statistics have no installed R implementation (TAM is not
# installed), so they are computed here in base R from their definitions.
#
# Known differences, recorded rather than hidden:
# - psych::alpha's raw.r correlates each item with rowSums(x, na.rm = TRUE); with missing data
#   flexpyctt's item_total is the correlation implied by the pairwise covariance matrix (the
#   matrix alpha uses), computed below as colSums(C) / sqrt(diag(C) * sum(C)). psych's r.drop is
#   already C-based and is compared directly.
# - psych::biserial divides by the n - 1 standard deviation of the total while using the
#   proportions p, q, so it equals the two-step biserial times sqrt((n - 1) / n).
# - psych::polyserial scales by sqrt((nrow - 1) / nrow) with nrow counting missing rows, so it is
#   recorded for complete data only.
suppressMessages({
  library(psych)
  library(ltm)
  library(polycor)
  library(lavaan)
  library(jsonlite)
})

as_rows <- function(m) {
  m <- as.matrix(m)
  lapply(seq_len(nrow(m)), function(i) {
    v <- as.list(unname(m[i, ]))
    lapply(v, function(x) if (is.na(x)) NULL else x)
  })
}

col_list <- function(m) lapply(seq_len(ncol(m)), function(j) unname(m[, j]))

# Multiple-choice responses: 2PL for the key, the wrong answers spread over the distractors with
# logits linear in theta (one distractor per item attracts low scorers most).
simulate_mc <- function(n, n_items, n_options, seed) {
  set.seed(seed)
  theta <- rnorm(n)
  a <- runif(n_items, 0.6, 2.0)
  b <- seq(-1.5, 1.5, length.out = n_items)
  key <- sample(1:4, n_items, replace = TRUE)
  out <- matrix(NA_integer_, n, n_items)
  for (j in seq_len(n_items)) {
    k <- n_options[j]
    p <- plogis(a[j] * (theta - b[j]))
    right <- runif(n) < p
    distractors <- setdiff(seq_len(k), key[j])
    slope <- c(-1.2, seq(-0.6, 0.3, length.out = length(distractors) - 1))
    icpt <- c(0.3, seq(0, -1.5, length.out = length(distractors) - 1))
    eta <- outer(theta, slope) + matrix(icpt, n, length(distractors), byrow = TRUE)
    prob <- exp(eta) / rowSums(exp(eta))
    pick <- apply(prob, 1, function(pr) sample(distractors, 1, prob = pr))
    out[, j] <- ifelse(right, key[j], pick)
  }
  colnames(out) <- sprintf("q%02d", seq_len(n_items))
  list(data = out, key = setNames(key, colnames(out)))
}

# Ordered polytomous scores 0..4 from a graded response model.
simulate_poly <- function(n, n_items, seed) {
  set.seed(seed)
  theta <- rnorm(n)
  a <- runif(n_items, 0.8, 2.2)
  out <- matrix(0L, n, n_items)
  for (j in seq_len(n_items)) {
    cuts <- sort(rnorm(4, 0, 1))
    pstar <- sapply(cuts, function(b) plogis(a[j] * (theta - b)))
    out[, j] <- as.integer(rowSums(matrix(runif(n), n, 4) < pstar))
  }
  colnames(out) <- sprintf("y%d", seq_len(n_items))
  out
}

add_mcar <- function(m, rate, seed) {
  set.seed(seed)
  m[matrix(runif(length(m)) < rate, nrow(m))] <- NA
  if (any(rowSums(!is.na(m)) == 0)) stop("a row lost every response")
  m
}

alpha_stats <- function(x) {
  a <- psych::alpha(as.data.frame(x), warnings = FALSE, check.keys = FALSE)
  C <- cov(x, use = "pairwise")
  list(
    raw_alpha = a$total$raw_alpha, std_alpha = a$total$std.alpha,
    drop_raw_alpha = a$alpha.drop$raw_alpha, drop_std_alpha = a$alpha.drop$std.alpha,
    raw_r = unname(a$item.stats$raw.r), r_drop = unname(a$item.stats$r.drop),
    mean = unname(a$item.stats$mean), sd = unname(a$item.stats$sd), n = unname(a$item.stats$n),
    feldt_lower = a$feldt$lower.ci[[1]], feldt_upper = a$feldt$upper.ci[[1]], n_obs = nrow(x),
    c_item_total = unname(colSums(C) / sqrt(diag(C) * sum(C)))
  )
}

# The two-step polyserial factor sd_n(y) / sum(dnorm(tau)) over the item's observed scores.
polyserial_factor <- function(y) {
  y <- y[!is.na(y)]
  tau <- qnorm(cumsum(table(y)) / length(y))
  tau <- tau[-length(tau)]
  sqrt(mean((y - mean(y))^2)) / sum(dnorm(tau))
}

polyserial_stats <- function(x) {
  complete <- !anyNA(x)
  total <- rowSums(x)
  out <- list(factor = unname(sapply(seq_len(ncol(x)), function(j) polyserial_factor(x[, j]))))
  if (complete) {
    out$polycor_total <- unname(sapply(seq_len(ncol(x)), function(j)
      polycor::polyserial(total, x[, j], ML = FALSE)))
    out$polycor_rest <- unname(sapply(seq_len(ncol(x)), function(j)
      polycor::polyserial(total - x[, j], x[, j], ML = FALSE)))
    out$psych_total <- unname(as.numeric(psych::polyserial(matrix(total), x)))
    out$point_biserial_total <- unname(as.numeric(cor(total, x)))
    out$point_biserial_rest <- unname(sapply(seq_len(ncol(x)), function(j)
      cor(total - x[, j], x[, j])))
    if (all(x %in% c(0, 1))) {
      out$psych_biserial_total <- unname(sapply(seq_len(ncol(x)), function(j)
        suppressWarnings(psych::biserial(total, x[, j]))[1, 1]))
      out$ltm_point_biserial_total <- unname(sapply(seq_len(ncol(x)), function(j)
        ltm::biserial.cor(total, x[, j], level = 2)))
    }
  }
  out
}

omega_stats <- function(m, n_obs) {
  om <- suppressWarnings(suppressMessages(
    psych::omega(m, nfactors = 1, fm = "minres", n.obs = n_obs, plot = FALSE)))
  f <- suppressWarnings(psych::fa(m, nfactors = 1, fm = "minres", rotate = "none"))
  lam <- as.numeric(f$loadings[, 1])
  if (sum(lam) < 0) lam <- -lam
  list(correlation = unname(m), omega_tot = om$omega.tot, fa_loadings = lam,
       omega_from_fa = 1 - sum(1 - lam^2) / sum(m))
}

lavaan_polychoric <- function(x) {
  df <- as.data.frame(x)
  fit <- withCallingHandlers(
    lavCor(df, ordered = names(df), missing = "pairwise", output = "fit"),
    warning = function(w) if (grepl("converge", conditionMessage(w))) stop(w))
  unname(unclass(lavInspect(fit, "sampstat")$cov))
}

reliability_case <- function(x) {
  x <- as.matrix(x)
  listwise <- x[complete.cases(x), , drop = FALSE]
  ltm_raw <- ltm::cronbach.alpha(as.data.frame(x), na.rm = TRUE)$alpha
  ltm_std <- ltm::cronbach.alpha(as.data.frame(x), standardized = TRUE, na.rm = TRUE)$alpha
  list(
    items = colnames(x), n = nrow(x), scores = as_rows(x),
    pairwise = alpha_stats(x), listwise = alpha_stats(listwise),
    ltm_alpha = ltm_raw, ltm_std_alpha = ltm_std, n_listwise = nrow(listwise),
    polyserial = polyserial_stats(x), polyserial_listwise = polyserial_stats(listwise),
    omega_pearson = omega_stats(cor(x, use = "pairwise"), nrow(x)),
    omega_polychoric = omega_stats(lavaan_polychoric(x), nrow(x))
  )
}

option_stats <- function(raw, key, n_groups) {
  scored <- t(t(raw) == key) + 0
  scored[is.na(scored)] <- 0
  total <- rowSums(scored)
  breaks <- quantile(total, probs = seq(0, 1, length.out = n_groups + 1), type = 7)
  if (anyDuplicated(breaks)) stop("duplicated breaks")
  group <- cut(total, breaks, include.lowest = TRUE, labels = FALSE)
  rows <- list()
  for (j in seq_len(ncol(raw))) {
    answered <- !is.na(raw[, j])
    chosen <- raw[answered, j]
    rest <- (total - scored[, j])[answered]
    tot <- total[answered]
    grp <- group[answered]
    for (opt in sort(unique(c(chosen, key[[j]])))) {
      picked <- chosen == opt
      pb <- if (sum(picked) %in% c(0, length(picked))) NA else cor(as.numeric(picked), rest)
      rows[[length(rows) + 1]] <- list(
        item = colnames(raw)[j], option = opt, is_key = opt == key[[j]], n = sum(picked),
        proportion = mean(picked), point_biserial = pb,
        mean_score = if (any(picked)) mean(tot[picked]) else NA,
        group_proportions = sapply(seq_len(n_groups), function(g) mean(picked[grp == g])))
    }
  }
  list(n_groups = n_groups, breaks = unname(breaks), total = unname(total), rows = rows)
}

mc_case <- function(name, raw, key, n_groups) {
  keyed <- psych::score.multiple.choice(unname(key), raw, score = FALSE)
  rf <- psych::responseFrequency(raw)
  c(list(name = name, raw = as_rows(raw), key = as.list(key),
         keyed_scores = as_rows(keyed),
         response_frequency = list(options = colnames(rf), values = as_rows(rf)),
         options = option_stats(raw, key, n_groups)),
    reliability_case(keyed))
}

mc <- simulate_mc(500, 10, c(4, 4, 4, 4, 4, 4, 4, 4, 4, 5), seed = 4601)
mc_missing <- add_mcar(mc$data, 0.05, seed = 4602)
poly <- simulate_poly(300, 6, seed = 4603)
poly_missing <- add_mcar(poly, 0.06, seed = 4604)

cases <- list(
  mc_case("mc_complete", mc$data, mc$key, 3),
  mc_case("mc_missing", mc_missing, mc$key, 4),
  c(list(name = "poly_complete"), reliability_case(poly)),
  c(list(name = "poly_missing"), reliability_case(poly_missing))
)

# Three items: a just-identified one-factor model (omega still defined).
three <- poly[, 1:3]
small <- list(scores = as_rows(three), omega_pearson = omega_stats(cor(three), nrow(three)))

res <- list(
  meta = list(case = "v46_item_analysis", r_version = R.version.string,
              psych_version = as.character(packageVersion("psych")),
              ltm_version = as.character(packageVersion("ltm")),
              polycor_version = as.character(packageVersion("polycor")),
              lavaan_version = as.character(packageVersion("lavaan")),
              converged = TRUE),
  cases = cases,
  three_items = small
)
out <- file.path("packages", "flexpyctt", "tests", "reference", "v46_item_analysis.json")
write_json(res, out, auto_unbox = TRUE, digits = NA, null = "null", na = "null", pretty = FALSE)
cat("wrote", out, "\n")
