# Local dependence and model-based parallel analysis (VALIDATION.md V26).
# Run from the repository root:  Rscript validation/r/m5_local_dependence_reference.R
# Writes tests/reference/v26_local_dependence.json. CI does not run R; the JSON is committed.
#
# mirt residuals(type = "LD" / "LDG2" / "Q3") on mirt fits (61 points on [-6, 6], TOL = 1e-10,
# optimizer NR): the Python test refits the same models with flexpyirt and compares.
#   LD:   lower triangle X^2, upper triangle signed Cramer's V.
#   LDG2: lower triangle G^2.
#   Q3:   residual correlations with EAP scores (fscores default).
# ltm::unidimTest (Drasgow & Lissak's modified parallel analysis) is called with data and the
# mirt 2PL estimates (thetas = cbind(d, a1), IRT = FALSE), so it simulates z ~ N(0, 1) as
# flexpyirt does; its tetrachorics come from polycor::polychor and the diagonal is the largest
# off-diagonal correlation.
# ltm ships its own (factor-coded) Science and LSAT data, so mirt's are named with mirt::.
source(file.path("validation", "r", "setup.R"))  # mirt from GitHub; mirt_version / mirt_sha
suppressMessages({
  library(mirt)
  library(ltm)
  library(jsonlite)
})

fit_mirt <- function(data, itemtype) {
  m <- mirt(data, 1, itemtype = itemtype, quadpts = 61, TOL = 1e-10, optimizer = "NR",
            SE = FALSE, verbose = FALSE, technical = list(NCYCLES = 5000))
  stopifnot(extract.mirt(m, "converged"))
  m
}

ld_case <- function(name, data, itemtype) {
  m <- fit_mirt(data, itemtype)
  ld <- residuals(m, type = "LD", verbose = FALSE)
  ldg2 <- residuals(m, type = "LDG2", verbose = FALSE)
  q3 <- residuals(m, type = "Q3", verbose = FALSE)
  J <- ncol(data)
  lower <- which(lower.tri(matrix(0, J, J)), arr.ind = TRUE)  # (row > col): i = col, j = row
  pairs <- lapply(seq_len(nrow(lower)), function(k) {
    r <- lower[k, 1]; cl <- lower[k, 2]
    list(i = cl - 1L, j = r - 1L, x2 = unclass(ld)[r, cl], cramer_v = unclass(ld)[cl, r],
         g2 = unclass(ldg2)[r, cl], q3 = unclass(q3)[cl, r])
  })
  pats <- as.matrix(data)
  list(name = name, itemtype = itemtype, items = colnames(data), n = nrow(data),
       data = lapply(seq_len(nrow(pats)), function(i) lapply(unname(pats[i, ]),
                                                           function(v) if (is.na(v)) NULL else v)),
       coef = lapply(seq_len(J), function(k) as.list(coef(m, simplify = TRUE)$items[k, ])),
       loglik = extract.mirt(m, "logLik"), pairs = pairs)
}

set.seed(2601)
cases <- list()
cases[[1]] <- ld_case("lsat7_2pl", as.data.frame(expand.table(mirt::LSAT7)), "2PL")
cases[[2]] <- ld_case("science_grm", as.data.frame(mirt::Science - 1L), "graded")

# Simulated 2PL with one locally dependent pair (items 1 and 2 share a nuisance factor) and
# 5% MCAR responses.
n <- 1200
theta <- rnorm(n)
u <- rnorm(n)
a <- c(1.4, 1.2, 1.0, 0.8, 1.6, 1.1, 0.9, 1.3)
d <- c(0.2, -0.4, 0.8, -1.0, 0.0, 1.2, -0.6, 0.5)
eta <- outer(theta, a) + matrix(d, n, length(a), byrow = TRUE)
eta[, 1:2] <- eta[, 1:2] + 1.5 * u
sim <- (matrix(runif(n * length(a)), n) < plogis(eta)) * 1L
sim[matrix(runif(length(sim)) < 0.05, n)] <- NA
colnames(sim) <- paste0("s", seq_along(a))
cases[[3]] <- ld_case("sim_2pl_ld_missing", as.data.frame(sim), "2PL")

# ltm::unidimTest on LSAT7 with mirt's 2PL estimates.
lsat <- expand.table(mirt::LSAT7)
m <- fit_mirt(lsat, "2PL")
cf <- coef(m, simplify = TRUE)$items
set.seed(2602)
ut <- unidimTest(data = lsat, thetas = cbind(cf[, "d"], cf[, "a1"]), IRT = FALSE, B = 4000)
unidim <- list(case = "lsat7_2pl", B = 4000, observed = ut$Tobs,
               boot_second_mean = mean(ut$T.boot[, 2]), boot_second_sd = sd(ut$T.boot[, 2]),
               boot_second_q = unname(quantile(ut$T.boot[, 2], c(0.5, 0.9, 0.95))),
               p_value = ut$p.value)

res <- list(
  meta = list(case = "v26_local_dependence", r_version = R.version.string,
              mirt_version = mirt_version_string(), mirt_sha = mirt_commit_sha(),
              ltm_version = as.character(packageVersion("ltm")),
              polycor_version = as.character(packageVersion("polycor")),
              converged = TRUE, TOL = 1e-10, quadpts = 61, theta_lim = c(-6, 6)),
  cases = cases, unidim_test = unidim
)
write_json(res, file.path("tests", "reference", "v26_local_dependence.json"), auto_unbox = TRUE,
           digits = NA, null = "null", pretty = FALSE)
cat("wrote tests/reference/v26_local_dependence.json\n")
