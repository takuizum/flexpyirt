# Jackknife slope index (VALIDATION.md V31).
# Run from the repository root:  Rscript validation/r/m5_jsi_reference.R
# Writes tests/reference/v31_jsi.json. CI does not run R; the JSON is committed.
#
# mirt residuals(type = "JSI", fold = FALSE) (Edwards, Houts & Cai, 2018) on mirt fits (61 points
# on [-6, 6], TOL = 1e-10). For each item j mirt refits the model to the data without item j
# (extra arguments such as TOL reach those refits) and reports, in row j,
#   (a_i - a_i^(-j)) / SE(a_i^(-j))
# for every other item i, with the SE from the refit (Oakes, mirt's default SE.type). The Python
# test refits the same models with flexpyirt and compares.
suppressMessages({
  library(mirt)
  library(jsonlite)
})

fit_mirt <- function(data, itemtype) {
  m <- mirt(data, 1, itemtype = itemtype, quadpts = 61, TOL = 1e-10, optimizer = "NR",
            SE = FALSE, verbose = FALSE, technical = list(NCYCLES = 5000))
  stopifnot(extract.mirt(m, "converged"))
  m
}

jsi_case <- function(name, data, itemtype) {
  m <- fit_mirt(data, itemtype)
  jsi <- unclass(residuals(m, type = "JSI", fold = FALSE, verbose = FALSE, TOL = 1e-10,
                           optimizer = "NR", technical = list(NCYCLES = 5000)))
  J <- ncol(data)
  pats <- as.matrix(data)
  list(name = name, itemtype = itemtype, items = colnames(data), n = nrow(data),
       data = lapply(seq_len(nrow(pats)), function(i) lapply(unname(pats[i, ]),
                                                           function(v) if (is.na(v)) NULL else v)),
       coef = lapply(seq_len(J), function(k) as.list(coef(m, simplify = TRUE)$items[k, ])),
       loglik = extract.mirt(m, "logLik"),
       jsi = lapply(seq_len(J), function(r) lapply(unname(jsi[r, ]),
                                                  function(v) if (is.na(v)) NULL else v)))
}

set.seed(3101)
cases <- list()
cases[[1]] <- jsi_case("lsat7_2pl", as.data.frame(expand.table(mirt::LSAT7)), "2PL")
cases[[2]] <- jsi_case("science_grm", as.data.frame(mirt::Science - 1L), "graded")

# Simulated 2PL, 10 items, items 1-2 and 6-7 share nuisance factors, 5% MCAR responses.
n <- 1000
theta <- rnorm(n)
a <- c(1.4, 1.2, 1.0, 0.8, 1.6, 1.1, 0.9, 1.3, 1.5, 0.7)
d <- c(0.2, -0.4, 0.8, -1.0, 0.0, 1.2, -0.6, 0.5, -0.2, 0.9)
eta <- outer(theta, a) + matrix(d, n, length(a), byrow = TRUE)
eta[, 1:2] <- eta[, 1:2] + 1.5 * rnorm(n)
eta[, 6:7] <- eta[, 6:7] + 1.0 * rnorm(n)
sim <- (matrix(runif(n * length(a)), n) < plogis(eta)) * 1L
sim[matrix(runif(length(sim)) < 0.05, n)] <- NA
colnames(sim) <- paste0("s", seq_along(a))
cases[[3]] <- jsi_case("sim_2pl_ld_missing", as.data.frame(sim), "2PL")

# Simulated 4-category GPCM, 8 items, items 3-4 locally dependent.
n <- 800
theta <- rnorm(n)
u <- rnorm(n)
a <- c(1.2, 0.9, 1.4, 1.1, 0.8, 1.3, 1.0, 1.5)
b <- rbind(c(-1, 0, 1), c(-0.5, 0.3, 1.2), c(-1.2, -0.2, 0.6), c(-0.8, 0.4, 1.0),
           c(-1.5, -0.5, 0.5), c(-0.3, 0.5, 1.4), c(-1.0, 0.1, 0.9), c(-0.6, 0.2, 1.1))
gp <- matrix(0L, n, length(a))
for (j in seq_along(a)) {
  th <- theta + if (j %in% 3:4) 1.2 * u else 0
  num <- cbind(0, sapply(1:3, function(k) a[j] * (k * th - sum(b[j, 1:k]))))
  p <- exp(num - apply(num, 1, max))
  p <- p / rowSums(p)
  gp[, j] <- apply(p, 1, function(pr) sample(0:3, 1, prob = pr))
}
colnames(gp) <- paste0("g", seq_along(a))
cases[[4]] <- jsi_case("sim_gpcm_ld", as.data.frame(gp), "gpcm")

res <- list(
  meta = list(case = "v31_jsi", r_version = R.version.string,
              mirt_version = as.character(packageVersion("mirt")),
              converged = TRUE, TOL = 1e-10, quadpts = 61, theta_lim = c(-6, 6)),
  cases = cases
)
write_json(res, file.path("tests", "reference", "v31_jsi.json"), auto_unbox = TRUE, digits = NA,
           null = "null", pretty = FALSE)
cat("wrote tests/reference/v31_jsi.json\n")
