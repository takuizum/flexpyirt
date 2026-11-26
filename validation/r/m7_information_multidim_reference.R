# M7 reference values (VALIDATION.md V43: information matrix, directional information and
# expected scores of multidimensional models; DESIGN.md §18.3) from R mirt.
# Run from the repository root:  Rscript validation/r/m7_information_multidim_reference.R
# Writes tests/reference/v43_information_multidim.json. CI does not run R; the JSON is committed.
#
# The information functions depend on the item parameters only, so nothing is estimated: each
# case sets chosen parameter values with pars = "values" and builds the mirt object with
# TOL = NaN (confirmatory models with mirt(), bifactor and two-tier models with bfactor()). The
# data are uniform random responses, only there so that mirt can build the object.
#
# Per case the file stores the parameters in flexpyirt's names, a set of theta points and
#   - iteminfo(item, Theta, multidim_matrix = TRUE) per item and point -> info_matrix [J][T][D][D]
#   - per direction (angles in degrees to each axis, with sum(cos^2) = 1):
#       iteminfo(item, Theta, degrees = ) per item                     -> iteminfo [T][J]
#       testinfo(m, Theta, degrees = )                                  -> testinfo [T]
#       sqrt(u' (sum_j I_j)^-1 u), u = cos(degrees), from the matrices  -> se [T]
#   - expected.item() per item and expected.test()                     -> expected_item, expected_test
suppressMessages({
  library(mirt)
  library(jsonlite)
})

set.seed(20261126)
N <- 400L

random_data <- function(n_cat, items) {
  dat <- sapply(n_cat, function(k) sample(rep_len(0:(k - 1L), N)))
  colnames(dat) <- items
  dat
}

# Names of a mirt item's parameters as flexpyirt names them.
flex_params <- function(sv_item, itemtype, k) {
  v <- setNames(sv_item$value, sv_item$name)
  if (itemtype %in% c("gpcm", "nominal")) {
    drop <- c("d0", "ak0", paste0("ak", k - 1L))
    if (itemtype == "gpcm") drop <- c(drop, grep("^ak", names(v), value = TRUE))
    v <- v[!names(v) %in% drop]
  }
  as.list(v)
}

set_values <- function(sv, values) {
  for (i in names(values)) {
    for (nm in names(values[[i]])) {
      sel <- sv$item == i & sv$name == nm
      stopifnot(sum(sel) == 1L)
      sv$value[sel] <- values[[i]][[nm]]
    }
  }
  sv$est <- FALSE
  sv
}

theta_points <- function(n_factors) {
  inner <- matrix(round(runif(12L * n_factors, -3, 3), 3), ncol = n_factors)
  extreme <- rbind(rep(-5, n_factors), rep(c(5, -5), length.out = n_factors),
                   rep(0, n_factors))
  rbind(inner, extreme)
}

compute_case <- function(m, description, dat, itemtype, structure, factors, directions) {
  items <- colnames(dat)
  n_factors <- length(factors)
  k <- unname(extract.mirt(m, "K"))
  Theta <- theta_points(n_factors)
  sv <- mod2values(m)
  item_objs <- lapply(seq_along(items), function(j) extract.item(m, j))
  info_matrix <- lapply(item_objs, function(it) {
    lapply(seq_len(nrow(Theta)), function(t) {
      unname(iteminfo(it, Theta[t, , drop = FALSE], multidim_matrix = TRUE))
    })
  })
  test_matrix <- lapply(seq_len(nrow(Theta)), function(t) {
    Reduce(`+`, lapply(info_matrix, function(x) x[[t]]))
  })
  dirs <- lapply(directions, function(deg) {
    u <- cos(deg * pi / 180)
    stopifnot(abs(sum(u^2) - 1) < 1e-12)
    ii <- sapply(item_objs, function(it) iteminfo(it, Theta, degrees = deg))
    ti <- testinfo(m, Theta, degrees = deg)
    quad <- sapply(test_matrix, function(M) drop(t(u) %*% M %*% u))
    # mirt's directional information is u' I u with its own information matrix.
    stopifnot(max(abs(quad - ti) / ti) < 1e-10)
    se <- sapply(test_matrix, function(M) sqrt(drop(t(u) %*% solve(M, u))))
    list(degrees = deg, iteminfo = unname(ii), testinfo = unname(ti), se = unname(se))
  })
  expected_item <- sapply(item_objs, function(it) expected.item(it, Theta))
  list(
    description = description, structure = structure, factors = factors, items = items,
    itemtype = itemtype, n_categories = k,
    params = setNames(lapply(seq_along(items), function(j) {
      flex_params(sv[sv$item == items[j], ], itemtype[j], k[j])
    }), items),
    theta = unname(Theta),
    info_matrix = info_matrix,
    directions = dirs,
    expected_item = unname(expected_item),
    expected_test = unname(expected.test(m, Theta))
  )
}

cases <- list()

# --- confirmatory, 2 factors, every family, cross-loadings ---------------------------------
items <- paste0("i", 1:10)
itemtype <- c("2PL", "2PL", "3PL", "graded", "nominal", "2PL", "4PL", "graded", "gpcm", "gpcm")
n_cat <- c(2, 2, 2, 4, 3, 2, 2, 3, 4, 3)
dat <- random_data(n_cat, items)
spec <- mirt.model("F1 = 1-5, 10\nF2 = 2, 5-10\nCOV = F1*F2")
values <- list(
  i1 = list(a1 = 1.3, d = -0.4),
  i2 = list(a1 = 0.9, a2 = 1.1, d = 0.5),
  i3 = list(a1 = 1.7, d = 0.3, g = 0.2),
  i4 = list(a1 = 1.4, d1 = 1.2, d2 = 0.1, d3 = -1.1),
  i5 = list(a1 = 0.8, a2 = 0.6, ak1 = 0.7, d1 = 0.4, d2 = -0.3),
  i6 = list(a2 = 1.5, d = 0.8),
  i7 = list(a2 = 1.2, d = -0.2, g = 0.15, u = 0.92),
  i8 = list(a2 = 1.1, d1 = 0.9, d2 = -0.7),
  i9 = list(a2 = 1.0, d1 = 0.6, d2 = 0.1, d3 = -0.8),
  i10 = list(a1 = 0.7, a2 = 0.9, d1 = 0.3, d2 = -0.5)
)
sv <- mirt(dat, spec, itemtype = itemtype, pars = "values")
sv$value[sv$name == "COV_21"] <- 0.4
m <- mirt(dat, spec, itemtype = itemtype, pars = set_values(sv, values), TOL = NaN,
          quadpts = 9L, verbose = FALSE)
cases$cfa2_mixed <- compute_case(
  m, "two correlated factors: 2PL, 3PL, 4PL, GRM, GPCM, nominal; four cross-loading items",
  dat, itemtype, "factors", c("F1", "F2"),
  list(c(0, 90), c(90, 0), c(45, 45), c(30, 60), c(150, 60))
)

# --- confirmatory, 3 factors ----------------------------------------------------------------
items <- paste0("i", 1:9)
itemtype <- c("2PL", "2PL", "graded", "2PL", "gpcm", "3PL", "2PL", "graded", "2PL")
n_cat <- c(2, 2, 3, 2, 3, 2, 2, 4, 2)
dat <- random_data(n_cat, items)
spec <- mirt.model("F1 = 1-3, 7\nF2 = 4-6\nF3 = 6-9\nCOV = F1*F2*F3")
values <- list(
  i1 = list(a1 = 1.2, d = 0.1), i2 = list(a1 = 0.8, d = -0.9),
  i3 = list(a1 = 1.5, d1 = 0.8, d2 = -0.6),
  i4 = list(a2 = 1.4, d = 0.4), i5 = list(a2 = 1.0, d1 = 0.2, d2 = -0.4),
  i6 = list(a2 = 0.9, a3 = 0.8, d = -0.3, g = 0.2),
  i7 = list(a1 = 0.6, a3 = 1.3, d = 0.7),
  i8 = list(a3 = 1.1, d1 = 1.0, d2 = 0.0, d3 = -1.2), i9 = list(a3 = 1.6, d = -0.5)
)
sv <- mirt(dat, spec, itemtype = itemtype, pars = "values")
m <- mirt(dat, spec, itemtype = itemtype, pars = set_values(sv, values), TOL = NaN,
          quadpts = 5L, verbose = FALSE)
cases$cfa3 <- compute_case(
  m, "three correlated factors: 2PL, 3PL, GRM, GPCM; two cross-loading items",
  dat, itemtype, "factors", c("F1", "F2", "F3"),
  list(c(0, 90, 90), rep(acos(1 / sqrt(3)) * 180 / pi, 3), c(60, 45, 60), c(45, 90, 135))
)

# --- bifactor: general + 3 specific factors ---------------------------------------------------
items <- paste0("i", 1:12)
itemtype <- rep(c("2PL", "2PL", "graded", "2PL"), 3)
n_cat <- rep(c(2, 2, 3, 2), 3)
dat <- random_data(n_cat, items)
specific <- rep(1:3, each = 4)
general_slopes <- c(1.2, 0.9, 1.4, 1.0, 1.1, 0.8, 1.3, 1.5, 0.7, 1.0, 1.2, 0.9)
specific_slopes <- c(0.6, 0.8, 0.5, 0.9, 0.7, 0.4, 1.0, 0.6, 0.8, 0.5, 0.7, 0.9)
values <- setNames(lapply(1:12, function(j) {
  v <- list(a1 = general_slopes[j])
  v[[paste0("a", specific[j] + 1L)]] <- specific_slopes[j]
  if (itemtype[j] == "graded") c(v, list(d1 = 0.9 - 0.1 * j, d2 = -0.5 - 0.05 * j))
  else c(v, list(d = 0.2 * j - 1.3))
}), items)
sv <- bfactor(dat, specific, itemtype = itemtype, pars = "values")
m <- bfactor(dat, specific, itemtype = itemtype, pars = set_values(sv, values), TOL = NaN,
             quadpts = 7L, verbose = FALSE)
cases$bifactor <- compute_case(
  m, "bifactor: general factor and three specific factors, 2PL + GRM",
  dat, itemtype, "bifactor", c("general", "s1", "s2", "s3"),
  list(c(0, 90, 90, 90), c(60, 60, 60, 60), c(45, 45, 90, 90), c(90, 90, 0, 90))
)

# --- two-tier: 2 correlated primary + 2 specific factors ---------------------------------------
items <- paste0("i", 1:8)
itemtype <- c("2PL", "graded", "2PL", "2PL", "2PL", "graded", "2PL", "2PL")
n_cat <- c(2, 3, 2, 2, 2, 3, 2, 2)
dat <- random_data(n_cat, items)
specific <- c(1, 1, 2, 2, 1, 1, 2, 2)
values <- list(
  i1 = list(a1 = 1.2, a3 = 0.6, d = 0.2), i2 = list(a1 = 1.0, a3 = 0.7, d1 = 0.8, d2 = -0.6),
  i3 = list(a1 = 0.9, a4 = 0.5, d = -0.4), i4 = list(a1 = 1.4, a2 = 0.5, a4 = 0.8, d = 0.6),
  i5 = list(a2 = 1.1, a3 = 0.4, d = -0.1), i6 = list(a2 = 1.3, a3 = 0.9, d1 = 1.1, d2 = 0.0),
  i7 = list(a2 = 0.8, a4 = 0.7, d = 0.9), i8 = list(a2 = 1.5, a4 = 0.6, d = -0.7)
)
model2 <- mirt.model("F1 = 1-4\nF2 = 4-8\nCOV = F1*F2")
sv <- bfactor(dat, specific, model2 = model2, itemtype = itemtype, pars = "values")
m <- bfactor(dat, specific, model2 = model2, itemtype = itemtype,
             pars = set_values(sv, values), TOL = NaN, quadpts = 7L, verbose = FALSE)
cases$two_tier <- compute_case(
  m, "two-tier: two correlated primary factors (one cross-loading) and two specific factors",
  dat, itemtype, "two_tier", c("p1", "p2", "s1", "s2"),
  list(c(0, 90, 90, 90), c(60, 60, 60, 60), c(45, 45, 90, 90), c(90, 90, 45, 135))
)

bundle <- list(
  meta = list(
    case = "v43_information_multidim",
    description = paste("V43: information matrix, directional information and expected",
                        "scores of multidimensional models vs mirt iteminfo(multidim_matrix =",
                        "TRUE) / iteminfo(degrees = ) / testinfo(degrees = ) / expected.item /",
                        "expected.test"),
    r_version = R.version.string,
    mirt_version = as.character(packageVersion("mirt")),
    # Nothing is estimated (TOL = NaN at chosen parameters), so there is no convergence to check.
    converged = TRUE
  ),
  cases = cases
)
path <- file.path("tests", "reference", "v43_information_multidim.json")
write_json(bundle, path, auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null")
cat("wrote", path, "\n")
