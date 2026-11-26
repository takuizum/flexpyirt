# Install R mirt from GitHub (philchalmers/mirt) for the reference scripts (VALIDATION.md, "R and
# mirt versions"). Run from the repository root:
#
#   Rscript validation/r/install_mirt.R          # default-branch HEAD
#   Rscript validation/r/install_mirt.R <ref>    # a commit SHA, tag or branch
#
# Installs into FLEXPYIRT_R_LIB if set, else the user library (R_LIBS_USER), the same library
# validation/r/setup.R puts first on .libPaths(). Missing CRAN dependencies of mirt are installed
# too; packages that are already installed are not upgraded.
args <- commandArgs(trailingOnly = TRUE)
ref <- if (length(args) >= 1L) args[[1L]] else "HEAD"
repos <- getOption("repos")  # e.g. a binary mirror set by the CI's setup-r
if (nzchar(Sys.getenv("FLEXPYIRT_CRAN")) || is.null(repos) || any(repos == "@CRAN@")) {
  repos <- c(CRAN = Sys.getenv("FLEXPYIRT_CRAN", unset = "https://cloud.r-project.org"))
}

lib <- Sys.getenv("FLEXPYIRT_R_LIB", unset = "")
if (!nzchar(lib)) lib <- Sys.getenv("R_LIBS_USER", unset = "")
if (!nzchar(lib)) stop("Neither FLEXPYIRT_R_LIB nor R_LIBS_USER is set.", call. = FALSE)
lib <- path.expand(lib)
dir.create(lib, showWarnings = FALSE, recursive = TRUE)
.libPaths(c(lib, .libPaths()))

if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes", lib = lib, repos = repos)
}
remotes::install_github("philchalmers/mirt", ref = ref, lib = lib, upgrade = "never",
                        dependencies = NA, repos = repos)

desc <- utils::packageDescription("mirt", lib.loc = lib)
if (is.null(desc$RemoteSha)) stop("mirt in ", lib, " has no RemoteSha.", call. = FALSE)
cat(sprintf("mirt %s (philchalmers/mirt@%s) installed in %s\n", desc$Version, desc$RemoteSha, lib))
