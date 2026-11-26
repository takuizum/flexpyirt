# Shared setup of every validation/r/*_reference.R script that uses mirt (VALIDATION.md, "R and
# mirt versions"). Sourced from the repository root, before `library(mirt)`:
#
#   source(file.path("validation", "r", "setup.R"))
#
# References are generated with R mirt installed from GitHub (philchalmers/mirt, default branch),
# not with a CRAN release. Install or update it with
#
#   Rscript validation/r/install_mirt.R          # default-branch HEAD
#   Rscript validation/r/install_mirt.R <sha>    # a given commit, tag or branch
#
# Library: the directory in the environment variable FLEXPYIRT_R_LIB if set, else the user library
# (R_LIBS_USER). It is put first on .libPaths(), so a GitHub mirt there wins over a CRAN mirt in a
# site library.
#
# A script stops if the mirt it would load was not installed from GitHub (no RemoteSha in its
# DESCRIPTION). FLEXPYIRT_ALLOW_CRAN_MIRT=1 lets it run anyway for local experiments; the meta then
# records mirt_sha = null, which tests/test_reference_meta.py rejects, so such JSON is never
# committed.
#
# Every script records mirt_version = mirt_version_string() and mirt_sha = mirt_commit_sha() in
# its `meta`.

flexpyirt_r_lib <- function() {
  lib <- Sys.getenv("FLEXPYIRT_R_LIB", unset = "")
  if (!nzchar(lib)) lib <- Sys.getenv("R_LIBS_USER", unset = "")
  path.expand(lib)
}

local({
  lib <- flexpyirt_r_lib()
  if (nzchar(lib) && dir.exists(lib)) .libPaths(c(lib, .libPaths()))
})

mirt_version_string <- function() as.character(utils::packageVersion("mirt"))

mirt_commit_sha <- function() {
  sha <- utils::packageDescription("mirt")$RemoteSha
  if (is.null(sha) || !nzchar(sha)) NA_character_ else sha
}

local({
  if (!requireNamespace("mirt", quietly = TRUE)) {
    stop("mirt is not installed; run `Rscript validation/r/install_mirt.R` first.", call. = FALSE)
  }
  if (is.na(mirt_commit_sha()) && !identical(Sys.getenv("FLEXPYIRT_ALLOW_CRAN_MIRT"), "1")) {
    stop(sprintf(paste0(
      "mirt %s in %s was not installed from GitHub. References are generated with mirt from ",
      "philchalmers/mirt: run `Rscript validation/r/install_mirt.R` (or set ",
      "FLEXPYIRT_ALLOW_CRAN_MIRT=1 for a local experiment that is not committed)."),
      mirt_version_string(), dirname(find.package("mirt"))), call. = FALSE)
  }
  message(sprintf("mirt %s (%s) from %s; %s", mirt_version_string(), mirt_commit_sha(),
                  dirname(find.package("mirt")), R.version.string))
})
