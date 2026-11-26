# Contributing

English | [日本語](CONTRIBUTING.ja.md)

## Packages

The repository is a uv workspace that releases three distributions with one version
(DESIGN.md §16.18):

| Distribution | Where | Contents |
|---|---|---|
| `flexpyirt` | repository root (`pyproject.toml`, `src/flexpyirt/`, `tests/`, `validation/`) | Estimation and everything that uses a fitted model. Depends on JAX and on flexpyctt |
| `flexpyctt` | `packages/flexpyctt/` (its own `pyproject.toml`, `README.md`, `src/`, `tests/`, `validation/`) | Item and test analysis that needs no fitted model. NumPy, SciPy and pandas only |
| `flexpyequate` | `packages/flexpyequate/` (its own `pyproject.toml`, `README.md`, `docs/DESIGN.md`, `src/`, `tests/`; reference scripts in `validation/r/equate_*.R`) | Equating of score distributions. NumPy, SciPy and pandas only; independent of the other two |

- **flexpyctt never imports JAX or flexpyirt.** A CI job installs it alone and runs its tests
  without JAX. Its functions take `rng` (a seed or `numpy.random.Generator`), not a JAX key.
- A tool that needs a `FitResult` belongs in flexpyirt; one that needs only data belongs in
  flexpyctt. flexpyirt re-exports flexpyctt's public names.
- **flexpyequate never imports JAX, flexpyirt or flexpyctt**, and neither of them depends on it
  (it is a development dependency of the root only, so that `uv sync` installs it). A CI job
  installs it alone and runs its tests. Equating that needs item parameters belongs in flexpyirt;
  flexpyequate's design is in
  [packages/flexpyequate/docs/DESIGN.md](packages/flexpyequate/docs/DESIGN.md).
- `packages/flexpyctt/README.md` and `packages/flexpyequate/README.md` are the PyPI pages of
  those packages: absolute links, no development information. The documentation site shows the
  flexpyequate README on its "Score equating" page (`docs/equating.md`,
  `scripts/mkdocs_hooks.py`).
- Lint and type checks of flexpyirt and flexpyctt are configured once, in the root
  `pyproject.toml`. flexpyequate has its own `[tool.ruff]` and `[tool.mypy]` (the same rules):
  `ruff check .` from the root applies them to its directory, and mypy runs from
  `packages/flexpyequate/`. Each package has its own pytest configuration and test directory.

## Branches

| Branch | Role | Created from / merged into |
|---|---|---|
| `main` | Released states only. No direct pushes | Accepts release PRs from `develop` only |
| `develop` | Integration branch; collects what goes into the next release | Accepts PRs from `feature/*` and `fix/*` |
| `feature/<slug>` | Features, documentation, refactoring and so on | From `develop`, PR into `develop` |
| `fix/<issue number>-<slug>` | Bug fixes. **Always tied to an issue** | From `develop`, PR into `develop` |

- `<slug>` uses lowercase letters, digits, `.`, `_` and `-` only (e.g. `feature/grm-docs`,
  `fix/12-oakes-se`).
- A fix branch's PR description contains `Closes #<issue number>`, and the issue must be open.
  (`develop` is not the default branch, so the workflow closes the issue **at release time**, not at
  merge time.)
- The `PR rules` check (`scripts/check_pr.py`) enforces these on every PR.

## PRs and reviews

- Every change goes in through a PR. `main` and `develop` are protected by rulesets and cannot be
  merged into until CI (`test (3.12/3.13)`) and `pr-rules` pass. CI also runs
  `flexpyctt without JAX (3.12/3.13)` and `flexpyequate without JAX (3.12/3.13)`.
- The repository owner reviews PRs before merging them (no automatic merging).
- Changes to numerical code must pass the agreement tests against R mirt
  ([docs/dev/VALIDATION.md](docs/dev/VALIDATION.md)).

## Releases

The version comes from **git tags** (`vX.Y.Z`, via hatch-vcs). No `pyproject.toml` holds a version;
releasing means tagging, and flexpyirt, flexpyctt and flexpyequate always get the same version. Untagged development commits get versions like `0.0.2.dev3+g1a2b3c4`.

1. Open a PR from `develop` into `main`. **Its title decides the version bump**:
   - `release(major): ...` / `release(minor): ...` / `release(patch): ...`
   - Merge it with a **merge commit** (the `main` ruleset allows merge commits only).
2. The merge runs `.github/workflows/release.yml`:
   1. `plan` job: computes the next version from the PR title and the latest `vX.Y.Z` tag.
   2. `publish` job: **waits for approval in the `release` environment**. Once the owner approves
      it on the Actions page, it tags `main` with `vX.Y.Z`, builds the three distributions
      (`uv build --package flexpyctt`, `uv build --package flexpyequate`,
      `uv build --package flexpyirt`) and creates a GitHub
      Release (wheels and sdists attached, release notes generated). It then closes the issues that
      fix PRs in this release `Closes`d.
   3. `pypi-flexpyctt` job, then `pypi` job, and `pypi-flexpyequate` job beside them: **publish
      the same wheels and sdists to PyPI** (`pypi` environment, Trusted Publishing; no API
      token), one job per PyPI project. flexpyctt goes first because flexpyirt depends on it;
      flexpyequate depends on neither, so its job waits only for the `publish` job and a failure
      in it does not hold back the other two (rerun the failed job).
   4. The same push to `main` runs `.github/workflows/docs.yml`, which deploys the documentation
      site to GitHub Pages.
3. **The first release** (no `v*` tag yet) is **v0.0.1**, whatever bump the title asks for.
4. Manual runs (`workflow_dispatch`) are dry runs only: they compute the next version and build
   the three distributions.

Merge commits on `main` are not merged back into `develop` (`develop` is protected and the bot
cannot push to it). So development version numbers on `develop` may not refer to the latest release
tag; releases are not affected.

Versions follow [Semantic Versioning](https://semver.org/). The public API may change before 1.0.0.

## Security (who can release)

- **Only repository admins can update `main`** (ruleset `main-release-gate`). The PR and CI
  requirements (ruleset `main`) apply to admins too. Only an admin who can merge develop into main
  can start a release.
- **The `publish` job runs in the `release` environment and needs the designated reviewer's (the
  owner's) approval.** The environment deploys from `main` only, so other branches and forks cannot
  create tags or Releases.
- **PyPI uploads use Trusted Publishing** (OIDC). Each PyPI project (`flexpyirt`, `flexpyctt`, `flexpyequate`)
  trusts only the publisher "repository `takuizum/flexpyirt`, workflow `release.yml`, environment
  `pypi`", so no API token lives in the repository. A new project needs a pending publisher
  registered on PyPI before its first release. Restrict the `pypi` environment to deployments from `main` as well. A version
  number uploaded to PyPI can never be reused, even after deleting it.
- Workflow `GITHUB_TOKEN`s are read-only by default (`permissions: contents: read`); only the
  `publish` job writes. PRs from forks get a read-only token and run only CI and PR rules.
- Give new collaborators write access only. Every additional admin is another person who can
  release.

## Local development

Manage Python environments with `uv` only (no pip).

```bash
uv sync --all-extras --dev
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
uv run pytest packages/flexpyctt
uv run pytest packages/flexpyequate
(cd packages/flexpyequate && uv run mypy)
```

`uv sync` installs the three packages in editable mode. Ruff covers all of them from the root
and mypy flexpyirt and flexpyctt; flexpyequate's mypy run is the last line. `uv run pytest` runs
flexpyirt's tests, `uv run pytest packages/flexpyctt` flexpyctt's and
`uv run pytest packages/flexpyequate` flexpyequate's.

### Tests

Tests run in parallel (pytest-xdist, `-n auto`, one module per worker); pass `-n 0` to run
serially. They come in two tiers:

| Tier | What | Where it runs |
|---|---|---|
| fast | `-m "not slow"`: unit tests, docs, the M1 mirt smoke test, split into 3 shards per Python (a few minutes on CI) | every PR, every push to `develop`/`main` |
| full | everything, including all mirt reference comparisons, split into 4 shards per Python | nightly on `develop`, release PRs (`develop` → `main`), manual runs (*Run workflow*) |

A mirt reference comparison is complete only when the full tier passes (nightly or on the
release PR); check the nightly run before ticking a ROADMAP item.

- Mark a test `@pytest.mark.slow` when it takes more than ~5 s, counting its share of a fit it
  caches with `functools.cache` (the first test that needs the fit pays for it). The fast-tier
  summary lists unmarked tests over 5 s ("tests over 5 s not marked slow").
- `tests/test_mirt_*.py` modules are slow as a whole (`pytestmark = pytest.mark.slow`), except
  `test_mirt_m1.py`.
- A module runs on one worker, so a fit that takes minutes goes in a module of its own (see
  `tests/_mirt_multigroup.py` and `tests/test_mirt_m4_multigroup_*.py`).
- CI keeps a persistent JAX compilation cache between runs (`JAX_COMPILATION_CACHE_DIR`).
- Full-tier shards are balanced by the seconds per module in `tests/durations.json`, fast-tier
  shards by those in `tests/durations-fast.json` (`--durations-file`). After adding or splitting
  heavy modules, refresh them: `uv run pytest --write-durations tests/durations.json` and
  `uv run pytest -m "not slow" -n 4 --write-durations tests/durations-fast.json` (or merge the
  `durations-*` artifacts of a CI run). A module missing from a file counts as 1 s per test.
- On CI the required checks `test (3.12)` and `test (3.13)` pass when the `checks` job (ruff,
  mypy, flexpyctt's and flexpyequate's suites) and every `fast` shard passed.

Run `uv run pytest -m "not slow"` locally for a quick check, and `uv run pytest --shard 2/4` to
run one full-tier shard as CI does
(`uv run pytest -m "not slow" --shard 2/3 --durations-file tests/durations-fast.json` for a fast-tier
one).

To regenerate the mirt reference values (needs R 4.4+; mirt is installed from GitHub,
`philchalmers/mirt`, into the user library or `FLEXPYIRT_R_LIB`; see
[VALIDATION.md](docs/dev/VALIDATION.md), "R and mirt versions"):

```bash
Rscript validation/r/install_mirt.R
validation/r/run_references.sh
```

`validation/r/run_references.sh m1 m2` reruns single scripts; a new script that uses mirt sources
`validation/r/setup.R` and is added to the stage lists of `run_references.sh`.

flexpyctt's reference scripts are in `packages/flexpyctt/validation/r/` and write to
`packages/flexpyctt/tests/reference/`; run them from the repository root too. flexpyequate's are
`validation/r/equate_*.R` and write to `packages/flexpyequate/tests/reference/`.

To rerun the wall-time benchmarks against R `mirt` (see [benchmarks/README.md](benchmarks/README.md)
for details and `--quick` mode):

```bash
uv run python benchmarks/data.py
uv run python benchmarks/run_flexpyirt.py
Rscript benchmarks/run_mirt.R
uv run python benchmarks/report.py
```

## Naming

Names follow PEP 8 and read as Python, not as R (DESIGN.md §15.46). `ruff` checks the `N` rules.

- `snake_case` for functions, methods, arguments and variables; `CapWords` for classes;
  `UPPER_SNAKE_CASE` for module constants.
- Do not copy R/mirt argument names. Use the Python spelling and mention mirt's name in the
  docstring: `n_categories` (not `n_cat`), `min_cell` (mirt `mincell`), `mean_log`/`sd_log` (R
  `meanlog`/`sdlog`), `bounds=(lower, upper)` (mirt `theta_lim`).
- Spell public names out (`is_empirical_histogram`, not `is_eh`). Well-known abbreviations are
  fine: model names (`TwoPL`, `GRM`, `GPCM`), the `n_` prefix for counts, and `se`, `vcov`,
  `loglik`, `theta`, `eta`. Avoid abbreviations that collide with common aliases, such as `pd`
  (pandas) or `df` (degrees of freedom vs DataFrame).
- Use one name for one concept across the whole API (`n_points`, `n_factors`, `n_categories`).
  Members of a family share a suffix (`NormalPrior`, `LogNormalPrior`, `BetaPrior`).
- `Literal` string options are lowercase `snake_case` (`"slope_intercept"`, `"stocking_lord"`).
  Established acronyms stay upper case (`"EAP"`, `"MAP"`, `"BH"`).
- Mathematical notation is allowed where it mirrors the formulas in DESIGN.md: upper-case symbols
  inside jitted kernels (`G`, `Q`, `D`, `K_f`; ruff `N803`/`N806` are ignored for this) and
  standard constants such as `LinkResult.A`/`B`.
- Name local closures by their role (`objective`, `backtrack_body`, `interp_weights`), not `obj`
  or `tmp`. Line-by-line ports of mirt code (e.g. `_itemfit.collapse_cells`) may keep mirt's local
  names so they can be compared; say so in a comment.
- Renaming anything that is stored in `FitResult` JSON needs a new `schema_version`, and older
  files must still load (rename the old keys on read).

## Documentation

Every document exists in English (`*.md`) and Japanese (`*.ja.md`) with the same content. Update
both when you change one. The PR and issue templates under `.github/` are single bilingual files
(English and Japanese side by side), because GitHub uses one default template.

The files in `docs/` also make up the bilingual documentation site
(<https://takuizum.github.io/flexpyirt/>, Japanese under `/ja/`; MkDocs, `mkdocs.yml`). The site is
deployed from `main`, so it shows the released state. To check it locally:

```bash
uv run --group docs mkdocs serve           # preview
uv run --group docs mkdocs build --strict  # the CI check (fails on broken links and anchors)
```

- Write links between documents as relative links (they work on GitHub and on the site). On the
  site, links that leave `docs/` point to the file on GitHub (`scripts/mkdocs_hooks.py`).
- `README.md` / `README.ja.md` are also shown on the PyPI project page, so their links are absolute
  (site URLs for documents, GitHub URLs for everything else). The README is for package users
  only: no development setup, test or CI details, branch flow, milestones, or links to
  CONTRIBUTING and `docs/dev/`.
- When you add a document to `docs/`, add it to `nav` in `mkdocs.yml` (and to the Japanese
  `nav_translations`).
- The site is for package users: how to use flexpyirt (`guide`, `reference`), the theory behind it
  (`theory`) and how its results are checked (`accuracy`). Developer documents live in `docs/dev/`
  ([design](docs/dev/DESIGN.md), [API sketch](docs/dev/API.md),
  [validation plan](docs/dev/VALIDATION.md), [roadmap](docs/dev/ROADMAP.md)); they are excluded from
  the site, so site pages must not link to them with relative links (use a GitHub URL if needed).
  Keep milestones, design-section numbers and validation case IDs out of user pages.
