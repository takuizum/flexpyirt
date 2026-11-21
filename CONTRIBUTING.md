# Contributing

English | [日本語](CONTRIBUTING.ja.md)

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
  merged into until CI (`test (3.11/3.12/3.13)`) and `pr-rules` pass.
- The repository owner reviews PRs before merging them (no automatic merging).
- Changes to numerical code must pass the agreement tests against R mirt
  ([docs/VALIDATION.md](docs/VALIDATION.md)).

## Releases

The version comes from **git tags** (`vX.Y.Z`, via hatch-vcs). `pyproject.toml` holds no version;
releasing means tagging. Untagged development commits get versions like `0.0.2.dev3+g1a2b3c4`.

1. Open a PR from `develop` into `main`. **Its title decides the version bump**:
   - `release(major): ...` / `release(minor): ...` / `release(patch): ...`
   - Merge it with a **merge commit** (the `main` ruleset allows merge commits only).
2. The merge runs `.github/workflows/release.yml`:
   1. `plan` job: computes the next version from the PR title and the latest `vX.Y.Z` tag.
   2. `publish` job: **waits for approval in the `release` environment**. Once the owner approves
      it on the Actions page, it tags `main` with `vX.Y.Z`, runs `uv build` and creates a GitHub
      Release (wheel and sdist attached, release notes generated). It then closes the issues that
      fix PRs in this release `Closes`d.
3. **The first release** (no `v*` tag yet) is **v0.0.1**, whatever bump the title asks for.
4. Manual runs (`workflow_dispatch`) are dry runs only: they compute the next version and build.

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
- Workflow `GITHUB_TOKEN`s are read-only by default (`permissions: contents: read`); only the
  `publish` job writes. PRs from forks get a read-only token and run only CI and PR rules.
- Give new collaborators write access only. Every additional admin is another person who can
  release.

## Local development

Manage Python environments with `uv` only (no pip).

```bash
uv sync --all-extras --dev
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
```

To regenerate the mirt reference values (needs R 4.4+ and mirt):

```bash
Rscript validation/r/m1_reference.R
Rscript validation/r/m2_reference.R
```

To rerun the wall-time benchmarks against R `mirt` (see [benchmarks/README.md](benchmarks/README.md)
for details and `--quick` mode):

```bash
uv run python benchmarks/data.py
uv run python benchmarks/run_flexpyirt.py
Rscript benchmarks/run_mirt.R
uv run python benchmarks/report.py
```

## Documentation

Every document exists in English (`*.md`) and Japanese (`*.ja.md`) with the same content. Update
both when you change one. The PR and issue templates under `.github/` are single bilingual files
(English and Japanese side by side), because GitHub uses one default template.
