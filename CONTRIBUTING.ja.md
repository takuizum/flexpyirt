# 開発フロー

[English](CONTRIBUTING.md) | 日本語

## ブランチ

| ブランチ | 役割 | 作り方 / マージ先 |
|---|---|---|
| `main` | リリース済みの状態だけを置く。直接 push しない | `develop` からのリリース PR だけを受け付ける |
| `develop` | 統合ブランチ。次のリリースに入る変更が集まる | `feature/*` と `fix/*` の PR を受け付ける |
| `feature/<slug>` | 機能追加・ドキュメント・リファクタリングなど | `develop` から切り、`develop` へ PR |
| `fix/<issue番号>-<slug>` | バグ修正。**必ず issue に対応させる** | `develop` から切り、`develop` へ PR |

- `<slug>` は小文字・数字・`.`・`_`・`-` だけ（例: `feature/grm-docs`, `fix/12-oakes-se`）。
- fix ブランチの PR 説明には `Closes #<issue番号>` を書く。issue は open でなければならない。
  （`develop` は既定ブランチではないので、issue はマージ時ではなく**リリース時**にワークフローが閉じる。）
- これらは PR ごとに `PR rules` チェック（`scripts/check_pr.py`）が検査する。

## PR とレビュー

- すべての変更は PR で入れる。`main` と `develop` はルールセットで保護され、CI（`test (3.11/3.12/3.13)`）と
  `pr-rules` が通らないとマージできない。
- PR はリポジトリのオーナーがレビューしてからマージする（自動マージしない）。
- 数値計算に関わる変更は、R mirt との照合テスト（[docs/VALIDATION.ja.md](docs/VALIDATION.ja.md)）が通ることが前提。

## リリース

バージョンは **git タグ**（`vX.Y.Z`）から決まる（hatch-vcs）。`pyproject.toml` にはバージョンを書かず、
リリースはタグを打つことで行う。タグのない開発中のコミットは `0.0.2.dev3+g1a2b3c4` のような開発版になる。

1. `develop` → `main` の PR を作る。**タイトルでバージョンの上げ方を決める**:
   - `release(major): ...` / `release(minor): ...` / `release(patch): ...`
   - PR は**マージコミット**でマージする（`main` のルールセットはマージコミットだけを許可）。
2. マージされると `.github/workflows/release.yml` が動く:
   1. `plan` ジョブ: PR タイトルと最新の `vX.Y.Z` タグから次のバージョンを決める。
   2. `publish` ジョブ: **`release` 環境の承認待ちになる**。オーナーが Actions 画面で承認すると、
      `main` にタグ `vX.Y.Z` を打って `uv build` し、GitHub Release を作成する（wheel / sdist 添付、
      リリースノート自動生成）。このリリースに含まれる fix PR が `Closes` していた issue を閉じる。
3. **最初のリリース**（`v*` タグがまだない）は、タイトルの上げ方に関係なく **v0.0.1**。
4. 手動実行（`workflow_dispatch`）は dry run 専用で、次のバージョンの計算とビルドだけを行う。

`main` へのマージコミットは `develop` に戻さない（`develop` は保護されていて、ボットは push できない）。
このため `develop` 上の開発版の番号は直前のリリースタグを参照しないことがあるが、リリースには影響しない。

バージョンは [Semantic Versioning](https://semver.org/lang/ja/) に従う。1.0.0 までは公開 API が変わりうる。

## セキュリティ（誰がリリースできるか）

- **`main` を更新できるのはリポジトリ管理者だけ**（ルールセット `main-release-gate`）。PR と CI の要件
  （ルールセット `main`）は管理者にも適用される。リリースを起こせるのは develop → main を
  マージできる管理者だけ。
- **`publish` ジョブは `release` 環境で動き、指定レビュアー（オーナー）の承認が必要**。環境は `main` からしか
  デプロイできないので、他のブランチやフォークからはタグも Release も作れない。
- ワークフローの `GITHUB_TOKEN` は既定で読み取り専用（`permissions: contents: read`）。書き込みは
  `publish` ジョブだけ。フォークからの PR は読み取り専用トークンで CI と PR rules だけが走る。
- 協力者を追加するときは write 権限にとどめる。管理者を増やすとリリースできる人が増える。

## ローカル開発

Python 環境は `uv` だけで管理する（pip は使わない）。

```bash
uv sync --all-extras --dev
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
```

mirt の参照値を作り直すとき（R 4.4+ と mirt が必要）:

```bash
Rscript validation/r/m1_reference.R
Rscript validation/r/m2_reference.R
```

R `mirt` との wall time ベンチマークを再実行するとき（詳細と `--quick` モードは
[benchmarks/README.ja.md](benchmarks/README.ja.md) を参照）:

```bash
uv run python benchmarks/data.py
uv run python benchmarks/run_flexpyirt.py
Rscript benchmarks/run_mirt.R
uv run python benchmarks/report.py
```

## ドキュメント

すべての文書は英語版（`*.md`）と日本語版（`*.ja.md`）があり、内容は同じ。片方を変えたら両方を直す。
`.github/` の PR・issue テンプレートは、GitHub が既定のテンプレートを 1 つしか使わないため、英日を併記した
1 つのファイルにしている。
