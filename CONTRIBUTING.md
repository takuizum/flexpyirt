# 開発フロー

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
- 数値計算に関わる変更は、R mirt との照合テスト（`docs/VALIDATION.md`）が通ることが前提。

## リリース

1. `develop` → `main` の PR を作る。**タイトルでバージョンの上げ方を決める**:
   - `release(major): ...` / `release(minor): ...` / `release(patch): ...`
   - PR は**マージコミット**でマージする（タイトルが `main` のマージコミットに残る）。
2. マージされると `.github/workflows/release.yml` が自動で:
   1. `pyproject.toml`（と `uv.lock`）のバージョンを上げて `main` にコミット（`chore(release): vX.Y.Z`）
   2. タグ `vX.Y.Z` を作成
   3. `uv build` して GitHub Release を作成（wheel / sdist を添付、リリースノートは自動生成）
   4. `main` を `develop` にマージし戻す
   5. このリリースに含まれる fix PR が `Closes` していた issue を閉じる
3. **最初のリリース**（`v*` タグがまだない）は、タイトルの上げ方に関係なく `pyproject.toml` の現在の
   バージョンをそのまま出す（v0.0.1）。
4. 手動実行（`workflow_dispatch`）は dry run 専用で、何も push しない。

バージョンは [Semantic Versioning](https://semver.org/lang/ja/) に従う。1.0.0 までは公開 API が変わりうる。

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
