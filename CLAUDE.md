# CLAUDE.md — flexpyirt 実装の引き継ぎ

このリポジトリは、JAX ベースの Bock–Aitkin EM (BA-EM) による IRT 推定ライブラリ **flexpyirt**の
設計段階のリポジトリです。まだ実装コードはありません。以下のドキュメントを読んでから作業を始めてください。

| ファイル | 内容 |
|---|---|
| `docs/DESIGN.md` | アーキテクチャ・数式・アルゴリズム（最重要） |
| `docs/API.md` | 公開 API のスケッチ（ユーザーから見た使い方） |
| `docs/VALIDATION.md` | R mirt との数値照合計画と許容誤差 |
| `docs/ROADMAP.md` | マイルストーンと受け入れ基準 |

## 最優先の原則

1. **正しさ > 速さ > 機能の多さ。** 各機能は R mirt（`mirt` 1.4x 系）と数値照合するテストが通るまで「完成」としない。
   R mirt の焼き直しではなく、数式から実装し、R mirt は**検証の参照値**としてのみ使う。
2. **機能を水増ししない。** ROADMAP のマイルストーン順に進め、未検証の機能を README に載せない。
   TODO にチェックを付けるのは照合テストが CI で通ってから。
3. **関数型コア + オブジェクト指向シェル。** jit される内部は純粋関数と pytree のみ。
   名前・DataFrame・検証・エラーメッセージは外側の Python 層で扱う（DESIGN.md §2）。
4. **float64 必須。** `jax.config.update("jax_enable_x64", True)` をパッケージ import 時ではなく
   `fit()` 入口でチェックし、無効なら明確なエラーを出す（ユーザーのグローバル設定を勝手に変えない）。
5. **小さくコミット。** 1 マイルストーン内でも「数式 → カーネル → 単体テスト → mirt 照合」の単位でコミットする。

## 開発環境

- Python >= 3.11, パッケージ管理は `uv`。
- 依存: `jax`, `numpy`, `scipy`（初期値・検証用）, `pandas`（任意依存で polars も）。
- 開発依存: `pytest`, `hypothesis`, `ruff`, `mypy`。
- R 参照値の生成: `validation/r/*.R`（R 4.4+, mirt）。生成した参照値は `tests/reference/*.json` に
  コミットし、CI では R を走らせない（R がない環境でもテストが通るように）。

## コーディング規約

- 公開関数・クラスは型ヒント必須、docstring は NumPy スタイル。
- 公開 API は keyword-only 引数（`*`）を基本にする。選択肢は `Literal[...]`。
- 数式を実装した関数には、docstring に対応する式（LaTeX）と出典（著者・年）を書く。
- jit 関数内で Python の分岐が shape に依存しないこと。可変長はパディング + マスク（DESIGN.md §5）。
- 例外は `flexpyirt.errors` に定義した具体的なクラスで投げる（`SpecificationError`, `IdentificationError` など）。

## 開発フロー（CONTRIBUTING.md）

- `feature/<slug>` と `fix/<issue番号>-<slug>` を `develop` から切り、`develop` へ PR。fix は issue 必須（`Closes #N`）。
- リリースは `develop` → `main` の PR。タイトル `release(major|minor|patch): ...` で上げ方を決め、マージ後にオーナーが `release` 環境を承認するとタグ `vX.Y.Z` と GitHub Release ができる。バージョンは git タグから（hatch-vcs、`pyproject.toml` に書かない）。
- **PR はオーナーがレビューしてマージする。Claude は PR を作るまでで、マージしない。**
- Python 環境は `uv` のみ（pip は使わない）。

## 作業開始時にやること

1. `docs/ROADMAP.md` の M0（足場）から着手する。
2. 迷ったら DESIGN.md の「未決事項」節を確認し、決めた内容は DESIGN.md に追記してからコードを書く。
