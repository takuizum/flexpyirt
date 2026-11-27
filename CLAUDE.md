# CLAUDE.md — flexpyirt 実装の引き継ぎ

このリポジトリは、JAX ベースの Bock–Aitkin EM (BA-EM) による IRT 推定ライブラリ **flexpyirt**の
リポジトリです（M2 まで実装済み）。推定済みのモデルを使わない道具（**flexpyctt**）と得点分布の等化（**flexpyequate**）は、JAX に依存しない別のパッケージとして、それぞれ自分のリポジトリから公開します（下の「関連パッケージ」）。以下のドキュメントを読んでから作業を始めてください（各文書は英語版 `*.md` と日本語版 `*.ja.md` があり、内容は同じ。更新するときは両方を直す）。

| ファイル | 内容 |
|---|---|
| `docs/dev/DESIGN.md` | アーキテクチャ・数式・アルゴリズム（最重要） |
| `docs/dev/API.md` | 公開 API のスケッチ（ユーザーから見た使い方） |
| `docs/dev/VALIDATION.md` | R mirt との数値照合計画と許容誤差 |
| `docs/dev/ROADMAP.md` | マイルストーンと受け入れ基準 |

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

## 関連パッケージ（DESIGN.md §16.18、§16.24）

パッケージごとにリポジトリとバージョンが別（2026-11-27 に分割）。このリポジトリにあるのは flexpyirt だけ。

| パッケージ | リポジトリ | 内容 |
|---|---|---|
| `flexpyirt` | takuizum/flexpyirt（ここ） | 推定と、`FitResult` を使うものすべて。JAX と flexpyctt に依存 |
| `flexpyctt` | takuizum/flexpyctt | モデルを使わない項目・テスト分析（ポリコリック相関、次元性の診断、古典的項目分析、観測得点の DIF）。NumPy・SciPy・pandas のみ |
| `flexpyequate` | takuizum/flexpyequate | 得点分布の等化（線形、等パーセンタイル、カーネル等化）。NumPy・SciPy・pandas のみ。他の 2 つとは独立 |

- flexpyirt は `flexpyctt>=0.X,<0.(X+1)` に依存する（`pyproject.toml`）。1.0 より前は flexpyctt のマイナーバージョンが互換性の
  単位で、パッチリリースは公開 API を変えない。flexpyctt のマイナーが上がったら、flexpyirt 側で範囲を上げて CI を通す。
- `FitResult` が要る道具は flexpyirt、データだけで済む道具は flexpyctt に置く。flexpyirt は flexpyctt の公開名を再公開する
  （`polychoric` だけは別で、flexpyirt のものは JAX 版）。再公開する名前は `src/flexpyirt/__init__.py` に明示してある。
- flexpyirt は flexpyctt の非公開ヘルパーを 4 つ import している（`_coerce_ordinal`、`_observed_correlation`、`_eigenvalues`、
  `_descending_eigenvalues`）。flexpyctt 側にシグネチャを守るテストがある。
- flexpyctt を変える必要があるときは、flexpyctt のリポジトリで変更してリリースしてから、ここで依存の範囲を上げる。
- `tests/reference/flexpyctt/v24_polychoric.json` と `v27_detect.json` は flexpyctt の参照値のコピー（生成スクリプトは flexpyctt の
  `validation/r/`）。flexpyctt で再生成したら、ここにもコピーする。
- 項目パラメータを使う等化（IRT 真値等化・観測得点等化、`link()`）は flexpyirt に置く。flexpyirt は flexpyequate に依存しない
  （開発用の依存にだけ入れている。`tests/test_equating.py` が独立した実装として照合に使う）。

## 開発環境

- Python >= 3.12, パッケージ管理は `uv`。
- 依存: `jax`, `numpy`, `scipy`（初期値・検証用）, `pandas`（任意依存で polars も）。
- 開発依存: `pytest`, `hypothesis`, `ruff`, `mypy`。
- R 参照値の生成: `validation/r/*.R`（R 4.4+, mirt）。生成した参照値は `tests/reference/*.json` に
  コミットし、CI では R を走らせない（R がない環境でもテストが通るように）。
- mirt は CRAN 版ではなく **GitHub の最新版**（`philchalmers/mirt` の既定ブランチ）で照合する
  （VALIDATION.md「R と mirt の版」）。`Rscript validation/r/install_mirt.R` で入れ、
  `validation/r/run_references.sh` で全参照値を作り直す。mirt を使うスクリプトは必ず
  `validation/r/setup.R` を `library(mirt)` の前に `source` し、`meta` に `mirt_version` と
  `mirt_sha` を書く（`tests/test_reference_meta.py` が検査）。現在の参照値: R 4.6.2、mirt 1.48.5
  （`2b342bb`）。
- 週次の `mirt-head.yml` ワークフローが mirt HEAD で参照値を作り直して比較し、ずれたら issue を立てる。
  参照値は自動では更新しない。ずれを読み、mirt の退行か修正かを判断してから手元で作り直して PR にする。
  ずれと許容誤差の変更（理由を書いたものだけ）は VALIDATION.md に記録する。

## コーディング規約

- 公開関数・クラスは型ヒント必須、docstring は NumPy スタイル。
- 公開 API は keyword-only 引数（`*`）を基本にする。選択肢は `Literal[...]`。
- 数式を実装した関数には、docstring に対応する式（LaTeX）と出典（著者・年）を書く。
- jit 関数内で Python の分岐が shape に依存しないこと。可変長はパディング + マスク（DESIGN.md §5）。
- 例外は `flexpyirt.errors` に定義した具体的なクラスで投げる（`SpecificationError`, `IdentificationError` など）。
- 命名は PEP 8 に従い、R/mirt の引数名を持ち込まない（詳細は CONTRIBUTING.md「命名」、DESIGN.md §15.46）。
  - 公開名は略さず、同じ概念は API 全体で同じ名前（`n_categories`, `n_points`, `n_factors`）。
    mirt の対応名は docstring に書く（`min_cell` ← mirt `mincell`）。
  - `Literal` の選択肢は小文字 `snake_case`（`"slope_intercept"`）。定着した頭字語（`"EAP"`）は大文字のまま。
  - 大文字の数式記号は DESIGN.md の式に対応する jit カーネル内だけ。ローカルのクロージャは役割で名付ける（`obj`・`tmp` 禁止。
    mirt を行ごとに移植した箇所は例外）。
  - JSON に保存される名前を変えたら `schema_version` を上げ、旧ファイルも読めるようにする。

## 開発フロー（CONTRIBUTING.md）

- `feature/<slug>` と `fix/<issue番号>-<slug>` を `develop` から切り、`develop` へ PR。fix は issue 必須（`Closes #N`）。
- リリースは `develop` → `main` の PR。タイトル `release(major|minor|patch): ...` で上げ方を決め、マージ後にオーナーが `release` 環境を承認するとタグ `vX.Y.Z` と GitHub Release ができる。バージョンは git タグから（hatch-vcs、`pyproject.toml` に書かない）。
  同じワークフローの `pypi` ジョブが PyPI に公開し（Trusted Publishing）、`docs.yml` がドキュメントサイト（`docs/` から MkDocs、英日）を GitHub Pages に公開する。
  リリースの前に、依存の範囲に入る flexpyctt が PyPI にあること。
- `README*.md` は PyPI にも表示されるのでリンクは絶対 URL。**README には開発関連の情報（開発手順・テスト/CI・ブランチ運用・マイルストーン・CONTRIBUTING や docs/dev へのリンク）を置かない。**`docs/` 内は相対リンク。
  サイト（`docs/*.md`）は利用者向け（使い方・理論・精度）。開発者向け文書は `docs/dev/`（サイトから除外）。
  利用者向けページにマイルストーン・DESIGN の節番号・照合ケース ID・開発手順を書かない。`uv run --group docs mkdocs build --strict` が CI と同じ検査。
- **PR はオーナーがレビューしてマージする。Claude は PR を作るまでで、マージしない。**
- Python 環境は `uv` のみ（pip は使わない）。

## 作業開始時にやること

1. `docs/dev/ROADMAP.md` の M0（足場）から着手する。
2. 迷ったら DESIGN.md の「未決事項」節を確認し、決めた内容は DESIGN.md に追記してからコードを書く。
