# 開発フロー

[English](CONTRIBUTING.md) | 日本語

## パッケージ

このリポジトリは uv ワークスペースで、2 つの配布物を同じバージョンで公開する（DESIGN.md §16.18）。

| 配布物 | 場所 | 内容 |
|---|---|---|
| `flexpyirt` | リポジトリのルート（`pyproject.toml`、`src/flexpyirt/`、`tests/`、`validation/`） | 推定と、推定済みのモデルを使うものすべて。JAX と flexpyctt に依存する |
| `flexpyctt` | `packages/flexpyctt/`（自分の `pyproject.toml`、`README.md`、`src/`、`tests/`、`validation/` を持つ） | 推定済みのモデルを使わない項目・テスト分析。依存は NumPy、SciPy、pandas だけ |

- **flexpyctt は JAX も flexpyirt も import しない。** CI のジョブが flexpyctt だけをインストールし、JAX なしで
  テストを実行する。関数は JAX のキーではなく `rng`（シードまたは `numpy.random.Generator`）を取る。
- `FitResult` が要る道具は flexpyirt に、データだけで済む道具は flexpyctt に置く。flexpyirt は flexpyctt の公開名を
  再公開する。
- `packages/flexpyctt/README.md` は flexpyctt の PyPI のページになる。リンクは絶対 URL にし、開発関連の情報は置かない。
- lint と型チェックの設定はルートの `pyproject.toml` に 1 つだけあり、両方のパッケージを対象にする。pytest の設定と
  テストのディレクトリはパッケージごとに持つ。

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

- すべての変更は PR で入れる。`main` と `develop` はルールセットで保護され、CI（`test (3.12/3.13)`）と
  `pr-rules` が通らないとマージできない。CI では `flexpyctt without JAX (3.12/3.13)` も走る。
- PR はリポジトリのオーナーがレビューしてからマージする（自動マージしない）。
- 数値計算に関わる変更は、R mirt との照合テスト（[docs/dev/VALIDATION.ja.md](docs/dev/VALIDATION.ja.md)）が通ることが前提。

## リリース

バージョンは **git タグ**（`vX.Y.Z`）から決まる（hatch-vcs）。どの `pyproject.toml` にもバージョンを書かず、
リリースはタグを打つことで行う。flexpyirt と flexpyctt はいつも同じバージョンになる。タグのない開発中のコミットは `0.0.2.dev3+g1a2b3c4` のような開発版になる。

1. `develop` → `main` の PR を作る。**タイトルでバージョンの上げ方を決める**:
   - `release(major): ...` / `release(minor): ...` / `release(patch): ...`
   - PR は**マージコミット**でマージする（`main` のルールセットはマージコミットだけを許可）。
2. マージされると `.github/workflows/release.yml` が動く:
   1. `plan` ジョブ: PR タイトルと最新の `vX.Y.Z` タグから次のバージョンを決める。
   2. `publish` ジョブ: **`release` 環境の承認待ちになる**。オーナーが Actions 画面で承認すると、
      `main` にタグ `vX.Y.Z` を打って 2 つの配布物をビルドし（`uv build --package flexpyctt`、
      `uv build --package flexpyirt`）、GitHub Release を作成する（wheel / sdist 添付、リリースノート自動生成）。このリリースに含まれる fix PR が `Closes` していた issue を閉じる。
   3. `pypi-flexpyctt` ジョブ、続いて `pypi` ジョブ: 同じ wheel / sdist を **PyPI に公開する**（`pypi` 環境、
      Trusted Publishing。API トークンは使わない）。PyPI のプロジェクトごとに 1 ジョブ。flexpyirt が flexpyctt に
      依存するので、flexpyctt を先に公開する。
   4. 同じ `main` への push で `.github/workflows/docs.yml` がドキュメントサイトを GitHub Pages に公開する。
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
- **PyPI への公開は Trusted Publishing**（OIDC）。PyPI のプロジェクト（`flexpyirt`、`flexpyctt`）ごとに
  「リポジトリ `takuizum/flexpyirt`、ワークフロー `release.yml`、環境 `pypi`」だけを信頼する publisher を登録するので、
  API トークンはリポジトリに置かない。新しいプロジェクトは、最初のリリースの前に PyPI で pending publisher を登録する。`pypi` 環境もデプロイ元を `main` に限定しておく。PyPI に一度上げた番号は
  削除しても再利用できない。
- ワークフローの `GITHUB_TOKEN` は既定で読み取り専用（`permissions: contents: read`）。書き込みは
  `publish` ジョブだけ。フォークからの PR は読み取り専用トークンで CI と PR rules だけが走る。
- 協力者を追加するときは write 権限にとどめる。管理者を増やすとリリースできる人が増える。

## ローカル開発

Python 環境は `uv` だけで管理する（pip は使わない）。

```bash
uv sync --all-extras --dev
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
uv run pytest packages/flexpyctt
```

`uv sync` は両方のパッケージを編集可能な形でインストールする。ruff と mypy はルートから両方を検査する。
`uv run pytest` は flexpyirt のテスト、`uv run pytest packages/flexpyctt` は flexpyctt のテストを実行する。

### テスト

テストは pytest-xdist で並列に走る（`-n auto`、1 モジュールを 1 ワーカーで実行）。直列にしたいときは `-n 0` を付ける。
テストは 2 層に分かれる。

| 層 | 内容 | 実行するタイミング |
|---|---|---|
| fast | `-m "not slow"`：単体テスト、ドキュメント、M1 の mirt スモークテスト（CI で数分） | すべての PR、`develop`/`main` へのすべての push |
| full | mirt 参照値との照合をすべて含む全テスト。Python ごとに 4 シャードに分割 | `develop` の夜間実行、リリース PR（`develop` → `main`）、手動実行（*Run workflow*） |

mirt 参照値との照合は、full 層（夜間実行またはリリース PR）が通って初めて完了とする。ROADMAP の項目にチェックを付ける前に夜間実行の結果を確認する。

- 約 5 秒を超えるテストには `@pytest.mark.slow` を付ける。`functools.cache` で共有するフィッティングの時間も含める（そのフィッティングを最初に使うテストが時間を払う）。
  fast 層の結果の最後に、`slow` なしで 5 秒を超えたテストが一覧される（"tests over 5 s not marked slow"）。
- `tests/test_mirt_*.py` はモジュールごと slow にする（`pytestmark = pytest.mark.slow`）。`test_mirt_m1.py` だけは例外。
- 1 モジュールは 1 ワーカーで走るので、数分かかるフィッティングは専用のモジュールに分ける（`tests/_mirt_multigroup.py` と `tests/test_mirt_m4_multigroup_*.py` を参照）。
- CI は JAX の永続コンパイルキャッシュ（`JAX_COMPILATION_CACHE_DIR`）を実行間で引き継ぐ。
- full 層のシャードは、`tests/durations.json` に記録したモジュールごとの秒数で均等に分ける。重いモジュールを足したり分けたりしたら、全テストを走らせて更新する：
  `uv run pytest --write-durations tests/durations.json`（または full 層の CI 実行が残す `durations-*` アーティファクトをまとめる）。ファイルにないモジュールは 1 テスト 1 秒として扱う。

手元で素早く確かめるなら `uv run pytest -m "not slow"`。CI と同じ 1 シャードだけ走らせるなら `uv run pytest --shard 2/4`。

mirt の参照値を作り直すとき（R 4.4+ と mirt が必要）:

```bash
Rscript validation/r/m1_reference.R
Rscript validation/r/m2_reference.R
```

flexpyctt の参照値のスクリプトは `packages/flexpyctt/validation/r/` にあり、`packages/flexpyctt/tests/reference/` に
書き出す。これらもリポジトリのルートから実行する。

R `mirt` との wall time ベンチマークを再実行するとき（詳細と `--quick` モードは
[benchmarks/README.ja.md](benchmarks/README.ja.md) を参照）:

```bash
uv run python benchmarks/data.py
uv run python benchmarks/run_flexpyirt.py
Rscript benchmarks/run_mirt.R
uv run python benchmarks/report.py
```

## 命名

名前は PEP 8 に従い、R ではなく Python として読めるものにする（DESIGN.md §15.46）。`ruff` が `N` 規則を検査する。

- 関数・メソッド・引数・変数は `snake_case`、クラスは `CapWords`、モジュール定数は `UPPER_SNAKE_CASE`。
- R/mirt の引数名をそのまま使わない。Python の綴りにし、対応する mirt の名前は docstring に書く:
  `n_categories`（`n_cat` ではない）、`min_cell`（mirt の `mincell`）、`mean_log`/`sd_log`（R の `meanlog`/`sdlog`）、
  `bounds=(lower, upper)`（mirt の `theta_lim`）。
- 公開名は略さない（`is_eh` ではなく `is_empirical_histogram`）。定着した略語は使ってよい: モデル名（`TwoPL`, `GRM`,
  `GPCM`）、個数を表す接頭辞 `n_`、`se`, `vcov`, `loglik`, `theta`, `eta`。よく使う別名とぶつかる略語は避ける
  （`pd` は pandas、`df` は自由度と DataFrame で紛らわしい）。
- 同じ概念には API 全体で同じ名前を使う（`n_points`, `n_factors`, `n_categories`）。同じ系統のクラスは接尾辞を揃える
  （`NormalPrior`, `LogNormalPrior`, `BetaPrior`）。
- `Literal` の文字列の選択肢は小文字の `snake_case`（`"slope_intercept"`, `"stocking_lord"`）。定着した頭字語は大文字のまま
  （`"EAP"`, `"MAP"`, `"BH"`）。
- DESIGN.md の数式に対応する場所では数式記号を使ってよい: jit されるカーネル内の大文字の記号（`G`, `Q`, `D`, `K_f`。
  このため ruff の `N803`/`N806` は無視している）や、`LinkResult.A`/`B` のような標準的な定数。
- ローカルのクロージャは役割で名付ける（`objective`, `backtrack_body`, `interp_weights`。`obj` や `tmp` は使わない）。
  mirt のコードを行ごとに移植した箇所（例: `_itemfit.collapse_cells`）は照合のため mirt の局所変数名を残してよく、
  その旨をコメントに書く。
- `FitResult` の JSON に保存される名前を変えるときは `schema_version` を上げ、古いファイルも読めるようにする
  （読み込み時に旧キーを置き換える）。

## ドキュメント

すべての文書は英語版（`*.md`）と日本語版（`*.ja.md`）があり、内容は同じ。片方を変えたら両方を直す。
`.github/` の PR・issue テンプレートは、GitHub が既定のテンプレートを 1 つしか使わないため、英日を併記した
1 つのファイルにしている。

`docs/` の文書はそのまま英日 2 言語のサイト（<https://takuizum.github.io/flexpyirt/>、日本語は `/ja/`）
にもなる（MkDocs、`mkdocs.yml`）。サイトは `main` から公開されるので、リリース済みの状態を表す。
ローカルでの確認:

```bash
uv run --group docs mkdocs serve           # プレビュー
uv run --group docs mkdocs build --strict  # CI と同じ検査（リンク切れ・アンカー切れで失敗）
```

- 文書内のリンクは相対リンクで書く（GitHub でもサイトでも動く）。`docs/` の外を指すリンクは、サイトでは
  GitHub 上のファイルに書き換わる（`scripts/mkdocs_hooks.py`）。
- `README.md` / `README.ja.md` は PyPI のプロジェクトページにも表示されるので、リンクは絶対 URL で書く
  （文書はサイトの URL、それ以外は GitHub の URL）。README はパッケージ利用者だけに向けたもので、開発環境の
  準備、テストや CI の詳細、ブランチ運用、マイルストーン、CONTRIBUTING や `docs/dev/` へのリンクは書かない。
- `docs/` に文書を足したら `mkdocs.yml` の `nav`（と日本語の `nav_translations`）にも足す。
- サイトはパッケージ利用者向け: 使い方（`guide`, `reference`）、理論（`theory`）、結果の検証方法
  （`accuracy`）。開発者向けの文書は `docs/dev/` に置く（[設計](docs/dev/DESIGN.ja.md)、
  [API スケッチ](docs/dev/API.ja.md)、[検証計画](docs/dev/VALIDATION.ja.md)、
  [ロードマップ](docs/dev/ROADMAP.ja.md)）。これらはサイトから除外されるので、サイトのページから相対リンクで
  指さない（必要なら GitHub の URL）。マイルストーン、設計書の節番号、照合ケース ID は利用者向けページに書かない。
