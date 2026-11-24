# ロードマップ

[English](ROADMAP.md) | 日本語

各マイルストーンは「受け入れ基準」をすべて満たして完了とする。順番を飛ばさない。

## M0 足場
- `uv` プロジェクト、`src/flexpyirt/` レイアウト、ruff / mypy / pytest の設定、GitHub Actions（Linux, Python 3.11–3.13）。
- `flexpyirt.errors`、x64 チェック関数、空の `Model` / `fit` / `FitResult` の型だけ。
- **受け入れ基準**: CI が通る。editable install（`uv run --isolated --no-project --with-editable .`）で import できる。環境管理は uv のみで pip は使わない。

## M1 単一群 2PL / Rasch の BA-EM
- Compile 層: パターン圧縮、ParameterTable、FreeMap（固定・等値の仕組みはここで作る）。
- Core: E-step、項目ごとの Newton M-step（vmap）、`lax.while_loop` の EM ループ、Normal(0,1) 固定。
- SE: Oakes と直接 Hessian。
- スコアリング: EAP。
- **受け入れ基準**: V1, V2, V10(Oakes) が許容誤差内。EM 単調性テストが通る。

## M2 多母集団・FIPC・GRM・欠測
- 群ごとの潜在分布（Normal 自由）、群間等値（`free_id` 共有）、識別性チェック。
- FIPC（固定項目 + 潜在分布推定）。
- GRM。欠測（MAR）。
- **受け入れ基準**: V3, V4, V5, V9 が許容誤差内。識別不能な仕様で `IdentificationError`。

## M3 EH・MAP・3PL/4PL・GPCM・SQUAREM
- [x] EmpiricalHistogram（`EmpiricalHistogram`、モーメント制約付き M-step による参照群の標準化、平方根尺度の SQUAREM。V6 で照合済み）。
- [x] 事前分布と MAP（`NormalPrior`, `LogNormalPrior`, `BetaPrior`, `Model.prior()`, `FitResult.log_posterior`、V7 で照合済み）。
- [x] 3PL / 4PL（`ThreePL`, `FourPL`, `HeywoodWarning`、V11 で照合済み）。
- [x] GPCM / PCM（`GPCM`, `PCM`、V8 で照合済み）。
- [x] SQUAREM（`BAEM(squarem=True)`、スキーム S3 と単調性・実行可能性の安全装置。全 mirt 照合スイートが SQUAREM あり・なしで通過）。
- [x] ベンチマーク（mirt との wall time。`benchmarks/` を参照）。
- **受け入れ基準**: V6, V7, V8, V11 が許容誤差内。SQUAREM ありでも照合が通る。

## M4 結果まわり・多次元の入口
- [x] `FitResult.summary()`、`FitResult._repr_html_()`、デルタ法による表示用 IRT パラメタ化の SE
  （`b_se` / `bk_se`、V12 で照合済み）。
- [x] `FitResult` の JSON 入出力（`FitResult.to_json`, `FitResult.from_json`）。
- [x] スコアリング（MAP, ML, WLE, plausible values。V13 で照合済み）。
- [x] 確認的 2–3 次元 MIRT（`Model.with_factors()`、直積格子、多変量正規の因子間相関、Oakes/Hessian SE、EAP・MAP・ML・WLE スコアリング。V14 で照合済み。多母集団は V21 で照合済み）。
- **受け入れ基準**: mirt の `fscores` と EAP / MAP が一致。2 次元確認的モデルが mirt と一致。

## M5 以降（候補）
- [x] 項目をまたぐ等値制約の M-step（`Model.equal_across_items()`、項目・項目モデルをまたぐ `free_id` の共有、連結成分ごとの同時 Newton。V16 で照合済み）。
- [x] Davidian 曲線の潜在分布（`DavidianCurve`、単一群 1 次元、解析的標準化、Oakes/Hessian SE。V19 で照合済み）。
- [x] 混合正規の潜在分布（`NormalMixture`、単一群 1 次元、mirt の混合のパラメタ化を解析的に標準化、Oakes/Hessian SE。V22 で照合済み）。
- [x] 微分可能なリンキング（`link()`, `LinkResult`: Haebara / Stocking–Lord の非対称・対称版、自動微分による Ogasawara デルタ法 SE。2PL・3PL・GRM・GPCM と 2PL + GPCM 混在の共通項目について V20 で照合済み）。
- [x] 項目適合度: S-X2（`FitResult.item_fit()`、1 次元、多母集団は群ごと。V15 で照合済み）。
- [x] DIF: 多群の尤度比検定（`FitResult.dif()`、`add` / `drop` の 2 方式、Holm / BH 補正。V17 で照合済み）。
- [x] M2 限定情報の適合度（`FitResult.m2()`、単一群 1 次元の $M_2$ / $M_2^*$、RMSEA と信頼区間、SRMSR、2 値の TLI / CFI。V18 で照合済み）。
- [x] 名義反応モデル（`Nominal(n_categories)`、mirt の `itemtype = "nominal"` と同じ Thissen, Cai & Bock (2010) のパラメタ化。V23 で 1 因子・2 因子を照合済み、3 因子は単体テストのみ）。
- [ ] 次元性診断（DESIGN.md §16.8。問い A「次元はいくつか」と問い B「本質的一次元性は認められるか」を分けて扱う。探索的 MIRT は行わない）:
  - [x] D1 ポリコリック・テトラコリック相関と平滑化（`polychoric()`、`PolychoricResult`。V24 で `lavaan::lavCor`、`psych::polychoric` / `cor.smooth` と照合済み）。
  - [x] D2 固有値、平行分析、改訂 MAP、EKC（`parallel_analysis()`、`velicer_map()`、`empirical_kaiser()`。V25 で EFAtools・psych・EFA.dimensions と照合済み）。
  - [x] D3 局所依存 Q3 / Q3* / LD-X2 / G2（`FitResult.local_dependence()`、`mirt::residuals` と照合）とモデルに基づく平行分析（`model_parallel_analysis()`、`ltm::unidimTest` と照合）、アンサンブル `dimensionality()`（V26 で照合済み。アンサンブルは再計算で確認）。
  - [ ] JSI（Edwards, Houts & Cai, 2018。`mirt::residuals(type = "JSI")` と照合）。
  - [x] D4 DETECT の確認的・探索的版（`detect()`、`exploratory_detect()`。総得点による条件付けで `sirt::conf.detect` / `expl.detect` と V27 で照合済み。カーネル平滑化による条件付けと遺伝的アルゴリズムによる分割の探索は未実装）。
  - [x] D5a flexpyirt に実装した ULS / ML の EFA（`efa()`。`factor_analyzer` は scikit-learn 1.8 以降で失敗する。DESIGN §16.8）と Hull 法（`hull()`）。V28 で EFAtools の `EFA` / `HULL` と `psych::fa` と照合済み。
  - [x] D5b 比較データ法（`comparison_data()`。V29 で EFAtools の `CD()` とその生成器と照合済み）。
  - [ ] D6 ブートストラップでバイアス補正する DIMTEST（シミュレーション研究の第 1 種の誤りと検出力で受け入れる）。
