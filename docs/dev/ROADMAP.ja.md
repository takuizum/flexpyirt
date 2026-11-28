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
- [x] IRT 真値等化・観測得点等化と素点-尺度得点変換表（#106。`irt_equate()`, `score_distribution()`, `scale_table()`。1 次元、全項目モデル、単一群・多母集団。V51 で equateIRT・SNSequate と、mirt と equate パッケージから組み立てた参照に照合済み）。等化得点の標準誤差は後続。
- [x] 項目適合度: S-X2（`FitResult.item_fit()`、1 次元、多母集団は群ごと。V15 で照合済み）。
- [x] DIF: 多群の尤度比検定（`FitResult.dif()`、`add` / `drop` の 2 方式、Holm / BH 補正。V17 で照合済み）。
- [x] M2 限定情報の適合度（`FitResult.m2()`、単一群 1 次元の $M_2$ / $M_2^*$、RMSEA と信頼区間、SRMSR、2 値の TLI / CFI。V18 で照合済み）。
- [x] 名義反応モデル（`Nominal(n_categories)`、mirt の `itemtype = "nominal"` と同じ Thissen, Cai & Bock (2010) のパラメタ化。V23 で 1 因子・2 因子を照合済み、3 因子は単体テストのみ）。
- [x] 次元性診断（DESIGN.md §16.8。問い A「次元はいくつか」と問い B「本質的一次元性は認められるか」を分けて扱う。探索的 MIRT は行わない）:
  - [x] D1 ポリコリック・テトラコリック相関と平滑化（`polychoric()`、`PolychoricResult`。V24 で `lavaan::lavCor`、`psych::polychoric` / `cor.smooth` と照合済み）。
  - [x] D2 固有値、平行分析、改訂 MAP、EKC（`parallel_analysis()`、`velicer_map()`、`empirical_kaiser()`。V25 で EFAtools・psych・EFA.dimensions と照合済み）。
  - [x] D3 局所依存 Q3 / Q3* / LD-X2 / G2（`FitResult.local_dependence()`、`mirt::residuals` と照合）とモデルに基づく平行分析（`model_parallel_analysis()`、`ltm::unidimTest` と照合）、アンサンブル `dimensionality()`（V26 で照合済み。アンサンブルは再計算で確認）。
  - [x] JSI（Edwards, Houts & Cai, 2018。`FitResult.jsi()`。推定し直す方法と、推定し直さない 2 つの近似。推定し直す方法は
    V31 で `mirt::residuals(type = "JSI")` と照合済み）。
  - [x] D4 DETECT の確認的・探索的版（`detect()`、`exploratory_detect()`。総得点による条件付けで `sirt::conf.detect` / `expl.detect` と V27 で照合済み。カーネル平滑化による条件付けと遺伝的アルゴリズムによる分割の探索は未実装）。
  - [x] D5a flexpyirt に実装した ULS / ML の EFA（`efa()`。`factor_analyzer` は scikit-learn 1.8 以降で失敗する。DESIGN §16.8）と Hull 法（`hull()`）。V28 で EFAtools の `EFA` / `HULL` と `psych::fa` と照合済み。
  - [x] D5b 比較データ法（`comparison_data()`。V29 で EFAtools の `CD()` とその生成器と照合済み）。
  - [x] D6 ブートストラップでバイアス補正する DIMTEST（`dimtest()`。V30 のシミュレーション研究で受け入れ済み: .05 での第 1 種の誤り 0.5〜1.0%（保守的）、検出力 0.99〜1.00）。
- [x] flexpyctt の観測得点による DIF（#102、DESIGN §16.20）: ETS 分類つきの Mantel–Haenszel（`mantel_haenszel()`）、順序のある項目の Mantel 検定（`mantel()`）、一般化 MH（`generalized_mantel_haenszel()`）、ロジスティック回帰（`logistic_dif()`）。純化つき。V47 で difR と `mantelhaen.test()` に照合済み。
- [ ] 順序ロジスティック回帰による DIF（#102 の 2 本目の PR。参照は `difR::difPolyLogistic`）。

## M6 単一群・1 次元の制限をなくす
各項目は mirt（または独立な実装）との照合が通って完了とする。参照先は issue に書く。
- [x] Davidian 曲線・混合正規の潜在分布を多母集団モデルで使えるようにする（#75、DESIGN §27）: モデルごとに 1 つの族、
  群ごとの形、アンカーつき・FIPC のデザイン、Oakes / Hessian SE、EAP / PV、群ごとの `item_fit()`（S-X2 は 1 群でも使える
  ようになった）、JSON は変更なし。V56 で照合（mirt `multipleGroup(customGroup = ...)` と独立な R の尤度）。
- [x] M2 を多母集団・欠測ありのデータ・MAP 推定に広げる（#76。DESIGN §16.23、V42 で照合）: 群について並べた統計量
  （$\Delta$ の群の行を $\sqrt{N_g/N}$ 倍）、ペアごとに利用可能な周辺と MCAR のもとでのその厳密な共分散、MAP 推定値での
  計算（mirt と同じ）。多母集団の `DavidianCurve`・EH・混合分布はエラーのまま。
- [x] 確認的多次元モデルでの M2 と S-X2（#77）: 単一群の `with_factors()` モデル（2〜3 因子、直積格子）で `FitResult.m2()` と `FitResult.item_fit()` が使える（V39 で照合済み）。双因子 / two-tier と多母集団の多次元モデルは引き続きエラー。
- [x] 4PL・名義反応モデル・多次元モデルのリンキング（#78。確認的 2・3 因子モデルでは `link(..., transformation=)` で対角・直交・斜交の $\mathbf A$ を選ぶ。V40 で照合済み）。
- [x] 群の数が多い場合（#79、DESIGN §12 の未決事項 2、§26）: E-step と Oakes の交差項を固定長の単一群パターンブロック上で
  回し（計算量は受験者数に比例、コンパイル時間は群の数によらない）、潜在分布の M-step は積み重ねた群パラメタ上で `vmap`、
  `with_groups()` はマッピングも受け付ける。10 群で mirt `multipleGroup` と照合（V55）。10・50・200 群のベンチマークは `benchmarks/`。
- [ ] 固定節点の準モンテカルロ求積による 4〜5 因子の確認的モデル（#164、`Quadrature.qmc()`、DESIGN §28）。実装し、細かい
  直積格子（3 次元）と厳密に分解した尤度（4 次元）で確認し、V58 で $13^4$ 格子の mirt の EM と照合した（mirt 自身の
  `QMCEM` はこのケースで失敗する）。V58 が夜間の CI で通ったらチェックを付ける。
- **受け入れ基準**: 各項目が VALIDATION.md の許容誤差内で mirt（または R による独立な実装）と一致。

## M7 双因子・診断・情報関数・調査設計（この順に実装する）
1. [x] 次元縮約つきの双因子モデル・two-tier モデル（#80）。
   - [x] 双因子モデル（`Model.with_bifactor()`、単一群、特殊因子数に制限なし、Oakes/Hessian SE、EAP スコアリング。
     V32 で mirt の `bfactor()` と照合済み）。
   - [x] two-tier モデル（`Model.with_two_tier()`、相関してよい主因子 1〜2 個と特殊因子、単一群、Oakes/Hessian SE、
     EAP スコアリング。V33 で mirt の `bfactor(model2 = )` と照合済み）。
2. [x] 局所依存（Q3, LD-X², JSI）を多母集団・正規以外の潜在分布・多次元 / 双因子モデルに広げる（#81）。
   - [x] `FitResult.local_dependence()`（Q3、Q3\*、LD-X²、G²）を多母集団、`EmpiricalHistogram` /
     `DavidianCurve` / `NormalMixture` の潜在分布、確認的な 2〜3 因子モデルに広げた（V26 で mirt の
     `residuals()` と照合。混合分布は密度を与えたときの統計量のみ）。
   - [x] 双因子・two-tier モデルの `FitResult.local_dependence()`。期待度数表は縮約した格子の上で求める
     （V38 で `bfactor()` の適合に対する mirt の `residuals()` と照合）。
   - 実装しない（決定済み、#81）: 単一群・1 次元・`Normal` 以外の JSI。JSI は単一群・1 次元のモデルについてのみ
     定義されており、多次元・多母集団への拡張は研究されておらず、参照となる実装もない（mirt の `type = "JSI"` は
     1 因子が必要）。これらのモデルでは `jsi()` は `SpecificationError` を出す（DESIGN.md §16.17）。
3. [x] 項目・テスト情報関数、信頼性、合計得点による EAP（#82）。公開する Lord–Wingersky / Poisson 二項分布の
   カーネル（#83）の上に作る。
   - [x] カーネル（#83）: `summed_score_distribution()`。Poisson 二項分布のカーネルとしての Lord–Wingersky 再帰
     （通常と対数空間）、DFT-CF 法、2 値項目用の正規・精密化正規・Poisson 近似。`jit` / `vmap` / 自動微分に対応。
     V36 で `poibin`・`dpois`・mirt の EAPsum の表と照合済み。
   - [x] 1 次元モデル（単一群・多母集団）（#82）: `item_information()`、`test_information()`、`expected_score()`、
     `reliability()`（周辺・経験的。潜在分布は `Normal`）、`summed_score_table()`、
     `FitResult.score(method="EAPsum")`。V37 で 2PL・3PL・GRM・GPCM・名義・混合形式のテスト・2 群モデルについて
     `iteminfo()`・`testinfo()`・`expected.item()` / `expected.test()`・`marginal_rxx()`・`empirical_rxx()`・
     `fscores(method = "EAPsum")` と照合済み。
   - [x] 多次元モデル（確認的 2–3 因子、双因子、two-tier。単一群と多母集団）: `information_matrix()`、
     `item_information(direction=)` / `test_information(direction=)` による方向別の情報量（合成値の SE つき）と
     `expected_score()`。V43 で `iteminfo(multidim_matrix = TRUE)`・`iteminfo(degrees = )`・
     `testinfo(degrees = )`・`expected.item()` / `expected.test()` と照合。
4. [ ] 標本ウェイト（#84）。
   - [x] `fit(..., weights=...)`: ウェイトを足し合わせるパターン圧縮による重み付き周辺尤度（EM、SQUAREM、多群、
     欠測、FIPC）。BIC はウェイトの和を使う。Oakes / Hessian の SE は警告つき（度数ウェイトでのみ正しい）。
     mirt の `survey.weights` と V34 で照合（2PL、GRM、2 群、FIPC、双因子）。標本設計に整合する SE は #85 で扱う。
     項目適合度、M2、局所依存、JSI、DIF はウェイト付きの推定結果に対してエラーを出す。
   - [ ] 標本設計に基づく SE: $\sum_iw_i^2s_is_i^\top$ による `fit(..., weights=, se="sandwich")`（DESIGN §17.6）。
     V57 で照合（夜間の CI が通ったらチェックする）。
5. [ ] ロバスト標準誤差: sandwich と cross-product（#85）。
   - [x] 標本ウェイトなし: `fit(se="sandwich" | "crossprod")`。単一群・多母集団、FIPC、EH、2〜3 因子、事前分布
     （Davidian 曲線・正規混合・双因子・two-tier モデルでは不可）。V35 で mirt の `SE.type = "sandwich"` / `"crossprod"` と照合済み。
   - [ ] 標本ウェイトとクラスタ: 重み付きサンドイッチ、ウェイトつき `crossprod`（mirt の定義、度数ウェイトの
     警告つき）、$G/(G-1)$ によるクラスタロバストなサンドイッチ `fit(..., cluster=)`（DESIGN §17.6）。mirt の
     重み付き Oakes 共分散と numDeriv の個人スコアから作った R の参照値と V57 で照合（夜間の CI が通ったら
     チェックする）。
- **受け入れ基準**: mirt の `bfactor()`、`residuals()`、`iteminfo()` / `testinfo()` / `marginal_rxx()` /
  `fscores(method = "EAPsum")`、`survey.weights`、`SE.type = "sandwich"` / `"crossprod"` との V 番号つき照合。

## 現場向け機能
- [x] 図（#103）: `item_probabilities()`、`empirical_item_curve()`、`wright_map_table()` が図の元になる数値を
  返し、`flexpyirt.plotting`（観測比率と群ごとの曲線（DIF）つきの `plot_item_curves`、`plot_information`、
  `plot_expected_score`、`plot_wright_map`）が任意依存 `flexpyirt[plot]` の matplotlib で描く。1 次元モデル、
  全項目族、単一群・多母集団。V48 で mirt の `itemfit(return.tables = TRUE)`、`probtrace()`、
  `coef(IRTpars = TRUE)` と TAM の `tam.threshold()` に照合。
- [x] DIF の分析手順（#104）: `dif_screen()` がアンカーを純化しながら全項目をスクリーニングし（`"all_others"`、
  `"sequential_free"`、`"sequential_constrain"`。尤度比検定または Wald 検定）、最終的な部分不変モデルを返す。
  `dif_effect_sizes()` が期待得点の符号つき・符号なしの差、ESSD、RMSD とテスト（DTF）の行を返す。
  `FitResult.dif(test="wald")`。1 次元の多母集団モデル、2 群以上。V49 で mirt の `DIF(scheme = "drop" /
  "drop_sequential" / "add_sequential", Wald = TRUE)`、`DRF()` の点推定、`empirical_ES()`、`RMSD_DIF()` に照合。
- [x] リンキングの共通項目の点検と項目パラメータドリフト（#105。`screen_common_items()`、
  `CommonItemScreeningResult`、`stack_screening_tables()`: 頑健 z・変位・期待得点の RMSD と反復的な純化、1 次元。V50 で照合済み）。
- [x] 受験者適合度と残差にもとづく項目適合度（#108）: `FitResult.person_fit()`（$l_z$、$l_z^*$、受験者の infit / outfit と
  $z$ 値）と `FitResult.item_fit(statistics=("infit", "outfit"))`。1 次元モデル（全項目モデル、欠測、多母集団）。
  V53 で mirt の `personfit()` / `itemfit(fit_stats = "infit")`、sirt の `personfit.stat()` と照合。$l_z^*$ は式の R での
  再計算と帰無分布のシミュレーションで照合（R パッケージの参照値はない。オーナーの承認待ち）。
- [ ] モデル比較表と回答データのシミュレーション（#109）: `compare_fits()`（AIC・SABIC・HQ・BIC・尤度比検定。
  mirt の `anova()`）と `simulate_responses()` / `SimulatedData`（`FitResult` または項目バンクを与えた `Model`
  から。全項目モデル・全潜在分布・全モデル構造。JAX のキー）。V54 で照合。照合テストが夜間 CI で通ったらチェックする。
- [x] カット得点による分類の正確さと一貫性（#107。`classification_accuracy()`、`ClassificationResult`: 合計得点上の
  カットの Lee, 2010 と $\theta$ 上のカットの Rudner, 2001, 2005、D 法と P 法、κ、カットごとの偽陽性・偽陰性率。
  1 次元モデル、単一群と多母集団。V52 で cacIRT と mirt からの再計算と照合）。

## 継続的な作業
- [ ] GitHub 上の最新の R mirt で照合し、参照値を定期的に確認する（#74）。
- [x] ユーザー定義の項目モデルの公開拡張 API と docs の例（#86）: `ItemModel` / `ItemKernel` の契約、JSON 用の
  `register_item_model`、ガイドの 2PL の実例（DESIGN §20）。2PL / GRM の再実装は組み込みと機械精度で一致する。
- [ ] 組み込みでないモデルの照合済みの例（GGUM を mirt の `itemtype = "ggum"` と照合、V44 など）。新しい項目モデルは
  本体に組み込まず、このように追加する。
- [x] flexpyctt: 古典的項目分析（キーによる採点、項目統計量、α / ω、選択肢分析。#101）。V46 で psych・ltm・polycor
  と照合済み。
