# 検証計画（R mirt との数値照合）

[English](VALIDATION.md) | 日本語

## 方針
- 照合の参照値は R `mirt` で生成し、JSON で `tests/reference/` にコミットする。CI では R を実行しない。
- 生成スクリプトは `validation/r/<case>.R`。各スクリプトの先頭に `sessionInfo()` 相当（R・mirt のバージョン）を JSON に書き出す。
- 照合は**同じ条件**で行う: 同じ求積（等間隔 61 点・$[-6,6]$ など）、同じ収束基準（mirt 側は `TOL` を厳しく、例: 1e-8）、加速なし、同じ事前分布。
- データはシミュレーション（固定シード）と公開データ（例: mirt 同梱の `LSAT7`, `Science`, `SAT12`）の両方。

## 許容誤差（初期値、必要に応じて見直す）
| 量 | 許容（最大絶対差） |
|---|---|
| 対数尤度 | 1e-6（相対） |
| 項目パラメタ（内部尺度 $a, d$） | 1e-4 |
| 潜在分布パラメタ（平均・SD） | 1e-4 |
| 標準誤差 | 1e-3（相対） |
| EAP | 1e-5 |
| MAP / ML / WLE（$\theta$） | 1e-4 |
| MAP / ML / WLE（SE） | 1e-3（相対） |
| 尺度リンキング $(A, B)$ / 目的関数（同一の項目推定値） | 1e-9 / 1e-9（相対） |
| 尺度リンキングのデルタ法 SE / `vcov`（同一の推定値・共分散） | 1e-5（相対） |
| 尺度リンキング $(A, B)$ / 目的関数 / SE（推定 + リンキングの一連） | 1e-6 / 1e-5（相対） / 1e-3（相対） |
| ポリコリック相関（lavaan / psych と）、閾値、平滑化（`cor.smooth` と）。2 つの実装とも | 1e-6 / 1e-3、1e-12、1e-12 |
| V24 のケースでの flexpyctt（NumPy）と flexpyirt（JAX）のポリコリック相関、Owen の $T$ による $\Phi_2$ と Plackett の積分 | 1e-9（フラグは一致）、1e-14 |
| 同じ行列での EKC・MAP・観測固有値、正規の参照の平均 / 95 パーセンタイル | 1e-11、$\lvert z\rvert\le4.5$ / 6e-3 |
| LD $X^2$ / $G^2$、符号付き $V$・Q3（推定し直したモデル）、モデルに基づく平行分析の観測値 / シミュレーションの平均 / 95 パーセンタイル | 1e-5（相対）、1e-6、2e-4 / $\lvert z\rvert\le4.5$ / 0.02 |
| DETECT の条件付き共分散、指標（両方の半分）、Ward の併合の高さ | 1e-12 |
| EFA の含意する $\Lambda\Lambda^\top$ / 共通性、CAF・CFI・RMSEA、$\chi^2$、Hull の適合度 / $st$ | 1e-4、2e-4、1e-4（相対）、1e-5 / 1e-2（相対） |
| CD: 主因子法の $\Lambda\Lambda^\top$、同じ乱数による 1 因子の母集団、順位和の $p$、平均 RMSE（採用された $k$ / 棄却された追加の $k$） | 1e-12、厳密、1e-10（相対）、15% / 60%（相対） |
| DIMTEST（シミュレーション研究。テストではその小さい版）: .05 での第 1 種の誤り / 検出力 | 20 回中 $\le 0.15$ / 10 回中 $\ge 8$ |
| 観測得点による DIF: MH・一般化 MH の統計量、$\hat\alpha_{MH}$ とその対数の分散、Mantel の統計量、累積オッズ比とその標準誤差。ロジスティック回帰の尤度比統計量、$\Delta R^2$、係数／標準誤差と Wald。検出結果と純化の経過 | 1e-10（相対）。1e-8 / 1e-6（相対）。一致 |
| mirt の推定値でのクロス積／サンドイッチの共分散（各要素を $\sqrt{v_{ii}v_{jj}}$ で割った差）。通しで当てはめたときのパラメタ／SE | 1e-9 / 1e-6、1e-6 / 1e-5（相対） |
| 古典的項目分析（α、標準化 α、項目を除いたときの値、Feldt の区間、項目–合計・項目–残りの相関、点双列相関、2 段階の多分系列・双列相関、キーによる得点、選択肢の統計量）と psych / ltm / polycor / 素の R。ω total とその負荷量と psych | 1e-12（得点は完全一致）。1e-6 / 1e-5 |
| mirt の $\theta$ での受験者・項目適合度（$l_z$、infit、outfit、$z$）、R で再計算した $l_z^*$、flexpyirt 自身の推定値での端から端まで | 1e-9（相対）、1e-7、1e-4 |

## ケース一覧（マイルストーンと対応）
| ID | 内容 | mirt 側の呼び出し | MS |
|---|---|---|---|
| V1 | 2PL 単一群（LSAT7） | `mirt(LSAT7, 1, "2PL", SE=TRUE, SE.type="Oakes")` | M1 |
| V2 | Rasch 単一群 | `mirt(..., "Rasch")` | M1 |
| V3 | GRM 単一群（Science） | `mirt(Science, 1, "graded")` | M2 |
| V4 | 2PL 2 群・部分アンカー・焦点群自由 | `multipleGroup(..., invariance=c(anchors, "free_means","free_var"))` | M2 |
| V5 | FIPC（固定項目 + 潜在分布推定） | `pars='values'` で `est=FALSE` | M2 |
| V6 | FIPC での EH（単一群・2 群） | `mirt(..., dentype = "EH", pars = <バンク固定>)` / `multipleGroup(...)` | M3 |
| V7 | 3PL + 事前分布（MAP） | `mirt.model` の `PRIOR = (1-30, g, expbeta, ...)` 相当 | M3 |
| V8 | GPCM・PCM（Science + シミュレーション） | `mirt(..., "gpcm")` / `mirt(..., "Rasch")`（多値） | M3 |
| V9 | 欠測あり（MAR） | NA を含むデータ | M2 |
| V10 | SE（Oakes。crossprod / sandwich は V35） | `SE.type=` | M1–M3 |
| V11 | 3PL・4PL 単一群（シミュレーション） | `mirt(..., "3PL")` / `mirt(..., "4PL")` | M3 |
| V12 | 表示用 IRT 閾値とデルタ法 SE（2PL・Rasch・GRM・GPCM・PCM・2 群 2PL） | `coef(m, IRTpars = TRUE, printSE = TRUE)` | M4 |
| V13 | 能力値スコアリング: MAP・ML・WLE（2PL・GRM・GPCM・3PL・2 群・欠測） | `fscores(m, method = "MAP"/"ML"/"WLE", response.pattern = ..., full.scores.SE = TRUE)` | M4 |
| V15 | S-X2 項目適合度（GRM・GPCM・分散自由の PCM・3PL・FIPC の 2PL、`mincell` 1 と 3、2 群の 2PL と GRM を群ごと） | `itemfit(m, fit_stats = "S_X2")` | M5 |
| V16 | 項目をまたぐ等値制約（LSAT7 の共通傾き、2PL + graded のモデル間制約、アンカーつき 2 群 + 共通傾き） | `mirt(..., mirt.model("... CONSTRAIN = (1-5, a1)"))` / `multipleGroup(...)` | M5 |
| V17 | V4 のデータでの尤度比検定による DIF（`a`+`d` と `d` の `add`、完全不変モデルからの `a`+`d` の `drop`） | `DIF(mg, which.par = ..., scheme = ...)` | M5 |
| V18 | M2 / M2* 限定情報の適合度（LSAT7 2PL、シミュレーション 2PL、3PL、分散自由の Rasch、GRM $K=3$、GPCM $K=4$） | `M2(m, quadpts = 61, theta_lim = c(-6, 6))` | M5 |
| V14 | 確認的 2–3 次元 MIRT（2PL 単純構造・交差負荷・GRM・3 次元 2PL） | `mirt(..., mirt.model("F1 = ...\nF2 = ...\nCOV = F1*F2"), SE = TRUE, SE.type = "Oakes")` | M4 |
| V19 | Davidian 曲線潜在分布（2PL $k=3$・GRM $k=2$・平均分散自由の FIPC 2PL $k=3$） | `mirt(..., dentype = "Davidian-k")` および `mirt(..., customGroup = ..., SE.type = "Oakes")` | M5 |
| V20 | 尺度リンキング: Haebara・Stocking–Lord（非対称・対称）と Ogasawara SE（2PL・GRM・GPCM・3PL・非標準の新フォーム格子・2PL + GPCM 混在） | 個別 `mirt()` 推定 + `probtrace()` / `expected.item()` による素の R 実装 | M5 |
| V21 | 多母集団の確認的 2 次元 MIRT（単純構造・交差負荷。両因子にアンカー、焦点群の平均・分散を自由） | `COV = F1*F2` での `multipleGroup(..., invariance = c(anchors, "free_means", "free_var"))` | M4 |
| V22 | 混合正規潜在分布（2PL $k=2$ の二峰、小さい成分を持つ GRM $k=2$、GPCM $k=3$） | 像の格子での `multipleGroup(..., dentype = "mixture-k", invariance = c(<全項目>, "free_means", "free_var"))` と、標準化した尤度の R による独立な実装（Newton による仕上げ、Richardson ヘッセ行列 SE、EAP） | M5 |
| V23 | 名義反応モデル（Science、順序のない・負の採点係数を含むシミュレーション、名義 + 2PL の混在、大半が非単調な項目、2 因子の確認的モデル） | GPCM の点（またはより良い別の初期値）から開始する `pars` を与えた `mirt(..., "nominal")`、`itemfit(fit_stats = "S_X2")` | M5 |
| V24 | ポリコリック・テトラコリック相関（シミュレーションのテトラコリック、2〜5 カテゴリの混在、10% MCAR、空のセルが 1 つの 2 x 2 表、LSAT7、Science）と平滑化 | lavaan の `lavCor(..., ordered = , missing = )`、psych の `polychoric(..., smooth = FALSE, correct = 0)`、`cor.smooth()` | M5 |
| V25 | 次元数: V24 の行列での EKC、MAP（2 乗と 4 乗）、PCA・SMC の固有値。正規データの平行分析の参照（3 つの大きさ） | EFAtools の `EKC()`、`PARALLEL()`、psych の `vss()$map`、EFA.dimensions の `MAP()`、4 乗版 MAP の R による独立な実装 | M5 |
| V26 | 局所依存（LSAT7 の 2PL、Science の GRM、依存するペアと 5% MCAR を含むシミュレーションの 2PL）とモデルに基づく平行分析（LSAT7） | `residuals(m, type = "LD" / "LDG2" / "Q3")`、ltm の `unidimTest(data = , thetas = cbind(d, a1), IRT = FALSE)` | M5 |
| V26（M7） | 2 群（4% MCAR を含む 2PL、GRM、2 因子の 2PL）、`EmpiricalHistogram`、`DavidianCurve(2)`、`NormalMixture(2)`（V22a の推定値）、2 因子（交差負荷と 3% MCAR を含む 2PL、GRM）、3 因子（2PL）の局所依存 | `multipleGroup()`、`dentype = "EH"` / `"Davidian-2"`（`use_dentype_estimate = TRUE`。Q3 は `Theta = fscores(m, use_dentype_estimate = TRUE)`）、`mirt.model()` の推定への `residuals()`。混合分布は、混合分布の重みを事前分布に入れた単一群の mirt オブジェクト | M7 |
| V27 | DETECT: 確認的（シミュレーションの 2 次元で正しい・誤った分割、1 次元、Science）と探索的（3 クラスタのシミュレーション、分割を指定） | sirt の `conf.detect(use_sum_score = TRUE, smooth = FALSE)`、`expl.detect(use_sum_score = TRUE, estsample = )` | M5 |
| V28 | EFA（ULS、ML。Hull の上限までのすべての $k$）と Hull 法（CAF、CFI、RMSEA）。V24 の行列のうち変数が 6 つ以上のものとシミュレーションの行列 2 つ | EFAtools の `EFA(method = , rotation = "none")`、`HULL(method = , gof = )`、psych の `fa(fm = "minres" / "ml", rotate = "none")` | M5 |
| V29 | 比較データ法: 主因子法、書き出した乱数による 1 因子の母集団の生成、順位和検定、3 因子の連続・2 因子の順序・1 因子の 2 値のデータでの `CD()` | EFAtools の `.paf_iter`、`.simulate_cfm_empirical`、`CD()`、`wilcox.test(x, y, "less")` | M5 |
| V30 | DIMTEST: 第 1 種の誤りと検出力（2PL。$N$ 1000〜2000、$J$ 20〜30。1 次元と相関 0.7〜0.8 の 2 次元。各 200 反復） | なし（DIMPACK 以外に実装がない）。`packages/flexpyctt/validation/simulation/dimtest_study.py` | M5 |
| V31 | ジャックナイフ傾き指標（LSAT7 の 2PL、Science の GRM、依存するペア 2 組と 5% の MCAR を含むシミュレーションの 2PL、依存するペア 1 組を含むシミュレーションの GPCM）。Newton 法と 1 ステップの近似は推定し直した値と比べる | `residuals(m, type = "JSI", fold = FALSE, TOL = 1e-10)` | M5 |
| V32 | 次元縮約による双因子モデル（特殊因子 3 つと一般因子のみの項目 3 つを含む 2PL、テストレット 3 つの GRM $K=4$） | `bfactor(data, model, itemtype = , SE = TRUE, SE.type = "Oakes", quadpts = 31)`; `fscores(method = "EAP", quadpts = 15)` | M7 |
| V33 | 次元縮約による two-tier モデル（相関する主因子 2 つと特殊因子 3 つ（1 つは両方の主因子にまたがる）の 2PL、交差負荷項目 1 つとテストレット 2 つの GRM $K=4$） | `bfactor(data, model, model2 = mirt.model("F1 = ...\nF2 = ...\nCOV = F1*F2"), SE = TRUE, SE.type = "Oakes", quadpts = 15)`; `fscores(method = "EAP")` | M7 |
| V34 | 標本ウェイト: 非整数ウェイト（合計 $\ne N$、ウェイト 0 が 5 行）の 2PL、合計を $N$ に正規化したウェイトの Science GRM、アンカー・focal 群の潜在分布の推定・5% の欠測を持つ 2 群、潜在分布を推定する FIPC、次元縮約による双因子 2PL。推定値、対数尤度、AIC、BIC、Oakes の SE | `mirt(..., survey.weights = w)`、`multipleGroup(..., survey.weights = w)`、`bfactor(..., survey.weights = w)`、`SE.type = "Oakes"` | M7 |
| V35 | ロバスト標準誤差: クロス積・サンドイッチの共分散行列（誤特定のもとでの 2PL と 5% の MCAR、3PL + GRM + GPCM + 名義、正規事前分布の MAP、focal の平均・分散が自由な 2 群、項目間の等値制約付き FIPC、相関のある 2 因子・3 因子、FIPC の EH） | `mirt(..., SE.type = "crossprod")` / `SE.type = "sandwich"`（EH のパンには `"Richardson"`） | M7 |
| V36 | 合計得点分布のカーネル（Poisson 二項分布: 確率ベクトル 5 例、$J$ 10〜300。多値: GRM $K=4$ と 2PL + GRM の混合） | `poibin::ppoibin(method = "RF" / "DFT-CF" / "RNA" / "NA")`、`poibin::dpoibin()`、`stats::dpois()`。mirt の `fscores(method = "EAPsum", full.scores = FALSE)` | M7 |
| V37 | mirt の推定値での情報関数・期待得点・信頼性・合計得点 EAP（2PL $J=12$。同じモデルで欠測 8%。3PL $J=12$。GRM と GPCM $K=4$、$J=8$。名義 $K=4$、$J=6$。2PL + 3PL + GRM + GPCM + 名義の混合 $J=14$。2 群の 2PL + GRM で焦点群の平均・分散を推定） | `iteminfo()`、`testinfo()`、`expected.item()`、`expected.test()`、`marginal_rxx()`、`empirical_rxx()`、`fscores(method = "EAPsum", full.scores = FALSE / TRUE)` | M7 |
| V38 | mirt の推定値での双因子・two-tier モデルの局所依存（双因子 2PL、$J=11$、特殊因子 3、一般因子のみの項目 2、欠測 3%。双因子 GRM $K=3$、$J=8$。two-tier 2PL、$J=10$、相関する主因子 2（交差負荷 1 項目）と特殊因子 2） | `residuals(bfactor(...), type = "LD" / "LDG2" / "Q3")`、two-tier は `bfactor(model2 = )`。Q3 は `Theta = fscores(m, quadpts = <適合時の値>)` | M7 |
| V39 | 確認的多次元モデルの $M_2$ / $M_2^*$ と S-X2（V14 の 4 モデル: 2D 2PL 単純構造、交差負荷のある 2D 2PL、2D GRM $K=4$、3D 2PL。因子間相関あり。GRM は `mincell` 1 と 3） | mirt の V14 の推定値での `M2(m, quadpts =, theta_lim =)`、`itemfit(m, fit_stats = "S_X2", quadpts =, theta_lim =)`。加えて、相関の列を $\Delta_2$ に入れた $M_2^*$ を R で厳密に計算した値 | M6 |
| V40 | 4PL・名義反応モデル・2PL + 名義混在の共通項目の尺度リンキング。多次元リンキング $\boldsymbol\theta^*=\mathbf A\boldsymbol\theta+\boldsymbol\beta$（対角・直交・斜交の $\mathbf A$。2 因子の 2PL と GRM、3 因子の 2PL）。Haebara・Stocking–Lord（非対称・対称）と Ogasawara の共分散 | 個別の `mirt()` 推定 + `probtrace()` / `expected.item()` による素の R の規準を `optim()` で最小化 | M6 |
| V42 | 多母集団・欠測データ・MAP 推定の $M_2^*$（大きさの異なる 2 群とアンカー項目の 2PL、欠測なしと 15% MCAR。2 群の GRM $K=3$。2 群の 2 因子 2PL。20% MCAR の単一群 2PL。2PL の MAP 推定） | mirt の推定値での `multipleGroup()` / `mirt()` の `M2(m, quadpts =, theta_lim =)`。加えて DESIGN §16.20 を R で独立に計算した値と、mirt の流儀で書いた R コード | M6 |
| V43 | 指定したパラメタでの多次元モデルの情報行列・方向別の情報量・期待得点（相関する 2 因子で 2PL・3PL・4PL・GRM・GPCM・名義の項目と交差負荷 4 項目。3 因子で 2PL・3PL・GRM・GPCM と交差負荷 2 項目。特殊因子 3 の双因子 2PL + GRM。相関する主因子 2 と特殊因子 2 の two-tier 2PL + GRM） | `pars =` と `TOL = NaN` で作った `mirt()` / `bfactor()` のオブジェクトでの `iteminfo(item, Theta, multidim_matrix = TRUE)`、`iteminfo(item, Theta, degrees = )`、`testinfo(m, Theta, degrees = )`、`expected.item()`、`expected.test()` | M7 |
| V45 | M-step の非拘束尺度（任意、`BAEM(m_step_scale="unconstrained")`）を V3・V4・V5・V7・V9・V11・V14 のケースで照合。`LogitNormalPrior`（`"natural"` は mirt に同等なものがないため独立な数値検証。`mode_scale="logit"` は mirt の `g` への `norm`） | 各ケースの既存の参照値。SciPy（`stats.norm`、`integrate.quad`）と式から書いた停留条件。`v45_3pl_logit_norm_map.json`（mirt）とテスト内で計算した flexMIRT の logit 尺度のモード | #15 |
| V46 | 古典的項目分析（flexpyctt）: キーによる採点、α（ペアワイズとリストワイズ）、項目統計量、点双列・双列・多分系列相関、ω total（Pearson とポリコリック）、選択肢分析。シミュレーションによる 10 項目の多肢選択データ（欠測なし、5% MCAR）と 6 項目の 0〜4 のデータ（欠測なし、6% MCAR） | なし（mirt ではない）: psych `alpha()`、`score.multiple.choice()`、`responseFrequency()`、`polyserial()`、`biserial()`、`fa()`、`omega(nfactors = 1)`。ltm `cronbach.alpha()`、`biserial.cor()`。polycor `polyserial(ML = FALSE)`。lavaan `lavCor()`。選択肢の統計量は素の R | flexpyctt |
| V47 | 観測得点による DIF（flexpyctt、#102）: 連続修正の有無、Holm / BH、純化、利用者のアンカー、外部のマッチング変数を含む Mantel–Haenszel（`difR::verbal`、均一・不均一 DIF を入れたシミュレーションの 2PL）。3 群と名義カテゴリの一般化 MH（シミュレーションの段階反応）。Mantel 検定と Liu–Agresti のオッズ比。ロジスティック回帰（both / uniform / non-uniform、尤度比 / Wald、純化、アンカー、マッチング） | difR 6.1.0 の `difMH()`、`difGMH()`、`difLogistic()`、`difMantel.poly()`、`liu_agresti_ccor()`。`stats::mantelhaen.test()`。R での直接の実装: `packages/flexpyctt/validation/r/ctt_dif_reference.R`。Liu & Agresti（1996、表 1）の計算例 | — |
| V48 | mirt の推定値での図の数値（2PL $J=10$、$N=1003$。欠測 10% の 2PL。GRM $K=4$、$J=6$。2PL + 3PL + GPCM + 名義の混合 $J=10$。2 群の 2PL + GRM で焦点群の平均・分散を推定。TAM の GPCM $K=4$、$J=6$）: 経験的項目曲線、トレースライン、サーストン閾値 | `itemfit(fit_stats = "X2", return.tables = TRUE, group.bins = 10, mincell.X2 = 0, Theta = fscores(method = "EAP"))`（群ごとは `extract.group()`）、`probtrace()`、`probtrace()` に対する `uniroot()`、`coef(IRTpars = TRUE)`。TAM `tam.threshold()` | #103 |
| V49 | アンカーを純化する DIF のスクリーニング（他の全項目をアンカー、逐次的に自由化、逐次的に制約。2 群・2PL 8 項目 + GRM $K=3$ 4 項目、3 群 2PL）、Wald 検定による DIF、mirt の推定値での期待得点の符号つき・符号なしの差、DTF、ESSD / ETSSD、RMSD | `DIF(scheme = "drop" / "drop_sequential" / "add_sequential", seq_stat = 0.05, p.adjust = "BH", max_run = )`、`DIF(Wald = TRUE)`、`mirt:::calc_DRFs(best_fitting = TRUE)`（`DRF()` の内部）、`empirical_ES()`、`RMSD_DIF(probfun = FALSE)` | M5 |
| V50 | リンキングの共通項目の点検（#105）: 頑健 z（Huynh & Meyer, 2010）・変位・期待項目得点曲線の RMSD。除外の順序、純化の各段階のリンキング係数、最終の項目表（2PL、2PL + GRM + GPCM 混在） | 個別の `mirt()` 推定 + 素の R による再計算（`median()`、`quantile(type = 7)`、`expected.item()`、V20 のリンキング）。ドリフト項目の照合に `equateIRT::dif.test` | 現場向け |
| V51 | IRT 真値等化・観測得点等化と素点-尺度得点変換表（2 値: 20 + 20 項目、3PL + 2PL、リンクあり。多値: 4 GRM $K=4$ + 6 2PL と 3 GPCM $K=3$ + 2 GRM $K=4$ + 5 3PL + 3 2PL、リンクあり。母集団は新・旧・合成 $w_1=0.5$。変換表は真値と EAPsum） | `equateIRT:::score_compute(method = "TSE" / "OSE")`、`SNSequate::irt.eq(method = "TS")`。mirt の `expected.test()`・`probtrace()`・`testinfo()`・`fscores(method = "EAPsum", full.scores = FALSE)` と `uniroot()`、`equate::equate(type = "equipercentile")` | #106 |
| V52 | カット得点による分類の正確さと一貫性（#107）: Lee（合計得点のカット）と Rudner（$\theta$ のカット）、D 法と P 法、カット 1〜3 個、整数でないカット、両方向のカット尺度の変換（3PL $J=20$、GRM $K=4$・$J=10$、2 群の 2PL $J=15$・焦点群の平均と分散は自由） | cacIRT 1.4 `class.Lee()`、`Lee.poly.D()` / `Lee.poly.P()`、mirt の `testinfo()` の SE を与えた `Rud.D()` / `Rud.P()`。mirt の `probtrace()` / `expected.test()` / `testinfo()` からの独立な再計算（κ、偽陽性・偽陰性率、カテゴリごとの指標、表） | field |
| V53 | EAP・MAP・ML・WLE の推定値での受験者適合度（$l_z$、$l_z^*$、受験者の infit / outfit と $z$）と項目の infit / outfit。V37 のデータと推定値（2PL、8% 欠測の 2PL、3PL、GRM と GPCM $K=4$、名義 $K=4$、2PL + 3PL + GRM + GPCM + 名義の混在、2PL + GRM の 2 群）と Rasch での再推定 | `personfit(m, Theta = )`、`itemfit(m, fit_stats = "infit", Theta = )`、`sirt::personfit.stat()`、式から R で再計算した $l_z^*$ | — |
| V54 | モデル比較表（V11a のデータで Rasch・2PL・3PL、V8b のデータで PCM・GPCM、V4 のデータで全項目不変の 2 群 2PL とアンカーのみのモデル・最後の項目だけ自由なモデル）と回答データのシミュレーション（2PL・Rasch・3PL・4PL・GRM・GPCM・名義反応の 1 次元と 2 因子での決定的な逆変換、潜在分布ごとの分布と復元の確認） | `anova(mod1, mod2, ...)`。同じ $\theta$ と一様乱数での `probtrace()`（乱数列は `simdata()` と一致させられない） | #109 |

## mirt 側の設定と既知の差（M1 で判明）
- **M-step 最適化**: 参照値は `optimizer = "NR"` で生成する。mirt 既定の M-step 最適化（Rasch は `nlminb`、
  2PL は BFGS）は対数尤度の変化が小さくなると反復上限が縮み、M-step がほぼ動かなくなって `TOL` 判定が
  EM の収束前に通る（LSAT7 の Rasch で分散が約 2e-5 ずれ、`TOL` を変えても結果が変わらない）。
- **潜在分布の M-step**: mirt は分散（・平均）を閉形式のモーメントで更新する。これは離散化した周辺尤度
  （mirt 自身が報告する対数尤度）の最大化ではないため、Rasch の分散が flexpyirt（厳密な M-step, DESIGN §4.4）
  とわずかに異なる（シミュレーション N=2000 で分散差 7.9e-5、対数尤度は flexpyirt が 5.6e-7 高い）。
  閉形式更新で mirt の値が 1e-10 で再現できることを確認済み。許容誤差（1e-4）内なので照合は通常どおり行い、
  加えて「flexpyirt の対数尤度が mirt 以上」を検査する。
- **潜在分布パラメタの照合方針（決定）**: flexpyirt は厳密な M-step のみを持つ（DESIGN §4.4）。mirt との差は
  格子の打ち切りから生じるので、打ち切りが無視できない条件（V4 の焦点群など、平均が 0 から離れる・分散が
  2 を超える）では、**mirt・flexpyirt ともに格子を広げて（例: `theta_lim = c(-8, 8)`、点数も比例して増やす）**
  打ち切り差を許容誤差より十分小さくしてから照合する。既定格子（61 点・$[-6,6]$）での照合は、打ち切り差が
  1e-5 未満の条件に限る。いずれの場合も「flexpyirt の対数尤度が mirt 以上」を検査する。
- **`theta_lim` の渡し方（M2 で判明）**: `mirt()` / `multipleGroup()` では `technical = list(theta_lim = ...)` の中に
  入れる。トップレベル引数にすると黙って無視され、既定の $[-6,6]$ になる（`fscores()` はトップレベルで受け付ける）。
  参照値 JSON の `meta.grid_check` に実際に使われた格子の範囲を記録している。
- **対数尤度の比較**: mirt が報告する対数尤度は最終反復の E-step のもので、必ずしも最終推定値での値と一致しない。
  「flexpyirt ≥ mirt」の検査は、全パラメタを mirt の推定値に固定して flexpyirt で評価した対数尤度と比べる（M2 以降）。
- **M2 の照合結果**: V3（GRM）、V4（2 群・アンカー 5 項目・焦点群自由、81 点 $[-8,8]$）、V5（FIPC、同格子）、
  V9（15% MCAR）のいずれも、パラメタ・対数尤度・SE が mirt と ~1e-8 で一致。格子を両側で揃えれば、
  厳密な M-step と mirt の閉形式更新の差（打ち切り由来）は無視できることも確認された。
  照合テスト: `tests/test_mirt_m2.py`。参照値の生成: `validation/r/m2_reference.R`。
- **3PL / 4PL の漸近線の尺度と SE の変換（V11、M3 で判明）**: mirt は $g, u$ を logit 尺度で最適化し、
  `coef(m, printSE = TRUE)` も `logit(g)` / `logit(u)` と logit 尺度の SE を返す（`rawug` を変えても同じ）。
  flexpyirt は自然尺度で推定するため（DESIGN §15.2）、`validation/r/m3_reference.R` では推定値を `plogis()`、
  SE をデルタ法 $SE_g = g(1-g)\,SE_{\mathrm{logit}(g)}$（$u$ も同様）で自然尺度に変換して照合し、変換前の
  logit 尺度の値も `raw_logit` に残している。内点の最尤解では観測情報行列がヤコビアンで厳密に変換される。
- **経験ヒストグラム（V6、M3）**: mirt の `dentype = "EH"` は毎サイクルでヒストグラムを標準化せずに推定し直すので、
  全項目が自由だと尺度が識別されず EM が漂流する（自由推定の 2PL は 50,000 サイクルで収束せず、ヒストグラムの分散は 1.5 まで
  ずれた）。flexpyirt は代わりに参照群を標準化する（DESIGN §15.34）が mirt にはこの方式がないため、V6 は固定項目が尺度を
  決めてどちらも標準化しない FIPC を使う: V6a（$J=20$、項目 1–10 を固定、$N=3000$、標準化した $\chi^2_4$ の特性）と
  V6b（3000 人 × 2 群、両群で同じバンクを固定、項目 11–20 を群間等値、焦点群の特性は $0.5-1.2\,\chi^2_4$）、
  $[-6,6]$ 上の 61 点。細かい格子では EH の尤度は多くのヒストグラム方向でほぼ平坦で最大点は尖っているため EM は這うように
  進み（mirt は `TOL = 1e-9` で 69,683 / 62,632 サイクル）、mirt の停止規則はヒストグラムを見ない。そのため最終的な重みを
  節点ごとには比較しない。mirt の解において: 対数尤度は mirt と 1.5e-15 で一致、flexpyirt の EM を 1 ステップ進めると
  項目は 1e-9（mirt の `TOL`）、ヒストグラムは 4e-7 動く（mirt は E-step 1 回分遅れたヒストグラムを報告する）、
  Hessian と Oakes の SE（ヒストグラム固定）は mirt の Richardson SE と 1.1e-7 で一致。flexpyirt の通常の推定（SQUAREM、
  `tol = 1e-6`）は 7,162 / 2,837 反復で収束し、mirt より高い対数尤度（1.6e-3 / 7.3e-4 上）で終わり、項目は mirt と 5e-5
  以内。mirt 1.46.1 の EH の Oakes SE は自身の Richardson SE より 0.1–1.3% 小さく（原因は未確認）、2 群 EH モデルでは
  Richardson SE の計算が失敗するので、V6b は Oakes == Hessian（4e-16）を確認する。テスト: `tests/test_mirt_v6_eh.py`。
  参照値の生成: `validation/r/m3_eh_reference.R`。
- **V11 のシミュレーション設計と mirt `NR` の初期値問題（M3）**: 事前分布なしの 3PL / 4PL 最尤推定では、
  項目が易しすぎ／難しすぎたり傾きが小さかったりして $N(0,1)$ の裾で漸近線が観測できないと、有限標本でも
  境界解（$g\to 0$ や $u\to 1$、情報行列が特異）が生じやすい。さらに 4PL では、mirt 既定の傾き初期値
  （`a = 0.851`）から logit 尺度で `optimizer = "NR"` を回すと初期反復で `logit(g) ≈ -15` に飛び、
  logit 尺度の勾配 $g(1-g)\,\partial Q/\partial g$ が 0 付近で消えるため対数尤度が真に低い境界点に捕捉される。
  許容誤差を緩めずに内点の収束した大域最尤解を照合するため、V11a は $J=15, N=5000$、
  $a\sim\mathrm{lognormal}(0.4,0.2)$、$d\sim U(-1.5,0.5)$、$g\sim U(0.12,0.25)$ とし、V11b は
  $J=12, N=20000$（パターン数は最大 $2^{12}=4096$）、$a\sim\mathrm{lognormal}(0.6,0.15)$、
  $d\sim U(-0.5,0.5)$、$g\sim U(0.10,0.22)$、$u\sim U(0.78,0.90)$ として mirt 側の初期値のみ $a=2.0$ とした
  （flexpyirt は既定の初期値のまま大域解に収束）。両ケースとも `HeywoodWarning` なしで収束し、パラメタは
  ~1e-8（3PL: 5.4e-9、4PL: 7.2e-8）、対数尤度は ~1e-15、Oakes SE は ~1e-7（3PL: 1.1e-8、4PL: 1.9e-7）、
  Hessian 対 mirt Richardson SE は ~6e-4（3PL: 6.8e-5、4PL: 5.6e-4。Richardson の数値微分誤差によるもので、
  flexpyirt 内の Oakes と Hessian は ~1e-11 で一致）、EAP は ~1e-15 で一致した。
  照合テスト: `tests/test_mirt_m3.py`。参照値の生成: `validation/r/m3_reference.R`。
- **GPCM・PCM のパラメタ対応と照合結果（V8、M3）**: mirt の `itemtype = "gpcm"`（および多値データに対する
  `itemtype = "Rasch"`。これは `a1 = 1` 固定・`COV_11` 自由推定の PCM に相当）は、`a1`、`ak0..ak{K-1}`
  （`0..K-1` 固定）、`d0`（`0` 固定）、`d1..d{K-1}`（自由）の列を返す。`validation/r/m3_gpcm_reference.R` は
  これらの固定値を確認したうえで、`a1` → `a`、`d1..d{K-1}` → `d1..d{K-1}` と対応付け、`ak*` と `d0` を除いて
  保存する。照合ケースは 3 件: V8a（`Science` を `0..3` に変換、GPCM、61 点 $[-6,6]$）、V8b（シミュレーション
  GPCM、$J=10, K=4, N=3000$、ステップ順が逆転した項目を含む、61 点 $[-6,6]$）、V8c（生成時の潜在分散 $1.5$ の
  シミュレーション PCM、mirt `itemtype = "Rasch"`、V4/V5 と同じく格子の打ち切りを無視できる 81 点 $[-8,8]$）。
  3 件とも許容誤差より十分に小さく一致した: 項目パラメタは ~1.4e-8（V8c の潜在分散は 6.1e-10）、対数尤度は
  ~3.7e-15、Oakes SE は ~1.3e-8、Hessian 対 mirt Richardson SE は ~1.1e-6（flexpyirt 内の Oakes と
  Hessian は ~1.1e-11）、EAP は ~8.0e-15。
  照合テスト: `tests/test_mirt_m3_gpcm.py`。参照値の生成: `validation/r/m3_gpcm_reference.R`。
- **SQUAREM の照合結果（M3）**: すべての mirt 照合テスト（`tests/test_mirt_m1.py`, `tests/test_mirt_m2.py`,
  `tests/test_mirt_m3.py`, `tests/test_mirt_m3_gpcm.py`, `tests/test_mirt_m3_priors.py`）を
  `BAEM(squarem=False)` と `BAEM(squarem=True)` の両方で実行し、同じ加速なし参照値と照合している。
  全 16 ケースが同一の不動点に許容誤差内で収束し（2PL/GRM/GPCM/PCM/3PL/4PL/FIPC/2 群/欠測/MAP の項目パラメタ最大差
  $\le 2.8\times 10^{-8}$、V2 シミュレーション Rasch の潜在分散差は通常 EM と同じ打ち切り由来の $7.9\times 10^{-5}$）、
  対数尤度（MAP 時は対数事後）は単調非減少で、EM 写像の評価回数を 1.3〜5.0 倍削減した（例:
  `v2_lsat7_rasch` 121 → 24、`v4_twogroup_2pl` 302 → 113、`v8_science_gpcm` 251 → 96、
  `v11_4pl` 531 → 272、`v7_2pl_map` 80 → 35、`v7_3pl_map` 64 → 37）。単体・Hypothesis 単調性・安全装置・評価回数削減テスト:
  `tests/test_m3_squarem.py`。
- **事前分布・MAP と R mirt 1.46 の検証（V7、M3）**: `mirt:::LL.Priors`、`mirt:::DerivativePriors`、
  `mirt:::BL.LL` の調査および R での数値停留・Hessian 検証により、目的関数の一致と mirt 1.46 側の 3 つの実装上の不具合を
  特定した:
  1. **目的関数（`mirt:::LL.Priors`）— DESIGN §4.7 と完全一致**: `norm(mu, sd)` は `dnorm(x, mu, sd, log = TRUE)`、
     `lnorm(mu, sd)` は `dlnorm(x, mu, sd, log = TRUE)`、`expbeta(a, b)` は
     `dbeta(plogis(logit_g), a, b, log = TRUE)` を**自然尺度・logit ヤコビアンなし**で評価する。厳密 MAP 推定値に
     全パラメタを固定して mirt で評価すると、`mirt@Fit$logLik` と `mirt@Fit$logPrior` は flexpyirt と $\le 10^{-14}$
     で一致する。
  2. **mirt 1.46 の `expbeta` 微分バグ**: `mirt:::DerivativePriors`（および C++ `computeDPars`）は、自然尺度の Beta
     微分 $(a-1)/g - (b-1)/(1-g)$ と $-(a-1)/g^2 - (b-1)/(1-g)^2$ に連鎖律係数 $g(1-g)$ を掛けずにそのまま
     **logit 尺度**の勾配・Hessian に加算している。このため `expbeta` では `optimizer = "NR"` が収束せず、勾配ベースの最適化
     （`BFGS`, `L-BFGS-B`, `nlminb`）は `logLik + logPrior` が真に低い非停留点（V7c では差 `4.89`、`max |Δg| = 0.095`）
     で停止し、mirt の `SE.type = "Oakes"` も `g_se` を約 10 分の 1 に過小評価する。
  3. **mirt 1.46 の `lnorm` Hessian 符号バグ**: `lnorm` の勾配は正確（そのため `NR` は ~1e-8 で厳密な MAP 推定値に収束する）
     だが、`DerivativePriors` / `computeDPars` の 2 階微分第 3 項の符号が逆（`+(log(a)-mu)/(a^2*s^2)` のところが
     `-(log(a)-mu)/(a^2*s^2)`）になっており、mirt の `Oakes` SE は真の事後 Hessian SE から約 5% ずれる（符号反転式とは
     `8.3e-10` で一致することを確認）。一方 `norm` 事前分布は勾配・Hessian とも正確で、`Oakes` SE は真の事後 Hessian SE と
     `9.6e-10` で一致する。
  4. **`SE.type = "Richardson"` は事前分布を含めない**: `mirt:::BL.LL` は `LL.Priors` を呼ばず周辺対数尤度のみを計算する
     ため、mirt の Richardson SE は**尤度のみ**の情報行列 $-\nabla^2\ell(\hat\eta_{\text{MAP}})$ に一致する（~1e-8 で確認）。

  以上を踏まえ、3 件の V7 ケース（$J=15, N=1000$、61 点 $[-6,6]$）を `tests/test_mirt_m3_priors.py` で照合する
  （参照値生成: `validation/r/m3_priors_reference.R`）:
  - **V7a（`v7_2pl_map`）**: 2PL、$a\sim\mathrm{NormalPrior}(1.2, 0.5)$、$d\sim\mathrm{NormalPrior}(0, 2)$。mirt
    （`NR`, `SE.type = "Oakes"`）と直接照合し、`loglik` 相対差 `3.0e-13`、`log_posterior` 相対差 `5.1e-15`、
    パラメタ `8.0e-9`、Oakes SE 対 mirt Oakes `3.4e-9`、Oakes 対 Hessian `1.2e-12`、EAP `6.4e-15`
    （mirt Richardson は $\hat\eta_{\text{MAP}}$ での flexpyirt 尤度のみ Hessian と一致することを確認）。
  - **V7b（`v7_2pl_lnorm_map`）**: 2PL、$a\sim\mathrm{LogNormalPrior}(0.2, 0.3)$、$d\sim\mathrm{NormalPrior}(0, 2)$。
    `loglik`（`4.0e-13`）、`log_posterior`（`1.4e-14`）、パラメタ（`2.6e-9`）、EAP（`7.1e-15`）は mirt（`NR`）と直接照合し、
    Oakes / Hessian SE は R で求めた厳密な事後 Hessian と照合（`1.8e-9`、Oakes 対 Hessian `2.3e-12`）。
  - **V7c（`v7_3pl_map`）**: 3PL、$a\sim\mathrm{LogNormalPrior}(0.2, 0.3)$、$d\sim\mathrm{NormalPrior}(0, 2)$、
    $g\sim\mathrm{Beta}(5, 17)$（mirt の `expbeta`）。R 上で mirt の厳密な目的関数 `logLik + logPrior` を最大化し
    （`max |grad| = 5.1e-13`、`mirt(..., pars = sv_fixed)` で `8.9e-15` まで相互検証、EAP は mirt の `fscores`）、
    `loglik` 相対差 `2.5e-13`、`log_posterior` 相対差 `8.1e-16`、パラメタ `2.9e-9`、Oakes / Hessian SE `5.7e-9`
    （Oakes 対 Hessian `5.7e-12`）、EAP `6.0e-15` で一致した。
  単体テスト（scipy `log_pdf` 照合、自動微分ガード、固定パラメタの事前分布無視、共有 `free_id` の重複なし、表編集、AIC/BIC 警告）:
  `tests/test_m3_priors.py`。
- **表示用 IRT パラメタ化とデルタ法 SE（V12、M4）**: mirt 1.46.1 の
  `coef(m, IRTpars = TRUE, printSE = TRUE)` は傾き・切片パラメタを $b = -d/a$（2PL / Rasch）、
  $b_k = -d_k/a$（GRM / `graded`）、ステップ困難度 $b_k = -(d_k - d_{k-1})/a$（$d_0 = 0$、GPCM / PCM）
  へ変換しており、flexpyirt の `FitResult.items` の定義と完全に一致する。一方で
  `mirt:::mirt2traditional` の `cls == "gpcm"` 分岐を調べたところ、mirt 1.46.1 には **GPCM の `b1_se` のみ**
  に影響する 1 つずれ（off-by-one）添字バグがあることが判明した:
  - `delta_index[[nfact + 1L]]` が `c(which.a, ncat + nfact + 2L)`（`d0` を含む `opar` 中の `a1` と `d1` の位置）で
    上書きされるのに対し、`fns` 内で返す `ret` は `d0` を除いているため `b1` は `ncat + nfact + 1L`、`b2` は
    `ncat + nfact + 2L` に置かれる。
  - その結果 `ret[index[2]]` は `b1 = -d1/a` ではなく `b2 = -(d2 - d1)/a` を取り出して `(a, d1)` のみで微分
    （`d2` 固定）するため、真の勾配 $(d_1/a^2,\, -1/a)$ ではなく $((d_2 - d_1)/a^2,\, 1/a)$ を使ってしまう
    （R 上で mirt の生 `b1_se` がこのバグ式と `< 1e-9` で一致し、`Science` では最大 268% ずれることを確認）。
  - mirt の `gpcm` 出力における点推定値 `b1..b{K-1}` と `b2_se..b{K-1}_se` は正確である（`vcov(m)` からの厳密計算と
    `< 1e-9` で一致）。PCM（`a = 1` 固定）では `a` の項が 0 で $|\partial b_2/\partial d_1| = 1 = |\partial b_1/\partial d_1|$
    となるためバグが相殺し、`b1_se..b3_se` すべて正確になる。

  以上を踏まえ、6 件のケースを `tests/test_mirt_m4_irtpars.py` で通常 EM と SQUAREM の両方で照合する
  （参照値生成: `validation/r/m4_irtpars_reference.R` → `tests/reference/v12_irtpars.json`、
  `optimizer = "NR"`, `accelerate = "none"`, `TOL = 1e-10`, `SE.type = "Oakes"`）:
  - **V12a（`lsat7_2pl`、2PL、61 点 $[-6,6]$）**: `loglik` 相対差 `3.4e-16`、`max |Δb| = 1.4e-8`、
    `b_se` 相対差 `1.5e-8`。
  - **V12b（`lsat7_rasch`、Rasch、61 点 $[-6,6]$）**: `loglik` 相対差 `3.1e-15`、`max |Δb| = 1.1e-7`、
    `b_se`（`= d_se`）相対差 `6.9e-8`。
  - **V12c（`science_grm`、GRM 0..3、61 点 $[-6,6]$）**: `loglik` 相対差 `3.7e-15`、
    `max |Δbk| = 2.7e-8`、`bk_se` 相対差 `1.1e-8`。
  - **V12d（`science_gpcm`、GPCM 0..3、61 点 $[-6,6]$）**: `loglik` 相対差 `3.1e-15`、
    `max |Δbk| = 1.5e-9`、`b2_se` / `b3_se` は mirt `IRTpars` と直接照合で相対差 `9.7e-10`、`b1_se` は
    R の `vcov(m)` による厳密デルタ法 SE と照合で相対差 `9.3e-10`。
  - **V12e（`science_pcm`、PCM 0..3、81 点 $[-8,8]$）**: `loglik` 相対差 `8.4e-16`、
    `max |Δbk| = 2.3e-9`、`bk_se`（3 列とも mirt `IRTpars` と直接照合）相対差 `3.4e-10`。
  - **V12f（`mg_2pl`、2 群 2PL、アンカー 4 項目、焦点群自由、81 点 $[-8,8]$）**: `loglik` 相対差 `1.1e-15`、
    `max |Δb| = 7.3e-9`、`b_se` 相対差 `1.5e-8`。
  単体テスト（解析式照合、固定パラメタ、`se=None`、非正定値 `vcov`、部分等値、混在モデル、`summary()`、`_repr_html_()`）:
  `tests/test_m4_result.py`。
- **EAP の照合**: 推定値の差が EAP に伝播しないよう、全パラメタを mirt の推定値に固定して
  （ParameterTable の `free=False`）スコアリングだけを照合する。許容誤差 1e-5。
- **能力値スコアリング: MAP・ML・WLE（V13、M4）**: mirt 1.46.1 の `mirt:::fscores.internal`、`mirt:::MAP`、
  `mirt:::ML`、`mirt:::WLE`、`mirt:::MAP.mirt`、`mirt:::WLE.mirt` を調査し、以下を確認した:
  - **範囲（境界）**: `theta_lim` は EAP の求積格子（および MAP / ML / WLE の EAP 初期値）に使われ、
    MAP・ML・WLE 本体は `[-max_theta, max_theta]`（既定 `max_theta = 20`）上で最適化する。
  - **目的関数と SE（既定 `expected.info = FALSE`）**: MAP は連続な正規潜在分布を事前分布とする
    $\log L(\theta\mid y) + \log\phi(\theta;\mu_g,\sigma_g^2)$、ML は $\log L(\theta\mid y)$、WLE は
    $\log L(\theta\mid y) + \tfrac{1}{2}\log I(\theta)$（$I(\theta)$ は観測項目の Fisher テスト情報量。
    Warm の方程式 $\partial\log L/\partial\theta + I'(\theta)/(2I(\theta)) = 0$ および Firth の Jeffreys
    事前分布と一致）を最大化する。3 手法とも mirt 既定の SE は最適点における負の目的関数の `nlm()` Hessian
    の逆平方根（観測曲率。`response.pattern` なしで `expected.info = TRUE` にすると $I(\hat\theta)^{-1/2}$
    へ切り替わり、短いテストや極端パターンで差が出る。mirt 1.46.1 では `response.pattern` 経由で
    `expected.info` が引き継がれない）。
  - **極端パターンと mirt 1.46.1 の不具合**: ML では全観測項目が最低／最高カテゴリなら `-Inf` / `+Inf`
    （SE は `NA`）、全欠測は `NA` / `NA`。mirt 1.46.1 で 2 つの境界での挙動を確認した: (a) 3PL の低得点パターンで
    $\log L(\theta)$ の上限が $-\infty$ にある場合、`nlm` の有限差分 Hessian が $\le 0$（`SE = NA`）に
    なると `mirt:::ML` が `SEest > 30` 判定をすり抜けて `-Inf` ではなく有限値 `F1 ≈ -17`（`SE = NA`）を
    返す（`v11_3pl` の 3,790 パターン中 31 件。`mirt:::ML` で SE が NaN になる分岐が `SEest > 30` →
    `±Inf` の変換を飛ばすことをソースで確認）。(b) 欠測データでは、最終項目が欠測の極端パターンの一部が
    `±Inf` ではなく `max_theta = 20`（`F1 ≈ 20`, `SE = NA`）に張り付く（挙動は観測したが、原因とした
    `fscores.internal` の欠測数の数え方は mirt の R ソースで該当箇所を確認できなかったため、説明は未確認と
    する）。どちらも V13 の照合には使っていない。flexpyirt はこれらを正しく検出し
    `±inf` と SE `NaN` を返す。これとは別に、3PL の二峰性パターンで厳密に凹な内部の局所最頻値 $\hat\theta$
    （$\mathrm{SE} \le 30$）の対数尤度より下側漸近線 $\lim_{\theta\to-\infty}\log L(\theta)$ が厳密に大きい場合、
    `mirt:::ML` の EAP 出発の局所 `nlm()` 探索は内部の局所最頻値で停止して有限値 `(F1, SE_F1)` を返すのに対し、
    flexpyirt は $f(\hat\theta)$ を $f(\pm 20)$ と比較して真の大域上限である `(-inf, NaN)`（本来の最尤推定）を
    意図的に返す。`v11_3pl` の全 3,790 ユニークパターン中 **25 パターン** でこの差異が生じる（V13d のシード付き
    抽出には含まれておらず、`tests/test_mirt_m4_scoring.py` でパターン index 1285 を直接検証している）。全欠測行については、mirt が全手法で `NA` に落とすのに対し、flexpyirt では
    ベイズ手法（EAP・MAP・PV）は事前分布、尤度のみの手法（ML・WLE）は `(NaN, NaN)` を返す。

  全パラメタを mirt の推定値に固定して 6 ケースを `tests/test_mirt_m4_scoring.py` で照合した
  （参照値生成: `validation/r/m4_scoring_reference.R` → `tests/reference/v13_fscores.json`、
  `optimizer = "NR"`, `accelerate = "none"`, `TOL = 1e-10`。許容誤差は $\theta$ 絶対 `1e-4`・SE 相対 `1e-3`、
  `-inf`/`+inf`/`NaN` は厳密一致。残る ~`1e-6`–`2e-4` の差は `nlm` の有限差分勾配・Hessian 由来）:
  - **V13a（`lsat7_2pl`、2PL、32 パターン、61 点 $[-6,6]$）**: `loglik` 相対差 `5.1e-16`、
    MAP $\theta$ `2.2e-6` / SE 相対 `1.8e-5`、ML $\theta$ `1.5e-6` / SE 相対 `4.1e-5`（`±inf` 2 件）、
    WLE $\theta$ `2.1e-6` / SE 相対 `5.1e-5`。
  - **V13b（`science_grm`、GRM 0..3、90 パターン、61 点 $[-6,6]$）**: `loglik` 相対差 `3.7e-15`、
    MAP $\theta$ `3.0e-6` / SE 相対 `3.6e-5`、ML $\theta$ `2.5e-6` / SE 相対 `8.7e-5`（`±inf` 2 件）、
    WLE $\theta$ `2.8e-6` / SE 相対 `1.5e-4`。
  - **V13c（`science_gpcm`、GPCM 0..3、90 パターン、61 点 $[-6,6]$）**: `loglik` 相対差 `2.8e-15`、
    MAP $\theta$ `2.2e-6` / SE 相対 `4.1e-5`、ML $\theta$ `2.0e-6` / SE 相対 `8.8e-5`（`±inf` 2 件）、
    WLE $\theta$ `2.5e-6` / SE 相対 `1.2e-4`。
  - **V13d（`v11_3pl`、3PL、112 パターン、61 点 $[-6,6]$）**: `loglik` 相対差 `1.5e-15`、
    MAP $\theta$ `3.8e-6` / SE 相対 `7.4e-5`、ML $\theta$ `5.8e-6` / SE 相対 `2.1e-4`（低得点発散を含む `±inf` 20 件）、
    WLE $\theta$ `3.2e-6` / SE 相対 `1.9e-4`。
  - **V13e（`twogroup_2pl`、2 群 2PL、ref/focal 計 104 パターン、81 点 $[-8,8]$）**:
    `loglik` 相対差 `1.9e-15`、MAP $\theta$ `2.9e-6` / SE 相対 `5.1e-5`、ML $\theta$ `2.6e-6` / SE 相対 `8.7e-5`
    （`±inf` 4 件）、WLE $\theta$ `3.3e-6` / SE 相対 `1.6e-4`。
  - **V13f（`missing_2pl`、15% MCAR 欠測 2PL、88 パターン、61 点 $[-6,6]$）**:
    `loglik` 相対差 `7.4e-16`、MAP $\theta$ `2.0e-6` / SE 相対 `4.6e-5`、ML $\theta$ `1.4e-6` / SE 相対 `1.1e-4`
    （`±inf` 8 件）、WLE $\theta$ `2.0e-6` / SE 相対 `1.5e-4`。
  Plausible values（`method="PV"`。$D > 1$ は `tests/test_mirt_m4_mirt.py`、DESIGN §15.42）は `tests/test_m4_scoring.py` で統計的に検証する（同一 `key` の再現性、
  同一パターンの回答者間での独立性、標本平均・標準偏差がモンテカルロ誤差内で EAP と事後 SD に一致すること）。
  あわせて平坦事前分布 MAP $\approx$ ML、全 7 項目モデルでの WLE 停留条件、極端パターン、多母集団、欠測、入力検証を
  単体テストで確認している。
- **多母集団の 2 次元 MIRT（V21、M4）**: シミュレーション 2 ケース（1,500 人 × 2 群。アンカー q01–q03 / q07–q09 の単純構造と、
  q07 が交差負荷しアンカーが q01–q03 / q08–q10 のもの）。焦点群の平均・分散は自由、相関は両群で自由、$[-6, 6]$ 上の次元あたり
  81 点（`validation/r/m4_mirt_multigroup_reference.R` → `tests/reference/v21_mirt_multigroup.json`、mirt 1.48、`NR`、
  `TOL = 1e-10`）。mirt は 1 つの因子の尾根（その因子の焦点群の平均・分散とその傾きの交換）に沿って最大点の手前で止まる:
  flexpyirt は mirt の推定値での mirt の対数尤度を正確に再現するが、自身の解はそれより 3.2e-5 高く、mirt の推定値でのスコアは
  その因子の焦点群の分散 / 平均で約 0.035 / 0.018（他は 1e-3 未満）、flexpyirt の解では 1e-6 未満。この差は次元あたり 31 点から
  81 点にしても縮まらなかった（issue #20 の 13）。そのため、その因子のブロック（その因子の項目の傾き・切片と焦点群の平均・分散）
  以外のパラメタと SE は通常の許容誤差（`1e-4`、SE 相対 `1e-3`）で比べ、ブロックは対数尤度（flexpyirt の方が高い）と停留性で
  確認する。Oakes と直接 Hessian の一致、EM と SQUAREM の単調性、スコアリングと JSON の往復保存も確認する。テスト:
  `tests/test_mirt_m4_multigroup.py`。
- **S-X2 項目適合度（V15、M5）**: 既存の参照値を再利用した 6 ケース（Science の V3 GRM と V8 GPCM、潜在分散が自由な
  V8 PCM、V11 3PL、8 項目を固定した V5 FIPC 2PL、および `mincell = 3` の V3）。mirt は参照値の推定値で、各パラメタの
  `est` フラグを保ったまま（自由パラメタを自由度から引くため）`TOL = NaN` とし、推定と同じ格子で
  `itemfit(m, fit_stats = "S_X2")` を計算する（mirt 1.48、`validation/r/m5_itemfit_reference.R` →
  `tests/reference/v15_sx2.json`）。flexpyirt はデータを推定し直す: 自由度はすべてのケースで一致し、S-X2 は相対 `2.2e-8`、
  p 値は `1.3e-8` で一致（推定値自体の一致の水準）。単体テストで Lord–Wingersky 漸化式を全パターン列挙と比べ、期待度数の行が
  条件付き確率であることを確認する。テスト: `tests/test_mirt_m5_itemfit.py`。多母集団（DESIGN §15.45）: mirt の群ごとの
  `itemfit()` を、V4 の 2 群 2PL では mirt の推定値で（`multipleGroup(TOL = NaN)` でも EM が 1 ステップ進むので、すべての
  パラメタを固定して推定してから各 `est` フラグを戻す）、Science（GRM）を行の偶奇で 2 群に分けすべての項目を等値・焦点群の
  平均と分散を自由にしたものでも計算する（`validation/r/m5_itemfit_multigroup_reference.R` →
  `tests/reference/v15_sx2_multigroup.json`）。Science では mirt の項目パラメタが約 2e-4 収束しきらない（flexpyirt の
  対数尤度のほうが高い）ので、flexpyirt を mirt の推定値で評価する。両ケースの両群で自由度は一致し、S-X2 は相対 `1e-6`
  以内。テスト: `tests/test_mirt_m5_itemfit_multigroup.py`。
- **確認的多次元モデルの $M_2$ / $M_2^*$ と S-X2（V39、M6。DESIGN §16.19）**: V14 の 4 モデル（2D 2PL 単純構造、
  `q07` が交差負荷する 2D 2PL、2D GRM $K=4$、3D 2PL。自由な相関は 2D で 1 つ、3D で 3 つ）を、mirt 1.48 で V14 の
  推定値に置き、すべての `est` フラグを保ったまま `TOL = NaN` で作り直し、`M2()` と `itemfit(fit_stats = "S_X2")` を
  推定時の直積格子（$[-6,6]$ 上の $31^2$ と $15^3$ ノード。どちらの関数も推定時の格子を読まないので明示的に渡す）で
  評価した（`validation/r/m6_fit_multidim_reference.R` → `tests/reference/v39_fit_multidim.json`）。ケース、R スクリプト、
  参照ファイルが別なので、V15 / V18 の拡張ではなく新しい ID とした。テスト: `tests/test_mirt_m6_fit_multidim.py`
  （slow tier。2D 単純構造の S-X2 の比較と、拒否するモデル構造のテストだけ fast tier）と、単体テスト `tests/test_m6_fit_multidim.py`（直積格子上の
  S-X2 の期待表・$\pi_2$・$\Xi_2$ を全反応パターンの列挙と比較、$\Delta_2$ の相関の列を中心差分と比較）。
  - **S-X2**: 4 モデルのすべての項目（および `mincell = 3` の GRM）で自由度が一致。mirt の推定値では S-X2 が相対
    `2.7e-14`、p 値が `1.2e-14` で一致。既定の初期値から再推定すると S-X2 が `1.2e-7`、p 値が `5.0e-8`（推定値どうしが
    一致する水準で、3D のケースが最大）。`collapseCells` の移植はそのまま使っており、セルの併合と求積に違いは
    見つからなかった。
  - **$M_2$（mirt の流儀）**: mirt の `M2()` は $\Delta_2$ を作るときに `GroupPars` の `est` フラグを `FALSE` にする
    ので、自由な相関は $df$ では引かれるのに $\Delta_2$ に列を持たない（V18 の `rasch_free_var` と同じ）。
    `_latent_in_delta=False` にすると、flexpyirt は mirt の生の $M_2$ を、mirt の推定値で相対 `7.4e-13`、再推定で
    `1.9e-8` で再現する。$df$ は一致し、SRMSR は `1.5e-14` / `3.4e-8`、TLI / CFI（2 値の 3 ケース）は `4.5e-9` で
    一致する。R スクリプトでも mirt の部品（`dP`、`buildXi2els`）から mirt の値を計算し直し、`1e-10` で一致することを
    確認している。
  - **$M_2$（flexpyirt の既定）**（すべての自由パラメタを $\Delta_2$ に入れる）: 自由な相関ごとに解析的な列を 1 つ
    加えた R の厳密な計算（R の中で、正規化した事前分布の中心差分と照合）と比較し、mirt の推定値で `2.5e-13`、再推定で
    `5.3e-9`。既定の値は mirt の生の値より、相対で `4.7e-5`（2D 単純構造）、`2.8e-7`（2D 交差負荷）、`2.5e-4`
    （2D GRM）、`6.9e-4`（3D）だけ小さい。
  - **数値計算上の注意**: GRM では、項目得点の周辺モーメントのもとでの $\Delta_2$ は条件が悪い。項目の微分を中心差分
    （刻み $10^{-6}$、誤差 $\sim10^{-10}$）で求めると $M_2$ が相対 `4e-6` ずれるので、R の参照値では項目の列に mirt の
    解析的な `dP`（2PL と graded では正しい）を使い、flexpyirt は自動微分を使う。
  - **照合していない範囲**: 多母集団の多次元モデルの S-X2（V21 の構成。`SpecificationError` のまま。双因子・two-tier
    では両方の統計量が同じ。多母集団の $M_2$ は V42）、多次元モデルでの 3PL / GPCM / 名義項目（項目カーネルは V18 / V15 で照合済みで、単体テストは 2PL + GRM の
    混合モデルを扱うが、mirt との比較はない）、mirt の既定以外の格子。
- **多母集団・欠測データ・MAP 推定の $M_2^*$（V42、M6、#76。DESIGN §16.23）**: mirt 1.46.1 で推定した 6 つの
  シミュレーションケース（`multipleGroup()` / `mirt()`、`TOL = 1e-10`、$[-6,6]$ の 61 点、2 因子は次元ごとに 31 点。
  `validation/r/m6_m2_groups_missing_map_reference.R` → `tests/reference/v42_m2_groups_missing_map.json`）:
  `twogroup_2pl`（2PL、$J=6$、700 / 300、項目 1–3 を群間で等値、焦点群の平均と分散は自由）、`twogroup_2pl_missing`
  （同じ設計で 15% MCAR）、`twogroup_grm`（GRM $K=3$、$J=5$、500 / 500、全項目等値）、`twogroup_2d`（2 因子 2PL、
  $J=8$、600 / 400、全項目等値、焦点群の平均・分散・相関は自由）、`missing_2pl`（2PL、$J=7$、$N=800$、20% MCAR）、
  `map_2pl`（2PL、$J=6$、$N=250$、$a$ に `lnorm(0, 0.5)`、$d$ に `norm(0, 2)`）。mirt の `M2()` を生の参照値とする。
  R スクリプトは DESIGN §16.23 の式を R で独立に書いた `exact`（すべての自由パラメタについての中心差分の $\Delta$ で、
  群の行を $\sqrt{N_g/N}$ 倍、群ごとの mirt の `buildXi2els` を $N_gN_{ab}/(N_aN_b)$ で再尺度化、多母集団の独立モデル）も
  計算し、同じコードを mirt の流儀にしたものが mirt の生の $M_2$ を再現することを R の中で確かめる。テスト:
  `tests/test_mirt_m6_m2_groups.py`（slow 層）と `tests/test_m5_m2.py` の単体テスト。
  - **mirt の推定値で**、flexpyirt の既定の値は `exact` と、2 値のケースすべてで相対 `5.5e-10`、GRM で `3.8e-7`
    （V39 と同じく R の参照値の中心差分の誤差）以内で一致し、$df$ は同一、TLI / CFI は `1.1e-10`、群ごとの SRMSR は mirt の値と
    `7.8e-15` 以内。`_mirt_convention=True` と `_latent_in_delta=False` では mirt の生の $M_2$ を `1.8e-14`、TLI / CFI を
    `9.4e-8`（mirt は帰無モデルを EM で推定する）以内で再現する。
  - **flexpyirt で推定し直すと**、対数尤度（MAP では対数事後）は mirt と `6.1e-10`、$M_2$ は `exact` と `3.0e-5`
    以内（推定値の一致は約 `1e-5`）。
  - **mirt と厳密な統計量の差**: mirt の値は `twogroup_2pl` と `twogroup_2d` で `0.45%`・`0.37%`（群の行の重みなし、
    潜在分布の列なし）、`twogroup_grm` で `2.6%`（潜在分布の列）、`missing_2pl` と `twogroup_2pl_missing` で `5.8%`・
    `18.3%`（`N.ratio` の重み付け）大きく、`map_2pl` では同一（mirt も flexpyirt も事前分布を使わない）。
  - **シミュレーションによる較正**（mirt との比較ではない。作業用スクリプトで実行し、ここと slow 層の単体テスト
    `test_m2_with_mcar_missing_follows_chi2_at_true_parameters` に要約）: 2PL、$J=6$、$N=600$、真値で 1000 反復
    （$df=21$）。全項目 30% の MCAR では厳密な統計量の平均 `20.86 ± 0.21`、分散 `42.9`（42）、.05 での棄却率 5.3%
    （KS $p=0.84$）、mirt の重み付けでは平均 `19.12`（KS $p=10^{-12}$）。項目ごとに異なる MCAR 率（0, 0, 10, 40, 60, 70%）
    では `20.86 ± 0.21`、5.6%（KS $p=0.12$）に対し、mirt は `19.90`、分散 `59.6`（KS $p=10^{-14}$）。推定したパラメタでは
    （$df=9$、とくに断らなければ 400 反復）: 欠測なし `9.22 ± 0.22`
    （棄却率 6.0%、KS $p=0.63$）、30% MCAR `9.04 ± 0.20`、4.8%（KS $p=0.61$）で mirt の重み付けは `8.94`、異なる率では
    `9.19 ± 0.22`、6.0%（KS $p=0.64$）に対し mirt は `7.17`、1.5%（KS $p=5\times10^{-17}$）、異なる率で $N=2000$（300 反復）
    では `8.92 ± 0.24`、4.7%（KS $p=0.82$）に対し mirt は `6.81` で棄却なし（KS $p=4\times10^{-14}$）。全項目等値で焦点群の平均・分散が自由な 900 / 150 の 2 群（400 反復、$df=28$）: 平均
    `28.17`、棄却率 5.5%（KS $p=0.29$）。mirt の流儀では `26.72`（KS $p=0.004$）。
  - **照合していない範囲**: 多母集団の `DavidianCurve`（エラー）、多値の帰無モデル（TLI / CFI は `NaN` のまま。§16.3）、
    MCAR でない MAR（統計量に参照分布がない。文書に記載）。
- **確認的 2–3 次元 MIRT（V14, M4）**: 4 つのシミュレーションケースを `tests/test_mirt_m4_mirt.py` で照合した
  （通常 EM・SQUAREM の双方で通過。参照値生成: `validation/r/m4_mirt_reference.R` →
  `tests/reference/v14_mirt2d.json`、`accelerate = "none"`, `TOL = 1e-10`, `SE.type = "Oakes"`、
  求積格子は mirt の既定直積格子 $[-6, 6]$ 上で 2 次元 $31^2 = 961$ 点・3 次元 $15^3 = 3{,}375$ 点）。
  2 次元の 3 ケースは `optimizer = "NR"`、3 次元ケースは mirt の `NR` が 3,375 点上の 3 次元 `GroupPars` で
  数値 Hessian により極端に遅くなるため `optimizer = "BFGS"`（`control = list(reltol = 1e-13, maxit = 500L)`）を
  用いた。全 4 ケースで `flexpyirt.loglik >= fixed_at_mirt.loglik`、EM / SQUAREM の厳密な単調性
  （SQUAREM により EM 写像評価回数は 2.2–2.8 倍削減）、および Oakes と直接 Hessian の一致
  （相対 $\le 1.2\times 10^{-14}$）を確認した:
  - **V14a（`mirt2d_2pl_simple`、2 次元 2PL、$J=12, N=2000$、`F1=1-6`, `F2=7-12`, $\rho=0.5$、$31^2$ 点）**:
    `loglik` 相対 `2.1e-15`、項目パラメタ `3.3e-9`、`cor_12` `1.0e-8`、項目 Oakes SE 相対 `3.7e-9`、
    `cor_12_se` 相対 `1.1e-8`、Oakes 対 Hessian `7.5e-15`、EAP $\theta$ `7.3e-15` / SD `8.0e-15`。
  - **V14b（`mirt2d_2pl_cross`、交差負荷 `q07` つき 2 次元 2PL、$J=12, N=2500$, $\rho=0.45$、$31^2$ 点）**:
    `loglik` 相対 `1.8e-15`、項目パラメタ `4.0e-9`、`cor_12` `7.3e-9`、項目 Oakes SE 相対 `4.9e-9`、
    `cor_12_se` 相対 `2.9e-9`、Oakes 対 Hessian `5.6e-15`、EAP $\theta$ `6.9e-15` / SD `7.0e-15`。
  - **V14c（`mirt2d_grm`、2 次元 GRM $K=4$、$J=10, N=2000$、`F1=1-5`, `F2=6-10`, $\rho=0.55$、$31^2$ 点）**:
    `loglik` 相対 `1.7e-15`、項目パラメタ `2.0e-9`、`cor_12` `4.6e-9`、項目 Oakes SE 相対 `2.2e-9`、
    `cor_12_se` 相対 `3.0e-9`、Oakes 対 Hessian `4.4e-15`、EAP $\theta$ `8.0e-15` / SD `1.2e-14`。
  - **V14d（`mirt3d_2pl`、3 次元 2PL、$J=9, N=2000$、`F1=1-3`, `F2=4-6`, `F3=7-9`, $\rho=(0.45, 0.25, 0.50)$、$15^3$ 点）**:
    `loglik` 相対 `4.7e-15`、項目パラメタ `1.9e-8`、相関 `4.9e-8`、項目 Oakes SE 相対 `9.0e-7`、
    相関 Oakes SE 相対 `7.3e-5`（mirt の 3 次元有限差分 Oakes ステップ由来。flexpyirt の Oakes 対 Hessian は
    `1.2e-14`）、EAP $\theta$ `1.1e-14` / SD `1.1e-14`。
  - **MAP（4 ケースすべて）**: 全パラメタを mirt の V14 推定値に固定し、EAP のパターンと、それを 3 項目欠測にした
    20 パターン（ケースあたり 220 パターン。mirt 1.48、`validation/r/m4_mirt_map_reference.R` →
    `tests/reference/v14_mirt_map.json`）で `fscores(method = "MAP")` と比較: $\theta$ は `4.3e-6`、SE は相対 `3.8e-5`
    以内で一致（mirt の `nlm` の許容誤差の水準。許容誤差は `1e-4` / `1e-3`）。
  - **ML と WLE（4 ケースすべて）**: MAP と同じ設定に、全最低・全最高と 1 因子だけ極端なパターンを加えた（ケースあたり 223
    パターン。`validation/r/m4_mirt_mlwle_reference.R` → `tests/reference/v14_mirt_mlwle.json`）。WLE は 892 行すべてで一致
    （$\theta$ `5.0e-6`、SE 相対 `1.7e-4`）。ML は両者とも内点の 644 行で一致（$\theta$ `6.5e-6`、SE `1.2e-4`）。mirt が
    ある次元で `±Inf` を返す 242 行では flexpyirt も同じ符号の無限大を返し、mirt が発散する次元を ±20 近くで SE 0 として
    止める 22 行では flexpyirt は符号付きの無限大を返す（DESIGN §15.41）。
  単体テスト（カーネル・多変量正規重みと正定値性ガード・直積格子形状・2 次元 Rasch / 混在 / 事前分布 / 固定・
  Oakes == Hessian == 参照 Oakes・識別性と仕様エラー）: `tests/test_m4_mirt.py`。
- 照合テスト: `tests/test_mirt_m1.py`（V1, V2, V10）。参照値の生成: `validation/r/m1_reference.R`。
- **項目をまたぐ等値制約（V16、M5）**: mirt の `CONSTRAIN`（`optimizer = "NR"`、`TOL = 1e-10`）。V16a: LSAT7 の
  2PL で全項目共通の傾き（$N(0,1)$、$[-6,6]$ の 61 点）。V16b: シミュレーション $N=2500$、項目 1–6 が 2PL・7–10 が
  3 カテゴリの graded で、2PL と graded にまたがる共通傾き（1, 2, 7）、項目 3・4 の共通切片、8–10 の共通傾き、9・10 の
  共通 `d1`。V16c: 各 1500 人の 2 群、アンカー q01–q05 は群間等値、q06–q10 の傾きは各群の中で共通、焦点群の平均と分散を
  自由推定（$[-8,8]$ の 81 点）。いずれも mirt と一致（通常の EM・SQUAREM とも）: パラメタ $2.1\times10^{-8}$、
  対数尤度 $4\times10^{-11}$、Oakes SE（相対）$1.3\times10^{-8}$、Hessian と mirt Richardson SE $2.9\times10^{-8}$、
  flexpyirt の Oakes と Hessian $5\times10^{-15}$。自由パラメタ数は mirt の `nest` と一致。テスト:
  `tests/test_mirt_m5_equal_items.py`。参照値生成: `validation/r/m5_equal_items_reference.R`。
- **尤度比検定による DIF（V17、M5）**: V4 のデータで mirt の `DIF()`（`optimizer = "NR"`、`TOL = 1e-10`、基準モデルと
  すべての再推定で $[-8,8]$ の 81 点）。アンカーつきモデル（q01–q05 を等値、焦点群の平均と分散を自由）からの
  `scheme = "add"`（q06–q15 の `a1`+`d` と `d`）と、完全不変モデルからの `scheme = "drop"`（全項目の `a1`+`d`）。mirt は
  情報量規準を（どちらの方式でも）項目を自由にしたモデル − 制約したモデルで報告し、p 値は既定で Holm 補正
  （`p.adjust = "holm"`）。mirt の `p` は `1 - pchisq(...)` なので非常に大きな $X^2$ では厳密に 0（flexpyirt は生存関数）。
  一致（通常の EM・SQUAREM とも）: $X^2$ と AIC/SABIC/HQ/BIC の差 $2.2\times10^{-11}$、`p`・`adj_p` $3\times10^{-11}$。
  テスト: `tests/test_mirt_m5_dif.py`。参照値生成: `validation/r/m5_dif_reference.R`。
- **M2 / M2* 限定情報の適合度（V18、M5）**: $[-6,6]$ の 61 点上の 6 ケース（`validation/r/m5_m2_reference.R` →
  `tests/reference/v18_m2.json`、mirt 1.46.1）: `lsat7_2pl`（V1）、`sim_2pl`（V1）、`threepl`（V11 3PL）、
  `rasch_free_var`（V2 潜在分散自由の Rasch）、`grm_sim`（GRM $K=3, J=10, N=2000$）、`gpcm_sim`
  （GPCM $K=4, J=8, N=2000$）。mirt 1.46.1 の `M2()` との照合で、mirt 側の 4 つの相違・不具合を確認した（すべて R で検証済み）:
  1. **$\Delta_2$ から自由な潜在分布パラメタを除外**: mirt は `M2()` 内で `GroupPars` の `est` を `FALSE` に強制する一方、
     `df` からは引いている。flexpyirt は Maydeu-Olivares & Joe (2005, 2006) に従い既定ですべての自由パラメタを
     $\Delta_2$ に含める。内部の `_m2.m2_statistic(..., _latent_in_delta=False)` は mirt の `rasch_free_var` の値を再現する
     （mirt の推定値で `M2` 相対 `9.7e-16`、`SRMSR` `4.9e-17`、`TLI` `1.6e-9`）。既定（`True`）は潜在分布を含めた厳密な R 参照値と
     一致し（mirt の推定値で `M2` 相対 `1.0e-12`、再推定では $[-6,6]$ での $7.9\times10^{-5}$ の分散差により `9.5e-6`）、
     mirt の `M2` とは `0.035` 異なる。
  2. **3PL/4PL での `dP("dich")` の不具合**: `dich` の `mirt:::dP` は `(a1, d)` に $(u-g)\sigma(1-\sigma)$
     ではなく 2PL の微分 $P_0 P_1$ を使い、第 3・4 列にも $((1-\sigma)g(1-g),\, \sigma u(1-u))$ ではなく
     カテゴリ確率 $(P_0, P_1)$ をそのまま入れている（`cbind(Theta * PQ, PQ, P)`）。このため `threepl` の `M2` が
     4.8% ずれる（`64.82` 対 厳密値 `68.11`）。
  3. **GPCM の `dP("gpcm")` 傾き微分の不具合**: `gpcm` の `mirt:::dP` は `a1` の微分で `den = rowSums(num)` で割るのを
     忘れており（`rowSums(Theta[, i] * aknum[[i]])`）、`gpcm_sim` の `M2` が 10.8% ずれる（`5.27` 対 厳密値 `5.90`）。
  4. **多値帰無モデルのランク落ち**: $K_j > 2$ では帰無 $\Delta_2$ の項目 $j$ の $K_j - 1$ 本の切片列が厳密に線形従属
     （$\operatorname{rank}=J < \sum(K_j-1)$）になるが、mirt はランクを確認せず `qr.Q(qr(delta), complete = TRUE)` の
     `ncol(delta)` 列目以降を取り出すため、多値帰無 $M_2^*$ が項目順に依存する。flexpyirt は多値項目を含むとき `tli` と
     `cfi` を `NaN` とする（DESIGN §16.3）。

  一致（通常の EM・SQUAREM とも、全ケースで `df` は完全一致）:
  - **mirt と直接比較（`lsat7_2pl`, `sim_2pl`, `grm_sim`）**: `M2` 相対 $\le 3.8\times10^{-11}$、
    `p` $\le 7.8\times10^{-11}$、`RMSEA` $\le 4.4\times10^{-13}$、信頼区間端点 $\le 1.4\times10^{-8}$、
    `SRMSR` $\le 1.4\times10^{-12}$、2 値の `TLI` $\le 4.8\times10^{-10}$、`CFI` $\le 2.4\times10^{-10}$。
  - **厳密な R の $M_2^*$（`threepl`, `gpcm_sim`、`expected.item()` の中心差分微分 + mirt の `buildXi2els`）**:
    `M2` 相対 $\le 1.6\times10^{-9}$、`p` $\le 7.1\times10^{-10}$、`RMSEA` $\le 3.7\times10^{-11}$、
    信頼区間端点 $\le 6.0\times10^{-9}$、`threepl` の `TLI` `1.8e-10` / `CFI` `0`、`SRMSR` は mirt と直接比較で
    $\le 8.5\times10^{-13}$。
  テスト: `tests/test_mirt_m5_m2.py`（mirt 照合）、`tests/test_m5_m2.py`（単体テスト）。
- **Davidian 曲線潜在分布（V19、M5）**: `mirt` 1.46.1（`Theta_DC_den`, `Mstep.DC.optim`, `EM.group`,
  `LoadGroupPars`, `updatePrior`）および `dcurver` 0.9.3 の調査から、厳密な離散化 MMLE との 4 つの相違を確認した
  （DESIGN §16.4）:
  1. 組み込みの `mirt(..., dentype = "Davidian-k")` は毎 EM 反復で期待度数に Woods の線形補間
     `standardizeQuadrature` を適用してから未正規化の `dcurver::ddc` を `nlminb` で最適化するため、EM の単調性が
     崩れ（mirt は `"Davidian"` で単調性検査を無効化）、周辺対数尤度が真に低い点で終わる（V19a の 2PL $k=3$ で
     `21.47` 点、V19b の GRM $k=2$ で `139.32` 点低い）。V19a の終了点は mirt 自身の離散化尤度
     （$\sum\log\sum_q L_q\,\mathrm{ddc}(\theta_q)/\sum_q\mathrm{ddc}(\theta_q)$）に対しても停留点ではない: `PHI_1..3` の
     スコアは $(-0.54, -3.40, 56.7)$ で、mirt の項目パラメタのまま `PHI` だけを最大化すると `4.67` 改善する。V19b では
     mirt は停留点 $N(0,1)$（`PHI_1 = 1.5707963`、スコア $(3.6\times10^{-4}, 0)$、`PHI` の再最適化で改善なし）に
     停滞する: `PHI_2` のスコアは正弦優先座標の $\phi_1=\pi/2$ の極特異点のために 0 になり、`PHI_1` のスコアはその 1 次の
     方向が純粋な位置・尺度の変化で自由な項目パラメタに吸収されるために 0 になる（後者はどの座標系でも生じる。
     flexpyirt は摂動つき既定初期値で $\boldsymbol\phi=\mathbf 0$ を離れる。DESIGN §16.4）。
  2. 組み込みの `dentype = "Davidian-k"` は `SE = TRUE`（`"No standard error method currently supported
     for Davidian curves"`）と `MEAN_1` / `COV_11` の自由推定（`"Estimating Davidian mean-variance
     hyper-parameters is not currently supported"`）を禁止している（いずれも `mirt:::EM.group`）。
  3. `fscores()` は `use_dentype_estimate = TRUE` を明示しない限り $N(0,1)$ 事前分布を使う（`EAP` と plausible
     values のみ対応し `MAP` は拒否）。
  4. `createGroup()`（格子上で正規化した `dcurver::ddc` を呼び、`TOL = 1e-10`, `accelerate = "none"`,
     `SE.type = "Oakes"`）経由で厳密な標準化 Davidian MMLE を mirt 内で検証する際、`createGroup` 既定の
     `derivType = "Richardson"` は粗い 2 階差分幅（`mirt:::numerical_deriv` が `delta * 1000 = 0.01`、対角で
     `2*delta = 0.02` を用いる）により $k=3$ の Oakes SE を最大 25.6% 歪めるため `derivType = "central"`
     （`delta = 1e-5`）を用いた。また `mirt:::BL.LL` は `dentype == "custom"` で `pars2` ではなく `pars` を参照する
     （潜在密度が摂動されない）不具合があるため、Richardson Hessian SE は R の厳密な周辺対数尤度から直接算出した
     （`h = 1e-4`）。

  **V19 参照値が検証するもの。** `exact` ブロックは DESIGN §16.4 の標準化 Davidian MMLE を R で書き直したもので、
  座標系・閉形式モーメント $(\mu_Z,\sigma_Z^2)$・標準化 $z_q=\mu_Z+\sigma_Z(\theta_q-\mu)/\sigma$ は flexpyirt と同じ定義
  （R で別途記述）であり、密度値は `dcurver::ddc`、E-step・EM・Oakes SE・EAP は `mirt`（`createGroup()`）、Hessian SE は
  別途記述した R の周辺対数尤度による。したがってこのモデルに対する flexpyirt の推定機構（E-step・潜在 M-step・収束・
  Oakes/Hessian SE・スコアリング）と、`dcurver` に対する密度を検証する。閉形式モーメントは独立に検査する: R スクリプトは
  各推定解と組み込み解で $z\,\mathrm{ddc}(z)$ と $z^2\,\mathrm{ddc}(z)$ を `stats::integrate` で積分し（最大絶対誤差
  `1.2e-9`、`moments_check` に保存。誤差が `1e-8` を超えれば停止）、`tests/test_m5_davidian.py` は台形則で検査する。
  `mirt_builtin` ブロックは mirt のパラメタにおける密度・`Prior`・`logLik`・EAP を mirt 自身の Davidian 実装と照合する。
  **参照値の収束。** mirt の EM（BFGS M-step、`TOL = 1e-10`）は厳密な周辺対数尤度の勾配が最大 `1.7e-3`（V19c。V19a/b
  は `3.8e-5`・`5.8e-5`、`max_grad_r_mirt_em` に保存）の点で停止する。そこで R スクリプトは mirt の解を R の厳密な
  周辺対数尤度上の Newton 法（中心差分の勾配と Hessian）で max|grad| < `2e-5` まで磨き（各ケース 1 ステップ、最終
  `1.8e-6`〜`3.6e-6`、`max_grad_r`）、磨いた値で `TOL = NaN`（EM 反復なし）として `mirt` を再実行し、その点での
  Oakes SE・`logLik`・EAP を得る。この仕上げの前は V19c の項目・潜在パラメタの差は `8.5e-6`・`7.3e-6` で、flexpyirt の
  対数尤度が参照値より `1.8e-8` 高かった。すなわち残差は参照値側にあり、flexpyirt 側ではなかった。

  歪んだ標準化 $\chi^2_4$ 特性の 3 シミュレーションケース（$N = 3000$）を `tests/test_mirt_m5_davidian.py` で照合する
  （通常 EM・SQUAREM とも。参照値生成: `validation/r/m5_davidian_reference.R` → `tests/reference/v19_davidian_*.json`。
  値は通常 EM と SQUAREM の最大）:
  - **V19a（`v19_davidian_2pl_k3`、2PL、$J=20, k=3$、61 点 $[-6,6]$）**: 厳密 MMLE 対 R `mirt(customGroup)` / R
    Hessian: `loglik` 相対 `2.3e-16`、項目パラメタ `5.7e-8`、`phi_1..3` `1.3e-8`、Oakes SE 対 mirt Oakes `4.6e-6`
    （項目 `6.3e-7`）、Hessian SE 対 R Richardson `2.8e-6`（潜在 `3.8e-7`）、flexpyirt Oakes 対 Hessian `9.7e-16`、
    EAP `6.2e-15`（SQUAREM で EM 評価回数 `113 → 39`）。組み込み `mirt(..., dentype="Davidian-3")` のパラメタ
    （`mean = mu_z`, `var = var_z`）では `Prior`・`logLik`（相対 `1.4e-15`）・`fscores(..., use_dentype_estimate=TRUE)`
    （`8.2e-15`）が完全一致し、厳密 MMLE は対数尤度を `+21.47` 改善する。
  - **V19b（`v19_davidian_grm_k2`、GRM $K=4$、$J=10, k=2$、61 点 $[-6,6]$）**: 厳密 MMLE: `loglik` 相対 `1.9e-16`、
    項目パラメタ `4.9e-8`、`phi_1..2` `9.1e-9`、Oakes SE 対 mirt Oakes `3.6e-7`、Hessian SE 対 R Richardson `1.9e-6`、
    flexpyirt Oakes 対 Hessian `1.1e-15`、EAP `6.0e-15`（SQUAREM `111 → 33`）。組み込み `Davidian-2` のパラメタでは
    `logLik`（相対 `1.9e-16`）・EAP（`6.7e-15`）が完全一致し、厳密 MMLE は対数尤度を `+139.32` 改善する。
  - **V19c（`v19_davidian_fipc_k3`、FIPC 2PL、`q01–q10` 固定、$J=20, k=3$、`mean`/`var` 自由、移動・拡大した歪み特性
    $0.4 + 1.2\cdot\text{skewed}$、81 点 $[-8,8]$）**: 厳密 MMLE: `loglik` 相対 `1.6e-15`、項目パラメタ `7.5e-8`、
    潜在パラメタ（`mean, var, phi_1..3`）`1.4e-7`、Oakes SE 対 mirt Oakes `6.0e-6`、Hessian SE 対 R Richardson
    `1.8e-6`、flexpyirt Oakes 対 Hessian `2.9e-15`、EAP `5.3e-15`（SQUAREM `107 → 33`）。
  残る `~1e-6` の SE の差は mirt と R の有限差分微分（`delta = 1e-5`、`h = 1e-4`）によるもので、flexpyirt の SE は
  厳密な自動微分による。許容誤差は上の表のとおり（パラメタ `1e-4`、SE 相対 `1e-3`）。
  単体テスト（単位球面・モーメント・ゼロ角での $N(0,1)$ 一致・多項式零点の自動微分ガードと求積点上に零点を持つ推定・
  初期角の符号・`degree` 1 と 10 の推定・Rasch/FIPC・識別性と仕様エラー・スコアリング・JSON 往復と不正な
  `DavidianCurve` JSON）: `tests/test_m5_davidian.py`。
- **微分可能な IRT 尺度リンキング（V20、M5）**: `mirt` は特性曲線法リンキングを持たないため、
  `validation/r/m5_linking_reference.R`（→ `tests/reference/v20_linking.json`）では共通項目を含む 2 つの
  フォームをそれぞれ `mirt` で推定し（`optimizer = "NR"`、`accelerate = "none"`、`TOL = 1e-10`、$N(0, 1)$、
  $[-6, 6]$ の等間隔 61 点、`SE.type = "Oakes"`。基準群 $\theta\sim N(0, 1)$、新群 $\theta\sim N(0.5, 1.2^2)$、
  真値 $A = 1.2, B = 0.5$。V20e の新フォームは固定した $N(0.3, 1.5)$ の下で推定し、対称版の逆方向の項が非標準の
  新フォーム格子を使う）、`mirt::probtrace()` / `mirt::expected.item()` と論文の定義式
  （Haebara, 1980; Stocking & Lord, 1983; Kim & Lee, 2006）に基づく独立した素の R 実装（`optim(BFGS)` + `nlm`
  と中心有限差分 Newton 仕上げ、`max |grad| <= 1.9e-11`）で参照値 $(A, B)$、目的関数値、および Ogasawara (2001) の
  デルタ法 SE を算出した。Haebara 法は 2 値項目の両カテゴリを含む全カテゴリで和をとる（Kim & Lee, 2006）。
  6 ケース × 4 方式（Stocking–Lord・Haebara の非対称・対称）について次の 3 通りで検証した:
  (i) mirt の項目推定値をそのまま入力した直接比較（推定誤差とリンキング誤差を分離）、(ii) mirt 自身の共通項目推定値と
  Oakes 共分散ブロックからのデルタ法 SE（SE の式を推定・共分散の差から分離）、(iii) flexpyirt で両フォームを推定して
  リンキングする一連の比較。さらに全ケース・全方式で Newton 法を 4 つの遠い初期値
  （$(A, B) = (0.25, -3), (4, 3), (0.5, 2.5), (3, -2)$）から走らせ、1e-9 以内で参照解に達することを確かめる。
  外部の R リンキングパッケージ（`plink`、`equateIRT` など）は使っていない。目的関数と SE は同じ公刊式の独立実装と
  照合しており、第三者ソフトウェアとは照合していない:
  - **V20a（`v20a_2pl`、2PL、各フォーム $J=20$、共通 10 項目、各群 $N=2000$）**: 直接比較 `max |Δ(A, B)| = 9.6e-13`、
    目的関数相対 `1.6e-13`、mirt の共分散での SE 相対 `2.2e-8`。一連の比較 `max |Δ(A, B)| = 1.9e-10`、目的関数相対
    `2.6e-9`、Ogasawara SE 相対 `2.2e-8`。
  - **V20b（`v20b_grm`、GRM $K=4$、各フォーム $J=16$、共通 8 項目、各群 $N=2000$）**: 直接比較 `max |Δ(A, B)| = 9.1e-13`、
    目的関数相対 `1.0e-13`、mirt の共分散での SE 相対 `8.3e-9`。一連の比較 `max |Δ(A, B)| = 9.5e-10`、目的関数相対
    `2.2e-9`、Ogasawara SE 相対 `8.1e-9`。
  - **V20c（`v20c_gpcm`、GPCM $K=4$、各フォーム $J=16$、共通 8 項目、各群 $N=2000$）**: 直接比較 `max |Δ(A, B)| = 2.5e-12`、
    目的関数相対 `9.5e-14`、mirt の共分散での SE 相対 `1.6e-8`。一連の比較 `max |Δ(A, B)| = 1.4e-9`、目的関数相対
    `7.1e-10`、Ogasawara SE 相対 `1.6e-8`。
  - **V20d（`v20d_3pl`、3PL、各フォーム $J=16$、共通 8 項目、各群 $N=4000$、内点 $g$）**: 直接比較
    `max |Δ(A, B)| = 3.6e-12`、目的関数相対 `1.2e-13`、mirt の共分散での SE 相対 `9.4e-8`。一連の比較
    `max |Δ(A, B)| = 3.0e-10`、目的関数相対 `5.2e-9`、Ogasawara SE 相対 `9.4e-8`。
  - **V20e（`v20e_2pl_srcgrid`、V20a のデータ、新フォームは固定 $N(0.3, 1.5)$ の下で推定）**: 直接比較
    `max |Δ(A, B)| = 9.1e-13`、目的関数相対 `1.4e-13`、mirt の共分散での SE 相対 `3.4e-9`。一連の比較
    `max |Δ(A, B)| = 4.2e-10`、目的関数相対 `7.0e-10`、Ogasawara SE 相対 `3.4e-9`。
  - **V20f（`v20f_mixed`、2PL + GPCM $K=4$ 混在、各フォーム $J=16$、共通 5 + 3 項目、各群 $N=2000$）**: 直接比較
    `max |Δ(A, B)| = 2.0e-11`、目的関数相対 `6.2e-14`、mirt の共分散での SE 相対 `2.1e-8`。一連の比較
    `max |Δ(A, B)| = 1.3e-10`、目的関数相対 `3.6e-10`、Ogasawara SE 相対 `2.1e-8`。
  したがって V20 でリンキングを照合済みの項目モデルは 2PL・3PL・GRM・GPCM と 2PL + GPCM 混在である（4PL と名義は下の
  V40）。Rasch・PCM は同じカーネルを共有し、厳密回復の単体テストでのみ確認している。
  照合テスト: `tests/test_mirt_m5_linking.py`。単体テスト（全 7 種の 1 次元モデル、ICC・閾値の不変性、恒等変換と厳密回復、
  対称版の可逆性、独自格子、EH・多母集団、1 つの多母集団推定を `source` と `target` の両方に使うときの同時共分散、
  非収束時に SE を出さないこと、Rasch・PCM の傾き $1/A$、`transform_theta`、勾配の有限差分検証、Ogasawara SE の
  中心有限差分検証、および $N(\hat\eta, \Sigma)$ からのパラメタ抽出ごとに重み固定でリンキングを解き直すモンテカルロ
  線形化検証（推定のブートストラップではない））: `tests/test_m5_linking.py`。
- **4PL・名義反応モデル・多次元モデルのリンキング（V40、M6、#78）**: V20 と同じ方法による拡張。
  `validation/r/m5_linking_extensions_reference.R` は共通項目を持つ 2 つのフォームを `mirt` で推定し
  （`accelerate = "none"`、`TOL = 1e-10`、`SE.type = "Oakes"`、どちらの推定も潜在平均 0・分散 1）、リンキングを flexpyirt と
  独立に素の R で計算する。規準は推定に使った格子の上で `mirt::probtrace()` / `mirt::expected.item()` から計算し、
  `optim(BFGS)` + `nlm` と中心差分の Newton 仕上げで最小化する（`max |grad| <= 4.2e-11`、Hessian の正定値性を確認）。
  Ogasawara (2001) の共分散は、R で書き下したカテゴリ確率と入れ子の中心差分（刻み 1e-5）から求める。各ケースを
  Stocking–Lord・Haebara の非対称・対称の 4 通りで計算し、(i) mirt の項目推定値と mirt の Oakes 共分散ブロックをそのまま
  使った比較（コアの関数で直接、および mirt の推定値に固定した `FitResult` に `link()` をライブラリ自身の初期値で適用）、
  (ii) end-to-end（両フォームを flexpyirt で推定）で照合する。参照解はシミュレーションの真の変換とも比べ、デルタ法の
  標準誤差の 4 倍以内で再現することを確認する。
  **1 次元**（`tests/reference/v40_linking_unidim.json`、`tests/test_mirt_v40_linking.py`。$[-6, 6]$ の 61 節点、
  `optimizer = "NR"`。基準 $\theta\sim N(0,1)$、新フォーム $\theta\sim N(0.5, 1.2^2)$、真値 $A = 1.2$、$B = 0.5$）。
  ソルバーは $(1, 0)$、$(0.4, -2)$、$(3, 2.5)$ からも出発させる。
  - **V40a（`v40a_4pl`、4PL、各フォーム $J=12$、共通 8 項目、各群 $N=20000$、$g$ と $u$ は内点。mirt は V11b と同じく
    $a = 2$ から出発）**: 直接比較 `max |Δ(A, B)| = 5.5e-12`、規準の相対差 `2.1e-14`、共分散の相対差 `4.1e-7`。
    end-to-end `max |Δ(A, B)| = 1.2e-9`、規準の相対差 `6.8e-8`、共分散の相対差 `4.2e-7`（項目パラメタは mirt と `2.5e-8` で一致）。
    共分散ブロックは $g$ と $u$（自然尺度）を含む。
  - **V40b（`v40b_nominal`、名義 $K=4$、各フォーム $J=12$、共通 6 項目、各群 $N=3000$。推定されたスコアリング係数には
    負の値がある）**: 直接比較 `max |Δ(A, B)| = 1.2e-12`、規準の相対差 `5.5e-14`、共分散の相対差 `4.2e-7`。
    end-to-end `max |Δ(A, B)| = 3.3e-9`、規準の相対差 `1.3e-8`、共分散の相対差 `4.4e-7`。Stocking–Lord 法は
    `expected.item()` と同じくカテゴリを $0..K-1$ で採点する。
  - **V40c（`v40c_mixed`、2PL + 名義 $K=3$、各フォーム $J=15$、共通 6 + 3 項目、各群 $N=3000$）**: 直接比較
    `max |Δ(A, B)| = 1.9e-12`、規準の相対差 `2.6e-14`、共分散の相対差 `1.6e-7`。end-to-end `max |Δ(A, B)| = 2.2e-10`、
    規準の相対差 `5.7e-10`、共分散の相対差 `1.6e-7`。
  **多次元**（`tests/reference/v40_linking_multidim.json`、`tests/test_mirt_v40_linking_multidim.py`。DESIGN §19）。
  因子間相関が自由な確認的モデル。基準の母集団は標準、新フォームの母集団は因子の平均 $\boldsymbol\mu$・標準偏差
  $\boldsymbol\sigma$ なので、真の変換は $\mathbf A = \operatorname{diag}(\boldsymbol\sigma)$、
  $\boldsymbol\beta = \boldsymbol\mu$ である。各ケースを $\mathbf A$ の 3 つの族で解く（1 ケース 12 通り）。R の参照実装は
  回転を閉じた式（2 因子は余弦・正弦、3 因子は Rodrigues の公式）で書いており、flexpyirt の行列指数関数とは独立である。
  コアの比較は恒等変換から出発する。差は 4 つの規準のうち最大のもので、共分散の差は $(D^2+D)$ 次の参照行列の最大要素に
  対する相対値である。対角と斜交の族はシミュレーションの変換を標準誤差の 4 倍以内で再現する（拡大率が 1 つしかない
  直交の族はこの変換を含まない）。

  | ケース | 族 | 直接比較 `max \|Δ(A, β)\|` | 直接比較の共分散 | end-to-end `max \|Δ(A, β)\|` | end-to-end の共分散 |
  |---|---|---|---|---|---|
  | V40d（`v40d_2pl_2d`）: 2PL、2 因子、各フォーム $J=16$、共通 10 項目（2 項目は両因子に負荷）、各群 $N=3000$、各次元 31 節点、$\boldsymbol\mu=(0.4,-0.3)$、$\boldsymbol\sigma=(1.2,0.8)$ | 対角 | `1.4e-11` | `2.9e-7` | `4.0e-10` | `2.9e-7` |
  | | 直交 | `3.2e-11` | `8.2e-7` | `4.1e-10` | `8.2e-7` |
  | | 斜交 | `3.2e-11` | `4.8e-7` | `3.6e-10` | `4.8e-7` |
  | V40e（`v40e_grm_2d`）: GRM $K=3$、2 因子、各フォーム $J=12$、共通 8 項目（2 項目は両因子に負荷）、各群 $N=3000$、各次元 31 節点、$\boldsymbol\mu=(-0.3,0.5)$、$\boldsymbol\sigma=(0.85,1.25)$ | 対角 | `4.3e-12` | `1.7e-7` | `3.8e-10` | `1.7e-7` |
  | | 直交 | `1.5e-11` | `8.6e-7` | `4.1e-10` | `8.6e-7` |
  | | 斜交 | `2.9e-11` | `1.2e-7` | `4.1e-10` | `1.2e-7` |
  | V40f（`v40f_2pl_3d`）: 2PL、3 因子、各フォーム $J=15$、共通 12 項目（3 項目は 2 因子に負荷）、各群 $N=3000$、各次元 15 節点、mirt は `optimizer = "BFGS"`（`reltol = 1e-13`）、$\boldsymbol\mu=(0.3,-0.2,0.4)$、$\boldsymbol\sigma=(1.2,0.9,1.1)$ | 対角 | `2.8e-11` | `6.1e-7` | `1.5e-8` | `6.2e-7` |
  | | 直交 | `2.2e-10` | `1.6e-6` | `7.0e-9` | `1.5e-6` |
  | | 斜交 | `1.5e-10` | `1.4e-6` | `2.3e-8` | `1.3e-6` |

  規準の値は相対差で `7.9e-14`（直接比較）、`1.3e-7`（end-to-end）まで一致する。mirt の推定値に固定した `FitResult` への
  `link()` は、最小二乗の初期値から Newton 反復 3–6 回でコアの比較と同じ定数（R との差は最大 `2.2e-10`）を返す。flexpyirt で
  推定した項目パラメタは mirt と `5.1e-10`（2 因子）、`1.3e-7`（3 因子。mirt は BFGS）で一致する。
  **照合していない範囲**: 多次元の 3PL/4PL・GPCM・名義の共通項目（変換とソルバーは厳密な再現の単体テストだけで確認）。
  多母集団の多次元推定を `source` / `target` にする場合。鏡映（$\det\mathbf A < 0$。探索しない）。双因子・two-tier モデル
  （拒否する）。外部の R リンキングパッケージは使っていない。単体テスト（全カーネル・全族での曲線の不変性、3 つの族の
  閉じた式、1 因子の規準が §16.5 に一致、有限差分の勾配、2・3 因子での既知の変換の厳密な再現、リンキングを解き直して確かめる
  デルタ法のヤコビアン、`transform_theta`、エラー）: `tests/test_m6_linking_multidim.py`。
- **分類の正確さと一貫性（V52、#107）**: DESIGN §18.4 を `tests/test_r_classification.py`
  （低速層。生成スクリプト `validation/r/classification_reference.R` →
  `tests/reference/v52_classification.json`、mirt 1.48、cacIRT 1.4、$[-6, 6]$ 上 61 点）で照合する。
  各ケースのパラメタを全パラメタ固定の flexpyirt `FitResult` に入れ、P 法のために mirt の EAP を
  保存する。ケース: `threepl`（3PL、$J=20$、$N=3000$。この規模の 3PL の推定が収束しなかったので、
  mirt でも生成値に固定）、`grm`（GRM $K=4$、$J=10$、$N=2000$、mirt の推定値）、`twogroup`
  （2PL $J=15$、群ごとに $N=1500$、全項目不変、焦点群の平均と分散は自由）。実行: Lee は合計得点の
  カット 12、(8, 14)、(7.5, 13.2) と、$\tau$ で変換する $\theta$ のカット (−0.5, 0.8)。Rudner は
  $\theta$ のカット 0.3、(−0.5, 0.8) と、$\tau^{-1}$（R では `uniroot`）で変換する合計得点のカット
  (8, 14)。それぞれ D 法（推定時の求積点と重み）と P 法（mirt の EAP）。参照は 1 回の実行に 2 つ:
  - **cacIRT**（Lathrop, 2015）: 3PL は `class.Lee()`、GRM は `Lee.poly.D()` / `Lee.poly.P()`、
    `Rud.D()` / `Rud.P()` には mirt の `testinfo()` からの SE $1/\sqrt{I(\theta)}$ を与える。
    "Simultaneous"（全体）と "Marginal"（カットごと）の正確さ・一貫性、条件つきの値。`class.Rud()`
    は使わない: cacIRT の `iif()` は 3PL の情報量で Birnbaum の式の $(1-c)^2$ を $(1-c^2)$ と書いて
    おり、$c>0$ だと SE が誤る（2PL では正しい）。最大差: 厳密な関数とは `1.2e-15`（D）・`2.4e-15`
    （P）。`Lee.poly.D()` とは `3.9e-12`（cacIRT が漸化式の前にカテゴリ確率を
    $[10^{-4}, 1-10^{-4}]$ に切り詰めるため）。cacIRT の `class.Lee()` はカットに等しい真の得点を
    D 法では下（`cut(right = TRUE)`）、P 法では上に入れる。flexpyirt はどちらも上（同点になる
    ケースはない）。
  - **R での再計算**: mirt の `probtrace()`・`expected.test()`・`testinfo()` とスクリプト内で書いた
    Lord–Wingersky の漸化式から、表 $A$ と $B$、κ、カットごとの偽陽性・偽陰性率、カテゴリごとの
    指標（いずれも cacIRT にない）と条件つきの値。スクリプトは cacIRT の厳密な関数を $10^{-12}$ で
    再現することを確かめる。最大差: D `2.0e-14`、P `4.4e-15`。flexpyirt の EAP と mirt の EAP の差
    `7.8e-15`。許容誤差は $10^{-12}$（D、P、EAP）、`Lee.poly.D()` とは $10^{-10}$。
  - **R を使わない検査**（`tests/test_classification.py`）: 2PL + 3PL + GRM $K=3$ + GPCM $K=3$ の
    混在テストの全 72 反応パターンの総当たり（項目の確率は NumPy で書き下す。表、条件つきの値、κ。
    `1e-14`）、`test_information()` を使った `scipy.stats.norm` との Rudner の照合、指標間の恒等式
    （表の和が 1、観測側の周辺が等しい、カットごとに正確さ + 偽陽性 + 偽陰性 = 1）、
    `expected_score()` との尺度変換の照合、多母集団、エラー（高速層）。低速層: 平行測定 2 回の
    モンテカルロ（$N=200\,000$、$\theta\sim N(0,1)$）が、2401 点の格子での D 法と正確さ・一貫性・
    正確さの表について Lee と Rudner ともモンテカルロ SE の 4.5 倍以内で一致する。
  - **離散化**: 既定の 61 点の格子では、D 法の正確さは連続な潜在分布での値（2401 点）と最大
    `7.6e-3`（Lee、5 項目の混在テスト）・`2.5e-3`（Rudner）違う。一貫性と κ は `1e-9`。真の
    カテゴリが変わるところで条件つきの正確さが跳ぶためで（DESIGN §18.4）、実装の誤差ではない。
    同じ格子なら cacIRT も同じ値になる。
  - **照合していないもの**: `sirt::class.accuracy.rasch`（Rasch、$\theta$ 上のカット）は cacIRT が
    同じ指標を扱うので使っていない。`Normal` 以外の潜在分布（同じ重みのコード経路だが R とは
    比べていない）。

- **共通項目の点検（V50、#105）**: 頑健 z・変位・期待得点の RMSD を特性曲線リンキングと純化と組み合わせて実装した R パッケージは
  ないので、`validation/r/v50_screening_reference.R`（→ `tests/reference/v50_screening.json`、mirt 1.48、equateIRT 2.5.2）が
  DESIGN §22 の規則を素の R で計算し直す。各フォームを `mirt`（`optimizer = "NR"`、`accelerate = "none"`、`TOL = 1e-10`、
  $[-6, 6]$ 上 61 節点、潜在分布 $N(0, 1)$）で推定し、各アンカー集合のリンキングは V20 の素の R の Stocking–Lord / Haebara
  （`max |grad| < 1e-9` を確認）。位置、`log_ratio_a`、頑健 z（`median()`、`quantile(type = 7)`、`IQR()`）、変位、
  `target` の格子上の `expected.item()` 曲線の RMSD は式どおりに計算し、純化のループも同じ除外規則で書き下す。
  `tests/test_r_v50_screening.py` は mirt の推定値に固定した `FitResult` で `screen_common_items()` を実行し、除外の順序、
  各段階の $(A, B)$ と規準値、最終の項目表、フラグ、`ScreeningWarning` を比較する。`source` の母集団は `target` の尺度で
  $N(0.4, 1.2^2)$。ドリフトは `source` のフォームに仕込む。
  - **V50a（`v50a_2pl`、2PL、各フォーム共通 12 + 固有 8 項目、各フォーム $N=2000$。ドリフト: `c01` $b + 0.8$、
    `c02` $a \times 1.8$、`c03` $b - 0.6$）**: 7 通り（頑健 z の `removal="largest"` と `"all"`、変位、RMSD、対称 Haebara での
    変位、`max_iter=1` の頑健 z、`min_items=11` の RMSD 0.02）。除外の順序は完全に一致（頑健 z は `c02, c01, c03, c10`、変位と
    RMSD は `c01, c03`）。全段階で `max |Δ(A, B)| = 2.3e-11`、項目ごとの量 `5.0e-12`、頑健 z `2.7e-10`。
    `equateIRT::dif.test`（Stocking–Lord リンキング後の Lord の Wald 検定、純化あり、$\alpha = 0.05$）は `c01, c02, c03, c12`
    にフラグを立てる。仕込んだ 3 項目であり、頑健 z も最初にこの 3 項目を外す。
  - **V50b（`v50b_mixed`、2PL + GRM $K=4$ + GPCM $K=4$、共通 6 + 3 + 3 項目、各フォーム $N=2500$。ドリフト: `d01`
    $b + 0.7$、`g01` 閾値 $+0.6$、`p02` $a \times 0.5$）**: 4 通り（頑健 z、変位、RMSD、Haebara と `removal="all"` の RMSD）。
    除外の順序は完全に一致（頑健 z は `p02, d01, g01`、RMSD は `g01, p02, d01`、変位は `d01, g01`）。
    `max |Δ(A, B)| = 1.0e-11`、項目ごとの量 `4.8e-12`、頑健 z `3.4e-10`。
  許容誤差: $(A, B)$ `1e-9`、規準値は相対 `1e-7`、項目ごとの量 `1e-9`、頑健 z `1e-8`。**独立にのみ確認したもの**
  （`tests/test_screening.py` の単体テスト）: 4PL と名義項目の位置（すべてのカーネルで $b^* = A\,b + B$）、移した `source`
  曲線の直接評価との RMSD の一致、ドリフトのない項目での $(A, B)$ の厳密な回復、わずかに揺らした項目に仕込んだドリフトの除外、Rasch の傾きの比の頑健 z が
  NaN になること、`items=` による部分集合、`stack_screening_tables()`。閾値は慣行であり（DESIGN §22）、V50 が確かめるのは
  規則が仕様どおりに計算されることで、閾値のよさではない。

- **混合正規潜在分布（V22、M5）**: flexpyirt の `NormalMixture`（DESIGN §16.7）は形状に mirt のクラスのパラメタ化を
  使い、混合を解析的に標準化する。したがって格子 $\theta_q$ 上では、像の格子 $z_q=M+\sqrt V\theta_q$ 上の mirt の
  `multipleGroup(dat, 1, itemtype = ..., dentype = "mixture-k", invariance = c(<全項目>, "free_means", "free_var"))`
  （項目は 1 組、クラス 1 は $N(0,1)$ に固定、クラス $2..k$ の `MEAN_1`・`COV_11`・`PI` が自由）に厳密に一致し、
  $a=a^{\text{mirt}}\sqrt V$、$d_k=d_k^{\text{mirt}}+s_k a^{\text{mirt}}M$（2PL/GRM では $s_k=1$、GPCM では $k$）である。
  そこで `validation/r/m5_mixture_reference.R`（→ `tests/reference/v22_mixture_*.json`）はケースごとに次を記録する:
  1. `exact`: flexpyirt の離散化尤度を R で独立に書き直したもの（2PL・GRM・GPCM のカテゴリ確率、成分ごとに正規化した
     重み、標準化）を、$[-6,6]$ 上の mirt の解から Newton 法（中心差分の勾配、Richardson ヘッセ行列）で
     max|grad| < 2e-5（最終 1.8e-6–3.6e-6）まで最大化した点と、Richardson ヘッセ行列 SE、EAP。
  2. `mirt`: 厳密な最大点の像の格子（非対称な `theta_lim`。V22a では `c(-7.76, 9.50)`、`optimizer = "NR"`、
     `TOL = 1e-10`）で組み込みの mirt を推定し直し、推定値と `fscores()` の EAP を flexpyirt の尺度へ写したもの。R での
     mirt 尺度の尤度は mirt の `logLik` を 1e-12 で（mirt の推定値はその停留点で max|grad| <= 3.6e-6）、
     `fscores()` の EAP を 1e-8 で再現する。
  mirt 1.46.1 について分かったこと:
  - `dentype = "mixture-k"` は既定でクラスごとの項目パラメタを推定する（混合 IRT モデル）。項目共通のモデルには
    `invariance = <全項目>` が要る（`ESTIMATION` から無害な再利用の警告が出る）。`multipleGroup` の第 3 位置引数は
    `group` なので `itemtype` は名前付きで渡す必要がある。
  - **mirt の混合モデルの Oakes SE は観測情報の SE ではない。** mirt 自身の尺度と格子で、mirt の離散化尤度の R
    Richardson ヘッセ行列による潜在分布パラメタ（`MEAN_1`・`COV_11`・`PI`）の SE は V22a で (0.159, 0.139, 0.135) だが、
    `SE.type = "Oakes"` は (0.0315, 0.0048, 0.0401) で、3 ケースを通じた比は 0.014–0.34。`SE.type = "complete"` は
    (0.0262, 0.0042, 0.0380) で、1/sqrt(N pi_1 pi_2) = 0.0380 はクラス既知のときの `PI` の完全データ SE である。項目の SE
    も小さすぎる（V22a の q01 `a1`: 0.0539 対 0.0898）。混合モデルでは他の観測情報 SE の種類は拒否される。そこで SE の
    基準は R の Richardson ヘッセ行列とし、mirt の尺度での Oakes・complete・R ヘッセ行列の SE を証拠として保存
    （`mirt.se_*_mirt_scale`）、flexpyirt の潜在分布パラメタの SE が mirt の尺度での R ヘッセ行列の SE と一致し
    （素の形状の座標は共通）、mirt の Oakes SE がその 0.6 倍未満であることをテストで確認する。
  - 混合モデルでは `fscores(method = "MAP")` は拒否される（flexpyirt も MAP を拒否する）。EAP の列名は
    `Class_1`, `SE_F1`。
  - クラスで拡大した mirt の EM は遅い（生成値から `TOL = 1e-10` で 3,092–4,555 (61 points, V22a/b) and 24,961 (121 points, V22c) 回の EM サイクル）。mirt のパラメタ化では
    flexpyirt の（節点レベルの）EM も、V22a で 4 通り中 2 通り、V22b で 4 通り中 3 通りの初期形状から劣った点
    （対数尤度で 0.73–53.2 低い）に止まった。標準化すると、すべての初期値から通常の EM で 168–661 反復（SQUAREM で
    27–93 反復）で最大値に達した（DESIGN §16.7）。
  - **格子の解像度（V22c）**: 成分が細く（標準化後 $\sigma_c\approx0.26$）3 カテゴリの項目が 20 あると、事後分布の幅が
    $[-6,6]$ 上 61 点の間隔 0.2 と同程度になる。61 点での最大値は 121 点・241 点（互いに 1e-9 で一致）と対数尤度で
    5.7e-3 異なり、mirt の固定格子と flexpyirt の動く標準化格子の最大値は対数尤度で 2.6e-4、パラメタで最大 6e-3 離れた。
    そこで V22c は $[-6,6]$ 上 121 点を使う（mirt は不動点が同じ `accelerate = "squarem"` を使ったが、それでも 24,961 回の EM
    サイクルを要した）。
  結果（通常の EM と SQUAREM の最大値。flexpyirt は既定の初期形状から `BAEM(tol=1e-9)` で推定。成分のラベルに
  意味はないため標準化した成分は平均で並べて比較。SE の検査は mirt のラベル付けの初期値から推定（尤度が同じゲージの
  選択））:
  - **V22a（`v22_mixture_2pl_k2`、2PL、$J=20$、$N=3000$、$0.65N(0,1)+0.35N(2.5,0.5^2)$、61 点）**: `loglik` 相対
    `4.5e-16`（exact・mirt とも）、項目パラメタ `3.5e-8` / `4.3e-8`、標準化した成分 `5.4e-8` / `5.5e-8`、mirt の
    `MEAN_1, COV_11, PI` に対する素の形状 `1.2e-8`。R ヘッセ行列に対する項目 SE 相対 `1.0e-5`（潜在分布 `3.9e-6`）、
    Oakes 対 Hessian `3.6e-14`。exact のパラメタでの EAP `4.7e-15`、mirt に対し `4.5e-9`（SE `3.1e-9`）。通常の EM 734 反復、
    SQUAREM 54 反復。
  - **V22b（`v22_mixture_grm_k2`、GRM $K=4$、$J=12$、$N=2500$、$0.8N(0,1)+0.2N(-2.2,0.4^2)$、61 点）**: `loglik` 相対
    `8.8e-16`、項目パラメタ `3.3e-8` / `1.7e-9`、成分 `9.7e-9` / `7.8e-9`、mirt に対する素の形状 `9.3e-9`。R ヘッセ行列に
    対する項目 SE 相対 `1.1e-5`（潜在分布 `3.2e-5`）、Oakes 対 Hessian `5.0e-15`。EAP `4.9e-15`、mirt に対し `2.6e-9`。
    通常の EM 227 反復、SQUAREM 34 反復（flexpyirt の既定の初期値は、素の成分 1 が小さい成分になるラベルの付け替えた解に
    収束する。尤度と標準化した成分は同じ）。
  - **V22c（`v22_mixture_gpcm_k3`、GPCM $K=3$、$J=20$、$N=4000$、$0.5N(0,1)+0.3N(2.3,0.45^2)+0.2N(-2.3,0.5^2)$、
    121 点）**: `loglik` 相対 `2.7e-16`、項目パラメタ `2.6e-8 / 2.6e-8`、成分 `9.6e-8 / 9.4e-8`、mirt に対する素の形状 `2.9e-7`。
    R ヘッセ行列に対する項目 SE 相対 `3.2e-5`（潜在分布 `1.9e-4`）、Oakes 対 Hessian `1.8e-13`。EAP `4.9e-15`、
    mirt に対し `1.9e-8 (SE 1.4e-8)`。通常の EM 416 反復、SQUAREM 55 反復。
  許容誤差は上の表のとおり。単体テスト（仕様の検証、素の初期形状、厳密な標準化、等間隔・Gauss–Hermite 格子での重み、
  NaN の防御、EM と SQUAREM の単調性、Oakes == Hessian、ラベルを付け替えた初期値、デルタ法の成分、Rasch/FIPC の
  識別性、入れ子の Normal、退化解の警告、3 成分、採点、拒否するメソッド、リンキングの重み、JSON）:
  `tests/test_m5_mixture.py`。mirt との照合: `tests/test_mirt_m5_mixture.py`。
- **名義反応モデル（V23、M5）**: mirt の `itemtype = "nominal"`（`optimizer = "NR"`、`TOL = 1e-10`、$[-6,6]$ の
  61 点。V23f は次元あたり 31 点）。V23a: `0..3` に符号化し直した `Science`。V23b: シミュレーション
  $J=10, K=4, N=3000$ で、内部の採点係数が入れ替わった項目（`q09`: $\mathrm{ak}_1=2.2 > \mathrm{ak}_2=0.7$）と
  負の採点係数をもつ項目（`q10`: $\mathrm{ak}_1=-0.4$）を含む。V23c: 名義 5 項目（$K=4$）と 2PL 5 項目、$N=2500$。
  V23e: シミュレーション $J=8, K=4, N=3000$ で、$\mathrm{ak}_k<0$ や $\mathrm{ak}_k>K-1$ を含む非単調な項目が 6 つ。
  V23f: 2 因子の確認的名義モデル（$J=8$、因子あたり 4 項目、$K=4$、$N=3000$、相関は自由推定）。
  **初期値**: mirt の名義モデルの既定初期値（`ak1 = ... = 0.5`、`d = 0`）では減衰なしの NR が `Science` で鞍点に止まる
  （対数尤度 $-1631.79$、Comfort の `a1` $\approx 0$。最尤解は $-1608.452$）ため、スクリプトは mirt の NR を GPCM の点
  （$\mathrm{ak}_k=k$、$d_k=\log(p_k/p_0)$）から開始する。さらに 1 次元の各ケースで mirt の別の初期値 5 通り（既定初期値の
  NR と BFGS、`GenRandomPars` による NR 3 通り）で推定して `alt_starts` に保存し、いずれかの方が高ければそこから NR を
  やり直す（`meta.reference_start`）。これは V23e で起こり、GPCM の点からの NR は局所最大（$-29561.14$）に止まり、
  既定初期値の BFGS は $-26107.48$ に達する。保存したどの初期値も参照値を上回らず、flexpyirt は常に自身の既定初期値から
  推定してすべての参照値（V23e を含む）に到達する。**一致**（通常の EM・SQUAREM とも。V23a–c, e の最大）: 対数尤度
  $3.6\times10^{-11}$（絶対。相対 $1.6\times10^{-15}$）、パラメタ $7.1\times10^{-8}$、Oakes SE（相対）$3.8\times10^{-8}$、
  Hessian と mirt Richardson SE $5.6\times10^{-7}$、flexpyirt の Oakes と Hessian $1.7\times10^{-13}$、EAP
  $8\times10^{-15}$、各ケース 40–44 パタンの MAP / ML / WLE で $\theta$ $3.7\times10^{-6}$・SE（相対）$1.9\times10^{-4}$、
  S-X2（`itemfit(fit_stats = "S_X2")`）は相対 $5.8\times10^{-9}$ で自由度も一致（Science: $8.235, 10.223, 6.246,
  12.483$、自由度 6, 9, 6, 9）。V23f: 対数尤度 $4.0\times10^{-11}$、パラメタ $7.0\times10^{-9}$、Oakes SE
  $2.5\times10^{-9}$、相関 $5.4\times10^{-12}$（SE は相対 $9.3\times10^{-11}$）、EAP は $10^{-5}$ 以内。**ML の既知の差**:
  mirt の `fscores(method = "ML")` は採点係数にかかわらず全項目カテゴリ 0 / 全項目最上位カテゴリのパタンに
  $-\infty$ / $+\infty$ を返し、V23e の真に $+\infty$ のパタン 1 つ（全項目最上位カテゴリではない）を `max_theta = 20` の
  境界に残す（$19.99989$、SE は NA）。flexpyirt は極端パタンをカテゴリ傾き $a\,\mathrm{ak}_c$ で判定し、テストではその
  判定を R でのパタン対数尤度の独立な走査（`fscores.ml_scan`: $[-20,20]$ 上 0.01 刻みの格子を `optimize()` で精密化）と
  照合し、内点の ML 推定値（V23b・V23c で 1 つ、V23e で 2 つ）をそれと比較する。多次元の名義モデルは 2 因子（V23f）で
  照合済み、3 因子は単体テストのみ。単体テスト（確率・勾配・GPCM / 2PL への帰着・初期値・識別性・`fix_from`・JSON・
  S-X2・スコアリング・事前分布・等値制約・多群と DIF・EH・2 次元・退化項目の `HeywoodWarning`）:
  `tests/test_m5_nominal.py`。テスト: `tests/test_mirt_m5_nominal.py`。参照値生成:
  `validation/r/m5_nominal_reference.R`。
- **ポリコリック相関（V24、M5）**: mirt との照合ではない。順序変数に対する lavaan の `lavCor`（`nlminb` に解析的な
  勾配とヘッセ行列を与えた two-step 最尤法、$|\rho|\le0.999$、既定の `zero.add = c(0.5, 0)`）を厳密な参照値とする。
  lavaan が非収束の警告を出したらスクリプトを止める。psych の `polychoric` は既定の許容誤差の `optimize()` と
  `mnormt::sadmvn` を使うので、1e-3 までの照合とし、欠測のないデータで、空のセルを補正したペアを除いて比べる
  （`global = TRUE` の psych は補正しない）。カテゴリ数が異なると psych はペアごとの閾値を使い、欠測があると結果が
  ずれる（MCAR のケースで 6e-3）ため、そのケースは lavaan とだけ比べる。さらに各ペアの two-step の勾配が推定値で
  0 になることも確かめる。2 変量正規分布の分布関数は Plackett の積分と単体テストで比べる。
- **次元数（V25、M5）**: 決定的な量は V24 の lavaan の行列から R で計算するので、両側が同じ入力を見る。
  EFA.dimensions と EFAtools を両方読み込むと EFA.dimensions の `PARALLEL()` が EFAtools のものを隠すため、
  スクリプトでは `EFAtools::` を明示する。EFA.dimensions の 4 乗版 MAP は $(\operatorname{tr}(R^{*4})-p)/(p(p-1))$
  （行列のべき乗。このデータで 1 を超える）なので、4 乗版は要素ごとの独立な実装と比べ、EFA.dimensions の値は記録
  するだけにする。正規データの参照は両側とも確率的なので、テストは固定のキーを使い、平均は $z$（反復から求めた標準
  誤差、独立な 2 つの平均なので $\sqrt2$ 倍）で、95 パーセンタイルは絶対誤差で比べる。
- **局所依存とモデルに基づく平行分析（V26、M5）**: mirt の推定は `quadpts = 61`、`TOL = 1e-10`、
  `optimizer = "NR"`。flexpyirt は `BAEM(tol=1e-9)` で推定し直すので、比較には推定の一致（対数尤度は相対 $10^{-9}$）
  も含まれる。ltm は独自の `Science` と `LSAT` のデータ（因子型）を持つので、スクリプトでは mirt のデータを
  `mirt::` で指定する。ltm の `unidimTest` は推定済みの ltm オブジェクトではなくデータとパラメタで呼ぶので、
  標準正規の推定での flexpyirt と同じく $\theta\sim N(0,1)$ からシミュレーションする。テトラコリック相関は polycor
  （`optimise()` の許容誤差）によるので、観測された統計量の一致はそれで決まる。
- **単一群・1 次元・正規分布以外の局所依存（V26、M7、issue #81）**:
  `validation/r/m7_local_dependence_reference.R`（mirt 1.48）が
  `tests/reference/v26_local_dependence_extensions.json` を書く。`tests/test_mirt_m7_local_dependence.py` は
  mirt の推定値を、すべてのパラメタを固定した flexpyirt の `FitResult` に入れる（`fit(..., table=)`。EH の重みは
  mirt の `Internals$Prior`）。比較するのは統計量だけになる（mirt の推定値での対数尤度は相対 $10^{-9}$ で一致、
  実測 $4\times10^{-15}$ 以下）。9 ケース、群とペアの組は 641。実測の最大差: $X^2$ $3\times10^{-12}$、$G^2$
  $8\times10^{-9}$（相対。許容誤差 $10^{-7}$）、$V$ $3\times10^{-15}$、Q3 $2\times10^{-15}$（絶対。許容誤差
  $10^{-8}$）。

  | ケース | mirt の呼び出し | 格子 | ペア |
  |---|---|---|---|
  | 2 群、2PL、4% MCAR | `multipleGroup(invariance = c(anchors, "free_means", "free_var"))` | 121 | 2 x 45 |
  | 2 群、GRM（4 カテゴリ） | 同じ | 121 | 2 x 15 |
  | 2 群、2 因子、2PL | 同じ（`mirt.model`） | $61^2$ | 2 x 45 |
  | EH、2PL、歪んだ特性値、3% MCAR | `mirt(dentype = "EH")`、`use_dentype_estimate = TRUE` | 61 | 66 |
  | 2 次の Davidian 曲線、同じデータ | `mirt(dentype = "Davidian-2")`、同じ | 61 | 66 |
  | 正規混合分布、V22a の推定値 | 混合分布の重みを `Internals$Prior` に入れる | 61 | 190 |
  | 2 因子、2PL、交差負荷、3% MCAR | `mirt(mirt.model(...))` | $31^2$ | 45 |
  | 2 因子、GRM（3 カテゴリ） | 同じ | $31^2$ | 28 |
  | 3 因子、2PL | 同じ | $15^3$ | 36 |

  mirt の既定との違い（すべてスクリプトで対処している）: (1) EH と Davidian の推定に対して、`residuals()` は
  `use_dentype_estimate = TRUE` を付けないと $N(0,1)$ の事前分布を使う。付けた場合、`type = "Q3"` は残差を
  格子点で評価してしまう（mirt 1.48）ので、Q3 は `Theta = fscores(m, use_dentype_estimate = TRUE)` で求める。
  (2) Q3 のスコアは `fscores()` から来る。その格子は推定の格子によらず 1 / 2 / 3 因子で各次元 121 / 61 / 31 点。
  単一群のケースは `Theta = fscores(m, quadpts = <推定の格子>)` を渡す。`Theta` を群ごとに渡せない
  `multipleGroup` のケースは `fscores()` の格子で推定する。mirt の既定の呼び出しは `q3_default` として保存した。
  推定した格子での flexpyirt との差は、$31^2$ 対 $61^2$ で $2\times10^{-9}$、$15^3$ 対 $31^3$ で
  $1.3\times10^{-4}$（M5 のケースの $4\times10^{-8}$ とも整合する。61 点対 121 点）。
  (3) mirt の混合分布のクラスには `residuals()` がない。混合分布のケースは密度を与えたときの統計量を確かめる
  もので、mirt の混合分布の推定との比較ではない（推定は V22）。(4) ある群で観測されないカテゴリがあると、mirt の
  `table()` は $K_i\times K_j$ より小さくなる。flexpyirt は表の大きさを保つ。これは単体テストだけで確かめている。
  エンドツーエンドの確認として、2 群の 2PL を flexpyirt で推定し直し、mirt（`TOL = 1e-6` で推定）と
  $4\times10^{-4}$（$X^2$、$G^2$）、$5\times10^{-6}$（$V$、Q3）で一致する。mirt は群をアルファベット順に
  並べるので、$N(0,1)$ の参照群は `focal` になる。テストも同じ群を標準化する。もう一方の群を標準化しても
  モデルは同じだが、離散化は同じにならない（格子は $\pm6$ で切れ、自由な群の SD は 1.34）。対数尤度は
  $1.9\times10^{-3}$、$X^2$ は $3\times10^{-3}$ ずれる。**照合しておらず、実装していないもの**: 単一群・
  1 次元・正規分布以外の `jsi()`（mirt は 1 因子のみ、Davidian 曲線の SE なし、混合分布は非対応、`multipleGroup`
  では群ごとに単一群の $N(0,1)$ で推定し直す別の統計量）。双因子・two-tier モデルは V38。
- **双因子・two-tier モデルの局所依存（V38、M7、issue #81）**:
  `validation/r/m7_local_dependence_bifactor_reference.R`（mirt 1.48）が
  `tests/reference/v38_local_dependence_bifactor.json` を書く。
  `tests/test_mirt_m7_local_dependence_bifactor.py`（fast tier）は、V26（M7）と同じく mirt の推定値を
  すべてのパラメタを固定した flexpyirt の `FitResult` に入れるので、比較するのは統計量だけになる（mirt の推定値での
  対数尤度は相対 $10^{-9}$ で一致、実測 $\le8\times10^{-16}$）。mirt の `residuals(type = "LD")` は適合時の
  `quadpts` による完全な直積格子 `thetaComb(theta, nfact)` の上で和をとり、flexpyirt は縮約した格子の上で和をとる
  （DESIGN.md §17.3）。両者は同じ和の並べ替えである。

  | ケース | mirt の呼び出し | 次元あたりの点数（mirt の格子） | ペア数 | $X^2$ 相対 | $G^2$ 相対 | $V$ 絶対 | Q3 絶対 |
  |---|---|---|---|---|---|---|---|
  | 双因子 2PL、特殊因子 3、一般因子のみの項目 2、欠測 3% | `bfactor(data, c(1,1,1,2,2,2,3,3,3,NA,NA))` | 21（$21^4$） | 55 | $5\times10^{-13}$ | $2\times10^{-11}$ | $2\times10^{-15}$ | $9\times10^{-16}$ |
  | 双因子 GRM（3 カテゴリ）、特殊因子 2 | `bfactor(..., itemtype = "graded")` | 21（$21^3$） | 28 | $2\times10^{-14}$ | $7\times10^{-13}$ | $3\times10^{-16}$ | $1\times10^{-15}$ |
  | two-tier 2PL、相関する主因子 2（交差負荷 1 項目）、特殊因子 2 | `bfactor(..., model2 = mirt.model("F1 = 1-5\nF2 = 5-10\nCOV = F1*F2"))` | 15（$15^4$） | 45 | $1\times10^{-11}$ | $2\times10^{-8}$ | $8\times10^{-15}$ | $3\times10^{-15}$ |

  許容誤差は相対 $10^{-7}$（$X^2$、$G^2$）、絶対 $10^{-8}$（$V$、Q3）。Q3 は
  `Theta = fscores(m, quadpts = <適合時の値>)` で求める。mirt の既定の呼び出しは `fscores()` 独自の格子で
  スコアを出し、`q3_default` として保存している（適合時の格子を使う flexpyirt との差は 3 ケースで
  $2\times10^{-6}$、$1\times10^{-8}$、$2\times10^{-4}$）。1 つめのケースでは、3 項目の特殊因子のうち 2 項目に
  余分な共有分散を入れてシミュレーションしたが、特殊因子がそれを吸収し、mirt でも flexpyirt でもどのペアも
  検出されない。単体テスト（`tests/test_m7_local_dependence_reduced.py`）は、因子分解した表を直積格子の総当たりの和と
  比較し（$7^4$ ノード。GRM 項目と欠測を含む双因子と two-tier。$X^2$ は相対 $10^{-10}$）、Q3 を `score()` の EAP での
  定義と比較し、`specific` に入れなかったテストレットが検出されることを確認する。**照合していないもの**: flexpyirt が
  推定した値はここでは比較しない（適合は V32 と V33）。2PL と GRM 以外の項目モデル。特殊因子が 4 つ以上の場合。
  Gauss–Hermite 格子（mirt の `residuals()` は等間隔）。**実装していないもの**: これらのモデルの `jsi()`
  （1 項目が主因子と特殊因子の傾きを持ち、mirt の `type = "JSI"` は 1 因子を要求する）。多母集団（双因子・two-tier
  モデルは単一群のみ）。
- **DETECT（V27、M5）**: sirt の `conf.detect()` は `smooth = FALSE` で呼ぶ。既定の `smooth = TRUE` で
  `use_sum_score = TRUE` だと、標準化した総得点から生の項目得点を引くので、残りの得点の群が誤る（DESIGN §16.13）。
  `expl.detect()` は自分で `smooth = FALSE` にする。その戻り値の `detect.unweighted` は検証用の半分の指標だけを
  持つので、推定用の半分の指標はスクリプトで推定用の行に `conf.detect()` を当てて計算し直す。
- **EFA と Hull（V28、M5）**: 回転しない負荷量は、含意する $\Lambda\Lambda^\top$（列の符号と等しい固有値の順に
  よらない）で比べる。`HULL()` の上限 $J$ は乱数を使う平行分析から決まるので、スクリプトは `settings$n_fac_max` を
  記録し、テストはそれを `max_factors` として渡す。EFAtools の `optim()` による解は flexpyirt より収束が粗いので、
  テストでは EFAtools の $\psi=1-h^2$ での目的関数値を flexpyirt の値が上回らないことも確かめる。残る差は Heywood の解
  （独自性が境界にある）で最も大きく、Hull は真の因子数を超える $k$ でそれを含む。
- **比較データ法（V29、M5）**: CD は両側とも乱数を使う。生成器は、EFAtools の乱数を R で再現し（`set.seed` の
  あと、生成器と同じ順に `sample()` / `rnorm()` を呼ぶ）、書き出して flexpyirt に与えることで厳密に確かめる。負荷量の
  符号が生成器で決まる 1 因子では厳密に一致する。選択は答えがはっきりしたデータで確かめ、平均 RMSE は各側が別の
  母集団を引くので緩く比べる（採用された因子数では 15%。棄却された追加の因子では 60%。こちらは 1 つの母集団による
  ばらつきがずっと大きく、EFAtools の値 0.048 に対して 6 つのシードで 0.044〜0.072 になる）。JSON の `auto_unbox` は要素が 1 つの $p$ 値のベクトルをスカラーにするので、スクリプトは
  `as.list()` で書く。
- **DIMTEST（V30、M5）**: 参照値との照合ではなくシミュレーション研究で受け入れる（表は DESIGN §16.16）。200 反復で
  .05 での第 1 種の誤りは 0.5〜1.0%（保守的。ブートストラップは偏りを除くが、帰無仮説の下での $T$ の SD は 0.72 程度）、
  検出力は 0.99〜1.00。`uv run python packages/flexpyctt/validation/simulation/dimtest_study.py`（約 4 分）で再実行できる。テストは帰無
  20 反復と対立 10 反復を実行する。
- **双因子モデル（V32、M7）**: シミュレーション 2 例を `tests/test_mirt_m7_bifactor.py` で照合する（通常の EM と
  SQUAREM。生成スクリプト `validation/r/m7_bifactor_reference.R` → `tests/reference/v32_bifactor.json`、mirt 1.46.1 の
  `bfactor()`、`optimizer = "NR"`、`accelerate = "none"`、`TOL = 1e-10`、`SE.type = "Oakes"`、$[-6, 6]$ 上で次元あたり
  31 点、すなわち特殊因子ごとに $31^2$ 格子 1 つ）。mirt の `fscores()` には双因子用の処理がなく 4 次元の直積格子全体で
  積分するが、これは同じ 1 次元格子上の分解した事後分布と一致する。そこで EAP は全パラメタを mirt の推定値に固定し、
  次元あたり 15 点（mirt では $15^4$ 節点）で比べる。どちらの例でも mirt の推定値での縮約対数尤度は mirt の値と相対
  $10^{-15}$ で一致し、EM と SQUAREM は単調で、Oakes は Hessian と $3\times10^{-16}$（共分散の要素）で一致する:
  - **V32a（`bifactor_2pl`、2PL、$J=15, N=2000$、`s1 = q01-q04`、`s2 = q05-q08`、`s3 = q09-q12`、`q13-q15` は
    一般因子のみ）**: `loglik` 相対 `2.2e-16`、項目パラメタ `1.9e-7`（EM、498 反復。SQUAREM は `5.3e-9`、80 反復）、
    Oakes SE 相対 `3.5e-7`（SQUAREM `8.9e-9`）、EAP $\theta$ `4.9e-14` / SD `8.7e-15`。
  - **V32b（`bifactor_grm`、GRM $K=4$、$J=12, N=2000$、4 項目のテストレット 3 つ）**: `loglik` 相対 `8.9e-16`、項目
    パラメタ `1.9e-8`（EM、188 反復。SQUAREM は `2.5e-9`、29 反復）、Oakes SE 相対 `2.5e-8`（SQUAREM `3.7e-9`）、
    EAP $\theta$ `4.9e-14` / SD `7.9e-15`。
  - 単体テスト（`tests/test_m7_bifactor.py`）は、縮約した尤度・期待度数・EAP を $(1+S)$ 次元の直積格子全体での
    総当たり積分と比べ（欠測、GRM 項目、3PL 項目、一般因子のみの項目を含めて $10^{-12}$ 以内）、特殊因子 1 つの
    双因子モデルを検証済みの 2 次元の経路（`with_factors(..., correlated=False)`、対数尤度 $10^{-12}$）と比べる。
- **two-tier モデル（V33、M7）**: シミュレーション 2 例を `tests/test_mirt_m7_two_tier.py` で照合する（通常の EM と
  SQUAREM。生成スクリプト `validation/r/m7_two_tier_reference.R` → `tests/reference/v33_two_tier.json`、
  mirt 1.48 の `bfactor(data, model, model2 = )`、`accelerate = "none"`、`TOL = 1e-10`、`SE.type = "Oakes"`、
  $[-6, 6]$ 上の次元あたり 15 点、つまり特殊因子ごとに $15^3$ 格子 1 つ）。V14 の 3 次元の例と同じく、M ステップの
  最適化は `BFGS`（`control = list(reltol = 1e-13, maxit = 500L)`）を使う。mirt の `NR` は `GroupPars` に数値
  ヘッセ行列を使うので、3,375 節点では非常に遅い。主因子間の相関は mirt の `COV_21`（分散はどちらも 1 固定）。
  mirt の `fscores()` は全次元の直積格子で積分するが、これは同じ 1 次元格子での分解した事後分布と等しい。EAP は
  全パラメタを mirt の推定値に固定して比較する。どちらの例でも、mirt の推定値での縮約対数尤度は mirt の値と
  $2.1\times10^{-15}$（相対）で一致し、`flexpyirt.loglik >= fixed_at_mirt.loglik` が成り立ち、EM と SQUAREM は
  単調で、Oakes はヘッセ行列と $6.2\times10^{-17}$（共分散の要素）で一致する:
  - **V33a（`two_tier_2pl`、2PL、$J=16, N=2000$、`F1 = q01-q08`、`F2 = q09-q16`、$\rho=0.5$、`s1 = q01-q04`、
    `s2 = q09-q12`、`s3 = q05, q06, q13, q14`。4 項目は主因子のみ）**: `loglik` 相対 `2.1e-15`、項目パラメタ
    `3.0e-7`（EM、192 反復。SQUAREM は `3.1e-7`、33 反復）、`cor_12` `2.2e-8`（SQUAREM `1.3e-8`）、項目の Oakes SE
    相対 `4.4e-7`（SQUAREM `4.6e-7`）、`cor_12_se` 相対 `7.3e-6`、EAP $\theta$ `6.9e-15` / SD `4.0e-15`（mirt 側は
    $9^5$ 節点）。
  - **V33b（`two_tier_grm`、GRM $K=4$、$J=12, N=2000$、`F1 = g01-g06`、`F2 = g06-g12`（`g06` が交差負荷）、
    $\rho=0.4$、`s1 = g01-g04`、`s2 = g07-g10`）**: `loglik` 相対 `1.1e-15`、項目パラメタ `7.8e-9`（EM、192 反復。
    SQUAREM は `1.6e-8`、30 反復）、`cor_12` `6.3e-9`（SQUAREM `3.9e-9`）、項目の Oakes SE 相対 `1.0e-7`（SQUAREM
    `1.1e-7`）、`cor_12_se` 相対 `1.6e-6`、EAP $\theta$ `7.1e-15` / SD `3.4e-15`（mirt 側は $15^4$ 節点）。
  - 単体テスト（`tests/test_m7_two_tier.py`）では、縮約した尤度・期待度数・EAP を、全 $(P+S)$ 次元の直積格子での
    総当たり積分と比べる（$10^{-12}$ 以内。欠測、GRM 項目、交差負荷項目、特殊因子のない項目を含む）。主因子 1 つの
    two-tier モデルは双因子の経路と、特殊因子 1 つのモデルは照合済みの 3 次元の経路（`with_factors()` で特殊因子の
    相関を 0 に固定。対数尤度が $10^{-12}$ で一致）と比べる。
- **合計得点分布のカーネル（V36、M7）**: `summed_score_distribution()`（DESIGN §18.1）を
  `tests/test_r_m7_sumscore.py` で照合する（fast tier。生成スクリプト `validation/r/m7_sumscore_reference.R` →
  `tests/reference/v36_sumscore.json`、poibin 1.6、mirt 1.48）。
  - **2 値項目（Poisson 二項分布）**: 確率ベクトル 5 例（`uniform_10`: $J=10$、$p\sim U(0.05, 0.95)$。
    `twopl_50`: $\theta=0.4$ での 2PL。`extreme_200`: $\theta=-3$ での 2PL。`rare_300`: $p\sim U(0.001, 0.02)$。
    `equal_40`: $p=0.3$）。絶対差の最大値: `"recursion"` と poibin の再帰式（累積分布）で `2.9e-15`、DFT-CF の
    確率関数で `5.8e-16`。`"fft"` と DFT-CF の確率関数で `6.1e-16`。`"normal"` と `method = "NA"` で `1.2e-15`。
    `"refined_normal"` と `"RNA"` で `1.3e-15`。`"poisson"` と `dpois` で `8.5e-16`（許容誤差は絶対 $10^{-13}$、
    `"poisson"` は相対 $10^{-11}$）。poibin の厳密値は累積分布か DFT-CF から来るので、確かめられるのは絶対精度だけ。
    裾の相対精度は、単体テストの総当たり列挙と二項分布の特別な場合で確かめる。
  - **多値項目**: GRM（$K=4$、$J=10$、$N=1500$）と、2PL 5 項目 + GRM 5 項目についての mirt の合計得点の表。
    参照値には mirt の 61 節点上のカテゴリ確率（`probtrace`）と重みを保存してあるので、試されるのはカーネルだけ。
    $N\sum_qw_qL(s\mid\theta_q)$ は mirt の `expected` 列と `3.1e-15`（相対）で一致し、カーネルから作った合計得点
    ごとの $\theta$ の EAP と事後 SD は mirt の値と `5.3e-15` / `3.1e-15` で一致する。
  - **S-$X^2$（V15）**: `_itemfit.lord_wingersky` はこのカーネルを呼ぶようになった。V15 の照合テストは、参照値と
    許容誤差を変えずに通る。
  - **単体テスト**（`tests/test_m7_sumscore.py`）: $K\in\{2,3,4\}$ が混在する場合の全反応パターンの列挙（相対
    $10^{-13}$、厳密な 2 方法とも絶対 $5\times10^{-16}$）、二項分布の特別な場合、$P=10^{-4000}$（2000 項目）での
    対数空間の再帰、全方法の `jit` / `vmap` / `grad`、勾配の恒等式
    $\partial P(S=s)/\partial p_j=P_{-j}(s-1)-P_{-j}(s)$。
  - **厳密な 2 方法の精度と速度**（61 節点、jit 済み、この検証用マシンの CPU、1 回あたりミリ秒）: $J=50, K=2$ で
    再帰 0.10、対数空間の再帰 2.7、FFT 0.51。$J=200, K=2$ で 1.0 / 49 / 6.8。$J=1000, K=2$ で 57 / 1200 / 174。
    $J=100, K=5$ で 2.9 / 178 / 9.7。FFT と再帰の絶対差の最大値はどの場合も $3\times10^{-16}$〜$1.1\times10^{-15}$
    だが、最小の確率は $10^{-31}$〜$10^{-308}$ なので、FFT はそこでは 1 桁も合わない。この実装では、厳密な方法の
    うち再帰のほうが精度も速度も上なので（特性関数の計算自体に $O(JS_{\max})$ かかる）、既定は再帰のままとし、
    FFT は比較用に提供する。
- **情報関数・期待得点・信頼性・合計得点 EAP（V37、M7）**: DESIGN §18.2 を `tests/test_mirt_m7_information.py` で
  照合する（slow tier、約 30 秒。生成スクリプト `validation/r/m7_information_reference.R` →
  `tests/reference/v37_information.json`、mirt 1.48、$[-6, 6]$ 上の 61 点）。どのモデルも mirt で推定し
  （`TOL = 1e-8`、収束）、その推定値を全パラメタ固定（`table=`、`free = False`）の flexpyirt の `FitResult` に
  入れるので、比べるのは統計量だけ。ケース: `twopl`（$J=12$、$N=2000$）、`twopl_missing`（$N=800$、8% MCAR）、
  `threepl`（$J=12$、$N=5000$）、`grm` と `gpcm`（$K=4$、$J=8$、$N=2000$）、`nominal`（$K=4$、$J=6$、$N=3000$）、
  `mixed`（2PL 4、3PL 3、GRM $K=4$ 3、GPCM $K=3$ 2、名義 $K=3$ 2。$N=5000$）、`twogroup`（2PL 6 + GRM $K=4$ 4、
  全項目不変、各群 $N=1500$。焦点群は $\hat\mu=0.480$、$\hat\sigma^2=1.590$）。全ケースでの差の最大値:
  - **`iteminfo()` / `testinfo()`**（$\theta\in\{-6,-4,-3.5,\dots,4,6\}$）: 項目情報量は絶対 `1.4e-14`
    （相対では `4.5e-9`。情報量が $10^{-6}$ 程度になる名義項目で生じる。mirt は
    $\sum_k[(P'_k)^2/P_k-P''_k]$ を計算しており、そこで桁落ちする）。テスト情報量は相対 `8.3e-13`。許容誤差は
    相対 $10^{-9}$ + 絶対 $10^{-12}$。
  - **`expected.item()` / `expected.test()`**: 絶対 `7.5e-15` / 相対 `2.1e-14`（許容誤差は同じ）。
  - **周辺信頼性**: `integrate(rel.tol = 1e-12)` で再計算した `marginal_rxx()` に対して `1.4e-15`、
    `marginal_rxx()` そのもの（R の既定の許容誤差、相対で約 $1.2\times10^{-4}$）に対して `4.5e-11`
    （許容誤差は絶対 $10^{-10}$ と $10^{-8}$）。2 群モデルの参照値は、標準化した等価な単一群モデル
    （$a^*=a\sigma$、$d^*=d+a\mu$、$N(0,1)$）の `marginal_rxx()` で、両群を通じて `1.0e-15`（厳しい許容誤差）と
    `1.4e-11`（既定）。mirt の `testinfo()` を使って
    $\sigma^2I/(\sigma^2I+1)\,\phi(\theta;\mu,\sigma^2)$ を直接 `integrate()` した値とも `1.0e-15` で一致する。
    mirt 自身の式を焦点群に当てはめた値（`marginal_rxx(extract.group(mg, "focal"))`）は 0.7288 で、不変な定義の
    値は 0.8337。これは定義の違い（DESIGN §18.2）であって精度の違いではない。テストでは両者が異なることを
    確かめている。
  - **`empirical_rxx()`**（`fscores(method = "EAP", full.scores.SE = TRUE)` に対して。欠測のあるケースと、
    2 群モデルの各群を含む）: `4.4e-16`（許容誤差 $10^{-9}$）。
  - **`fscores(method = "EAPsum", full.scores = FALSE)`**: EAP は `6.2e-15`、事後 SD は `9.4e-16`
    （許容誤差は絶対 $10^{-9}$）。`probability` $\times N$ と `expected` は相対 `7.8e-15`（許容誤差 $10^{-9}$）。
  - **`fscores(method = "EAPsum", full.scores = TRUE)`**（一意な反応パターンごと）: EAP は `6.2e-15`、SD は
    `9.4e-16`。`twopl_missing` では一意なパターン 660 個のうち 453 個に欠測があり、mirt はちょうどそれらに `NA` を
    返し、flexpyirt は NaN を返す（mirt は欠測のあるデータに対して `full.scores = FALSE` の表を作らないので、
    このケースに表の参照値はない）。
  - **照合しておらず `SpecificationError` になるもの**: 多次元モデルの信頼性・合計得点の表・EAPsum（情報量と
    期待得点は V43）。`EmpiricalHistogram`・
    `DavidianCurve`・`NormalMixture` での信頼性・合計得点の表・EAPsum。`multipleGroup` の全受検者をまとめた
    `empirical_rxx()`（`empirical_rxx_pooled` として保存）に対応するものは flexpyirt にない。flexpyirt は群ごとに
    報告する。
  - **単体テスト**（`tests/test_m7_information.py`、fast tier）: 2PL（$a^2PQ$）と 3PL（Birnbaum）の閉形式
    （$10^{-12}$）、2PL / 3PL / 4PL / GRM / GPCM / 名義のカテゴリ確率の中心差分（$10^{-8}$）、カテゴリ確率から
    求めた期待得点、6 項目の混合テストの全 288 反応パターンの総当たりによる合計得点の表（$10^{-12}$）、
    全期待値の法則、潜在尺度の線形変換に対する周辺信頼性の不変性（$10^{-13}$）、信頼性の範囲と、
    シミュレーションデータでの周辺・経験的信頼性の近さ、エラーの経路。
- **多次元の情報行列・方向別の情報量・期待得点（V43、M7）**: DESIGN §18.3 を
  `tests/test_mirt_m7_information_multidim.py` で照合する（低速層、約 40 秒。参照値生成
  `validation/r/m7_information_multidim_reference.R` → `tests/reference/v43_information_multidim.json`、
  mirt 1.46.1）。推定はしない: 各ケースで項目パラメタを `pars = "values"` で指定し、`TOL = NaN` で mirt の
  オブジェクトを作る（確認的モデルは `mirt()`、双因子と two-tier は `bfactor()` / `bfactor(model2 = )`）。
  flexpyirt には同じ値をすべて固定して与える。ケース: `cfa2_mixed`（10 項目: 2PL、3PL、$u<1$ の 4PL、
  GRM $K=3,4$、GPCM $K=3,4$、名義 $K=3$。4 項目が両方の因子に負荷）、`cfa3`（9 項目、2PL・3PL・GRM・GPCM、
  交差負荷 2 項目）、`bifactor`（12 項目、2PL + GRM $K=3$、一般因子と特殊因子 3）、`two_tier`（8 項目、
  2PL + GRM $K=3$、交差負荷 1 項目のある相関する主因子 2 と特殊因子 2）。点は各 15 個: $[-3,3]^D$ の一様
  乱数 12 点、全座標 $-5$、$\pm5$ の交互、原点。方向は $\sum_m\cos^2\alpha_m=1$ を満たす角度で 4–5 個
  （軸、対角線、余弦が負の方向を含む）。参照値生成スクリプトは、mirt の `testinfo(degrees = )` が mirt 自身の
  `multidim_matrix = TRUE` の行列から求めた $\mathbf u^\top(\sum_j\mathbf I_j)\mathbf u$ に一致すること
  （$10^{-10}$）も確かめる。全ケースでの最大の差:
  - **項目情報行列**（`iteminfo(multidim_matrix = TRUE)`）: 絶対 `1.1e-15`、相対 `1.0e-12`（極端な点で
    大きさ $10^{-4}$ 程度の要素）。テスト情報行列（その和）は絶対 `1.8e-15`。
  - **方向別の項目・テスト情報量**（`iteminfo` / `testinfo(degrees = )`、`direction = cos(degrees)`）:
    項目は相対 `1.0e-12`、テストは相対 `5.8e-14`。
  - **合成値の SE** $\sqrt{\mathbf u^\top\mathbf I^{-1}\mathbf u}$（mirt の行列の和を R の `solve()` で
    逆行列にしたもの。mirt の `plot(type = "SE")` は代わりに $1/\sqrt{I_{\mathbf u}}$ を描く。DESIGN §18.3）: 相対 `2.8e-14`。
  - **`expected.item()` / `expected.test()`**: 絶対 `4.8e-14` / 相対 `4.7e-15`。
  許容誤差はすべて相対 $10^{-10}$ と絶対 $10^{-13}$（傾きを 0 に固定したため 0 になる要素のため）。
  - **単体テスト**（`tests/test_m7_information_multidim.py`。単純構造と多母集団のテストは低速層）: 2 因子の
    単純構造モデルが、軸の方向で 2 つの 1 次元の部分テストに一致する（項目・テスト情報量、合成値の SE、期待得点。
    $10^{-12}$）、その行列がブロック対角。行列が厳密に対称・半正定値で、組み込みの項目族では階数 1。2PL の閉じた式
    $P(1-P)\mathbf a\mathbf a^\top$（$10^{-12}$）。方向別の情報量 $=\mathbf u^\top\mathbf I\mathbf u$、方向の
    スケールと符号に対する不変性、合成値の SE と直接の逆行列の一致、$\mathrm{SE}_{\mathbf u}\ge I_{\mathbf u}^{-1/2}$。
    数値的に特異な行列で SE が無限大。双因子モデルで各項目の行列が一般因子と自分の特殊因子の外で 0。多母集団モデルで
    群ごとのパラメタを使う。1 次元の `information_matrix` がスカラーの情報量。`theta` の形（配列、因子名で引く
    DataFrame、1 点）とエラーの経路。
- **図の数値（V48、#103）**: DESIGN §21 を `tests/test_mirt_v48_plots.py` で照合する（slow tier、約 20 秒。
  生成スクリプト `validation/r/v48_plots_reference.R` → `tests/reference/v48_plots.json`、mirt 1.48、TAM 4.3-25、
  $[-6, 6]$ の 61 点）。V37 と同じく mirt がすべてのモデルを推定し（`TOL = 1e-8`、収束）、その推定値をすべて固定した
  `FitResult` に入れる。区間は回答者の順序に依存するので、データは行ごとに保存する（ユニークなパターンと各行の
  パターン番号）。ケース: `twopl`（$J=10$、$N=1003$。余り 3 のため最初の区間に 1 人、最後の区間に 2 人多い）、
  `twopl_missing`（$N=1200$、10% MCAR。区間は項目ごとにその項目に回答した回答者で作る）、`grm`（$K=4$、
  $J=6$、$N=1500$）、`mixed`（2PL 3、3PL 3、GPCM $K=4$ 2、名義 $K=3$ 2。$N=3000$）、`twogroup`（2PL 5 + GRM
  $K=4$ 3、全項目を等値、群あたり $N=805$、焦点群の平均・分散を推定）、`gpcm_tam`（TAM 自身の GPCM 推定）。
  全ケースでの最大差:
  - **EAP 推定値**（区間の入力）と `fscores(method = "EAP")`: `7.1e-15`（許容 $10^{-9}$）。
  - **`empirical_item_curve()`** と `itemfit(fit_stats = "X2", return.tables = TRUE, group.bins = 10,
    mincell.X2 = 0, Theta = eap)`（項目ごと。mirt は 1 回の呼び出しで 1 項目の表を返す。`mincell.X2 = 0` で
    セル併合を止める。`multipleGroup` の推定結果は拒否されるので、群ごとに `extract.group()` とその群の
    `Theta` の行で呼ぶ。mirt 自身の群ごとの `itemfit()` も同じことをしている）: 区間の人数は mirt の行和と
    完全に一致、観測比率も一致（`0`）、期待確率 `4.6e-15`（許容 $10^{-9}$）、区間平均と mirt のラベル
    （小数 4 桁に丸めてある）`5.0e-5`（丸めの分。許容 $5\times10^{-5}$）。期待表は
    $N\,P(\bar\theta_b)$ なので、区間平均も全精度で確かめている。
  - **`item_probabilities()`** と $\theta\in\{-6,-4,-3.5,\dots,4,6\}$ での `probtrace()`: `2.9e-15`
    （許容 $10^{-12}$）。
  - **`wright_map_table()` の項目位置** と、$[-20, 20]$ での $\sum_{c\ge k}$`probtrace()`$-1/2$ の
    `uniroot(tol = 1e-13)`: `2.1e-14`。`coef(IRTpars = TRUE)` の $b$（2PL）・$b_k$（GRM）と: `1.1e-14`
    （許容 $10^{-9}$）。3PL 項目の閾値は当然 $b$ より小さい。TAM 自身の GPCM 推定値（傾きと切片は `B` と
    `AXsi` から読む）での `tam.threshold()` と: `8.4e-5`。TAM の二分法は区間が $10^{-4}$ を下回ると止まる
    （許容 $10^{-4}$）。
  - **照合していないもの**: mirt の $X^2$ / $G^2$ とセル併合、`empirical.plot` の Clopper–Pearson 区間
    （計算しない）、図そのもの。
  - **単体テスト・スモークテスト**（`tests/test_plotting.py`、fast tier）: mirt の区間の人数の規則、欠測の
    ある区間の総当たりでの再計算、閾値の閉形式（2PL の $-d/a$、GRM の $-d_k/a$、交点のある・ない 3PL 項目）、
    DIF のある 2 群、エラーの経路。すべての描画関数を Agg バックエンドで描き、描いたデータが DataFrame と
    等しいこと。matplotlib を import できないときの `MissingDependencyError`。CI では matplotlib のない環境で
    `scripts/check_plotting_without_matplotlib.py` も実行する。
- **DIF のスクリーニング、Wald 検定、効果量（V49、#104）**: DESIGN §16.22 を
  `tests/test_mirt_m5_dif_screen.py` で照合する（slow tier、約 7 分。生成スクリプト
  `validation/r/m5_dif_screen_reference.R` → `tests/reference/v49_dif_screen.json`、mirt 1.48。すべての推定と
  推定し直しで `optimizer = "NR"`、`accelerate = "none"`、`TOL = 1e-10`、$[-8,8]$ の 81 点。時間のかかる
  `DIF()` の結果は `V49_CACHE=<file.rds>` で生成スクリプトの実行間に保存できる）。データ: 1200 人の 2 群（2PL 8
  項目と段階反応 4 項目 $K=3$。q02 の識別力、q04・q07 の切片、q09 の切片に DIF。焦点群 $N(0.4, 1.3)$）と 800 人の
  3 群（2PL 8 項目、g3 で q03 の切片に DIF）。
  - **`"all_others"`** を `DIF(<完全不変>, scheme = "drop", p.adjust = "BH")` と全パラメタおよび切片だけで照合
    （2 群と 3 群）: $X^2$ は $1.5\times10^{-11}$、`df` は完全一致、検出された項目は同じ。
  - **逐次戦略** を `DIF(scheme = "drop_sequential" / "add_sequential", seq_stat = 0.05, p.adjust = "BH")` と
    `max_run` 2・3・`Inf` で照合（flexpyirt の `max_iter` = `max_run` − 1）: 各ラウンド数で検出された項目が同じ、
    検出された項目の最終検定（mirt の出力）の $X^2$ は $1.5\times10^{-11}$、最終モデルの対数尤度
    （`return_seq_model = TRUE`）は相対 $1.5\times10^{-15}$。このデータでは 1 ラウンド目から検出結果が変わらない
    （どちらの戦略も 2 ラウンドで収束）。反復そのものは単体テスト（`tests/test_m5_dif_screen.py`）で確かめる。
    mirt の逐次ループは `items2test` 内の位置で項目名を引くので、生成スクリプトは先頭からの `1:k` を検定する
    （アンカーは最後の項目）。
  - **Wald** を Oakes の SE（`SE.type = "Oakes"`）つきのアンカーモデルでの `DIF(Wald = TRUE)` と照合（両方を
    独立に推定）: $W$ は相対 $2.7\times10^{-7}$、`p` / `adj_p`（Holm）は相対 $1.2\times10^{-7}$。
  - **効果量** は部分不変モデル（q02、q04、q07、q09 が自由）と不変モデルの mirt の推定値を、$[-6,6]$ の 201 点
    （DRF の `quadpts`・`theta_lim` と `RMSD_DIF()` の格子）の固定 `FitResult` に入れて照合: 符号つき・符号なしの差
    （焦点群と参照群の密度）と DTF を `mirt:::calc_DRFs(best_fitting = TRUE)` と $1.1\times10^{-15}$、EAP は
    $6.7\times10^{-15}$、ESSD と ETSSD を `empirical_ES()`（`Theta.focal` は同じ格子の EAP）と
    $2.2\times10^{-15}$、両群・両モデルの RMSD を `RMSD_DIF(probfun = FALSE)` と $9.7\times10^{-16}$。許容誤差:
    $X^2$ $10^{-8}$、$p$ 相対 $10^{-3}$（mirt の `p` は `1 - pchisq()` で小さな $p$ では不正確）、対数尤度 相対
    $10^{-8}$、$W$ 相対 $10^{-5}$、効果量 $10^{-12}$。
  - **mirt との違い**: mirt 1.48 の `DRF(den.type = "focal" / "reference")` は止まる（`add_std.DRF()` が未定義の
    `VAR` を参照する）ので、生成スクリプトは `DRF()` の点推定を計算する `calc_DRFs()` を呼ぶ。mirt の `DRF()` の
    既定の密度（`best_fitting = FALSE`、データの事後期待度数）は再現しない（DESIGN §16.22）。逐次方式で mirt は
    検出された項目だけを返すが、`DIFScreenResult.table` は検定した全項目を持つ。
- **標本ウェイト（V34、M7）**: mirt の `survey.weights`（mirt 1.48、`optimizer = "NR"`、`TOL = 1e-10`、
  `SE.type = "Oakes"`）。参照ファイルは行ごとのウェイトとともに元の行を保存するので、ウェイトを足し合わせる
  パターン圧縮も比較の対象になる。V34a: シミュレーションの 2PL（$J=10$、$N=1000$）、合計点と関係する対数正規
  ウェイト、正規化なし（$\sum w=1541.7$）、ウェイト 0 が 5 行。V34b: Science の GRM（$N=392$）、合計を $N$ に
  正規化した一様ウェイト。V34c: 各 800 人の 2 群、アンカー q01–q05、focal 群の平均と分散を推定、5% の MCAR 欠測、
  2 群で合計の異なるガンマウェイト（$[-8,8]$ の 81 点）。V34d: FIPC（q01–q05 を固定、潜在分布の平均と分散を推定、
  81 点）、合計を $N$ に正規化した対数正規ウェイト。V34e: 双因子 2PL（`bfactor()`、$J=9$、3 項目ずつの特殊因子 3 つ、
  $N=1000$、$[-6,6]$ の各軸 21 点）、正規化しない対数正規ウェイト（$\sum w=1467.6$）: 対数尤度
  $4.7\times10^{-16}$（相対）、パラメタ $9.5\times10^{-7}$、Oakes・Hessian の SE $2.2\times10^{-6}$（相対）、
  AIC・BIC $3.8\times10^{-11}$。V34a–d の差の実測値（通常の EM・SQUAREM の最大）: 対数尤度
  $1.8\times10^{-15}$（相対）、パラメタ $7.6\times10^{-8}$、Oakes SE $1.1\times10^{-7}$（相対）、flexpyirt の
  Hessian SE と mirt の Oakes $1.1\times10^{-7}$（相対）、AIC・BIC $5.1\times10^{-11}$。いずれも共通の許容誤差内。
  mirt の BIC はウェイトの和を $N$ として使う（flexpyirt も同じ。DESIGN §17.4）。ウェイトの合計が行数と異なる
  V34a がこれを確かめる。どの参照値も同じデータのウェイトなしの推定とは異なる（対数尤度が $10^{-3}$ 超、
  パラメタが $10^{-3}$ 超の差）ので、ウェイトを無視すればテストは落ちる。**R と照合していないもの**: 整数ウェイトと
  行の複製の一致（欠測のある 2PL、通常の EM と SQUAREM での 2 群 GRM、経験ヒストグラムの FIPC、MAP、双因子モデルと two-tier モデル）、
  ウェイト 0 の行と行の削除の一致、ウェイトの定数倍は、恒等式として `tests/test_m7_weights.py` で確かめる。
  Oakes・Hessian の SE は mirt と一致するが、標本ウェイトのもとでは正しい分散推定量ではない（DESIGN §17.4）。
  項目適合度、M2、局所依存、JSI、DIF、モデルに基づく平行分析は、ウェイト付きの推定結果に対してエラーを出す。
  テスト: `tests/test_mirt_m7_weights.py`。参照値生成: `validation/r/m7_weights_reference.R`。
- **ロバスト標準誤差（V35、M7）**: `fit(se="crossprod" | "sandwich")`（DESIGN §17.5）を mirt 1.48 の
  `SE.type = "crossprod"` / `"sandwich"` と照合する。生成スクリプトは mirt のパラメタ表（`mod2values`）と
  `vcov` の行列全体を書き出し、各行にそのパラメタの最初の `parnum` を付けるので、テストは mirt の解を再現して
  行列全体を比べる。
  **ケース**（`validation/r/m7_robust_se_reference.R`、`tests/reference/v35_*.json`）: 歪んだ特性と 5% の
  MCAR を持つ 3PL のデータに当てはめた 2PL（$J=12$、$N=2000$。わざと誤ったモデルで、サンドイッチは下の
  尺度で Oakes の共分散と最大 0.49 違う）、3PL 4 項目 + GRM(4) 3 項目 + GPCM(3) 3 項目 + 名義(3) 2 項目
  （$N=5000$）、$a\sim N(1.2, 0.5)$・$d\sim N(0, 2)$ の事前分布を付けた 2PL の MAP（$N=300$）、アンカー
  6 項目で focal の平均・分散が自由な 2 群（$[-8,8]$ に 81 点）、5 項目を固定し潜在分布の平均・分散を
  自由にして 4 項目の傾きを等値にした FIPC、交差負荷のある相関 2 因子（$31^2$ 点）、相関 3 因子（$15^3$ 点）、
  FIPC の EH。
  **mirt の解での比較**（flexpyirt の問題を mirt の推定値で組み立てる。尺度は行列全体の
  $\max_{ij}\lvert v_{ij}-v^{\text{mirt}}_{ij}\rvert/\sqrt{v^{\text{mirt}}_{ii}v^{\text{mirt}}_{jj}}$）:
  対数尤度は相対 $\le3.6\times10^{-15}$ で一致し、パラメタの対応が正しいことが確かめられる。クロス積の
  共分散は $\le1.3\times10^{-11}$（2PL $8.9\times10^{-15}$、混合 $4.9\times10^{-13}$、MAP
  $2.6\times10^{-15}$、2 群 $1.3\times10^{-11}$、FIPC $9.9\times10^{-15}$、2 因子 $6.2\times10^{-15}$、
  3 因子 $2.0\times10^{-14}$）、サンドイッチは mirt が NR の M-step を使う 6 ケースで $\le2.7\times10^{-8}$
  （$2.3\times10^{-10}$、$1.4\times10^{-9}$、$1.1\times10^{-10}$、$2.7\times10^{-8}$、
  $1.1\times10^{-8}$、$1.6\times10^{-9}$）、3 因子で $1.4\times10^{-5}$。サンドイッチの差は mirt の
  Oakes の Hessian（差分近似）による。許容誤差は $10^{-9}$ と $10^{-6}$（3 因子のサンドイッチは $10^{-4}$。後述）。
  **通しの比較**（flexpyirt 自身の初期値からの `fit()`、`BAEM(tol=1e-9)`）: mirt が NR の M-step を使う
  6 ケースで、どちらの種類でも自由パラメタは mirt と $4.1\times10^{-8}$ 以内、標準誤差は相対
  $4.6\times10^{-8}$ 以内。3 因子では $5.0\times10^{-7}$ と $6.6\times10^{-6}$。許容誤差は $10^{-6}$ と $10^{-5}$。
  3 因子のケースでは mirt は BFGS を使い（V14 と同じ。$15^3$ の格子では NR が非常に遅い）、推定値でのスコアが
  $5.6\times10^{-6}$ のところで止まる（NR では $\le2.9\times10^{-7}$）。そのためこのケースだけ許容誤差を
  100 倍に広げる（mirt の解でのサンドイッチとパラメタは $10^{-4}$、標準誤差は $10^{-3}$）。
  **下方漸近線。** mirt の `vcov` は $\operatorname{logit}g$ についてのものなので、テストは mirt の行列を
  ヤコビアン $g(1-g)$ で変換する。
  **事前分布。** mirt は flexpyirt と同じく $-\nabla^2\log\pi$ をクロス積に加える（`SE.simple` の
  `Igrad + Iprior`）。照合するのは正規事前分布だけである。mirt の `lnorm`・`expbeta` の事前分布の Hessian は
  誤っている（V7）ので、これらは単体テストで確かめる（事前分布の有無による行列の差が、対数事前分布の
  自動微分の Hessian に一致する）。
  **経験ヒストグラム。** mirt のクロス積は flexpyirt と同じくヒストグラムを固定するので、mirt の項目推定値と
  ヒストグラムで直接比べる: $6.2\times10^{-15}$。mirt の `"sandwich"` は Oakes の Hessian を使うが、EH ではこれは
  ヒストグラムを固定した Hessian ではない（V6）。その行列は flexpyirt と 0.049（標準誤差で最大 2.4%） 違う。そこでサンドイッチは、
  mirt の Richardson の共分散 $V_R$（ヒストグラムを固定した Hessian。DESIGN §15.35 の定義）と mirt の
  クロス積の共分散 $V_C$ から組み立てた $V_R V_C^{-1} V_R$ と比べる: $1.4\times10^{-8}$（Richardson は差分近似の
  Hessian。flexpyirt の Oakes の共分散と $V_R$ の差は $7.1\times10^{-9}$）。許容誤差は $10^{-6}$。EH の参照値は mirt で `TOL = 1e-7` を使う
  （約 11,000 サイクル。$10^{-9}$ には 100,000 サイクルでも届かなかった）。mirt 自身の推定値とヒストグラムでの
  比較なので影響はない。EH の通しの比較は、V6 に書いた理由で行わない。
  **`"sandwich.Louis"`。** パンを Louis の式で計算した同じ推定量である。潜在分布のパラメタが自由でない
  ケース（2PL、混合、MAP）では mirt の行列は `"sandwich"` と一致する（flexpyirt はどちらとも
  $\le8.8\times10^{-10}$ で一致）。潜在分布の平均・分散・相関が自由なケース（2 群、FIPC、2 因子）では mirt の
  Louis のパンが誤っており（`SE.type = "Louis"` 自体が行列を返さない）、`"sandwich.Louis"` は上の尺度で
  1.0〜7.0 ずれる。別の推定量ではないので提供しない。
  **対象外。** `DavidianCurve`・`NormalMixture`: mirt はどちらの種類も計算しないので、flexpyirt は
  `SpecificationError` を出す。双因子・two-tier モデルも `SpecificationError`（mirt は対応している。スコアのカーネルが
  次元縮約にまだ対応していない）。多母集団の多次元モデルと 4PL・PCM・Rasch の項目は同じコードを通るが、V35 に
  専用のケースはない。標本ウェイトつきの版とクラスタロバスト版は未実装で、`weights=` と併用すると
  `SpecificationError` を出す。
  テスト: `tests/test_mirt_m7_robust_se.py`。単体テスト（`tests/test_m7_robust_se.py`）: チャンク分割した
  スコアと密な前向き自動微分のヤコビアンの一致（任意のチャンクサイズ。欠測、事前分布付きの混合形式、
  共有パラメタのある 2 群、2 因子、EH）、$\sum_p c_ps_p$ と勾配の一致、正しいモデルで $N=20{,}000$ のとき
  標本誤差の範囲で $B=H$（標準誤差は 3% 以内）、$B=H$ を強制するとサンドイッチが Oakes の共分散に
  一致すること、行を 2 倍にすると共分散が半分になること、`b_se` と `link()` が選んだ共分散を読むこと、
  JSON の往復、拒否するケース。
- **モデル比較と回答データのシミュレーション（V54、#109、DESIGN §24）**:
  - **`compare_fits()` と `anova()`**: 各モデルを mirt（`optimizer = "NR"`、`accelerate = "none"`、
    `TOL = 1e-10`）と flexpyirt（`BAEM(tol=1e-9)`）で同じグリッドを使って推定し、表全体を比べる: V11a の
    データ（$[-6, 6]$ の 61 点）で Rasch / 2PL / 3PL、V8b のデータで PCM / GPCM、V4 のデータ（$[-8, 8]$ の
    81 点）で全項目不変の 2 群 2PL（焦点群の平均と分散は自由）とアンカー q01–q05 のみのモデル、最後の項目だけ
    群間で自由なモデル（$p=0.188$ なので、0 から離れた $p$ 値も確かめる）。`n_params` は mirt の `nest` と
    厳密に一致し、対数尤度と 4 つの情報量規準は $5\times10^{-11}$、$X^2$ は $1.4\times10^{-11}$、$p$ は
    $7\times10^{-13}$ で一致する（許容誤差は絶対値で $10^{-7}$、$10^{-7}$、$10^{-9}$）。mirt は
    $p<10^{-16}$ で $p=0$ を表示する（`1 - pchisq()`）。flexpyirt は上側確率を返す。
  - **`simulate_responses()` の決定的な部分**: 乱数列は R と一致させられない（`simdata()` は R の乱数生成器を
    使う）ので、参照値は項目パラメタ・$\theta$・一様乱数を固定して作る。9 項目（Rasch、2PL ×2、3PL、4PL、
    graded $K=3,4$、GPCM $K=4$、採点係数が自由な名義反応 $K=4$）を 400 個の $\theta$（$\pm6$、$\pm3$、0 を含む）
    で、2 因子モデル（2PL、graded $K=3$、3PL。交差負荷の項目 1 つ）を 300 点で評価し、flexpyirt の
    カテゴリ確率は `probtrace()` と $10^{-12}$ で一致し、R で $x=\#\{k<K-1: u>F_k\}$ により得た回答を
    厳密に再現する。
  - **`simulate_responses()` の分布の確認**（モンテカルロ、$|z|<4.5$）: 6 万人分の生成データ（Rasch・2PL・
    3PL・4PL・GRM・GPCM・名義反応の混在、$N(0.3, 1.4)$）と 2 群（3 万人と 4 万人）のカテゴリ比率と合計得点の
    分布が、モデルが導く周辺分布（同じカーネルと Lord–Wingersky 再帰（V36）の 81 点 Gauss–Hermite 求積）と
    一致する。潜在特性は指定した共分散を持つ（2 因子、特殊因子 2 つの双因子。同じ特殊因子の項目どうしは相関が
    高い）。`NormalMixture` と推定した `DavidianCurve` からの値は厳密な累積分布関数に対する
    Kolmogorov–Smirnov 検定（0.1% 水準）を通り、推定した `EmpiricalHistogram` からの値はグリッドの点で、
    頻度が推定された重みと一致する。
  - **復元**: 混在形式のテストで 2 万人分の生成データに `fit()` を当てはめると、すべての項目パラメタが生成値から
    4.5 SE 以内、各群 8000 人の 2 群の当てはめで焦点群の平均と分散（0.5、1.3）が 4.5 SE 以内に戻る。`FitResult`
    と、その推定値を与えた `Model` は同じキーから同一のデータを作り、データは `key` から再現できる。
  テスト: `tests/test_mirt_v54_compare_fits.py`、`tests/test_v54_simulate.py`。参照値の生成:
  `validation/r/v54_compare_simulate_reference.R`。

- **IRT 等化と素点-尺度得点変換表（V51、#106）**: DESIGN §23 を `tests/test_r_equating.py` で照合
  （生成スクリプト `validation/r/v51_irt_equating_reference.R` → `tests/reference/v51_irt_equating.json`。
  mirt 1.48、equateIRT 2.5.2、SNSequate 1.3.5、equate 2.0.9）。パラメタは推定しない: 項目パラメタは R で決めて保存し、
  潜在分布は $[-6,6]$ の等間隔 61 点上の $N(0,1)$、リンクは $\theta_Y=A\theta_X+B$（2 値 $A=1.15$, $B=-0.25$、
  多値 $A=0.9$, $B=0.3$）。速いテスト層は保存したパラメタから直接フォームを作り、カーネルと変換の規則を確かめる。
  遅い層は同じ値に固定した推定結果で `irt_equate()`・`score_distribution()`・`scale_table()` を通して全比較を繰り返す。
  最大差:
  - **2 値の真値等化**（20 + 20 項目、3PL はそれぞれ 12・9 項目、$\sum c=2.317$・$1.477$）: 曲線の値域内の得点は
    equateIRT 自身の Newton 解（`score_compute`。$|\Delta\theta|<10^{-6}$ で止まり誤差は約 $10^{-12}$）と比べて、
    等化得点 `2.4e-12`、$\theta$ `1.2e-12`（許容 $10^{-9}$）。全得点で SNSequate と比べて、$\sum c$ 以下の
    Kolen & Brennan の補間と最高点は `2.4e-15`（許容 $10^{-11}$）。値域内は SNSequate が使う `uniroot()` の既定許容値
    による `1.2e-4`（許容 $10^{-3}$。補助的な確認）。
  - **2 値の観測得点等化**: 両母集団に同じ節点と重みを与えた equateIRT の OSE、$w_1=1$, $0$, $0.5$: `5.7e-14`
    （許容 $10^{-11}$）。
  - **多値・混合形式の真値等化**（新: 4 GRM $K=4$ + 6 2PL、最高点 18。旧: 3 GPCM $K=3$ + 2 GRM $K=4$ + 5 3PL + 3 2PL、
    最高点 20、$\sum c=0.795$）: 多値フォームの IRT 等化を行う R パッケージはないので、新フォームの `expected.test()` を
    `uniroot(tol = 1e-13)` で解き、旧フォームの `expected.test()` を $A\theta+B$ で評価した参照を作った:
    $\theta$ `5.5e-13`、等化得点 `9.3e-13`（許容 $10^{-9}$）。
  - **モデル上の分布**: `probtrace()` 上の R の Lord–Wingersky 再帰は、R 内で mirt の
    `fscores(method = "EAPsum", full.scores = FALSE)` の `expected` / $N$ と $10^{-12}$ で一致。flexpyirt の分布は
    どの母集団でも両者と相対 `5.6e-15`（許容は相対 $10^{-11}$）。
  - **多値の観測得点等化**: 2 つの合成分布に対する `equate::equate(type = "equipercentile")`: `4.5e-9`。equate は
    パーセンタイル順位と累積分布を小数 10 桁に丸める（`equate:::equip`）ので、得点では約 $10^{-10}/g(y)$
    （許容 $10^{-8}$）。同じ分布で flexpyequate の等パーセンタイル等化とは `2.5e-14` で一致し、単体テストでは
    確率 0 を含む乱数の分布で両者を $10^{-12}$ で比べる。
  - **旧多値フォームの素点-尺度得点変換表**（slope 10、intercept 50、境界 25 と 75、整数に丸め）: 真値
    （`expected.test()` の `uniroot()` による $\theta$、SE は $1/\sqrt{\texttt{testinfo}}$）: $\theta$ `2.3e-12`、SE 相対
    `1.2e-12`。EAPsum（mirt `fscores`）: $\theta$ `5.6e-15`、SE 相対 `1.6e-15`。丸めた尺度得点は一致。
  - **R で照合していないもの**: 4PL の上側漸近線の補間、多母集団のフォームと母集団、リンクは
    `tests/test_equating.py` で `expected_score()`・`item_information()`・`summed_score_table()` と DESIGN §23 の式に
    対して単体テストする。EH・Davidian・混合の母集団は `link()` と同じ推定格子（`_fitted_grid`）から作り、等化としての
    テストは別にない。
- **受験者適合度と残差にもとづく項目適合度（V53、#108）**: DESIGN §25。V37 のデータと mirt の推定値を再利用する
  （全パラメタ固定、$[-6, 6]$ の 61 点）。ケースと推定法（EAP・MAP・ML・WLE）ごとに、R スクリプトは mirt の
  `fscores()`、全データでの `personfit(m, Theta = )`（最初の 300 個の一意なパターンを保存）と
  `itemfit(m, fit_stats = "infit", Theta = )`（EAP・MAP・WLE。ML の無限大の推定値は mirt が置き換えるため除く）を計算する。
  **mirt の $\theta$ で**（保存した推定値でカーネルを評価）: $l_z$・infit・outfit とその $z$ 値は 8 ケース・4 推定法の
  すべてで $8.3\times10^{-9}$（相対）以内、$l_z^*$ は DESIGN §25 の 3 項を R で再計算した値（`probtrace()` と中心差分、
  `iteminfo()`）と $|\hat\theta|\le10$ で $2.8\times10^{-8}$ 以内。それより外（3PL の平坦な裾での mirt の ML、V13 の欠測時の
  極値）では中心差分の精度が落ち、flexpyirt は NumPy での解析的な計算と一致する。**端から端まで**（flexpyirt 自身の推定値。
  mirt との差は EAP $5.6\times10^{-15}$、MAP $2.9\times10^{-6}$、ML $1.2\times10^{-5}$、WLE $2.6\times10^{-6}$ まで）:
  受験者の統計量は $2.0\times10^{-5}$ 以内、項目の統計量は $3.1\times10^{-5}$ 以内。**Rasch**: mirt の WLE 推定値で
  $l_z$・infit・outfit が `sirt::personfit.stat()` と $3.7\times10^{-13}$（相対）以内で一致（mirt と sirt の差は
  $1.8\times10^{-15}$）。**mirt との既知の差**（DESIGN §25 の 5 項）: 受験者の `z_outfit` は $n^2$（回答項目数）を使うが
  mirt の `personfit()` は $J^2$ を使い、mirt の `itemfit()` は項目の `z.outfit` で項目ごとの回答者数を行方向に再利用する。
  どちらも欠測があるときだけ効くので、参照値ではこれらの $z$ 値を mirt 自身の残差モーメント（`attr(m, "inoutfitreturn")`）
  から計算し直し、欠測のないデータでは mirt の値とも直接比べる。**$l_z^*$ には R パッケージの参照値がない**（`PerFit` は
  ここではインストールできず、その `lzstar()` は 2 値項目のみ）。R での再計算に加え、`tests/test_person_fit.py` で
  2PL での Snijders (2001) の閉じた式（ML・MAP・WLE、$10^{-10}$）と比べ、帰無分布をシミュレーションする（2PL、$J=20$、
  $N=3000$、真のパラメタ）: ML と WLE で $l_z^*$ の平均は標本誤差の範囲で 0、SD は $[0.93, 1.07]$ に入り、$l_z$ の SD は
  0.93 未満。これで $l_z^*$ を照合済みとしてよいかはオーナーが判断する（#108）。テスト:
  `tests/test_mirt_v53_person_fit.py`、`tests/test_person_fit.py`。参照値の生成: `validation/r/person_fit_reference.R`。

## 照合以外の検証
- **古典的項目分析（V46、flexpyctt、#101）**:
  `packages/flexpyctt/validation/r/ctt_item_analysis_reference.R`（R 4.6.2、psych 2.6.9、ltm 1.2.0、
  polycor 0.8.2、lavaan 0.7.3）が `packages/flexpyctt/tests/reference/v46_item_analysis.json` を書き、
  `packages/flexpyctt/tests/test_item_analysis.py` で照合する。シミュレーションの 4 例: 多肢選択 10 項目（$N=500$、
  正答と誤答の引きつけ方を変え、1 項目は 5 選択肢）の欠測なしと 5% MCAR、0〜4 で採点する 6 項目（$N=300$、
  段階反応モデル）の欠測なしと 6% MCAR。ほかに丁度識別の ω を見る 3 項目と、0〜4 の欠測なしデータの 1 項目を逆転採点したもの（psych の既定の
  `flip = TRUE` がその項目を反転する。`flip = FALSE` の値も記録し、0.18 低く、反転しない式と一致する）。観測された差: 閉じた式（α、標準化 α、
  項目を除いたときの値、Feldt の上下限、平均、SD、項目–合計・項目–残りの相関、点双列・多分系列・双列相関、
  選択肢の割合・点双列相関・平均得点・群ごとの割合）は $6\times10^{-15}$ 以下、キーによる得点は完全一致、
  ω total は $7\times10^{-8}$ 以下、負荷量は $1.4\times10^{-6}$ 以下（psych の `fa()` が minres の最適化を早く
  止めるため）、lavaan とのポリコリック行列は $2\times10^{-8}$ 以下。反転する項目は psych が反転するものと一致し、反転したときの ω は
  手で逆転採点したデータの ω と一致する。既知の違い（それぞれテストで確かめる）:
  (1) psych の `raw.r` は `rowSums(na.rm = TRUE)` との相関なので、欠測があるときは R で求めた $C$ に基づく
  `colSums(C)/sqrt(diag(C) * sum(C))` と比べ、`raw.r` そのものとは欠測なし・リストワイズのときだけ比べる
  （DESIGN §16.21）。`r.drop` は psych でも $C$ に基づくので全例で比べる。(2) ltm の素の
  `cronbach.alpha(na.rm = TRUE)` は合計の分散を欠測のない行から、項目の分散をすべての観測された得点から求め、
  どちらの規則でもない。欠測のないデータでだけ比べる（標準化 α は欠測のない行を使い、`"listwise"` と一致する）。
  (3) psych の `biserial()` は 2 段階の双列相関の $\sqrt{(n-1)/n}$ 倍になる。(4) psych の `polyserial()` は欠測の
  ある行も $n$ に数えるので、欠測のないデータで比べる。欠測があるときは 2 段階の係数 $\sigma_j/\sum_k\phi(\tau_{jk})$
  を R で計算し直して比べる。(5) 選択肢の統計量にはインストール済みの R 実装がない（TAM の `tam.ctt3` は未導入）。
  スクリプトが定義どおり素の R で計算し、選択肢の割合は psych の `responseFrequency()` とも一致する。issue に挙げた
  lavaan の 1 因子 CFA は使わなかった（共分散行列への ML は推定量も尺度も違う）。
- **観測得点による DIF（V47、flexpyctt、#102）**: 参照値は difR 6.1.0 と base R によるもので、
  `packages/flexpyctt/tests/reference/v47_dif.json` にコミットする。一致の程度: MH の $\chi^2$、$\hat\alpha_{MH}$ と
  Phillips–Holland の分散、一般化 MH の統計量（difGMH と、群 x カテゴリ x 層の表への `mantelhaen.test()`）、Mantel の
  統計量とオッズ比は相対 $10^{-10}$ まで（どれも閉じた式）。純化の経過（`difPur`）、反復回数、収束の判定、最終的な
  検出結果は一致。ロジスティック回帰: `difLogistic()` は既定の `epsilon = 1e-8` で `glm()` を呼ぶので、このデータでは
  係数と統計量が最尤推定値から相対 $2\times10^{-4}$ までずれる。そこでスクリプトは `stats::glm.control` を
  `epsilon = 1e-14` のものに置き換える。すると尤度比統計量、$\Delta R^2$、係数は $10^{-8}$ まで、標準誤差と Wald
  統計量は $10^{-6}$ まで一致する（`glm()` の `vcov()` は 1 つ前の反復の IRLS の重みを使い、逸脱度の基準は
  $\sqrt{\epsilon}$ 程度のステップで止まる）。テストで確認している既知の差: (1) `mantelhaen.test()` は
  $\lvert\sum A-\sum E(A)\rvert<1/2$ のとき連続修正を省く（difR と flexpyctt は省かない。シミュレーションのデータで
  3 項目）。(2) `difMantel.poly()` は出力を丸め（統計量は小数 3 桁、オッズ比は 4 桁）、純化では検出された項目を
  その項目自身のマッチング得点からも外す。そこで Mantel の統計量と純化の経過は、difR のほかの方法と同じ規則で R で
  計算し直し、`difMantel.poly()` は丸めた値での照合にとどめる。(3) `liu_agresti_ccor(se = "liu_agresti")` は
  $T=\sum ad/N-\hat\psi\sum bc/N$ の焦点群の累積度数による微分を $a/N-\hat\psi b/N$ としており（$a/N$ の符号が誤り）、
  $\sum ad/N$ ではなく $\sum bc/N$ で割るので、その `SE_log_Psi` は $SE(\ln\hat\psi)$ でも $SE(\hat\psi)$ でもない
  （$\psi=1.48$、3 層、$N=20000$ を 300 反復したモンテカルロの SD 0.0296 に対して 0.031。`verbal` のある項目で
  Phillips–Holland の 0.36 に対して 0.14）。flexpyctt と R スクリプトの `la_eq3()` は代わりに Liu & Agresti（1996）の
  式 (3) を使う。小さな層が多数ある場合にも妥当な推定量である（DESIGN §16.20）。R の実装は論文の記法で Python とは
  独立に書いたもので、論文の計算例（表 1、28 施設: $\ln\hat\theta=-1.153$、SE .571）を再現する。Python のテストも
  Mantel の 4.84 を含めて同じ公表値を確認する。推定値を代入するデルタ法（レビュー前の実装。層が大きい確認では
  モンテカルロの 0.0296 に対して 0.0291）はこの例で 0.375 になり、1 層 4 人では約 40% 過小になる。Python の
  モンテカルロのテスト（4 人と 10 人の層が 150 個、4 カテゴリの比例オッズの項目、400 反復）で式 (3) の SE の平均が
  $\ln\hat\psi$ のモンテカルロ SD と 10% 以内で一致すること、別のテストで層が大きい 0 / 1 項目では
  Phillips–Holland の標準誤差と 1% 以内で一致することを確認する。
  (4) `liu_agresti_ccor()` は flexpyctt の $1/\hat\psi$ を返す（difR の向き: 1 より大きいと焦点群に有利）。
- **パラメタ回復シミュレーション**: 条件（N, J, 群数, アンカー数）× 反復 100 で bias / RMSE を出し、`validation/recovery/` に結果と図を置く。
- **M-step の非拘束尺度と `LogitNormalPrior`（V45、#15）**: DESIGN §15.47–§15.48。
  1. *両方の M-step 尺度での内点解*: V3・V4・V5・V9（`tests/test_mirt_m2.py`: 対数尤度、項目と潜在分布のパラメタ。
     V4 の焦点群の分散は $\log\sigma^2$ で動く）、V11（`tests/test_mirt_m3.py`: 3PL/4PL、$\operatorname{logit}g$・
     $\operatorname{logit}u$、Oakes と Hessian の SE も）、V7（`tests/test_mirt_m3_priors.py`: MAP。`LogNormalPrior`
     のついた傾きは $\log a$ で動く）、V14（`tests/test_mirt_m4_mirt.py`: 2–3 因子、相関は
     $2\operatorname{atanh}\rho$ で動く）の mirt 照合を `BAEM(m_step_scale="unconstrained")` でも自然尺度と同じ
     許容誤差で行い、すべて通る。`tests/test_m_step_scale.py` は、変換と Newton ステップの単体テスト（同じ内点の
     最大点、境界へは内側から近づく、固定スロットは動かない、棄却されたステップは厳密に元の値）、シミュレーション
     した 3PL の MAP（`tol=1e-8` で推定値は自然尺度と 1e-5 以内、対数事後は 1e-12、Oakes の共分散は相対 1e-4。
     SQUAREM あり・なし）、焦点群の分散が自由な 2 群モデル、境界のケース（$g=0$ で生成した項目: 両尺度ともその項目で
     $g<10^{-4}$ に終わって `HeywoodWarning` を出し、対数尤度は 1e-5 で一致）を加える。
  2. *`LogitNormalPrior`*: mirt には自然尺度でこの密度を持つ事前分布がない（`g` への `norm` は mirt 内部の
     $\operatorname{logit}g$ に Jacobian なしで作用し、flexMIRT の `Guessing : Normal` と同じ）ので、既定の
     `mode_scale="natural"` には mirt の参照値はない。代わりに `tests/test_logitnormal_prior.py` で次を確かめる: `log_pdf` が `scipy.stats.norm` による
     $\log\phi(\operatorname{logit}x;\mu,s)-\log x-\log(1-x)$ と 4 組の $(\mu,s)$ で一致（相対 1e-13）。
     `scipy.integrate.quad` で密度の積分が 1（1e-10）、$\operatorname{logit}X$ の平均が $\mu$（1e-9）、分散が $s^2$
     （相対 1e-9）。Core の `eval_log_prior` が `log_pdf` と一致し、使わないスロットの勾配は 0 で有限。シミュレーション
     した 3PL（$J=8$、$N=1200$、$g$ に `LogitNormalPrior(-1.3863, 0.6)`、$d$ に `NormalPrior(0, 3)`、`tol=1e-10`）の
     MAP が両方の M-step 尺度で、テスト内に式から書いた事前分布の勾配
     $-1/g+1/(1-g)-(\operatorname{logit}g-\mu)/(s^2g(1-g))$ を使った $\ell(\eta)+\sum\log\pi$ の停留点であり
     （勾配の最大絶対値は自然尺度 3e-8、非拘束尺度 3e-9。テストは 1e-6）、厳密な局所最大（Hessian が負定値）。
     尤度の勾配は mirt で照合済みの周辺対数尤度のもの。同じテストで flexMIRT の logit 尺度の目的関数（Jacobian を
     除いた核）を Newton 法で最大化すると、どの $g$ も `LogitNormalPrior` のときより $1/2$ に近く、差は最大 0.037
     で、2 つの推定値が違うことを確かめる（DESIGN §15.48）。
  3. *`LogitNormalPrior(..., mode_scale="logit")`（logit 尺度のモード、DESIGN §15.48）*: ここでは mirt が参照値。
     mirt の `PRIOR = (..., g, norm, μ, s)` は内部の $\operatorname{logit}g$ に Jacobian なしでかかる正規事前分布。
     `validation/r/m7_logit_norm_prior_reference.R` はシミュレーションした 3PL（$J=8$、$N=1200$、$g$ に
     `norm(-1.3863, 0.6)`、$d$ に `norm(0, 3)`。NR、`TOL = 1e-10`、Oakes の SE）を推定し、R の中で mirt の
     `logPrior` が $\sum\log\phi(\operatorname{logit}g)+\sum\log\phi(d)$ に一致すること（1e-9）と、mirt の推定値が
     スクリプト内に書いた勾配で logit 尺度の目的関数の停留点であること（勾配の最大絶対値 2e-9）を確かめる。mirt は
     $\operatorname{logit}g$ とその SE を出力するので、参照値には $g$ とデルタ法の
     $\mathrm{SE}(g)=g(1-g)\,\mathrm{SE}(\operatorname{logit}g)$ を保存する。flexpyirt は両方の M-step 尺度で、mirt の
     対数尤度と対数事後（テストは相対 1e-9。実測は絶対 4e-10 と 5e-12）、$a,d,g$（テストは 1e-6。実測 2e-8）、
     $a,d,g$ の Oakes の SE（テストは相対 1e-5。実測 3e-8）に一致する。mirt とは独立に、2 の V45 のデータで、両方の
     M-step 尺度の MAP が、テスト内に書いた勾配で $\ell+\sum\log\phi(\operatorname{logit}g;\mu,s)+\sum\log\pi(d)$ の
     停留点であり（最大で自然尺度 2e-8、非拘束尺度 2e-9。テストは 1e-6）、厳密な局所最大であること、そこで
     $\lambda=\operatorname{logit}g$ についての Newton 法で求めた flexMIRT のモードに一致し（5e-9。テストは 1e-7）、
     対数事後も一致すること（1e-12。テストは 1e-8）、Hessian による SE がその独立な目的関数の logit 尺度の Hessian の
     逆行列からのデルタ法の SE に一致すること（相対 7e-9。テストは 1e-6）、`"natural"` のモードとは $g$ で最大 0.037
     違うことを確かめる。単体テストでは、`log_pdf` を $\operatorname{logit}x$ の `scipy.stats.norm.logpdf` と照合し
     （1e-13）、Core のコード `PRIOR_LOGITNORMAL_LOGIT = 5` とその勾配、`mode_scale` の検証、JSON のキー（往復と、
     ないときは `"natural"`）を確かめる。
- **EM の単調性**: すべての照合ケースで、反復ごとの（MAP 時は事後）対数尤度が非減少であることをテストで確認。
- **SE の内部整合**: Oakes の SE と、収束点での周辺対数尤度の直接 `jax.hessian` による SE が一致すること（小規模ケース）。
- **不変性**: 項目の並べ替え、群ラベルの付け替え、パターン圧縮の有無で結果が変わらないこと。
- **ユーザー定義の項目モデル（#86、DESIGN §20）**: `tests/test_custom_items.py` は公開の `ItemModel` / `ItemKernel`
  プロトコルだけを使って 2PL と GRM を実装し直す。1 つの混合モデルで、組み込みの推定と $10^{-12}$ まで一致する
  （対数尤度、反復回数、推定値、Oakes SE、表示用閾値とその SE、EAP/MAP/ML/WLE の得点、S-X2、M2、項目情報量）。
  2 因子モデルは $10^{-5}$ まで一致する（初期値が違う）。これは拡張の仕組みの検証であり、新しいモデルの検証ではない。
  組み込みでない例のモデル（GGUM など）は、検証済みとして示す前にそれ自身の mirt との照合が要る（GGUM には V44 を予約）。
