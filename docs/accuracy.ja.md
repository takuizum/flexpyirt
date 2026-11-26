# 精度と検証

[English](accuracy.md) | 日本語

flexpyirt の結果をどう確かめているか、R `mirt` とどこまで一致するか、両者が意図して異なるのはどこか。

## 結果の確かめ方

flexpyirt は mirt を移植したものではなく、数式から実装している（[理論](theory.ja.md) を参照）。mirt
（Chalmers, 2012）は独立な参照値として使う:

- 参照値は R `mirt` で計算してソースコードと一緒に保存し、変更のたびにテストで照合する。機能は mirt と一致して
  はじめて（mirt に相当する機能がない場合は R の独立な実装と一致してはじめて）対応済みとする。
- 両者を**同じ条件**で動かす: 同じ求積格子、mirt 側はずっと厳しい収束判定、加速なし、同じ事前分布。
- データは mirt に同梱の公開データ（`LSAT7`、`Science`）と、固定シードのシミュレーションデータ（多母集団、欠測、
  固定項目、境界解になりやすいモデルを含む）。

mirt との一致に加えて、参照値によらず成り立つべき性質もテストしている:

- **EM の単調性**: どのケースでも、対数尤度（MAP では対数事後）は反復の間に決して減らない（SQUAREM の有無とも）。
- **標準誤差の内部整合**: Oakes の恒等式と周辺対数尤度の直接の Hessian が丸め誤差の範囲で一致する。
- **不変性**: 受検者や項目の並べ替えで結果が変わらない。

## 許容誤差

テストで許す最大の差。実際にはほとんどのケースでずっと近く一致する（パラメタ・対数尤度・標準誤差はおおむね
$10^{-8}$ 程度）。

| 量 | 許容誤差 |
|---|---|
| 対数尤度 | $10^{-6}$（相対） |
| 項目パラメタ（傾き–切片の尺度） | $10^{-4}$（絶対） |
| 潜在分布の平均・分散 | $10^{-4}$（絶対） |
| 標準誤差 | $10^{-3}$（相対） |
| EAP スコア | $10^{-5}$（絶対） |
| MAP / ML / WLE スコアとその SE | $10^{-4}$（絶対）、$10^{-3}$（相対） |

## 照合の対象

| 機能 | 照合相手 |
|---|---|
| 2PL、Rasch、3PL、4PL、GRM、GPCM、PCM、Nominal | `mirt(..., itemtype = ...)`: 推定値、対数尤度、Oakes の SE |
| 欠測（MAR） | `NA` を含むデータでの `mirt()` |
| 標本ウェイト | `mirt(..., survey.weights = w)` と `multipleGroup(..., survey.weights = w)`: 推定値、対数尤度、AIC、BIC、Oakes の SE（2PL、graded、欠測を含む 2 群、固定項目パラメタ推定）。双因子モデルは `bfactor(..., survey.weights = w)` |
| アンカー項目つき多母集団 | `multipleGroup(..., invariance = c(<アンカー>, "free_means", "free_var"))` |
| 項目間の等値制約 | `mirt.model(... CONSTRAIN = ...)` |
| 固定項目パラメタ推定 | パラメタを固定した `mirt()`（`pars = "values"`、`est = FALSE`） |
| 事前分布と MAP | `mirt.model(... PRIOR = ...)` |
| 表示用閾値とデルタ法の SE | `coef(m, IRTpars = TRUE, printSE = TRUE)` |
| EAP、MAP、ML、WLE スコア | `fscores(m, method = ...)` |
| 確認的 2–3 因子モデル（多母集団も） | `mirt.model` と `COV = F1*F2` を使った `mirt()` / `multipleGroup()` |
| 経験ヒストグラム | 固定項目つきの `mirt(..., dentype = "EH")` |
| Davidian 曲線 | mirt の中で当てはめた厳密な標準化 Davidian 尤度（`createGroup()`）と、mirt 組み込みの `dentype = "Davidian-k"`（下記参照） |
| 混合正規 | 全項目を不変にした `multipleGroup(..., dentype = "mixture-k")` と、R の独立な実装 |
| S-X2 項目適合度 | `itemfit(m, fit_stats = "S_X2")` |
| 項目・テスト情報量、期待得点 | mirt の推定値での `iteminfo()`・`testinfo()`・`expected.item()`・`expected.test()`（2PL、3PL、GRM、GPCM、名義、混合形式、2 群）。相対 $10^{-9}$ まで |
| 多次元の情報量 | すべての項目モデルと交差負荷を含む 2 因子・3 因子モデル、双因子モデル、two-tier モデルの情報行列（`iteminfo(..., multidim_matrix = TRUE)`）、方向別の項目・テスト情報量（`degrees =` を渡した `iteminfo()` / `testinfo()`）と期待得点。相対 $10^{-10}$ まで |
| 信頼性 | `marginal_rxx()`（積分の許容誤差を厳しくした再計算とも照合。平均・分散が 0・1 でない群は、標準化した等価モデルで照合）と `empirical_rxx()`。$10^{-10}$ まで |
| 合計得点による EAP | `fscores(m, method = "EAPsum")` の表と受検者ごとの得点（欠測のある反応パターンが `NA` になる点も含む）。$10^{-9}$ まで |
| $M_2$ / $M_2^*$ | `M2(m)`。mirt と異なる場合は R による厳密な計算（下記参照） |
| DIF 検定 | `DIF(mg, scheme = "add" / "drop")` |
| ロバスト標準誤差 | `mirt(..., SE.type = "crossprod")`・`SE.type = "sandwich"`: mirt の推定値での共分散行列全体と、通しで当てはめたときの標準誤差 |
| SQUAREM | 上のすべての照合を `BAEM(squarem=True)` でも行う |
| M-step の非拘束尺度 | 3PL・4PL、事前分布と MAP、欠測、多母集団、項目パラメタ固定の較正、2–3 因子の確認的モデルの照合を `BAEM(m_step_scale="unconstrained")` でも同じ許容誤差で行う |
| `LogitNormalPrior` | 既定の `mode_scale="natural"` は mirt に同等のものがないため独立に検証する: 密度を SciPy と照合し（積分が 1、logit $X$ の平均と分散が指定どおり）、MAP 推定値が式から書いた対数事後の停留点であることを $10^{-6}$ で確かめる。`mode_scale="logit"` は mirt の `g` への `norm` 事前分布に一致し（対数尤度と対数事後は相対 $10^{-9}$、$a$・$d$・$g$ は $10^{-6}$、Oakes の SE は相対 $10^{-5}$）、独立に計算した flexMIRT の logit 尺度のモードにも一致する |
| ポリコリック相関 | 順序変数に対する lavaan の `lavCor()` と $10^{-6}$ まで、psych の `polychoric()` と $10^{-3}$ まで（その最適化の許容誤差による）、平滑化は psych の `cor.smooth()` |
| 平行分析、MAP、経験的カイザー基準 | 同じ相関行列に対する EFAtools の `EKC()`・`PARALLEL()`、psych の `vss()`、EFA.dimensions の `MAP()`、4 乗版 MAP の R による独立な実装と $10^{-11}$ まで。正規データの参照固有値は Monte Carlo 誤差の範囲で |
| Q3、LD $X^2$ / $G^2$ | mirt で推定した同じモデルでの `residuals(m, type = "Q3" / "LD" / "LDG2")` |
| ジャックナイフ傾き指標 | 同じモデルでの `residuals(m, type = "JSI")`。既定（推定し直す方法）は $10^{-5}$ で一致。推定し直さない近似は推定し直した値と比べ、15 項目程度からは 0.02 程度で一致するが、4〜5 項目のテストでは 1 以上ずれる |
| モデルに基づく平行分析 | 同じ項目パラメタでの ltm の `unidimTest()`。観測された統計量と、Monte Carlo 誤差の範囲でシミュレーションの分布 |
| 因子分析（ULS、ML）と Hull 法 | 同じ行列での EFAtools の `EFA()`・`HULL()` と psych の `fa()`。含意する負荷量は $10^{-4}$、適合度は $2\times10^{-4}$ まで、Hull の選択は同じ |
| 比較データ法 | EFAtools の `CD()`。同じ乱数から同じ母集団、同じ順位和検定、同じ提案 |
| DIMTEST | 単独のプログラム DIMPACK 以外に実装がないので、シミュレーションで確認する。5% 水準での第 1 種の誤りは 0.5〜1%（保守的）、相関 0.7〜0.8 の 2 次元に対する検出力は 99〜100% |
| DETECT（確認的・探索的） | 同じ分割での sirt の `conf.detect(use_sum_score = TRUE, smooth = FALSE)` と `expl.detect(use_sum_score = TRUE)`。条件付き共分散・指標・分割が $10^{-12}$ まで |
| 尺度リンキング（Haebara、Stocking–Lord。1 次元と確認的多次元モデル） | mirt の反応関数を使って規準を計算する R の独立な実装（mirt 自体に特性曲線法のリンキングはない） |
| 古典的項目分析（キーによる採点、α、項目–合計・項目–残りの相関、点双列・双列・多分系列相関、ω total、選択肢の統計量） | psych の `alpha()`、`score.multiple.choice()`、`polyserial()`、`biserial()`、`omega()`、ltm の `cronbach.alpha()` と `biserial.cor()`、polycor の `polyserial()`、素の R。閉じた式は $10^{-12}$、ω は $10^{-6}$ まで |

ポリコリック相関、平行分析、MAP、経験的カイザー基準、因子分析、Hull 法、比較データ法、DETECT、DIMTEST、古典的項目分析は、対になる
パッケージ flexpyctt（NumPy / SciPy。JAX は使わない）の関数である。flexpyirt 自身の `polychoric()` は同じ推定量の
JAX による実装で、同じ照合を通り、すべての照合ケースで 2 つは $10^{-9}$ まで一致する。

## 検証ケース

以下の各ケースはリポジトリのテストである。参照値は R で生成し（V30 はシミュレーション研究による）ソースと一緒に
保存してあり、テストがそれと照合する。推定したモデルは上の許容誤差で mirt と照合する。独自の許容誤差を持つ量は
ケース一覧の後の表に示す。ケースごとの R の呼び出し、設定、測った最大の差は
[検証計画](https://github.com/takuizum/flexpyirt/blob/main/docs/dev/VALIDATION.ja.md)にある。V26 には M7 で
追加した第 2 部がある。

| ID | 機能 | モデルとデータ | 照合先 |
|---|---|---|---|
| V1 | 2PL | `LSAT7`、単一群 | `mirt(..., "2PL")`、Oakes の SE |
| V2 | Rasch | 単一群、潜在分散を推定 | `mirt(..., "Rasch")` |
| V3 | GRM | `Science` | `mirt(..., "graded")` |
| V4 | 多母集団 | 2PL、2 群、一部のアンカー項目、焦点群の平均・分散を推定 | `multipleGroup(..., invariance = c(<アンカー>, "free_means", "free_var"))` |
| V5 | FIPC | 固定項目、潜在分布を推定 | `pars = "values"`・`est = FALSE` での `mirt()` |
| V6 | 経験ヒストグラム | FIPC のもとで 1 群・2 群 | `mirt(..., dentype = "EH")`、`multipleGroup(...)` |
| V7 | 事前分布と MAP | 事前分布つきの 3PL | `mirt.model(... PRIOR = ...)` |
| V8 | GPCM、PCM | `Science` とシミュレーション | `mirt(..., "gpcm")`、`mirt(..., "Rasch")` |
| V9 | 欠測反応 | MAR のデータ | `NA` を含むデータでの `mirt()` |
| V10 | 標準誤差 | Oakes の恒等式（ロバスト SE は V35） | `mirt(..., SE.type = "Oakes")` |
| V11 | 3PL、4PL | シミュレーション、単一群 | `mirt(..., "3PL" / "4PL")` |
| V12 | IRT 閾値とデルタ法 SE | 2PL、Rasch、GRM、GPCM、PCM、2 群の 2PL | `coef(m, IRTpars = TRUE, printSE = TRUE)` |
| V13 | MAP・ML・WLE スコア | 2PL、GRM、GPCM、3PL、2 群、欠測 | `fscores(m, method = ...)` |
| V14 | 確認的 2〜3 因子モデル | 単純構造と交差負荷の 2PL、GRM、3 因子の 2PL | `mirt.model(... COV = ...)` での `mirt()` |
| V15 | S-X2 項目適合度 | GRM、GPCM、PCM、3PL、FIPC。最小セル 1 と 3。2 群では群ごと | `itemfit(m, fit_stats = "S_X2")` |
| V16 | 項目間の等値制約 | 共通の傾き、モデルをまたぐ制約の 2PL + GRM、2 群 | `mirt.model(... CONSTRAIN = ...)` |
| V17 | DIF 検定 | V4 のデータ、`add` と `drop` の方式 | `DIF(mg, scheme = ...)` |
| V18 | $M_2$ / $M_2^*$ | 2PL、3PL、分散を推定する Rasch、GRM、GPCM | `M2(m)` と R による厳密な計算 |
| V19 | Davidian 曲線 | 2PL、GRM、平均・分散を推定する FIPC の 2PL | `mirt(..., dentype = "Davidian-k")` と mirt 内での厳密な尤度 |
| V20 | 尺度リンキング | 2PL、GRM、GPCM、3PL、2PL + GPCM の混在。Haebara と Stocking–Lord の両方の対称形、SE | 別々の `mirt()` 推定 + R による独立な基準関数 |
| V21 | 多母集団の確認的モデル | 2 因子、両因子にアンカー、焦点群の平均・分散を推定 | `COV = F1*F2` での `multipleGroup()` |
| V22 | 混合正規 | 2 成分の 2PL と GRM、3 成分の GPCM | 全項目を不変とした `multipleGroup(..., dentype = "mixture-k")` と R による独立な実装 |
| V23 | 名義反応モデル | `Science`、順序のない得点係数、2PL との混在、2 因子 | `mirt(..., "nominal")`、`itemfit()` |
| V24 | ポリコリック相関 | テトラコリック、2〜5 カテゴリ、10% の MCAR、空セル、`LSAT7`、`Science`。平滑化 | lavaan `lavCor()`、psych `polychoric()`、`cor.smooth()` |
| V25 | 次元数 | V24 の行列での EKC・MAP・固有値。平行分析の参照値 | EFAtools `EKC()`・`PARALLEL()`、psych `vss()`、EFA.dimensions `MAP()` |
| V26 | 局所依存、モデルに基づく平行分析 | `LSAT7` の 2PL、`Science` の GRM、依存するペアと MCAR を含むシミュレーションの 2PL | `residuals(m, type = "LD" / "LDG2" / "Q3")`、ltm `unidimTest()` |
| V26 (M7) | 1 群を超える局所依存 | 2 群。EH・Davidian・混合の潜在分布。2 因子・3 因子 | 対応する mirt の推定結果での `residuals()` |
| V27 | DETECT | 確認的（2 次元・1 次元・`Science`）と探索的 | sirt `conf.detect()`、`expl.detect()` |
| V28 | EFA、Hull 法 | ULS と ML、すべての因子数。CAF、CFI、RMSEA | EFAtools `EFA()`・`HULL()`、psych `fa()` |
| V29 | 比較データ法 | 母集団の生成、順位和検定、3 つのデータ | EFAtools `CD()` とその内部の各段階 |
| V30 | DIMTEST | 2PL、$N$ 1000〜2000、$J$ 20〜30、条件ごとに 200 回 | シミュレーション研究（DIMPACK 以外に参照実装がない） |
| V31 | ジャックナイフ傾き指標 | `LSAT7` の 2PL、`Science` の GRM、シミュレーションの 2PL と GPCM | `residuals(m, type = "JSI")` |
| V32 | 双因子モデル | 特殊因子 3 つと一般因子だけの項目を含む 2PL、テストレット 3 つの GRM | `bfactor()`、`fscores()` |
| V33 | two-tier モデル | 相関する主因子 2 つの 2PL、交差負荷項目を含む GRM | `bfactor(..., model2 = )` |
| V34 | 標本ウェイト | 2PL、GRM、欠測を含む 2 群、FIPC、双因子の 2PL | `survey.weights` つきの `mirt()`・`multipleGroup()`・`bfactor()` |
| V35 | ロバストな標準誤差 | 誤指定の 2PL、混在形式、MAP、2 群、FIPC、2 因子・3 因子、EH | `mirt(..., SE.type = "crossprod" / "sandwich")` |
| V36 | 合計得点の分布 | $J$ 10〜300 の Poisson 二項分布、GRM と混在形式 | poibin `ppoibin()`・`dpoibin()`、`dpois()`、`fscores(method = "EAPsum")` |
| V37 | 情報量・期待得点・信頼性・合計得点 EAP | 2PL（欠測ありも）、3PL、GRM、GPCM、名義、混在形式、2 群 | `iteminfo()`、`testinfo()`、`expected.item()`、`marginal_rxx()`、`empirical_rxx()`、`fscores(method = "EAPsum")` |
| V38 | 双因子・two-tier モデルの局所依存 | MCAR を含む双因子の 2PL、双因子の GRM、two-tier の 2PL | `bfactor()` の推定結果での `residuals()` |
| V39 | 確認的モデルの $M_2$ と S-X2 | V14 の 4 つのモデル | `M2()`、`itemfit()` と R による厳密な $M_2^*$ |
| V40 | リンキング: 4PL・名義・多次元 | 4PL、名義、2PL + 名義の混在。対角・直交・斜交の $\mathbf A$ による 2 因子・3 因子モデル | 別々の `mirt()` 推定 + R による独立な基準関数 |

mirt で推定したモデルとの単純な比較ではないケースの許容誤差:

| 量 | 許容誤差 |
|---|---|
| 同じ項目推定値での尺度リンキングの $(A,B)$ / 基準関数 | $10^{-9}$ / $10^{-9}$（相対） |
| 同じ推定値と共分散での尺度リンキングの SE | $10^{-5}$（相対） |
| 推定からリンキングまで通した $(A,B)$ / 基準関数 / SE | $10^{-6}$ / $10^{-5}$（相対） / $10^{-3}$（相対） |
| ポリコリック相関（lavaan / psych と）、閾値、平滑化 | $10^{-6}$ / $10^{-3}$、$10^{-12}$、$10^{-12}$ |
| EKC・MAP・観測固有値、正規参照の平均 / 95 パーセンタイル | $10^{-11}$、$\lvert z\rvert\le4.5$ / $6\times10^{-3}$ |
| LD $X^2$ / $G^2$、符号付き $V$・Q3、モデルに基づく平行分析の観測値 / シミュレーションの平均 / 95 パーセンタイル | $10^{-5}$（相対）、$10^{-6}$、$2\times10^{-4}$ / $\lvert z\rvert\le4.5$ / 0.02 |
| DETECT の条件付き共分散・指標・Ward の高さ | $10^{-12}$ |
| EFA の $\Lambda\Lambda^\top$ / 共通性、CAF・CFI・RMSEA、$\chi^2$、Hull の適合度 / $st$ | $10^{-4}$、$2\times10^{-4}$、$10^{-4}$（相対）、$10^{-5}$ / $10^{-2}$（相対） |
| 比較データ法: PAF、同じ乱数からの母集団、順位和検定の $p$、RMSE の平均 | $10^{-12}$、完全一致、$10^{-10}$（相対）、残す $k$ で 15%・次の $k$ で 60%（相対） |
| DIMTEST（シミュレーション研究の小さな再実行）: 帰無仮説のもとでの .05 での棄却 / 検出力 | 20 回中 3 回以下 / 10 回中 8 回以上 |
| mirt の推定値でのクロス積・サンドイッチ共分散（要素は $\sqrt{v_{ii}v_{jj}}$ に対する相対）、推定から通したパラメタ / SE | $10^{-9}$ / $10^{-6}$、$10^{-6}$ / $10^{-5}$（相対） |

## mirt との既知の違い

これらの違いは意図したもので、flexpyirt は自分が報告する量を厳密に最大化している。ふつうの使い方では小さいか
現れないが、2 つのプログラムを比べると目にすることがある。

- **有限の格子上の潜在平均・分散.** mirt は正規の潜在分布を期待度数の平均・分散で更新し、flexpyirt は離散化した
  尤度を厳密に最大化する（[理論 4 節](theory.ja.md#4-bockaitkin-em-アルゴリズム)）。格子が分布を切る場合（分散が
  大きい群や平均が 0 から遠い群）は推定値が少しずれ（例: Rasch の分散で $10^{-4}$ 程度）、対数尤度は flexpyirt の方が
  高い。両方で格子を広げると（`Quadrature.equispaced(81, bounds=(-8, 8))` と mirt の
  `technical = list(theta_lim = c(-8, 8))`）一致する。
- **報告される対数尤度.** mirt は最後の E ステップでの対数尤度を報告し、最終的な推定値での値と一致するとは限らない。
  flexpyirt は推定値での値を報告する。
- **3PL / 4PL の漸近線.** mirt は $g$ と $u$ をロジット尺度で推定し、$\mathrm{logit}(g)$ とロジット尺度の SE を報告する。
  flexpyirt は $g$ と $u$ を自然な尺度で推定・報告する。推定値は変換すれば一致する
  （$g=\sigma(\mathrm{logit}\,g)$、$SE_g=g(1-g)\,SE_{\mathrm{logit}\,g}$）。
- **経験ヒストグラム.** mirt はヒストグラムを標準化せずに推定し直すので、全項目が自由だと尺度が識別されず EM が
  漂流する。flexpyirt は参照群のヒストグラムを M ステップの中で平均 0・分散 1 に制約する。固定項目が尺度を決める
  場合（FIPC）は両者が一致する。
- **Davidian 曲線.** mirt の `dentype = "Davidian-k"` は毎サイクル期待度数を補間で標準化するため、EM の単調性が
  崩れ、より低い対数尤度で終わることがある。flexpyirt は離散化した尤度を厳密に最大化するので、推定値は mirt の
  組み込みの Davidian 推定とは異なり、テストしたケースでは対数尤度が高い。mirt の組み込みの Davidian 曲線には
  標準誤差も、潜在平均・分散の自由推定もない。
- **混合正規.** mirt の `dentype = "mixture-k"` は既定でクラスごとの項目パラメタを推定する（混合 IRT モデル）。
  flexpyirt の `NormalMixture` は項目パラメタが全体で共通で、全項目をクラス間で等値にした mirt のモデルに
  $\theta$ の尺度を除いて対応する。混合モデルに対する mirt の Oakes の SE は観測情報量の SE ではないが、
  flexpyirt の SE は観測情報量によるものである。
- **Nominal モデル.** mirt の既定の初期値では鞍点で止まることがある（例: `Science`）。flexpyirt の既定の初期値は、
  テストしたすべてのケースで最良の解に達した。ML スコアでは、mirt は全最低・全最高*カテゴリ*のパターンにすべて
  $\pm\infty$ を返すが、flexpyirt はカテゴリ傾き $a\,\mathrm{ak}_c$ で判断するので、$\mathrm{ak}_1<0$ での全 0 の
  ようなパターンは有限の推定値になる。
- **$M_2$.** flexpyirt は自由な潜在パラメタを微分行列 $\Delta_2$ に含め（mirt は含めない。たとえば Rasch の分散や、確認的多次元モデルの因子間相関。
  後者ではテストしたケースで flexpyirt の $M_2$ が最大 0.07% 小さい）、
  3PL / 4PL と GPCM の項目では厳密な微分を使う（mirt 1.46 の微分はここで不正確）。これらのモデルでは R による
  厳密な計算と照合しており、$M_2$ は mirt の値と異なる（テストした 3PL・GPCM のケースで 5–11%）。TLI と CFI は
  2 値モデルでのみ報告する。多値項目では独立帰無モデルがランク落ちするためである。
- **多次元の ML スコア.** ML 推定値のある次元が発散すると、flexpyirt は $\pm\infty$（SE は NaN）を報告するが、
  mirt は境界付近（$|\theta|\approx20$）で SE 0 のまま止まることがある。

すべてのケース、mirt の呼び出し、測った差を載せた完全な検証計画は、GitHub の
[検証計画](https://github.com/takuizum/flexpyirt/blob/main/docs/dev/VALIDATION.ja.md) にある。
## 他のパッケージとの既知の違い

- **EFA.dimensions の 4 乗版 MAP。** EFA.dimensions (0.1.9.3) は「4 乗」の MAP 統計量を行列のべき乗 $(\operatorname{tr}(R^{*4})-p)/(p(p-1))$ で計算しており、Velicer, Eaton & Fava (2000) のように各偏相関を 4 乗していない（値が 1 を超えることがある）。flexpyirt は要素ごとの 4 乗を使う。2 乗版は EFA.dimensions・psych と一致する。
- **sirt の `conf.detect()` の総得点による DETECT。** `use_sum_score = TRUE` で既定の `smooth = TRUE` のまま
  だと、sirt 4.2 は総得点を標準化してから 2 項目の得点を引くため、「残りの得点」の群が残りの得点の群にならず、
  指標が変わる（あるテストケースで DETECT 1.60 が 6.73 になる）。`smooth = FALSE` と `expl.detect()` では sirt と
  flexpyirt は一致する。
- **psych の `alpha()` と ltm の `cronbach.alpha()` の欠測の扱い。** 欠測があるとき、psych の `raw.r` は各項目と
  観測された得点の和（欠測は 0 点）との相関で、α と `r.drop` はペアワイズの共分散を使う。flexpyctt の `item_total`
  はそのペアワイズの共分散から求める相関なので `raw.r` とは違う（欠測 6% のテストケースで最大 0.014）。ltm の素の α は
  `na.rm = TRUE` のとき合計の分散を欠測のない行から、項目の分散をすべての回答から求める。flexpyctt の
  `missing="listwise"` はどちらも欠測のない行から求める。欠測がなければすべて一致する。
- **psych の `biserial()`** は合計得点の分母 $n-1$ の標準偏差で割るので、2 段階の双列相関（polycor の値）より
  $\sqrt{(n-1)/n}$ 倍小さい。

## 参考文献

- Chalmers, R. P. (2012). mirt: A multidimensional item response theory package for the R environment. *Journal of Statistical Software*, 48(6).
