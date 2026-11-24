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
| $M_2$ / $M_2^*$ | `M2(m)`。mirt と異なる場合は R による厳密な計算（下記参照） |
| DIF 検定 | `DIF(mg, scheme = "add" / "drop")` |
| SQUAREM | 上のすべての照合を `BAEM(squarem=True)` でも行う |
| ポリコリック相関 | 順序変数に対する lavaan の `lavCor()` と $10^{-6}$ まで、psych の `polychoric()` と $10^{-3}$ まで（その最適化の許容誤差による）、平滑化は psych の `cor.smooth()` |
| 平行分析、MAP、経験的カイザー基準 | 同じ相関行列に対する EFAtools の `EKC()`・`PARALLEL()`、psych の `vss()`、EFA.dimensions の `MAP()`、4 乗版 MAP の R による独立な実装と $10^{-11}$ まで。正規データの参照固有値は Monte Carlo 誤差の範囲で |
| Q3、LD $X^2$ / $G^2$ | mirt で推定した同じモデルでの `residuals(m, type = "Q3" / "LD" / "LDG2")` |
| モデルに基づく平行分析 | 同じ項目パラメタでの ltm の `unidimTest()`。観測された統計量と、Monte Carlo 誤差の範囲でシミュレーションの分布 |
| 因子分析（ULS、ML）と Hull 法 | 同じ行列での EFAtools の `EFA()`・`HULL()` と psych の `fa()`。含意する負荷量は $10^{-4}$、適合度は $2\times10^{-4}$ まで、Hull の選択は同じ |
| 比較データ法 | EFAtools の `CD()`。同じ乱数から同じ母集団、同じ順位和検定、同じ提案 |
| DIMTEST | 単独のプログラム DIMPACK 以外に実装がないので、シミュレーションで確認する。5% 水準での第 1 種の誤りは 0〜2%（保守的）、相関 0.7〜0.8 の 2 次元に対する検出力は 99〜100% |
| DETECT（確認的・探索的） | 同じ分割での sirt の `conf.detect(use_sum_score = TRUE, smooth = FALSE)` と `expl.detect(use_sum_score = TRUE)`。条件付き共分散・指標・分割が $10^{-12}$ まで |
| 尺度リンキング（Haebara、Stocking–Lord） | mirt の反応関数を使って規準を計算する R の独立な実装（mirt 自体に特性曲線法のリンキングはない） |

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
- **$M_2$.** flexpyirt は自由な潜在パラメタを微分行列 $\Delta_2$ に含め（mirt は含めない。たとえば Rasch の分散）、
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

## 参考文献

- Chalmers, R. P. (2012). mirt: A multidimensional item response theory package for the R environment. *Journal of Statistical Software*, 48(6).
