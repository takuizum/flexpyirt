# API リファレンス

[English](reference.md) | 日本語

flexpyirt の公開 API（`from flexpyirt import ...` で使えるもの）と、対になるパッケージ flexpyctt の公開 API の一覧。
使い方の流れは[ユーザーガイド](guide.ja.md)、数式とアルゴリズムは [理論](theory.ja.md) を参照。
各関数の docstring にも同じ内容と対応する式・出典がある（`help(flexpyirt.fit)` など）。

**flexpyctt** は、推定済みのモデルを使わない道具をまとめたパッケージである。[相関](#相関)と、[次元数](#次元数)のうち
モデルを使わないものが入る（どちらのパッケージのものかは各項目に書く）。依存は NumPy、SciPy、pandas だけなので、
JAX なしでインストールでき（`pip install flexpyctt`）、64 ビットの設定も要らない。flexpyirt をインストールすると
一緒に入る。関数は `from flexpyctt import ...` で使う。flexpyirt も同じ名前で再公開しているが、`polychoric` だけは
別で、`flexpyirt.polychoric` は同じ推定量の JAX による実装である。乱数を使う関数は、flexpyctt では `rng`（整数の
シードまたは `numpy.random.Generator`）、flexpyirt では JAX の PRNG `key` を受け取る。

公開関数の引数は、最初の 1〜2 個を除いてキーワード専用（`*` の後ろ）。

- 推定: [`fit`](#fit), [`FitResult`](#fitresult)
- 尺度リンキング: [`link`](#link), [`LinkResult`](#linkresult)
- 相関（flexpyctt）: [`polychoric`](#polychoric), [`PolychoricResult`](#polychoricresult)
- 次元数（`model_parallel_analysis` と `dimensionality` 以外は flexpyctt）: [`parallel_analysis`](#parallel_analysis), [`velicer_map`](#velicer_map), [`empirical_kaiser`](#empirical_kaiser), [`efa`](#efa), [`hull`](#hull), [`comparison_data`](#comparison_data), [`model_parallel_analysis`](#model_parallel_analysis), [`detect`](#detect), [`exploratory_detect`](#exploratory_detect), [`dimtest`](#dimtest), [`dimensionality`](#dimensionality) とその結果
- 情報関数と信頼性: [`item_information`](#item_information), [`test_information`](#test_information), [`expected_score`](#expected_score), [`reliability`](#reliability)
- 合計得点: [`summed_score_table`](#summed_score_table), [`summed_score_distribution`](#summed_score_distribution)
- モデル比較とシミュレーション: [`compare_fits`](#compare_fits), [`simulate_responses`](#simulate_responses), [`SimulatedData`](#simulateddata)
- モデル: [`Model`](#model)
- 項目モデル: [`TwoPL`](#twopl), [`Rasch`](#rasch), [`ThreePL`](#threepl), [`FourPL`](#fourpl), [`GRM`](#grm), [`GPCM`](#gpcm), [`PCM`](#pcm), [`Nominal`](#nominal)
- 事前分布: [`NormalPrior`](#normalprior), [`LogNormalPrior`](#lognormalprior), [`BetaPrior`](#betaprior)
- 潜在分布: [`Normal`](#normal)
- 設定: [`BAEM`](#baem), [`Quadrature`](#quadrature)
- 例外と警告: [`flexpyirt.errors`](#flexpyirterrors)

---

## 推定

### `fit`

```text
fit(spec, data, *, group=None, table=None, item_names=None, method=None,
    quadrature=None, se="oakes", weights=None) -> FitResult
```

Bock–Aitkin EM による周辺最尤推定（事前分布を付けた場合は Bayes modal / MAP 推定。Bock & Aitkin, 1981）。

| 引数 | 型 | 説明 |
|---|---|---|
| `spec` | `Model` | モデルの指定 |
| `data` | DataFrame / polars DataFrame / 2 次元配列 | 行が回答者・列が項目。値は `0..K-1`、欠測は NaN。列はモデルの項目とちょうど一致すること |
| `group` | 配列（長さ = 行数）, 任意 | 多母集団モデルの群ラベル。`Model.with_groups()` で宣言した群名 |
| `table` | DataFrame, 任意 | `Model.parameter_table()` を編集した表。指定すると `spec` の固定・等値・事前分布・初期値より優先 |
| `item_names` | list[str], 任意 | `data` が配列のときの列名 |
| `method` | `BAEM`, 任意 | EM の設定（既定 `BAEM()`） |
| `quadrature` | `Quadrature`, 任意 | 求積格子（既定: 1 因子は 61 点、2 因子は各軸 31 点、3 因子は各軸 15 点、いずれも $[-6, 6]$） |
| `se` | `"oakes"` / `"hessian"` / `"sandwich"` / `"crossprod"` / `None` | 標準誤差の計算方法: 観測情報行列の逆行列（`"oakes"`・`"hessian"`）、パターンごとのスコアのクロス積の逆行列（`"crossprod"`。mirt の `SE.type = "crossprod"`）、ロバストなサンドイッチ共分散（`"sandwich"`。mirt の `SE.type = "sandwich"`）。どれも MAP 時は事前分布の Hessian を含む。`"sandwich"`・`"crossprod"` は `DavidianCurve`・`NormalMixture`、双因子・two-tier モデル、`weights` との併用では `SpecificationError`。`None` なら計算しない |
| `weights` | 配列（長さ = 行数）, 任意 | 標本ウェイト。有限で非負の数（mirt の `survey.weights`）。推定値は $\sum_i w_i\log L_i$ を最大化する。ウェイトは与えられたまま使う（再スケールしない）。ウェイト 0 の行は無視する。整数ウェイトは行の繰り返しと同じ。このとき `se="oakes"` / `"hessian"` は重み付き情報行列の逆行列で、度数ウェイトでのみ正しく、標本ウェイトでは正しくない（`FlexpyirtWarning` で知らせる） |

**戻り値**: [`FitResult`](#fitresult)

**例外**: `PrecisionError`（JAX の 64 ビットが無効）、`SpecificationError`（仕様やデータの誤り）、
`IdentificationError`（識別不能）。いずれも推定を始める前に出る。

**警告**: `ConvergenceWarning`（`max_iter` に到達、または SQUAREM が通常 EM へ頻繁に退避）、`HeywoodWarning`（自由な漸近線が境界から `1e-4` 以内、自由な傾きが $|a| > 20$、または `NormalMixture` の成分の退化）、`FlexpyirtWarning`（情報行列が正定値でない、アンカーが 1 パラメタだけ、`weights` と `se="oakes"` / `"hessian"` の併用）。

### `FitResult`

`fit()` の戻り値。変更不可。

| 属性 | 型 | 説明 |
|---|---|---|
| `items` | DataFrame | 1 項目 1 行（多母集団では `group` 列つきで群 × 項目）。`model`、内部パラメタ（`a` または `a1`…`aD`、`d` / `d1`…、`g`, `u`）、`<param>_se`、および 1 次元モデルでは表示用の `b` / `bk`（$-d/a$、GRM は $-d_k/a$、GPCM/PCM は $-(d_k-d_{k-1})/a$、$d_0=0$）とデルタ法による `b_se` / `bk_se`（`vcov` と JAX 自動微分から計算。`se=None` または閾値に入るパラメタがすべて固定なら NaN） |
| `latent` | DataFrame | 1 群 1 行。1 因子では `group`, `mean`, `var`, `mean_se`, `var_se`、2–3 因子では `group`, `mean_1`…`mean_D`, `var_1`…`var_D`, `cor_12`… とその `_se` 列（固定値の SE は NaN）。`EmpiricalHistogram` では各ヒストグラムの `group`, `mean`, `var`, `sd`, `skewness`, `kurtosis`（超過尖度）。`NormalMixture` では `mean`, `var`、素の形状 `rmean_c`, `rvar_c`, `logit_c` とその SE、続いて標準化した成分 `pi_c`, `mean_c`, `var_c`（$c = 1..k$）とデルタ法の `_se` 列 |
| `latent_weights` | DataFrame / None | `EmpiricalHistogram` のみ: `group`, `node`, `weight`（群 × 求積節点ごとに 1 行）。`Normal` では None |
| `parameters` | DataFrame | 検証済みの ParameterTable に `estimate`（推定値。固定なら固定値）と `se` を足したもの |
| `loglik` | float | 推定値での周辺対数尤度（尤度のみ、事前分布は含めない） |
| `log_posterior` | float | 推定値での対数事後確率（$\ell(\hat\eta) + \log\pi(\hat\eta)$）。事前分布なしでは `loglik` と同値 |
| `aic`, `bic` | float | $-2\ell + 2k$、$-2\ell + k\log N$（$k$ = `n_params`, $N$ = `n_obs`。`weights` を付けた推定ではウェイトの和）。事前分布が有効なときは `FlexpyirtWarning` を出す |
| `n_params` | int | 自由パラメタ数（等値制約は 1 つと数える） |
| `n_obs` | int | 回答者数（`weights` を付けた推定ではウェイトが正の行の数） |
| `converged` | bool | 収束基準を満たしたか |
| `n_iter` | int | 採択された EM（または SQUAREM）の反復回数 |
| `n_em_evaluations` | int | EM 写像（E-step + M-step）の総評価回数。通常 EM では `n_iter` と同値、SQUAREM では採択 1 回あたり 2 または 3 |
| `n_safeguards` | int | SQUAREM の単調性・実行可能性の安全装置が介入した反復数（通常 EM では 0） |
| `history` | DataFrame | `iter`, `loglik`, `log_posterior`（その反復に入るときの値。反復ごとに単調非減少）, `max_change`, `em_evaluations`（累積 EM 写像評価回数）, `safeguard`（SQUAREM 安全装置が発動したか） |
| `vcov` | ndarray / None | 自由パラメタの共分散行列（`free_id` 順）。`se=None` なら None |
| `se_method` | str / None | `"oakes"` / `"hessian"` / `"sandwich"` / `"crossprod"` / None |

`weights` を付けて推定した結果は、`summary()` にウェイトの和を表示する。その `item_fit()`、`m2()`、
`local_dependence()`、`jsi()`、`dif()` は `SpecificationError` を出す（これらの統計量はウェイトなしのデータに
対して定義されている）。`model_parallel_analysis()` も同じ。`score()` はふだんどおり使える。

#### `FitResult.log_posterior`

```text
res.log_posterior -> float
```

自然尺度で評価した自由パラメタ全体の対数事後確率 $\ell(\hat\eta) + \sum_i \log\pi_i(\hat\eta_i)$
（`free_id` ごとに 1 回計上、固定パラメタの事前分布は無視）。事前分布がないときは `res.loglik` と同値。

#### `FitResult.summary`

```text
summary(*, digits=4) -> str
```

モデルと推定種別（`MMLE` / `MAP`）、回答者数・項目数・自由パラメタ数、群情報、収束状況と反復回数・EM 写像評価回数
（SQUAREM 安全装置が働いた場合はその回数）、SE 計算方法、求積格子、周辺対数尤度・対数事後確率・AIC・BIC
（事前分布が有効な場合は実行時警告を出さずに本文に注意文を記載）、潜在分布表、および標準誤差つきの項目パラメタ表
（デルタ法 `b_se` / `bk_se` を含む）を読みやすいテキスト形式で返す。`digits` が非負整数でない場合は
`SpecificationError` を送出する。

Jupyter ノートブックでは `FitResult._repr_html_()` が同じ内容を外部リソースなしの素の HTML/CSS
（項目名・群名は HTML エスケープ済み）で描画する。`repr(res)` は短い 1 行要約のまま保つ。

#### `FitResult.score`

```text
score(data, *, method="EAP", group=None, item_names=None, key=None, n_draws=5) -> DataFrame
```

推定済みモデルで回答者を採点する（[ユーザーガイド 7 節](guide.ja.md#7-スコアリング)）。`"EAPsum"` 以外の手法では
欠測反応を尤度（WLE では $I(\theta)$ も）から除く。

| 引数 | 型 | 説明 |
|---|---|---|
| `data` | DataFrame / polars DataFrame / 2 次元配列 | 推定と同じ項目の反応データ（推定に使ったものでなくてよい） |
| `method` | `"EAP"` / `"MAP"` / `"ML"` / `"WLE"` / `"PV"` / `"EAPsum"` | 採点手法（既定 `"EAP"`）: EAP（推定時の求積格子上の事後平均・事後 SD; Bock & Mislevy, 1982）、MAP（連続正規事前分布 $N(\mu_g,\sigma_g^2)$ による事後最頻値と観測曲率 SE）、ML（最尤推定と観測情報量 SE。有限最大値のないパターンは $-\infty$/$+\infty$・SE NaN、全欠測は NaN）、WLE（Warm, 1989; Firth, 1993: $\partial\log L/\partial\theta + I'(\theta)/(2I(\theta))=0$ を解く。極端パターンでも有限、全欠測は NaN、SE は WLE 目的関数の観測曲率）、PV（501 点格子上の区分線形逆 CDF 補間による回答者ごと独立な `n_draws` 個の事後標本; Mislevy, 1991）、EAPsum（合計得点を与えたときの事後平均と事後 SD。回答者の群の [`summed_score_table`](#summed_score_table) を引く。欠測が 1 つでもある行は NaN。1 次元で潜在分布が `Normal` のモデルのみ） |
| `group` | 配列（任意） | 多母集団モデルでは必須。行ごとの群ラベル |
| `item_names` | list[str]（任意） | `data` が配列のときの列名 |
| `key` | `jax.Array`（任意） | 明示的な JAX PRNG キー（`jax.random.key()` または `jax.random.PRNGKey()`）。`method="PV"` では必須、他の手法では `None` |
| `n_draws` | int | `method="PV"` で回答者ごとに抽出する plausible values の個数（既定 `5`、1 以上の整数） |

**戻り値**: 回答者 1 行ずつの `DataFrame`（`data` が DataFrame ならその index を引き継ぐ）。1 次元モデルでは
`"EAP"`, `"MAP"`, `"ML"`, `"WLE"`, `"EAPsum"` が `theta`, `se` 列、`"PV"` が `pv1`…`pv{n_draws}` 列。多次元モデルでは
宣言した因子順に `theta_<factor>`, `se_<factor>` 列（`"PV"` は `pv<m>_<factor>`）。

**例外**: 不正な `method`、`EmpiricalHistogram`・`DavidianCurve`・`NormalMixture` で `method="MAP"` を指定した場合、`"PV"` で
`key` が未指定・不正、決定的手法に `key` を渡した場合、または `n_draws` が正の整数でない場合に `SpecificationError`。

#### `FitResult.item_fit`

```text
item_fit(data, *, min_cell=1.0, group=None, item_names=None) -> DataFrame
```

Orlando–Thissen の S-X2 項目適合度（[ガイド 5 節](guide.ja.md#項目適合度s-x2)）。各項目について、合計得点
$1,\dots,S-1$ ごとの各カテゴリの観測度数とモデルによる期待度数を比べる。期待度数は、その項目を除いた
Lord–Wingersky 漸化式と、推定時の求積格子上の推定済み潜在分布の重みで計算する。度数の少ないセルは mirt の
`itemfit()` と同じ方法でまとめ（`min_cell` は mirt の `mincell`）、$df_j$ = 残ったセル数 − 行数 − その項目の自由パラメタ数 −
潜在分布の自由パラメタ数（mirt の約束）。確認的多次元モデル（`with_factors()`、単一群）では、推定時の直積格子の
すべてのノードで再帰を回して多変量正規の重みで積分し、自由な因子間相関を潜在分布のパラメタとして数える。

| 引数 | 説明 |
|---|---|
| `data` | 推定した項目への欠測のない反応（通常は推定に使ったデータ） |
| `min_cell` | まとめる前のセルあたり最小期待度数（既定 `1.0`） |
| `group` | 群ラベル（多母集団モデルでは必須。群ごとに S-X2 を計算する） |
| `item_names` | `data` が配列のときの列名 |

**戻り値**: `item`, `s_x2`, `df`, `rmsea`（$\sqrt{\max(0, X^2/df-1)/(N-1)}$）, `p` 列の `DataFrame`。まとめた結果
`df <= 0` になる項目は `df = 0` で統計量は NaN。多母集団モデルでは先頭に `group` 列がつく（群と項目ごとに 1 行）。

**例外**: 双因子・two-tier・多母集団の多次元・`EmpiricalHistogram`・`DavidianCurve`・`NormalMixture` のモデル、欠測を含むデータ、（群内で）観測されないカテゴリがある
項目、正でない `min_cell` の場合に `SpecificationError`。

#### `FitResult.m2`

```text
m2(data, *, item_names=None, ci=0.90) -> DataFrame
```

Maydeu-Olivares & Joe (2005, 2006) による限定情報の適合度統計量 $M_2$ / $M_2^*$
（[ガイド 5 節](guide.ja.md#限定情報の適合度m2)）。$J$ 個の項目得点平均と $\binom{J}{2}$ 個の交差積から計算する
（mirt の既定 `M2(type = "M2*")` に対応。2 値項目では厳密に $M_2$）。ヤコビ行列
$\Delta_2 = \partial\pi_2/\partial\eta^\top$ は JAX 自動微分により、すべての自由な項目・潜在分布パラメタを含む
（共有パラメタは 1 回）。確認的多次元モデル（`with_factors()`）では推定時の直積格子の上で積分し、$\Delta_2$ は自由な
因子間相関ごとに 1 列を持つ（mirt の `M2()` は潜在分布の列を入れないので、値がわずかに大きい）。

| 引数 | 説明 |
|---|---|
| `data` | 推定した項目への欠測のない反応（通常は推定に使ったデータ） |
| `item_names` | `data` が配列のときの列名 |
| `ci` | RMSEA 信頼区間の信頼水準（`(0, 1)` の開区間、既定 `0.90`） |

**戻り値**: 1 行の `DataFrame`。列は `m2`, `df`（$J + \binom{J}{2} - k$）, `p`, `rmsea`,
`rmsea_<lo>`, `rmsea_<hi>`（`ci` に従う。例: `0.90` で `rmsea_5`, `rmsea_95`）, `srmsr`,
`tli`, `cfi`。`df <= 0` または $\Delta_2$ がランク落ちしているとき（1 次・2 次モーメントから局所識別できないとき）は
`FlexpyirtWarning` を出し、`m2`・`p`・RMSEA・信頼区間・`tli`・`cfi` を `NaN` とする（`df` と `srmsr` は有限のまま）。
多値モデルでは `tli` と `cfi` を `NaN` とする（独立帰無モデルは 2 値項目でのみ定義されるため）。2 値モデルで帰無モデルの $M_2$ が当てはめモデルの $M_2$
以下のときは `FlexpyirtWarning` を出し `tli`・`cfi` を `NaN` とする。

**例外**: JAX 64 ビットが無効なとき `PrecisionError`。双因子・two-tier・多母集団・`EmpiricalHistogram`・`NormalMixture`・事前分布つきの
MAP モデル、欠測を含むデータ、不正な `ci`、項目数または回答者数が 2 未満のとき `SpecificationError`。

#### `FitResult.local_dependence`

```text
local_dependence(data, *, group=None, item_names=None) -> LocalDependenceResult
```

すべての項目ペアの局所依存の統計量を求める
（[ガイド 18 節](guide.ja.md#18-局所依存と次元性のレポート)）。Yen (1984) の Q3（EAP の $\hat\theta$ による残差
$x_{ij} - E(X_j \mid \hat\theta_i)$ の相関）、Q3\* $=$ Q3 $-$ Q3 の平均（Christensen, Makransky & Horton, 2017）、
推定した格子でモデルが含意する分割表と比べる Chen & Thissen (1997) の LD $X^2$ / $G^2$（$df = (K_i-1)(K_j-1)$、
$p$ 値、符号付きの Cramér の $V$）。mirt の `residuals(type = "Q3" / "LD" / "LDG2")` と同じ。欠測はペアごとに扱う。

期待度数の表と EAP には、推定した格子の上の推定した潜在分布を使う。`Normal`、`EmpiricalHistogram`、
`DavidianCurve`、`NormalMixture` と、2〜3 因子のモデルでは直積格子の上の多変量正規分布（EAP はベクトル）。
多母集団モデルでは `group` が必要で、すべての統計量を群ごとに、その群の受験者・項目パラメタ・潜在分布で求める。

双因子・two-tier モデル（`with_bifactor()`、`with_two_tier()`）では、期待度数表を完全な直積格子ではなく縮約した
格子の上で求める。同じ特殊因子内のペアは主因子とその特殊因子について和をとり、それ以外のペアは各項目の特殊因子を
先に積分してから主因子について和をとる。Q3 は全因子の EAP を使う。結果は、同じ格子での `bfactor()` の適合に対する
mirt の `residuals()`（直積格子を作る）と一致する。同じ特殊因子内のペアは、その因子を超える依存について検定される。

mirt の既定との違いが 2 つある。`dentype = "EH"` / `"Davidian-k"` の推定では、mirt は
`residuals(m, type = "LD", use_dentype_estimate = TRUE)` や
`residuals(m, type = "Q3", Theta = fscores(m, use_dentype_estimate = TRUE))` と呼ばないかぎり標準正規の
事前分布を使う。また、mirt の Q3 は `fscores()` 独自の格子（1 / 2 / 3 因子で各次元 121 / 61 / 31 点）で
スコアを求めるが、flexpyirt は推定した格子を使う。

| 引数 | 型 | 既定値 | 説明 |
|---|---|---|---|
| `data` | DataFrame または配列 | | 推定した項目への回答。欠測は NaN |
| `group` | 配列 | `None` | 群のラベル（多母集団モデルでは必須） |
| `item_names` | list of str | `None` | `data` が配列のときの列名 |

**戻り値**: [`LocalDependenceResult`](#localdependenceresult)

**例外**: `PrecisionError`。不正なデータや群のラベルは `SpecificationError`。

#### `FitResult.jsi`

```text
jsi(data, *, item_names=None, method="refit", newton_steps=1, tol=1e-8, max_iter=100) -> JSIResult
```

単一群・1 次元・`Normal` 潜在分布で項目が 3 つ以上のモデルについて、Edwards, Houts & Cai (2018) のジャックナイフ
傾き指標を求める（[ガイド 18 節](guide.ja.md#18-局所依存と次元性のレポート)）。各項目 $j$ について項目 $j$ を除いて
モデルを推定し直し（その反応を欠測として扱うので、潜在分布・制約・事前分布は指定どおり残る）、他のすべての項目
$i$ について

$$\mathrm{JSI}_{ji} = \frac{\hat a_i - \hat a_i^{(-j)}}{\mathrm{SE}(\hat a_i^{(-j)})}$$

を求める。自由な傾きを持たない項目は NaN。`data` は推定に使ったデータ。

| 引数 | 意味 |
|---|---|
| `method` | `"refit"`（既定）: 全項目モデルの推定値から、最大のステップが `tol` 未満になるまで Newton–Raphson で推定し直し、SE も推定し直したモデルから求める。mirt の `residuals(type = "JSI")` と同じ。`"newton"`: Newton 法を `newton_steps` ステップだけ行う（Taylor 近似）。`"one_step"`: 全項目モデルの共分散行列で 1 ステップ進め、分母にも全項目モデルの SE を使う。全体で E ステップ $J$ 回分ほど |
| `newton_steps` | `"newton"` のステップ数 |
| `tol`, `max_iter` | `"refit"` の収束条件 |

近似は 15 項目程度から推定し直した値と 0.02 程度で一致する。10 項目未満では 1 項目を除いたときの推定値の変化が
大きすぎるので、警告（`FlexpyirtWarning`）を出す。

**戻り値**: [`JSIResult`](#jsiresult)

**例外**: `PrecisionError`。対応しないモデル、項目が 3 つ未満、不正な指定、標準誤差のない推定結果での `"one_step"` は
`SpecificationError`。

**警告**: 推定し直しが収束しなければ `ConvergenceWarning`。

#### `FitResult.dif`

```text
dif(data, *, group, items, params=None, scheme="add", p_adjust="holm", item_names=None) -> DataFrame
```

多母集団モデルの群間の尤度比検定による DIF（[ガイド 9 節](guide.ja.md#特異項目機能dif)）。mirt の `DIF()` に相当。
項目ごとに、その項目の `params` を群間で等値にして（`scheme="add"`）または群ごとに自由にして（`scheme="drop"`）
モデルを推定し直し（同じ `BAEM` 設定と求積、この推定値から開始）、
$X^2 = 2(\ell_{\text{free}} - \ell_{\text{constrained}})$、$df = |k_{\text{base}} - k_{\text{new}}|$ を求める。

| 引数 | 説明 |
|---|---|
| `data`, `group` | この結果を推定したときの反応と群ラベル |
| `items` | 検定する項目（項目ごとに 1 回推定し直す） |
| `params` | 検定する内部パラメタ（既定: 各項目の全パラメタ） |
| `scheme` | `"add"`（制約する。推定済みモデルが基準）または `"drop"`（自由にする） |
| `p_adjust` | `adj_p` の補正。`"holm"`（既定、mirt と同じ）、`"BH"`（Benjamini–Hochberg）、`"none"` |
| `item_names` | `data` が配列のときの列名 |

**戻り値**: `item`, `converged`, `aic`, `sabic`, `hq`, `bic`（項目を自由にしたモデル − 制約したモデル。負なら DIF を
支持）, `x2`, `df`, `p`, `adj_p` 列の `DataFrame`。

**例外**: 単一群のモデル、事前分布つきの MAP 推定、未知の項目やパラメタ、検定するものがない項目（`"add"` で
すでに等値、`"drop"` ですでに自由）では `SpecificationError`。推定し直すモデルが識別できない（例: 唯一のアンカーを
自由にする）と `IdentificationError`。

#### `FitResult.to_json`

```text
to_json(path=None, *, indent=2) -> str
```

推定結果を厳格な RFC 8259 JSON 文字列に直列化し、`path`（`str` または `os.PathLike`）が指定されたときは
UTF-8 ファイルにも書き出す。`pickle` を使わないバージョン付き JSON 形式で、最上位に
`"format": "flexpyirt.FitResult"`、`"schema_version": 7`（version 1–6 のファイルも読める）、`"flexpyirt_version"` を記録する。多次元の結果は因子構造も保存する。有限な
float64 値は `repr` 厳密な JSON 数値として、非有限値（`NaN`, `+inf`, `-inf`）は浮動小数点型フィールドに
`"NaN"`, `"Infinity"`, `"-Infinity"` として明示的に符号化するため、すべての属性がビット単位で正確に往復する。

| 引数 | 型 | 説明 |
|---|---|---|
| `path` | `str` / `os.PathLike`（任意） | UTF-8 JSON を書き出すファイルパス（省略可） |
| `indent` | `int` / `None` | `json.dumps` に渡すインデント幅（既定 `2`、`None` で整形なしのコンパクト出力） |

**戻り値**: `str`（直列化された JSON 文字列）。

**例外**: `indent` が不正、または未登録のモデル・事前分布型が含まれる場合に `SpecificationError`。

#### `FitResult.from_json`

```text
FitResult.from_json(source) -> FitResult
```

ファイルパス（`str` または `os.PathLike`）または JSON 文字列から再推定なしで `FitResult` を復元する。項目モデル・
潜在分布・事前分布・求積格子・推定設定は明示的な名前レジストリで復元する（`eval` や `pickle` は一切使わない）。
復元した結果はすべての属性・`items`・`latent`・`summary()`・`_repr_html_()`・`aic`/`bic`・`score()` で完全に
同一の結果を返す。スコアリングに 64 ビット浮動小数点が必要なため入口で `check_x64()` を呼ぶ。

| 引数 | 型 | 説明 |
|---|---|---|
| `source` | `str` / `os.PathLike` | `to_json()` で書き出した JSON ファイルパス、または JSON 文字列 |

**戻り値**: `FitResult`

**例外**: JAX の 64 ビット浮動小数点が無効なら `PrecisionError`。不正・非厳格な JSON（生の `NaN`/`Infinity`
リテラルなど）、未知の `format` タグ・スキーマ版・登録名、または内部整合性違反なら `SpecificationError`。存在しない
ファイルパスなら `FileNotFoundError`。

---

## モデル

### `Model`

モデルの指定を表す変更不可能なオブジェクト。メソッドはすべて新しい `Model` を返す。

#### `Model.uniform`

```text
Model.uniform(item_model, *, items) -> Model
```

`items`（項目名の列）のすべてに同じ項目モデルを使う。例: `Model.uniform(TwoPL(), items=df.columns)`。

#### `Model.from_items`

```text
Model.from_items(items) -> Model
```

`{項目名: 項目モデル}` の辞書から作る（順序は辞書の順）。項目ごとに違うモデルを混ぜられる。

#### `Model.fix`

```text
model.fix(*, item, **values) -> Model
```

項目 `item` のパラメタを内部パラメタ（傾き・切片形式）の値で固定する。多母集団では全群で固定。
例: `model.fix(item="q01", a=1.2, d=0.3)`。存在しない項目・パラメタは `SpecificationError`。

#### `Model.fix_from`

```text
model.fix_from(bank, *, items=None, parameterization="slope_intercept") -> Model
```

項目バンクの値で項目パラメタを固定する（FIPC）。潜在分布を推定するには `.with_latent(Normal.free())` を併用する。

| 引数 | 説明 |
|---|---|
| `bank` | `item` 列と、パラメタの列を持つ DataFrame。`"slope_intercept"` なら `a`, `d`（GRM/GPCM/PCM は `d1`, `d2`, …）、`"irt"` なら `a`, `b`（GRM/GPCM/PCM は `b1`, …）。Rasch・PCM 項目は `a` 列がなくてよい。漸近線 `g`（3PL/4PL）・`u`（4PL）はどちらのパラメタ化でもそのまま渡す（3PL 項目では `u` 列を省略できる） |
| `items` | 固定する項目（名前のリストか `str -> bool` の関数）。既定はバンクにあってモデルにもある項目すべて |
| `parameterization` | `"slope_intercept"` または `"irt"`（$d = -ab$、GPCM/PCM は $d_k = -a\sum_{c=1}^{k} b_c$ に変換） |

#### `Model.with_factors`

```text
model.with_factors(factors, *, correlated=True) -> Model
```

2 または 3 次の確認的因子負荷パターンを宣言する（[ガイド 14 節](guide.ja.md#14-確認的多次元モデル)）。
`factors` は因子名（順序つき、2 または 3 因子）から各因子に負荷する項目（項目名の列または `str -> bool` 関数）への
辞書。交差負荷を許し、各因子は 1 項目以上を持ち、すべてのモデル項目が 1 因子以上に負荷する必要がある。負荷しない因子の
傾きは `0` に固定され、項目パラメタ名は `a1`…`aD`, `d`（または `d1`…）になる。潜在分布は多変量正規分布（平均 `0` 固定、
分散 `1` 固定。全項目が `Rasch` / `PCM` の場合は分散も自由）で、`correlated=True` のとき既定で因子間相関を自由推定
（mirt の `COV = F1*F2` 相当）、`correlated=False` で `0` に固定する。`with_groups()`（順序は問わない）と
組み合わせると、各群の `Normal` がすべての因子に適用される（[ガイド 14 節](guide.ja.md#14-確認的多次元モデル)）。多次元モデルでの
`with_latent()` と `EmpiricalHistogram` は未対応（`SpecificationError`）。

#### `Model.with_bifactor`

```text
model.with_bifactor(*, specific, general="general") -> Model
```

双因子モデルを宣言する（[ガイド 14 節](guide.ja.md#双因子モデル)）。全項目が負荷する一般因子（名前は `general`）と、
`specific`（因子名から項目への辞書、順序つき）の特殊因子からなる。各項目が負荷する特殊因子は高々 1 つで、どこにも
挙げない項目は一般因子のみに負荷し、各特殊因子は 2 項目以上が必要。項目パラメタは `a1`（一般因子）、`a2`…`aD`
（特殊因子。他の特殊因子への傾きは `0` に固定）、`d`（または `d1`…）で、mirt の `bfactor()` と一致する。因子は直交し、
潜在パラメタの行は `mean_1`…`mean_D`（`0` 固定）と `var_1`…`var_D`（`1` 固定）。E ステップは Gibbons–Hedeker の
次元縮約（特殊因子ごとに 2 次元格子 1 つ。既定は $[-6, 6]$ 上の 31 点）。群、`with_latent()`、`with_factors()` との
併用、Rasch / PCM 項目、編集したテーブルでの潜在パラメタの自由推定は `SpecificationError`。スコアリングは `"EAP"` のみ。

#### `Model.with_two_tier`

```text
model.with_two_tier(*, primary, specific, correlated=True) -> Model
```

two-tier モデルを宣言する（[ガイド 14 節](guide.ja.md#two-tier-モデル)）。`primary` の主因子と `specific` の特殊因子
（どちらも因子名から項目への辞書、順序つき。主因子は 1〜2 個）からなる。各項目は主因子の少なくとも 1 つに負荷し、
特殊因子には高々 1 つに負荷する。各特殊因子は 2 項目以上が必要。項目パラメタは `a1`…`aP`（主因子）、`a(P+1)`…`aD`
（特殊因子。負荷しない因子への傾きは `0` に固定）、`d`（または `d1`…）で、mirt の `bfactor(data, model, model2 = )` と
一致する。潜在パラメタの行は `mean_1`…`mean_D`（`0` 固定）、`var_1`…`var_D`（`1` 固定）と、主因子が 2 つのときの
`cor_12`（`correlated=True` なら自由）。特殊因子は他のすべての因子と直交する。E ステップは、主因子の直積格子と
特殊因子 1 つずつについて積分する（既定は、主因子 2 つで $[-6, 6]$ 上の次元あたり 15 点、1 つで 31 点）。主因子が
3 つ以上、主因子に負荷しない項目、群、`with_latent()`、`with_factors()` / `with_bifactor()` との併用、Rasch / PCM 項目、
編集したテーブルでの潜在平均・分散の自由推定は `SpecificationError`。スコアリングは `"EAP"` のみ。

#### `Model.with_latent`

```text
model.with_latent(dist) -> Model
```

単一群モデルの潜在分布を `Normal`・`EmpiricalHistogram`・`DavidianCurve`・`NormalMixture`（1 次元モデルのみ）で指定する。多母集団モデルでは使えない（`with_groups()` で指定する）。

#### `Model.with_groups`

```text
model.with_groups(**latents) -> Model
```

多母集団モデルにする。キーワード引数の順で群を定義し、**最初の群が参照群**。2 群以上が必要。
例: `model.with_groups(ref=Normal.standard(), focal=Normal.free())`。

#### `Model.equal_across_groups`

```text
model.equal_across_groups(*, items, params=None) -> Model
```

`items` のパラメタを群間で等しくする（アンカー項目）。`params` を省略すると全パラメタ、
`params=["a"]` のように指定するとそのパラメタだけ。`items` は名前のリストか `str -> bool` の関数、
`None` なら全項目。何度でも重ねて呼べる。

#### `Model.equal_across_items`

```text
model.equal_across_items(*, items, params) -> Model
```

`params`（例: `["a"]`, `["a1"]`, `["d"]`, `["d1", "d2"]`）の各パラメタを、選んだ `items`（2 項目以上）で共通の
1 つのパラメタにする（mirt の `CONSTRAIN` と同じく群ごと）。`equal_across_groups` と組み合わせると群間でも共有され、
重なる等値制約はまとめられる。固定パラメタには影響しない。共有パラメタの既定の初期値は各項目の初期値の平均。
編集した表で該当行に同じ `free_id` を与えるのと同じ。

#### `Model.prior`

```text
model.prior(*, param, dist, items=None) -> Model
```

選択した項目の内部パラメタ `param` に事前分布 `dist`（`NormalPrior`, `LogNormalPrior`, `BetaPrior`）を全群共通で付ける
（MAP 推定、[ガイド 11 節](guide.ja.md#11-事前分布と-map-推定)）。事前分布はヤコビアンなしの自然尺度で評価される。
固定パラメタの事前分布は無視され、群間で共有されるパラメタ（同じ `free_id`）は 1 回だけ数えられる。潜在分布パラメタ
（`"mean"`, `"var"`, `"phi_*"`, `"rmean_*"`, `"rvar_*"`, `"logit_*"` など）への指定は `SpecificationError`。

| 引数 | 説明 |
|---|---|
| `param` | 内部項目パラメタ名（`"a"`, `"d"`, `"g"`, `"u"`, `"d1"`, …） |
| `dist` | `NormalPrior`, `LogNormalPrior`, `BetaPrior` のいずれか |
| `items` | 項目名のリスト、`str -> bool` の関数、または全項目を表す `None` |

#### `Model.parameter_table`

```text
model.parameter_table(data, *, group=None, item_names=None) -> DataFrame
```

`data` に対する既定の ParameterTable を返す。列は `group`, `item`, `model`, `param`, `value`, `free`,
`lower`, `upper`, `prior`, `free_id`。編集して `fit(..., table=...)` に渡す（[ガイド 12 節](guide.ja.md#12-parametertable-を編集する)）。

#### その他の属性

| 属性・メソッド | 説明 |
|---|---|
| `item_names` | 項目名のタプル |
| `is_multigroup` | `with_groups()` で群を宣言したか |
| `is_multidimensional` | `with_factors()`・`with_bifactor()`・`with_two_tier()` のいずれかで因子を宣言したか |
| `n_factors` | 潜在因子数（`1`, `2`, または `3`。双因子モデルでは `1 + S`、two-tier モデルでは `P + S`） |
| `bifactor` | `with_bifactor()` で宣言したか |
| `n_primary_factors` | `with_two_tier()` で宣言したモデルの主因子数（それ以外は `0`） |
| `quadrature_dimension` | E ステップが積分する格子の次元（双因子モデルでは `2`、two-tier モデルでは `P + 1`） |
| `latent_parameter_names()` | 各群の Normal 潜在パラメタ名 |
| `factor_names` | 因子名のタプル（1 次元モデルでは空タプル） |
| `resolved_latent()` | 単一群モデルの潜在分布（既定則を適用したもの） |
| `resolved_groups()` | `(群名, Normal)` のタプル。単一群なら `(("all", ...),)` |
| `items`, `fixed`, `latent`, `groups`, `equal`, `priors`, `factors`, `correlated_factors` | 指定の中身（通常は直接使わない） |

---

## 項目モデル

### `TwoPL`

```text
TwoPL()
```

2 パラメタロジスティックモデル $P(y=1\mid\theta)=\sigma(a\theta+d)$（Birnbaum, 1968）。内部パラメタ `a`, `d`。
表示用に $b=-d/a$。mirt の `itemtype = "2PL"` と同じ。

### `Rasch`

```text
Rasch()
```

Rasch モデル $P(y=1\mid\theta)=\sigma(\theta+d)$（Rasch, 1960）。`a` は 1 に固定、`d` を推定。全項目が Rasch の
単一群モデルでは、既定で潜在分布の分散を推定する（mirt の `itemtype = "Rasch"` と同じ）。

### `ThreePL`

```text
ThreePL()
```

3 パラメタロジスティックモデル $P(y=1\mid\theta)=g+(1-g)\sigma(a\theta+d)$（Birnbaum, 1968）。
内部パラメタは `a`, `d`, `g`（自由、自然尺度で $0 < g < 1$）と `u`（1 に固定）、表示用に $b=-d/a$。
mirt の `itemtype = "3PL"` と同じモデル・パラメタ名（mirt は `g` を logit 尺度で推定するが、flexpyirt は
自然尺度で推定する）。`FourPL` と `FourPLKernel` を共有する。`g = 0` に固定すると `TwoPL` と厳密に一致する。

### `FourPL`

```text
FourPL()
```

4 パラメタロジスティックモデル $P(y=1\mid\theta)=g+(u-g)\sigma(a\theta+d)$（Barton & Lord, 1981）。
内部パラメタ `a`, `d`, `g`, `u` をすべて自由推定（自然尺度で $0 < g < u < 1$）、表示用に $b=-d/a$。
mirt の `itemtype = "4PL"` と同じ。`ThreePL` と `FourPLKernel` を共有する。

### `GRM`

```text
GRM(n_categories)
```

段階反応モデル（Samejima, 1969）。カテゴリは `0..n_categories-1`。
$P(y\ge k\mid\theta)=\sigma(a\theta+d_k)$、$d_1>\dots>d_{K-1}$。内部パラメタ `a`, `d1`…`d{K-1}`、
表示用に $b_k=-d_k/a$。mirt の `itemtype = "graded"` と同じ。`n_categories` が同じ GRM 項目はまとめて推定される。

### `GPCM`

```text
GPCM(n_categories)
```

一般化部分採点モデル（Muraki, 1992）。カテゴリは `0..n_categories-1`（`n_categories >= 2`）。
$P(y=k\mid\theta)=\exp(k\,a\theta+d_k)/\sum_{c=0}^{K-1}\exp(c\,a\theta+d_c)$（$d_0=0$ 固定）。
内部パラメタ `a`, `d1`…`d{K-1}`（順序制約なし）、表示用にステップ困難度 $b_k=-(d_k-d_{k-1})/a$（$d_0=0$）。
mirt の `itemtype = "gpcm"` と同じ。`n_categories = 2` では `TwoPL` と一致する。`n_categories` が同じ `PCM` と
`GPCMKernel` を共有する。

### `PCM`

```text
PCM(n_categories)
```

部分採点モデル（Masters, 1982）。カテゴリは `0..n_categories-1`（`n_categories >= 2`）。`a = 1` に固定した GPCM で、
$P(y=k\mid\theta)=\exp(k\theta+d_k)/\sum_{c=0}^{K-1}\exp(c\theta+d_c)$（$d_0=0$ 固定）。
表示用にステップ困難度 $b_k=-(d_k-d_{k-1})$。全項目が `Rasch` または `PCM` の単一群モデルでは、既定で
潜在分布の分散を推定する（多値データに対する mirt の `itemtype = "Rasch"` と同じ）。`n_categories = 2` では
`Rasch` と一致する。`n_categories` が同じ `GPCM` と `GPCMKernel` を共有する。

### `Nominal`

```text
Nominal(n_categories)
```

Thissen, Cai & Bock (2010) の全体傾き × 採点関数パラメタ化による名義反応モデル（Bock, 1972）。カテゴリは
`0..n_categories-1`（`n_categories >= 2`）。
$P(y=k\mid\theta)=\exp(a\,\mathrm{ak}_k\,\theta+d_k)/\sum_{c=0}^{K-1}\exp(a\,\mathrm{ak}_c\,\theta+d_c)$
（$\mathrm{ak}_0=0$、$\mathrm{ak}_{K-1}=K-1$、$d_0=0$ 固定）。内部パラメタは `a`（多次元モデルでは
`a1`…`aD`）、`ak1`…`ak{K-2}`、`d1`…`d{K-1}`（すべて自由推定・順序制約なし、表示用の `b` / `bk` 列はなし）。
mirt の `itemtype = "nominal"` と同じ。$\mathrm{ak}_k=k$ に固定すると `GPCM(n_categories)` と厳密に一致する
（`n_categories = 2` では自由な `ak` パラメタがなく、`Nominal(2)` は `GPCM(2)` / `TwoPL` と一致する）。
`NominalKernel(n_categories)` を用いる。

---

## 事前分布

ヤコビアンなしの自然尺度で項目パラメタに課す事前分布の不変データクラス（mirt と同じ）。いずれもハイパーパラメタを検証し
（不正なら `SpecificationError`）、`support` と `log_pdf(x) -> Array` を持つ。

### `NormalPrior`

```text
NormalPrior(mean, sd)
```

$(-\infty, \infty)$ 上の正規事前分布:
$\log\pi(x) = -\tfrac{1}{2}\log(2\pi) - \log\sigma - \frac{(x-\mu)^2}{2\sigma^2}$
（Mislevy, 1986、mirt の `norm`）。有限な `mean` と正の有限な `sd` が必要。潜在分布 `Normal` と名前が衝突しないよう
`NormalPrior` としている。

### `LogNormalPrior`

```text
LogNormalPrior(mean_log, sd_log)
```

$(0, \infty)$ 上の対数正規事前分布（$\log X \sim N(\mu, s^2)$）:
$\log\pi(x) = -\log x - \log s - \tfrac{1}{2}\log(2\pi) - \frac{(\log x - \mu)^2}{2s^2}$
（$x \le 0$ で $-\infty$。Mislevy, 1986、mirt の `lnorm`）。有限な `mean_log` と正の有限な `sd_log` が必要。
傾き `a` によく使う。

### `BetaPrior`

```text
BetaPrior(a, b)
```

$(0, 1)$ 上のベータ事前分布:
$\log\pi(x) = (a - 1)\log x + (b - 1)\log(1 - x) - \log B(a, b)$
（$(0, 1)$ の外で $-\infty$。Swaminathan & Gifford, 1986、mirt の `expbeta` の自然尺度対数事前目的関数と一致、
logit ヤコビアンなし）。正の有限な `a`, `b` が必要。3PL / 4PL の漸近線 `g`, `u` によく使う。

---

## 潜在分布

### `Normal`

```text
Normal(mean=0.0, var=1.0, free_mean=False, free_var=False)
Normal.standard()
Normal.free(*, mean=0.0, var=1.0)
```

正規分布 $N(\mu,\sigma^2)$。求積格子の上で密度を正規化した重みを使う。`mean` と `var` は固定値、
または推定するときの初期値。

| 生成 | 意味 |
|---|---|
| `Normal.standard()` | $N(0,1)$、何も推定しない |
| `Normal(free_var=True)` | 平均 0 固定、分散を推定 |
| `Normal.free()` | 平均・分散を推定 |

`var` は正、`mean` は有限でなければ `SpecificationError`。推定した `var` は格子で切る前の正規分布の
パラメタ（[ガイド 13 節](guide.ja.md#13-求積格子の選び方)）。

### `EmpiricalHistogram`

```text
EmpiricalHistogram(*, min_weight=1e-10)
```

経験ヒストグラム（Woods, 2007）。求積節点上の自由な重み $w_q \ge$ `min_weight`（和は 1）を毎 EM 反復で推定する
（[ガイド 8 節](guide.ja.md#8-潜在分布と識別性)）。固定項目が尺度を決めない限り（FIPC 以外）、参照群のヒストグラムを
平均 0・分散 1 に制約する（Rasch / PCM は平均のみ）。多母集団では全群が EH でなければならない。重みは `n_params` に
パラメタとして数える（標準化する参照群は $Q-3$、Rasch / PCM は $Q-2$、FIPC と各焦点群は $Q-1$）。標準誤差は推定した
ヒストグラムを固定した条件付きのもの。1 次元モデルのみ。

| 引数 | 説明 |
|---|---|
| `min_weight` | 各節点の重みの下限。$(0, 1)$ の範囲で、使う格子について $1/Q$ 未満 |

**例外**: `min_weight` が不正、群間で `EmpiricalHistogram` と `Normal` を混在させた、または多次元モデルの場合に
`SpecificationError`。切片を固定していて非ゼロの傾きを固定していない場合（分散のみの制約は未対応）は `IdentificationError`。

### `DavidianCurve`

```text
DavidianCurve(degree, *, mean=0.0, var=1.0, free_mean=False, free_var=False)
DavidianCurve.free(degree, *, mean=0.0, var=1.0)
```

次数 `degree`（`1..10`）の半ノンパラメトリック Davidian 曲線潜在密度（Zhang & Davidian, 2001; Woods & Lin, 2009;
[ガイド 8 節](guide.ja.md#davidian-曲線)）。正規化確率論的 Hermite 多項式と $k$ 個の角度パラメタ `phi_1`…`phi_k`
による単位球面係数で $h_k(z;\boldsymbol\phi) = P_k(z;\boldsymbol\phi)^2\varphi(z)$ を構成し
（$\boldsymbol\phi = \mathbf 0$ で非特異に $N(\mu,\sigma^2)$ を再現）、$h_k$ の厳密な連続モーメントで解析的に標準化することで
任意の $\boldsymbol\phi$ について連続平均・分散が `mean`・`var` に一致する。求積重みは格子上で正規化するため EM は
単調（非減少）で、自由な項目パラメタ・潜在パラメタすべてに対して厳密な Oakes / Hessian 標準誤差が得られる。単一群の 1 次元モデルのみ。
Fisher 情報行列は $\boldsymbol\phi = \mathbf 0$ で `phi_1`・`phi_2` 方向（`degree=1` では `phi_1`）に特異なため、正規に近い
特性では角度の標準誤差が信頼できず、`Normal` に対する尤度比検定は標準的でなく（AIC / BIC を使う）、局所最大が複数ありうる
（[ガイド 8 節](guide.ja.md#davidian-曲線)）。

| 引数 | 説明 |
|---|---|
| `degree` | 整数の多項式次数 $k \in \{1,\dots,10\}$（角度パラメタ `phi_1`…`phi_k` の個数） |
| `mean`, `var` | 連続平均と正の分散の値（自由推定時は初期値） |
| `free_mean`, `free_var` | `mean`・`var` を推定するか（識別性の規則は `Normal` と同じ） |

**例外**: `degree`・`mean`・`var`・フラグが不正、または `Model.with_groups()`・`Model.with_factors()` と組み合わせた場合に
`SpecificationError`。`free_mean`・`free_var` に必要な固定切片・固定傾きがない場合は `IdentificationError`。

### `NormalMixture`

```text
NormalMixture(n_components, *, mean=0.0, var=1.0, free_mean=False, free_var=False,
              means=None, variances=None, weights=None)
NormalMixture.free(n_components, *, mean=0.0, var=1.0, means=None, variances=None, weights=None)
```

成分数 $k$ = `n_components`（`2..10`）の有限混合正規潜在密度で、項目パラメタは共通の 1 組（Mislevy, 1984;
[ガイド 8 節](guide.ja.md#混合正規)）。形状は mirt の `dentype = "mixture-k"` のクラスのパラメタ化に従い、素の成分 1 が
$N(0,1)$、成分 $c \ge 2$ は素の平均 `rmean_c`・素の分散 `rvar_c`・ロジット `logit_c` $=\log(\pi_c/\pi_1)$ を持つ。
混合を厳密なモーメントで解析的に標準化し、任意の形状で連続平均・分散が `mean`・`var` に一致する。各成分を格子上で
正規化してから混合する。`FitResult.latent` は標準化した成分 `pi_c`・`mean_c`・`var_c` とデルタ法の SE を返す。
Oakes / Hessian 標準誤差は項目パラメタと潜在分布パラメタの両方を含む。単一群の 1 次元モデルのみで、等間隔格子と
Gauss–Hermite 格子に対応する。成分は交換可能（ラベルスイッチング）で、尤度に局所最大が複数ありうるので、いくつかの
初期形状から推定し直して `loglik` が最も高いものを採る。`score(method="MAP")`・`item_fit()`・`m2()` は
`SpecificationError`。

| 引数 | 説明 |
|---|---|
| `n_components` | 整数の成分数 $k \in \{2,\dots,10\}$ |
| `mean`, `var` | 混合の連続平均と正の分散の値（自由推定時は初期値） |
| `free_mean`, `free_var` | `mean`・`var` を推定するか（識別性の規則は `Normal` と同じ） |
| `means`, `variances`, `weights` | 長さ $k$ の初期形状（成分の平均、正の分散、和が 1 の正の重み）。任意のアフィン尺度でよい。既定は `(0, 1.5, -1.5, 3, -3, ...)`、すべて `1`、`1/k` |

**例外**: `n_components`・`mean`・`var`・フラグ・初期形状が不正、または `Model.with_groups()`・`Model.with_factors()` と
組み合わせた場合に `SpecificationError`。`free_mean`・`free_var` に必要な固定切片・固定傾きがない場合は
`IdentificationError`。

---

## 尺度リンキング

### `link`

```text
link(source, target, *, items=None, method="stocking_lord", symmetric=False,
     transformation="diagonal", nodes=None, weights=None, source_group=None,
     target_group=None, tol=1e-10, max_iter=200) -> LinkResult
```

別々に推定した `source`（新フォーム）尺度を、共通項目を通じて `target`（基準フォーム）尺度へ結ぶ
線形変換 $\theta^* = A\theta + B$（$A > 0$）の係数 $(A, B)$ を推定する（[ガイド 15 節](guide.ja.md#15-尺度リンキング)、
[理論](theory.ja.md)）。2PL・3PL・4PL・GRM・GPCM・名義反応モデルと 2PL + GPCM 混在・2PL + 名義混在の共通項目について
R の独立な実装と照合済みで、Rasch・PCM は同じカーネルを共有し単体テストでのみ確認している。因子数 $D > 1$ が同じ
確認的モデル（`Model.with_factors`。因子は位置で対応させる）は
$\boldsymbol\theta^* = \mathbf A\boldsymbol\theta + \boldsymbol\beta$ で結び、同じ目的関数を直積格子の上で使う
（2 因子・3 因子の 2PL と 2 因子の GRM について R と照合済み）。Newton 最適化と Ogasawara (2001) のデルタ法
標準誤差の微分は JAX の自動微分で計算する。SE は独立な推定を仮定し（1 つの多母集団推定を `source` と `target` の
両方に使うときは同時共分散を使う）、求積重みを固定し、収束しないときは報告しない。

| 引数 | 型 | 説明 |
|---|---|---|
| `source` | `FitResult` | 新フォーム側の推定結果 |
| `target` | `FitResult` | 基準フォーム側の推定結果（因子数は `source` と同じ） |
| `items` | iterable[str] / Mapping[str, str], 任意 | 共通項目（既定 `None`: 名前が一致する全項目）。項目名が異なるときは `{source_item: target_item}` の辞書。共通ペアは同じ項目カーネル・カテゴリ数でなければならない |
| `method` | `"stocking_lord"` / `"haebara"` | 目的関数: Stocking & Lord (1983) の TCC 二乗差（既定）または Haebara (1980) のカテゴリ曲線二乗差（多値拡張: Kim & Lee, 2006） |
| `symmetric` | bool | `True` で逆方向の二乗差を加えた対称版（Kim & Lee, 2006）。既定 `False` |
| `transformation` | `"diagonal"` / `"orthogonal"` / `"oblique"` | 因子数 $D > 1$ のときの $\mathbf A$ の族（1 因子では効果がない）。`"diagonal"`（既定）は $\operatorname{diag}(A_m)$、$A_m > 0$ で、同じ確認的構造で推定した 2 つの結果のあいだの変換。`"orthogonal"` は拡大と回転の積 $k\,\mathbf R$（Li & Lissitz, 2000）。`"oblique"` は $\det\mathbf A > 0$ の任意の行列（Oshima, Davey & Lee, 2000）。鏡映は含まない |
| `nodes` | 配列, 任意 | 求積節点 $\theta_q$。1 次元、または因子数 $D > 1$ では形状 `(Q, D)`（既定: `target` 群の推定格子、`symmetric=True` の逆方向は `source` 群の格子） |
| `weights` | 1 次元配列, 任意 | 非負の重み（和が 1 になるよう正規化。既定: 推定済み潜在分布の重み、`nodes` のみ指定時は一様重み） |
| `source_group`, `target_group` | str, 任意 | 多母集団 `FitResult` から結ぶ群を選ぶための群名 |
| `tol` | float | $\|\nabla F(A, B)\|_\infty$ の収束許容誤差（既定 `1e-10`） |
| `max_iter` | int | Newton 最大反復回数（既定 `200`） |

**戻り値**: [`LinkResult`](#linkresult)

**例外**: `PrecisionError`（JAX 64 ビット無効）、`SpecificationError`（`FitResult` 以外、因子数の不一致、双因子・two-tier の結果、群指定のない
多母集団結果、共通項目なし、共通項目でのカーネル・カテゴリ数不一致、不正な引数・格子・重み）。

**警告**: `grad_norm <= tol` に達する前に `max_iter` に到達したとき `ConvergenceWarning`。

### `LinkResult`

`link()` の戻り値。変更不可。

| 属性・メソッド | 型 | 説明 |
|---|---|---|
| `A`, `B` | float / ndarray | 線形リンキング係数（$\theta^* = A\theta + B$, $A > 0$）。因子数 $D > 1$ では `(D, D)` 行列 $\mathbf A$ と `(D,)` ベクトル $\boldsymbol\beta$ |
| `A_se`, `B_se` | float / ndarray | 陰関数定理によるデルタ法標準誤差（Ogasawara, 2001）。計算できない場合は NaN。形は `A`・`B` と同じ（対角の $\mathbf A$ の構造的な 0 では 0） |
| `vcov` | ndarray / None | `(A, B)` の漸近共分散行列 `(2, 2)`。因子数 $D > 1$ では `(A.ravel(), B)` の `(D*D + D, D*D + D)` 共分散行列で、階数は高々変換の自由パラメタ数。自由な共通項目がある側で `se=None`、全共通パラメタが固定、または $H_{cc}$ が正定値でない場合は `None` |
| `criterion` | float | `(A, B)` での最小化された目的関数値 |
| `method` | `"stocking_lord"` / `"haebara"` | 使用した基準 |
| `symmetric` | bool | 対称版かどうか |
| `transformation` | `"diagonal"` / `"orthogonal"` / `"oblique"` | $\mathbf A$ の族（1 因子では常に `"diagonal"`） |
| `converged` | bool | `grad_norm <= tol` を満たしたか |
| `n_iter` | int | 実行した Newton 反復回数 |
| `grad_norm` | float | `(A, B)` での $\|\nabla F(A, B)\|_\infty$ |
| `common_items` | tuple[str, ...] | 共通項目として使用した `source` 側の項目名 |
| `items` | DataFrame | `source` の全項目を基準尺度へ変換した表（`item`, `model`, `common`、内部の傾き–切片パラメタと表示用 IRT 閾値 `b` / `bk`）。そのまま `Model.fix_from()` に渡せる |
| `transform_theta(theta)` | scalar / ndarray / Series / DataFrame | 能力値を $\theta^* = A\theta + B$ で変換（採点 DataFrame の `theta` と全 `pv1..` 列。`se` は $A\,\mathrm{se}$ でリンキング誤差を含まない）。因子数 $D > 1$ では `theta_<因子>`・`pv<m>_<因子>` 列、または最後の軸の長さが $D$ の配列。`se_<因子>` は対角の $\mathbf A$ では定数倍、それ以外では NaN |
| `summary(*, digits=4)` | str | 読みやすいテキスト要約（Jupyter では `_repr_html_()` で HTML 表示） |

## 相関

パッケージ: flexpyctt（`from flexpyctt import polychoric`）。

### `polychoric`

```text
polychoric(data, *, smooth=True, zero_cell_correction=0.5, tol=1e-10,
           max_iter=100) -> PolychoricResult
```

順序反応のポリコリック相関行列（2 値項目ではテトラコリック相関）を two-step の最尤法で求める
（[ガイド 16 節](guide.ja.md#16-ポリコリック相関)）。各項目の閾値を周辺比率から求め、そのうえで各ペアの相関を、離散化した
2 変量正規分布のもとでのペアの分割表の尤度を最大にする値とする（Olsson, 1979）。2 変量正規分布の確率は Owen (1956) の
$T$ 関数で計算する。順序変数に対する lavaan の `lavCor` と一致する。

`flexpyirt.polychoric` は引数と結果が同じで、同じ推定量を JAX で計算する（2 変量正規分布の確率は Genz, 2004 の方法）。
JAX の 64 ビットが無効なときは `PrecisionError` を出す。2 つは $10^{-9}$ 程度で一致する。

| 引数 | 型 | 説明 |
|---|---|---|
| `data` | DataFrame / 2 次元配列 | 1 行が 1 回答者、1 列が 1 項目。順序のあるカテゴリの符号（任意の数値で、昇順がカテゴリの順序）、欠測は NaN。各項目に観測されたカテゴリが 2 つ以上必要。観測されない符号はカテゴリとみなさない |
| `smooth` | bool | 正定値でない行列を平滑化する（psych の `cor.smooth` と同じ規則）。`NotPositiveDefiniteWarning` を出す（既定 `True`） |
| `zero_cell_correction` | float | 空のセルがちょうど 1 つの 2 x 2 表で、空のセルを含む対角にこの値を足し、もう一方の対角から引く（周辺は変わらない。lavaan の既定の規則）。補正しないと推定値は $\pm 1$ になる。`0` で無効（既定 `0.5`） |
| `tol` | float | 1 観測あたりの勾配に対する収束判定（既定 `1e-10`） |
| `max_iter` | int | ペアごとの Newton 反復の上限（既定 `100`） |

欠測は利用可能なケースで扱う。閾値は項目への観測された回答をすべて使い、ペアの分割表は両方に回答した回答者を使う。

**戻り値**: [`PolychoricResult`](#polychoricresult)

**例外**: `SpecificationError`（数値でないデータ、項目が 2 つ未満、観測された
カテゴリが 1 つだけの項目、両方に回答した回答者がいないペア、またはその回答者の中で一方の項目が一定）。

**警告**: `HeywoodWarning`（ペアの尤度が $|\rho| \to 1$ で最大）、`ConvergenceWarning`（収束しないペアがある）、
`NotPositiveDefiniteWarning`（行列を平滑化した）。

### `PolychoricResult`

`polychoric()` の戻り値。変更不可。

| 属性 | 型 | 説明 |
|---|---|---|
| `correlation` | DataFrame | 項目 x 項目の相関行列（`smooth=True` で必要なときは平滑化後） |
| `unsmoothed_correlation` | DataFrame | 平滑化前のペアごとの推定値 |
| `thresholds` | dict[str, ndarray] | 項目ごとに、観測された $K$ カテゴリの $K - 1$ 個の閾値 $\tau_k = \Phi^{-1}(P(Y \le k))$ |
| `categories` | dict[str, ndarray] | 項目ごとに、観測された回答の値（昇順） |
| `n_obs` | DataFrame | 両方の回答が観測された回答者数（対角は項目への回答数） |
| `converged`, `at_boundary`, `zero_cell_corrected` | DataFrame (bool) | ペアごとに、収束したか、$\lvert\rho\rvert \to 1$ で最大か（推定値は $\pm\tanh 8$）、`zero_cell_correction` を適用したか |
| `smoothed` | bool | 平滑化で行列が変わったか |
| `min_eigenvalue` | float | `unsmoothed_correlation` の最小固有値 |

## 次元数

いずれも「次元はいくつか」を診断として答える。1 つの手法だけを信頼せず、いくつかを組み合わせて判断する
（[ガイド 17 節](guide.ja.md#17-次元数)）。

パッケージ: flexpyctt。ただし [`model_parallel_analysis`](#model_parallel_analysis)、
[`dimensionality`](#dimensionality)、`LocalDependenceResult`、`JSIResult` は推定済みのモデルを使うので flexpyirt のもの。

### `parallel_analysis`

```text
parallel_analysis(data, *, rng, correlation="polychoric", eigen_type="pca",
                  reference="permutation", n_replications=100, quantile=0.95,
                  zero_cell_correction=0.5) -> ParallelAnalysisResult
```

Horn (1965) の平行分析。観測された相関行列の固有値のうち、共通因子をもたない参照データの固有値の `quantile`
（Glorfeld, 1995）を上回るものを先頭から数える。

| 引数 | 型 | 説明 |
|---|---|---|
| `data` | DataFrame / 2 次元配列 | 回答者 x 項目（順序のある符号。`correlation="pearson"` なら連続量でもよい）。欠測は NaN |
| `rng` | int / `numpy.random.Generator` | 参照データのシードまたは生成器（例 `rng=0`） |
| `correlation` | `"polychoric"` / `"pearson"` | データと並べ替えの反復で使う相関（正定値でなければどちらも平滑化） |
| `eigen_type` | `"pca"` / `"smc"` | 相関行列の固有値、または対角を重相関係数の 2 乗にした縮約行列の固有値 |
| `reference` | `"permutation"` / `"normal"` | 項目ごとに独立に並べ替える（周辺分布を保つ。Garrido, Abad & Ponsoda, 2013）、または同じ大きさの独立な正規データ（Pearson） |
| `n_replications` | int | 参照データの数（既定 `100`） |
| `quantile` | float / None | 参照固有値の (0, 1) の分位点（既定 `0.95`）。`None` で平均 |
| `zero_cell_correction` | float | `polychoric()` に渡す |

**戻り値**: [`ParallelAnalysisResult`](#parallelanalysisresult)

**例外**: `SpecificationError`（不正な指定やデータ、項目が 3 つ未満）。

**警告**: 並べ替えの反復を飛ばしたとき（欠測のため、並べ替えたペアに使える回答がない）に `FlexpyirtWarning`。

### `ParallelAnalysisResult`

| 属性 | 型 | 説明 |
|---|---|---|
| `n_factors` | int | `reference` を上回る観測固有値の数（先頭から） |
| `observed` | ndarray | 観測固有値（降順） |
| `reference`, `reference_mean` | ndarray | 位置ごとの参照の分位点（または平均）と平均 |
| `replications` | ndarray | `(n_replications - n_skipped, p)` の参照固有値 |
| `correlation` | DataFrame | 観測された相関行列 |
| `eigen_type`, `reference_kind`, `correlation_kind`, `quantile`, `n_skipped` | | 使った設定と飛ばした反復の数 |

### `velicer_map`

```text
velicer_map(correlation, *, power=4) -> MAPResult
```

Velicer の最小平均偏相関（MAP）検定。最初の $m$ 個の主成分を偏相関で除いたあと、非対角の偏相関の `power` 乗の
平均を $m = 0, \dots, p - 1$ で最小にする。`power=2` は元の検定（Velicer, 1976）、`power=4` は改訂版（Velicer, Eaton &
Fava, 2000）。`correlation` は行列（DataFrame / 配列）または `PolychoricResult`。

### `MAPResult`

| 属性 | 型 | 説明 |
|---|---|---|
| `n_factors` | int | 最小にする $m$ |
| `statistic` | ndarray | $m = 0, \dots, p - 1$ の平均偏相関の統計量（偏分散が 0 になって以降は NaN） |
| `power` | int | 2 または 4 |

### `empirical_kaiser`

```text
empirical_kaiser(correlation, *, n_obs=None) -> EmpiricalKaiserResult
```

経験的カイザー基準（Braeken & van Assen, 2017）。$j$ 番目の固有値を
$\max\{(1+\sqrt{p/n})^2 (p-\sum_{i<j}\lambda_i)/(p-j+1),\ 1\}$ と比べる。`correlation` が `PolychoricResult` でなければ
`n_obs` が必要（`PolychoricResult` ならペアごとの $n$ の最小値を使う）。

### `EmpiricalKaiserResult`

| 属性 | 型 | 説明 |
|---|---|---|
| `n_factors` | int | 参照値を上回る固有値の数（先頭から） |
| `eigenvalues`, `reference` | ndarray | 固有値と参照値 |
| `n_obs` | int | 使った標本サイズ |

### `efa`

```text
efa(correlation, *, n_factors, method="uls", n_obs=None) -> EFAResult
```

相関行列（DataFrame、配列、`PolychoricResult`）の回転しない探索的因子分析。ULS（minres。Harman & Jones, 1966）または
ML（Jöreskog, 1967）で、独自性を $1 - \mathrm{SMC}$ から $[0.005, \max(1, \mathrm{SMC})]$ の範囲で最小化する。psych の
`fa(rotate = "none")`、EFAtools の `EFA(rotation = "none")` と同じ。`n_obs`（`PolychoricResult` ならペアごとの $n$ の
最小値が既定）を与えると、適合度に Bartlett の補正をした $\chi^2$、その $p$ 値、CFI、RMSEA が加わる。CAF と $df$ は
常に報告する。独自性が境界に達すると `HeywoodWarning`、最適化が途中で止まると `ConvergenceWarning` を出す。

### `EFAResult`

| 属性 | 型 | 説明 |
|---|---|---|
| `loadings` | DataFrame | 回転しない負荷量（列の和が非負になるよう符号をそろえる） |
| `uniquenesses`, `communalities` | Series | $\psi_i$ と $h_i^2$ |
| `fit` | dict | `caf`、`df`、`n_obs` があれば `chi2`、`p`、`cfi`、`rmsea` |
| `method`, `converged`, `heywood`, `objective` | | 推定法と診断 |

### `hull`

```text
hull(correlation, *, n_obs=None, max_factors=None, rng=None, method="uls",
     fit_index="caf", n_replications=1000) -> HullResult
```

Hull 法（Lorenzo-Seva, Timmerman & Kiers, 2011）。EFAtools の `HULL()` と同じ。$k = 0, \dots, J$ 因子の解を適合度
（CAF、CFI、$1-$RMSEA）と自由度で並べ、上側の凸包の上で scree 比
$st_k = \frac{(f_k - f_{k-1})/(df_k - df_{k-1})}{(f_{k+1} - f_k)/(df_{k+1} - df_k)}$ を最大にする $k$ を選ぶ。既定の $J$ は
平行分析の結果に 1 を足したもの（`rng` と `n_obs` が要る）で、自由度が正になる範囲に抑える。変数は 6 つ以上。

### `HullResult`

| 属性 | 型 | 説明 |
|---|---|---|
| `n_factors` | int | 選ばれた因子数 |
| `table` | DataFrame | $k$ ごとの `n_factors`、`fit`、`df`、`st`、`on_hull`、`heywood`、`converged` |
| `fit_index`, `method`, `max_factors` | | 使った設定 |

### `comparison_data`

```text
comparison_data(data, *, rng, max_factors=None, n_population=10000, n_samples=500,
                alpha=0.30, max_iter=50) -> ComparisonDataResult
```

比較データ法（Ruscio & Roche, 2012）。EFAtools の `CD()` と同じ。$k = 1, 2, \dots$ について、観測された Pearson 相関と
各変数の分布を再現する $k$ 個の共通因子をもつ母集団を生成し（Ruscio & Kaczetow, 2008）、データと同じ大きさの標本を
引いて、その固有値と観測固有値の RMSE を記録する。片側の Mann–Whitney 検定が `alpha` で RMSE を下げると言う限り
因子を加える。欠測のない数値データ。

### `ComparisonDataResult`

| 属性 | 型 | 説明 |
|---|---|---|
| `n_factors` | int | 提案される因子数 |
| `rmse` | DataFrame | 標本（行）と試した $k$（列）ごとの固有値の RMSE |
| `p_values` | Series | $k$ と $k - 1$ を比べる順位和の $p$（$k \ge 2$） |
| `population_rmse` | Series | 各母集団が観測相関をどれだけ再現するか |
| `max_factors` | int | 試せた最大の $k$ |

### `model_parallel_analysis`

```text
model_parallel_analysis(fit, data, *, key, n_replications=100, eigen_type="max",
                        zero_cell_correction=0.5, item_names=None) -> ModelParallelAnalysisResult
```

Drasgow & Lissak (1983) の修正平行分析。観測されたポリコリック（テトラコリック）相関行列の第 2 固有値を、`fit` から
シミュレーションしたデータ（大きさと欠測は同じ、$\theta \sim N(\hat\mu, \hat\sigma^2)$）での分布と比べる。
$p = (1 + \#\{T_b \ge T\})/(B + 1)$。`eigen_type="max"` は ltm の `unidimTest` と同じく対角に非対角の最大の相関を
置く。`"smc"` と `"pca"` も使える。`fit` は単一群・1 次元・`Normal` 潜在分布のモデルであること。

### `ModelParallelAnalysisResult`

| 属性 | 型 | 説明 |
|---|---|---|
| `statistic` | float | 観測された第 2 固有値 |
| `p_value` | float | Monte Carlo の $p$ 値 |
| `observed` | ndarray | 観測固有値 |
| `replications` | ndarray | シミュレーションしたデータの `(B, p)` の固有値 |
| `eigen_type`, `n_skipped` | | 使った設定と飛ばしたシミュレーションの数 |

### `LocalDependenceResult`

`FitResult.local_dependence()` の戻り値。

| 属性 | 型 | 説明 |
|---|---|---|
| `pairs` | DataFrame | ペアごとに 1 行: `item_i`, `item_j`, `n`, `q3`, `q3_star`, `x2`, `g2`, `df`, `p_x2`, `p_g2`, `cramer_v`。多母集団モデルでは先頭に `group` 列が付き、群とペアごとに 1 行 |
| `q3` | DataFrame | 項目 x 項目の Q3 行列。多母集団モデルでは群ごとの行列を `(group, item)` の行インデックスで並べる |
| `q3_mean` | float または Series | 非対角の Q3 の平均。多母集団モデルでは群をインデックスにした Series（各群の Q3\* はその群の平均を使う） |
| `max_q3_star` | float | 最大の Q3\*（全群で） |

### `JSIResult`

`FitResult.jsi()` の戻り値。

| 属性 | 型 | 説明 |
|---|---|---|
| `matrix` | DataFrame | 除いた項目（行）x 項目（列）の $\mathrm{JSI}_{ji}$。対角は NaN |
| `pairs` | DataFrame | ペア $i<j$ ごとに 1 行: `item_i`, `item_j`, `jsi_ij`（$j$ を除いたときの項目 $i$ の傾き）, `jsi_ji`, `folded`（両者の和。mirt の `fold = TRUE`） |
| `method` | str | 使った方法 |
| `converged` | Series | 除いた項目ごとに、推定し直しが収束したか |

### `detect`

```text
detect(data, clusters) -> DetectResult
```

与えた項目の分割に対する確認的 DETECT（Zhang & Stout, 1999）。各ペアの条件付き共分散は、総得点で条件付けた共分散と
残りの得点で条件付けた共分散の平均。DETECT $= 100 \times$ $\delta_{ij}\,\mathrm{ccov}_{ij}$ の平均（同じクラスタなら
$\delta = 1$、異なれば $-1$）で、ASSI と RATIO（Zhang, 2007）も求める。`data` は欠測のない数値の項目得点。
`clusters` は項目 → ラベルの対応、または列の順のラベル。よく使われる読み方（Jang & Roussos, 2007; Zhang, 2007）は、
DETECT $< 0.2$ で本質的に一次元、$0.2$〜$0.4$ で弱い、$0.4$〜$1$ で中程度、$> 1$ で強い多次元性。ASSI $> 0.25$ または
RATIO $> 0.36$ でその分割について一次元性からの本質的なずれ。

### `DetectResult`

| 属性 | 型 | 説明 |
|---|---|---|
| `indices` | DataFrame | 行 `DETECT`, `ASSI`, `RATIO`, `MADCOV100`, `MCOV100`、列 `unweighted`, `weighted` |
| `detect`, `assi`, `ratio` | float | 重みなしの指標 |
| `conditional_covariance` | DataFrame | 項目 x 項目の条件付き共分散 |
| `clusters` | Series | 使った分割 |
| `n_obs` | int | 回答者数 |

### `exploratory_detect`

```text
exploratory_detect(data, *, max_clusters=4, rng=None,
                   estimation_rows=None) -> ExploratoryDetectResult
```

行を推定用と検証用の半分に分け（`rng` で無作為に、または `estimation_rows` で指定）、推定用の半分で
$\max(\mathrm{ccov}) - \mathrm{ccov}$ に Ward 法を当てて項目をクラスタに分け、$2, \dots,$ `max_clusters` 個への各分割の
DETECT・ASSI・RATIO を両方の半分で報告する。

### `ExploratoryDetectResult`

| 属性 | 型 | 説明 |
|---|---|---|
| `table` | DataFrame | クラスタ数ごとの `cluster_sizes` と、推定用・検証用の半分での `DETECT` / `ASSI` / `RATIO` |
| `clusters` | dict[int, Series] | クラスタ数ごとの分割 |
| `n_clusters` | int | 推定用の半分での DETECT が最大のクラスタ数 |
| `estimation_rows`, `validation_rows` | ndarray | 2 つの半分の行位置 |

### `dimtest`

```text
dimtest(data, *, rng, assessment_items=None, training_fraction=1/3, n_bootstrap=100,
        min_group=20, max_assessment_size=None) -> DimtestResult
```

欠測のない 2 値データに対する DIMTEST（Stout, 1987; Nandakumar & Stout, 1993; Stout, Froelich & Gao, 2001）。項目を
検定用の部分テスト（AT）と分割用の部分テスト（PT）に分ける。本質的に一次元なら、PT の得点が同じ回答者の中での AT の
項目の共分散は 0 に近い。その標準化した和 $T_L$ を、カーネル平滑化した項目反応関数から生成した 1 次元データでの平均で
補正し、$T = (T_L - \bar T_G)/\sqrt{1 + 1/B}$、$p = 1 - \Phi(T)$ とする。`assessment_items` を与えなければ、回答者の
無作為な `training_fraction` で条件付き共分散（DETECT）により AT を選び、残りで検定する。検定は保守的である
（シミュレーションで第 1 種の誤りは名目水準を下回る）。小さい $p$ は第 2 の次元の強い証拠だが、大きい $p$ は一次元性の
弱い証拠にとどまる。

### `DimtestResult`

| 属性 | 型 | 説明 |
|---|---|---|
| `statistic`, `p_value` | float | $T$ と $1 - \Phi(T)$ |
| `t_l`, `t_g` | float, ndarray | $T_L$ とブートストラップの $T_L$ |
| `assessment_items`, `partitioning_items` | list[str] | AT と PT |
| `n_groups` | int | 使った PT の得点の群の数 |
| `training_rows`, `test_rows` | ndarray | AT を選ぶ行と検定する行 |

### `dimensionality`

```text
dimensionality(data, *, key, fit=None, n_replications=100, zero_cell_correction=0.5,
               alpha=0.05, q3_star_threshold=0.2, rmsea_threshold=0.05,
               detect_threshold=0.2, max_clusters=4, item_names=None) -> DimensionalityResult
```

いくつかの診断を実行して並べ、2 つの合意を別々に示す（単独で信頼できる手法はない。Auerswald & Moshagen, 2019）。

- **次元はいくつか**: `parallel_analysis()`（ポリコリック、PCA、並べ替え、95%）、`velicer_map()`（改訂版）、
  `empirical_kaiser()`、項目が 6 つ以上なら `hull()`（CAF、ULS）、欠測がなければ `comparison_data()`。合意は最頻値（同数ならすべて）と中央値、および一致した手法の数。
- **本質的一次元性は認められるか**: 欠測のないデータで `exploratory_detect()`（選んだ分割の検証用の半分での
  DETECT $<$ `detect_threshold`）、欠測のない 2 値データで `dimtest()`（$p \ge$ `alpha`）、`fit` を与えればさらに `model_parallel_analysis()`（$p \ge$ `alpha`）、最大の
  Q3\* $<$ `q3_star_threshold`（0.2 はよく使われる目安で、臨界値はテストと標本による）、`FitResult.m2()` の
  RMSEA $\le$ `rmsea_threshold`（近似的な適合。欠測があると使えず、その旨を示す）。合意は支持する手法の数（使えた
  手法のうち）。手法に重みは付けない。

### `DimensionalityResult`

| 属性 | 型 | 説明 |
|---|---|---|
| `number_of_dimensions` | DataFrame | `method`, `n_factors`, `detail`, `source` |
| `unidimensionality` | DataFrame | `method`, `statistic`, `threshold`, `supports_unidimensionality`（使えないときは `None`）, `detail`, `source` |
| `n_factors_mode`, `n_factors_median`, `n_agreeing` | tuple[int, ...], float, int | 次元数についての合意 |
| `n_supporting`, `n_unidimensionality_methods` | int | 本質的一次元性についての合意 |
| `results` | dict | 個々の結果 |
| `summary()` | str | テキストのレポート（`str(result)` も同じ） |

---

## 情報関数と信頼性

推定済みの 1 次元モデルに対する関数（[ガイド 5 節](guide.ja.md#情報関数期待得点信頼性)）。多次元モデルでは
どれも `SpecificationError`。

### `item_information`

```text
item_information(res, theta, *, group=None) -> DataFrame
```

潜在特性の値 `theta` での各項目の Fisher 情報量
$I_j(\theta)=\sum_kP_{jk}(\theta)\,(\partial\log P_{jk}(\theta)/\partial\theta)^2$（Birnbaum, 1968;
Samejima, 1969）。どの項目モデルでも自動微分で計算する。mirt の `iteminfo()` が返す値と同じ。

| 引数 | 型 | 説明 |
|---|---|---|
| `res` | `FitResult` | 推定済みの 1 次元モデル |
| `theta` | float / 1 次元配列 | 潜在特性の値（有限） |
| `group` | str（任意） | 群名。多母集団モデルでは必須 |

**戻り値**: `theta` の値ごとに 1 行（インデックス。名前は `theta`）、項目ごとに 1 列の `DataFrame`。

### `test_information`

```text
test_information(res, theta, *, group=None) -> DataFrame
```

テスト情報量 $I(\theta)=\sum_jI_j(\theta)$ と条件付き標準誤差 $1/\sqrt{I(\theta)}$（mirt の `testinfo()`）。
引数は [`item_information`](#item_information) と同じ。

**戻り値**: `theta` をインデックスとし、列 `information` と `se` をもつ `DataFrame`。

### `expected_score`

```text
expected_score(res, theta, *, group=None) -> DataFrame
```

項目の期待得点 $E_j(\theta)=\sum_kk\,P_{jk}(\theta)$ とテスト特性曲線 $T(\theta)=\sum_jE_j(\theta)$。
カテゴリの得点はデータと同じ $0,\dots,K_j-1$（mirt の `expected.item()` と `expected.test()`）。引数は
[`item_information`](#item_information) と同じ。

**戻り値**: `theta` をインデックスとし、項目ごとの列と列 `total` をもつ `DataFrame`。`total` という名前の
項目があると `SpecificationError`。

### `reliability`

```text
reliability(res, data=None, *, group=None, item_names=None) -> DataFrame
```

周辺信頼性と、`data` を渡した場合は EAP 得点の経験的信頼性。

- **周辺信頼性**: 潜在分布 $N(\mu,\sigma^2)$、テスト情報量 $I(\theta)$ に対して
  $\bar\rho=\int\frac{\sigma^2I(\theta)}{\sigma^2I(\theta)+1}\,\phi(\theta;\mu,\sigma^2)\,d\theta$
  （Green et al., 1984; Thissen & Wainer, 2001）。$N(0,1)$ では mirt の `marginal_rxx()` と一致し、潜在尺度を
  線形変換しても変わらない。積分は、推定に使った求積によらず、$\mu\pm8\sigma$ 上の等間隔 801 点で評価する。
- **経験的信頼性**: `data` の受検者の EAP 得点と標準誤差（`score(data, method="EAP")`）から
  $\hat\rho=s^2_{\hat\theta}/(s^2_{\hat\theta}+\overline{\mathrm{SE}^2})$。$s^2_{\hat\theta}$ は標本分散
  （分母 $N-1$）。mirt の `empirical_rxx()` と同じ。欠測のある行も採点して含める。

| 引数 | 型 | 説明 |
|---|---|---|
| `res` | `FitResult` | 潜在分布が `Normal` の、推定済みの 1 次元モデル |
| `data` | DataFrame / 2 次元配列（任意） | 推定と同じ項目の反応データ。渡すと経験的信頼性が加わる |
| `group` | 配列（任意） | `data` の行ごとの群ラベル（多母集団モデルで `data` を渡すときは必須） |
| `item_names` | list[str]（任意） | `data` が配列のときの列名 |

**戻り値**: 群ごとに 1 行の `DataFrame`（多母集団モデルでは先頭に `group` 列）。列は `marginal` と、`data` を
渡した場合は `empirical`（`data` の中の受検者が 2 人未満の群は NaN）。

**例外**: 多次元モデル、潜在分布が `EmpiricalHistogram`・`DavidianCurve`・`NormalMixture` の場合、`data` なしで
`group` / `item_names` を渡した場合は `SpecificationError`。

---

## 合計得点

### `summed_score_table`

```text
summed_score_table(res) -> DataFrame
```

推定済みモデルの合計得点 EAP の表（Thissen, Pommerich, Billeaud & Williams, 1995。
[ガイド 7 節](guide.ja.md#合計得点による-eap)）。合計得点 $s$ ごとに、推定に使った求積節点 $\theta_q$ と潜在重み
$w_q$、[`summed_score_distribution`](#summed_score_distribution) で求めた得点の確率 $L(s\mid\theta_q)$ から、
$P(s)=\sum_qL(s\mid\theta_q)w_q$、事後平均 $\sum_q\theta_qL(s\mid\theta_q)w_q/P(s)$、事後標準偏差を求める。
mirt の `fscores(method = "EAPsum", full.scores = FALSE)` の表と同じで、その `expected` 列は $N\,P(s)$。

**戻り値**: 合計得点 $0,\dots,\sum_j(K_j-1)$ ごとに 1 行の `DataFrame`（多母集団モデルでは群と得点ごとに 1 行で、
先頭に `group` 列）。列は `score`、`theta`、`se`、`probability`。

**例外**: 多次元モデル、潜在分布が `EmpiricalHistogram`・`DavidianCurve`・`NormalMixture` の場合は
`SpecificationError`。

### `summed_score_distribution`

```text
summed_score_distribution(probs, *, method="recursion", log=False) -> Array
```

独立な項目得点の合計得点の分布（[ガイド 5 節](guide.ja.md#合計得点の分布)）。`probs` の形は `(..., J, K)` で、
`J` 項目のカテゴリ確率（各行の和は 1。カテゴリ数が `K` より少ない項目は 0 で埋める）。たとえば、ある $\theta$ での
IRT モデルのカテゴリ確率。先頭の軸はバッチ軸。結果の形は `(..., J * (K - 1) + 1)` で、合計得点 $0, \dots, J(K-1)$ の
確率が入る。2 値項目なら Poisson 二項分布になる。`jax.jit`・`jax.vmap` の中で使え、微分もできる。値の検査はしない。

| `method` | 内容 | 精度 |
|---|---|---|
| `"recursion"`（既定） | Lord–Wingersky 再帰 | 厳密。裾も含めて、どの確率も相対誤差が小さい |
| `"fft"` | 特性関数の離散フーリエ変換 | 絶対誤差が約 `1e-16` の範囲で厳密。約 `1e-15` より小さい確率は信頼できない |
| `"normal"` | 連続修正つきの正規近似 | 2 値項目のみ。和がちょうど 1 になるとは限らない |
| `"refined_normal"` | 歪度を補正した正規近似 | 2 値項目のみ。和がちょうど 1 になるとは限らない |
| `"poisson"` | Poisson 近似 | 2 値項目のみ。すべての確率が小さいときに使える |

`log=True` は対数確率を返す。`"recursion"` では再帰そのものを対数空間で行い、0 にアンダーフローするはずの裾の確率も
保つ（そのぶん遅い）。他の方法では結果の対数をとる。形が合わない、`method` が未知、`K != 2` で近似を指定した、の
いずれかで `SpecificationError`。

---

## モデル比較とシミュレーション

### `compare_fits`

```text
compare_fits(*results, names=None) -> DataFrame
```

同じデータへの 2 つ以上の当てはめを、最も制約の強いモデルから弱いモデルの順に並べ、情報量規準と尤度比検定を
まとめる（[ガイド 5 節](guide.ja.md#モデルの比較aicbic尤度比検定)）。$\ell$ を周辺対数尤度、$k$ を自由パラメタ数、
$N$ を回答者数（重み付きの当てはめでは重みの和）として、$\mathrm{AIC}=-2\ell+2k$（Akaike, 1974）、
$\mathrm{SABIC}=-2\ell+k\log\frac{N+2}{24}$（Sclove, 1987）、$\mathrm{HQ}=-2\ell+2k\log\log N$（Hannan & Quinn,
1979）、$\mathrm{BIC}=-2\ell+k\log N$（Schwarz, 1978）。2 行目以降は直前の行と
$X^2=2(\ell_m-\ell_{m-1})$、自由度 $\mathit{df}=k_m-k_{m-1}$ で検定する（Wilks, 1938）。mirt の
`anova(mod1, mod2, ...)` に当たる。

| 引数 | 型 | 説明 |
|---|---|---|
| `*results` | `FitResult` | 同じ回答への 2 つ以上の当てはめ。制約の強いものから |
| `names` | list[str]、省略可 | `model` 列のラベル（既定は `model1`, `model2`, ...） |

**戻り値**: 当てはめごとに 1 行の `DataFrame`: `model`, `n_params`, `loglik`, `aic`, `sabic`, `hq`, `bic`,
`x2`, `df`, `p`。1 行目の `x2`・`df`・`p` は NaN。`df <= 0` のとき `p` は NaN。

**例外**: 当てはめが 2 つ未満、`FitResult` でない引数、`names` が当てはめごとに異なる 1 つのラベルでない、回答者数が
異なる、のいずれかで `SpecificationError`。

**警告**: パラメタ数が増えない、パラメタを増やしたのに対数尤度が下がる、事前分布のある当てはめがある、のいずれかで
`FlexpyirtWarning`。

入れ子関係は確かめられないので利用者の責任。境界の補正（分散を 0 に固定、混合成分を減らす、`DavidianCurve` 対
`Normal`）はしない。

### `simulate_responses`

```text
simulate_responses(source, *, n_obs=None, key, parameters=None, theta=None, group=None)
    -> SimulatedData
```

推定済みのモデル、または項目パラメタを与えた `Model` から項目反応を生成する
（[ガイド 5 節](guide.ja.md#回答データのシミュレーション)）。各回答者の $\boldsymbol\theta_i$ を所属する群の潜在分布から
引き（または `theta` から取り）、各回答を一様乱数の逆変換
$X_{ij}=\min\{k: u_{ij}\le\sum_{c\le k}P_{jc}(\boldsymbol\theta_i)\}$ で引く（Devroye, 1986）。潜在特性は、
`Normal` は推定された平均・分散・相関、`NormalMixture` は成分ごと、`DavidianCurve` は連続密度（細かいグリッド上の
累積分布関数の逆変換）、`EmpiricalHistogram` は求積点を推定された重みで（推定どおりの離散分布）引く。すべての項目
モデルとモデルの構造（1 次元、`with_factors`、双因子、two-tier、多母集団）に対応する。

| 引数 | 型 | 説明 |
|---|---|---|
| `source` | `FitResult` / `Model` | 当てはめ（その推定値を使う）または仕様 |
| `n_obs` | int / dict[str, int]、省略可 | 回答者数。多母集団モデルでは群ごとの人数、または群 → 人数の対応。`theta` を与えないときは必須 |
| `key` | `jax.Array` | JAX の PRNG キー（`jax.random.key(0)`）。同じキーからは同じデータ |
| `parameters` | DataFrame、省略可 | `Model` のとき（必須）: `Model.fix_from()` と同じ傾き・切片表現の項目バンクで、すべての項目パラメタを含む |
| `theta` | 配列、省略可 | 生成の代わりに使う潜在特性: 1 因子は `(N,)`、それ以外は `(N, D)` |
| `group` | 配列、省略可 | 多母集団モデルで `theta` を与えるとき: 各行の群ラベル |

`Model` から生成するときの潜在分布は仕様のもの: `Normal` の `mean` と `var`（因子間相関 0）、または
`NormalMixture` の `means`・`variances`・`weights`。`EmpiricalHistogram` と `DavidianCurve` は推定前に形がないので
`theta` が要る。

**戻り値**: [`SimulatedData`](#simulateddata)。

**例外**: 64 ビットの JAX でないと `PrecisionError`。`key`・`n_obs`・`theta`・`group`・`parameters` が不正
（たとえばバンクに項目パラメタが足りない）だと `SpecificationError`。

欠測は生成しない。テストデザインや欠測の機構には `responses` の要素を NaN にする。

### `SimulatedData`

[`simulate_responses`](#simulate_responses) が返す frozen dataclass。

| 属性 | 型 | 説明 |
|---|---|---|
| `responses` | DataFrame | 項目ごとの `int64` の列（モデルの項目名）、コード `0..K-1`。`fit()` に渡せる |
| `theta` | ndarray | 潜在特性。1 因子は `(N,)`、それ以外は `(N, D)` |
| `group` | ndarray または None | 各行の群ラベル（多母集団モデル）。それ以外は `None` |

---

## 設定

### `BAEM`

```text
BAEM(tol=1e-4, max_iter=2000, m_step_newton_steps=3, squarem=False)
```

| 引数 | 説明 |
|---|---|
| `tol` | 連続する採択反復点の間での自由パラメタの最大絶対変化がこれ未満で収束（mirt の `TOL` と同じ定義） |
| `max_iter` | 採択される最大反復回数。到達すると `ConvergenceWarning` |
| `m_step_newton_steps` | 1 回の M-step で進める Newton 法のステップ数。各ステップで期待対数尤度は減らない |
| `squarem` | SQUAREM スキーム S3 加速（Varadhan & Roland, 2008）と単調性・実行可能性の安全装置を有効にする（既定 `False`） |

### `Quadrature`

```text
Quadrature.equispaced(n_points=61, *, bounds=(-6.0, 6.0))
Quadrature.default(n_factors=1, *, bounds=(-6.0, 6.0))
Quadrature.gauss_hermite(n_points=21)
```

等間隔の求積格子（両端を含む）。2・3 次元では直積格子として使う。`Quadrature.default(n_factors)` は
mirt の既定と同じく、1 因子で 61 点、2 因子で各軸 31 点（$31^2 = 961$）、3 因子で各軸 15 点
（$15^3 = 3{,}375$）、区間 $[-6, 6]$ を返す。`nodes()` は 1 次元軸 `(n_points,)`、`grid(n_factors)` は
直積格子 `(n_points**n_factors, n_factors)` を返す。軸あたり点数は 3 以上、範囲は有限で `lower < upper`。

`Quadrature.gauss_hermite(n_points)` は `Normal` 潜在分布用に Gauss–Hermite 節点 $\sqrt2\,x_q$（次元あたり 3–200 点、
2・3 次元では直積則）を使い、推定した平均・共分散に合わせて重みを付け直す。`base_log_weights(n_factors)` は
節点ごとの固定の対数重み（等間隔格子では 0）を返す。`NormalMixture` でも使える（各成分はその平均・分散の `Normal` と同じ重みを受け取る）。`EmpiricalHistogram`・`DavidianCurve` とは併用できない。

---

## 例外と警告

### `flexpyirt.errors`

`from flexpyirt.errors import ...` で使う。

#### `FlexpyirtError`

flexpyirt が意図して出すすべての例外の基底クラス。

`SpecificationError` と警告のクラスは flexpyctt と共通で（`flexpyctt.errors` で定義し、ここで再公開している）、
`FlexpyirtError` と `FlexpyirtWarning` は flexpyctt の基底クラス `FlexpycttError`、`FlexpycttWarning` と同じクラス
である。したがって、flexpyirt のクラスで捕まえれば flexpyctt の関数が出したものも捕まえられる。

#### `SpecificationError`

仕様とデータの矛盾。列の過不足、範囲外の値、データに現れないカテゴリ、存在しない項目・パラメタの固定、
ParameterTable の不正な編集、群ラベルの不一致など。推定の前に出る。`ValueError` を継承。

#### `IdentificationError`

尺度が識別できない指定。例: アンカーなしで焦点群の平均・分散を推定、固定項目なしで単一群の平均を推定。
推定の前に出る。`ValueError` を継承。

#### `PrecisionError`

JAX の 64 ビット浮動小数点が無効。`jax.config.update("jax_enable_x64", True)` で有効にする。
`RuntimeError` を継承。

#### `FlexpyirtWarning`

flexpyirt が出すすべての警告の基底クラス（`UserWarning` を継承）。情報行列が正定値でない場合、
焦点群の平均が 1 つのパラメタだけでつながっている場合、または事前分布が有効なモデルで `FitResult.aic` /
`FitResult.bic` を参照した場合に直接出る。

#### `ConvergenceWarning`

`max_iter` までに収束しなかった、または SQUAREM が通常 EM（$\theta_2$）へ頻繁に退避した（5 回以上かつ全反復の 50% 超）。

#### `HeywoodWarning`

推定後、自由な項目パラメタが有限の境界から `1e-4` 以内に到達した（3PL / 4PL で `g` が 0 付近、`u` が 1 付近
など）か、自由な傾きが $|a| > 20$ になった場合に出る。該当する項目名をすべて列挙する。`NormalMixture` の解が退化した
場合（重みが `1e-3` 未満、標準偏差が格子間隔の半分未満、平均が格子の外のいずれかの成分、または 2 つの成分の一致）にも出る。

#### `NotPositiveDefiniteWarning`

`polychoric()` でペアごとの行列が正定値でなく、`smooth=True` により平滑化したときに出る。
