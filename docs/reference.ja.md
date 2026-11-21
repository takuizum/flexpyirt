# API リファレンス

[English](reference.md) | 日本語

flexpyirt の公開 API（`from flexpyirt import ...` で使えるもの）の一覧。使い方の流れは
[ユーザーガイド](guide.ja.md)、数式とアルゴリズムは [設計書](DESIGN.ja.md) を参照。
各関数の docstring にも同じ内容と対応する式・出典がある（`help(flexpyirt.fit)` など）。

公開関数の引数は、最初の 1〜2 個を除いてキーワード専用（`*` の後ろ）。

- 推定: [`fit`](#fit), [`FitResult`](#fitresult)
- モデル: [`Model`](#model)
- 項目モデル: [`TwoPL`](#twopl), [`Rasch`](#rasch), [`ThreePL`](#threepl), [`FourPL`](#fourpl), [`GRM`](#grm), [`GPCM`](#gpcm), [`PCM`](#pcm)
- 事前分布: [`NormalPrior`](#normalprior), [`LogNormal`](#lognormal), [`Beta`](#beta)
- 潜在分布: [`Normal`](#normal)
- 設定: [`BAEM`](#baem), [`Quadrature`](#quadrature)
- 例外と警告: [`flexpyirt.errors`](#flexpyirterrors)

---

## 推定

### `fit`

```text
fit(spec, data, *, group=None, table=None, item_names=None, method=None,
    quadrature=None, se="oakes") -> FitResult
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
| `quadrature` | `Quadrature`, 任意 | 求積格子（既定 61 点・$[-6, 6]$） |
| `se` | `"oakes"` / `"hessian"` / `None` | 標準誤差の計算方法（MAP 時は事前分布の Hessian を含む）。`None` なら計算しない |

**戻り値**: [`FitResult`](#fitresult)

**例外**: `PrecisionError`（JAX の 64 ビットが無効）、`SpecificationError`（仕様やデータの誤り）、
`IdentificationError`（識別不能）。いずれも推定を始める前に出る。

**警告**: `ConvergenceWarning`（`max_iter` に到達、または SQUAREM が通常 EM へ頻繁に退避）、`HeywoodWarning`（自由な漸近線が境界から `1e-4` 以内、または自由な傾きが $|a| > 20$）、`FlexpyirtWarning`（情報行列が正定値でない、アンカーが 1 パラメタだけ）。

### `FitResult`

`fit()` の戻り値。変更不可。

| 属性 | 型 | 説明 |
|---|---|---|
| `items` | DataFrame | 1 項目 1 行（多母集団では `group` 列つきで群 × 項目）。`model`、内部パラメタ（`a`, `d` / `d1`…, `g`, `u`）、`<param>_se`、表示用の `b` / `bk`（$-d/a$、GRM は $-d_k/a$、GPCM/PCM は $-(d_k-d_{k-1})/a$、$d_0=0$）とデルタ法による `b_se` / `bk_se`（`vcov` と JAX 自動微分から計算。`se=None` または閾値に入るパラメタがすべて固定なら NaN） |
| `latent` | DataFrame | 1 群 1 行。`group`, `mean`, `var`, `mean_se`, `var_se`（固定値の SE は NaN） |
| `parameters` | DataFrame | 検証済みの ParameterTable に `estimate`（推定値。固定なら固定値）と `se` を足したもの |
| `loglik` | float | 推定値での周辺対数尤度（尤度のみ、事前分布は含めない） |
| `log_posterior` | float | 推定値での対数事後確率（$\ell(\hat\eta) + \log\pi(\hat\eta)$）。事前分布なしでは `loglik` と同値 |
| `aic`, `bic` | float | $-2\ell + 2k$、$-2\ell + k\log N$（$k$ = `n_params`, $N$ = `n_obs`）。事前分布が有効なときは `FlexpyirtWarning` を出す |
| `n_params` | int | 自由パラメタ数（等値制約は 1 つと数える） |
| `n_obs` | int | 回答者数 |
| `converged` | bool | 収束基準を満たしたか |
| `n_iter` | int | 採択された EM（または SQUAREM）の反復回数 |
| `n_em_evaluations` | int | EM 写像（E-step + M-step）の総評価回数。通常 EM では `n_iter` と同値、SQUAREM では採択 1 回あたり 2 または 3 |
| `n_safeguards` | int | SQUAREM の単調性・実行可能性の安全装置が介入した反復数（通常 EM では 0） |
| `history` | DataFrame | `iter`, `loglik`, `log_posterior`（その反復に入るときの値。反復ごとに単調非減少）, `max_change`, `em_evaluations`（累積 EM 写像評価回数）, `safeguard`（SQUAREM 安全装置が発動したか） |
| `vcov` | ndarray / None | 自由パラメタの共分散行列（`free_id` 順）。`se=None` なら None |
| `se_method` | str / None | `"oakes"` / `"hessian"` / None |

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
score(data, *, method="EAP", group=None, item_names=None) -> DataFrame
```

回答者ごとの EAP（`theta`）と事後標準偏差（`se`）を返す（Bock & Mislevy, 1982）。事前分布は推定した潜在分布
（多母集団では回答者の群の分布）、求積格子は推定と同じ。欠測は尤度から除く。

| 引数 | 説明 |
|---|---|
| `data` | 推定と同じ項目の反応データ（推定に使ったものでなくてよい） |
| `method` | `"EAP"` のみ（MAP / ML / WLE / PV は M4 予定） |
| `group` | 多母集団モデルでは必須。行ごとの群ラベル |
| `item_names` | `data` が配列のときの列名 |

戻り値の index は `data` が DataFrame ならその index。

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
model.fix_from(bank, *, items=None, parameterization="slope-intercept") -> Model
```

項目バンクの値で項目パラメタを固定する（FIPC）。潜在分布を推定するには `.with_latent(Normal.free())` を併用する。

| 引数 | 説明 |
|---|---|
| `bank` | `item` 列と、パラメタの列を持つ DataFrame。`"slope-intercept"` なら `a`, `d`（GRM/GPCM/PCM は `d1`, `d2`, …）、`"irt"` なら `a`, `b`（GRM/GPCM/PCM は `b1`, …）。Rasch・PCM 項目は `a` 列がなくてよい。漸近線 `g`（3PL/4PL）・`u`（4PL）はどちらのパラメタ化でもそのまま渡す（3PL 項目では `u` 列を省略できる） |
| `items` | 固定する項目（名前のリストか `str -> bool` の関数）。既定はバンクにあってモデルにもある項目すべて |
| `parameterization` | `"slope-intercept"` または `"irt"`（$d = -ab$、GPCM/PCM は $d_k = -a\sum_{c=1}^{k} b_c$ に変換） |

#### `Model.with_latent`

```text
model.with_latent(dist) -> Model
```

単一群モデルの潜在分布を `Normal` で指定する。多母集団モデルでは使えない（`with_groups()` で指定する）。

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

#### `Model.prior`

```text
model.prior(*, param, dist, items=None) -> Model
```

選択した項目の内部パラメタ `param` に事前分布 `dist`（`NormalPrior`, `LogNormal`, `Beta`）を全群共通で付ける
（MAP 推定、[ガイド 11 節](guide.ja.md#11-事前分布と-map-推定)）。事前分布はヤコビアンなしの自然尺度で評価される。
固定パラメタの事前分布は無視され、群間で共有されるパラメタ（同じ `free_id`）は 1 回だけ数えられる。潜在分布パラメタ
（`"mean"`, `"var"`）への指定は `SpecificationError`。

| 引数 | 説明 |
|---|---|
| `param` | 内部項目パラメタ名（`"a"`, `"d"`, `"g"`, `"u"`, `"d1"`, …） |
| `dist` | `NormalPrior`, `LogNormal`, `Beta` のいずれか |
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
| `resolved_latent()` | 単一群モデルの潜在分布（既定則を適用したもの） |
| `resolved_groups()` | `(群名, Normal)` のタプル。単一群なら `(("all", ...),)` |
| `items`, `fixed`, `latent`, `groups`, `equal`, `priors` | 指定の中身（通常は直接使わない） |

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
GRM(n_cat)
```

段階反応モデル（Samejima, 1969）。カテゴリは `0..n_cat-1`。
$P(y\ge k\mid\theta)=\sigma(a\theta+d_k)$、$d_1>\dots>d_{K-1}$。内部パラメタ `a`, `d1`…`d{K-1}`、
表示用に $b_k=-d_k/a$。mirt の `itemtype = "graded"` と同じ。`n_cat` が同じ GRM 項目はまとめて推定される。

### `GPCM`

```text
GPCM(n_cat)
```

一般化部分採点モデル（Muraki, 1992）。カテゴリは `0..n_cat-1`（`n_cat >= 2`）。
$P(y=k\mid\theta)=\exp(k\,a\theta+d_k)/\sum_{c=0}^{K-1}\exp(c\,a\theta+d_c)$（$d_0=0$ 固定）。
内部パラメタ `a`, `d1`…`d{K-1}`（順序制約なし）、表示用にステップ困難度 $b_k=-(d_k-d_{k-1})/a$（$d_0=0$）。
mirt の `itemtype = "gpcm"` と同じ。`n_cat = 2` では `TwoPL` と一致する。`n_cat` が同じ `PCM` と
`GPCMKernel` を共有する。

### `PCM`

```text
PCM(n_cat)
```

部分採点モデル（Masters, 1982）。カテゴリは `0..n_cat-1`（`n_cat >= 2`）。`a = 1` に固定した GPCM で、
$P(y=k\mid\theta)=\exp(k\theta+d_k)/\sum_{c=0}^{K-1}\exp(c\theta+d_c)$（$d_0=0$ 固定）。
表示用にステップ困難度 $b_k=-(d_k-d_{k-1})$。全項目が `Rasch` または `PCM` の単一群モデルでは、既定で
潜在分布の分散を推定する（多値データに対する mirt の `itemtype = "Rasch"` と同じ）。`n_cat = 2` では
`Rasch` と一致する。`n_cat` が同じ `GPCM` と `GPCMKernel` を共有する。

---

## 事前分布

ヤコビアンなしの自然尺度で項目パラメタに課す事前分布の不変データクラス（DESIGN §4.7）。いずれもハイパーパラメタを検証し
（不正なら `SpecificationError`）、`support` と `log_pdf(x) -> Array` を持つ。

### `NormalPrior`

```text
NormalPrior(mean, sd)
```

$(-\infty, \infty)$ 上の正規事前分布:
$\log\pi(x) = -\tfrac{1}{2}\log(2\pi) - \log\sigma - \frac{(x-\mu)^2}{2\sigma^2}$
（Mislevy, 1986、mirt の `norm`）。有限な `mean` と正の有限な `sd` が必要。潜在分布 `Normal` と名前が衝突しないよう
`NormalPrior` としている。

### `LogNormal`

```text
LogNormal(meanlog, sdlog)
```

$(0, \infty)$ 上の対数正規事前分布（$\log X \sim N(\mu, s^2)$）:
$\log\pi(x) = -\log x - \log s - \tfrac{1}{2}\log(2\pi) - \frac{(\log x - \mu)^2}{2s^2}$
（$x \le 0$ で $-\infty$。Mislevy, 1986、mirt の `lnorm`）。有限な `meanlog` と正の有限な `sdlog` が必要。
傾き `a` によく使う。

### `Beta`

```text
Beta(a, b)
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
Quadrature.equispaced(n_points=61, *, lim=(-6.0, 6.0))
```

1 次元の等間隔の求積格子（両端を含む）。既定は mirt の既定と同じ 61 点・$[-6, 6]$。
`nodes()` で格子点の配列を返す。点数は 3 以上、範囲は有限で `lower < upper`。

---

## 例外と警告

### `flexpyirt.errors`

`from flexpyirt.errors import ...` で使う。

#### `FlexpyirtError`

flexpyirt が意図して出すすべての例外の基底クラス。

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
など）か、自由な傾きが $|a| > 20$ になった場合に出る。該当する項目名をすべて列挙する。
