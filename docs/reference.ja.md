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
| `quadrature` | `Quadrature`, 任意 | 求積格子（既定: 1 因子は 61 点、2 因子は各軸 31 点、3 因子は各軸 15 点、いずれも $[-6, 6]$） |
| `se` | `"oakes"` / `"hessian"` / `None` | 標準誤差の計算方法（MAP 時は事前分布の Hessian を含む）。`None` なら計算しない |

**戻り値**: [`FitResult`](#fitresult)

**例外**: `PrecisionError`（JAX の 64 ビットが無効）、`SpecificationError`（仕様やデータの誤り）、
`IdentificationError`（識別不能）。いずれも推定を始める前に出る。

**警告**: `ConvergenceWarning`（`max_iter` に到達、または SQUAREM が通常 EM へ頻繁に退避）、`HeywoodWarning`（自由な漸近線が境界から `1e-4` 以内、または自由な傾きが $|a| > 20$）、`FlexpyirtWarning`（情報行列が正定値でない、アンカーが 1 パラメタだけ）。

### `FitResult`

`fit()` の戻り値。変更不可。

| 属性 | 型 | 説明 |
|---|---|---|
| `items` | DataFrame | 1 項目 1 行（多母集団では `group` 列つきで群 × 項目）。`model`、内部パラメタ（`a` または `a1`…`aD`、`d` / `d1`…、`g`, `u`）、`<param>_se`、および 1 次元モデルでは表示用の `b` / `bk`（$-d/a$、GRM は $-d_k/a$、GPCM/PCM は $-(d_k-d_{k-1})/a$、$d_0=0$）とデルタ法による `b_se` / `bk_se`（`vcov` と JAX 自動微分から計算。`se=None` または閾値に入るパラメタがすべて固定なら NaN） |
| `latent` | DataFrame | 1 群 1 行。1 因子では `group`, `mean`, `var`, `mean_se`, `var_se`、2–3 因子では `group`, `mean_1`…`mean_D`, `var_1`…`var_D`, `cor_12`… とその `_se` 列（固定値の SE は NaN）。`EmpiricalHistogram` では各ヒストグラムの `group`, `mean`, `var`, `sd`, `skewness`, `kurtosis`（超過尖度） |
| `latent_weights` | DataFrame / None | `EmpiricalHistogram` のみ: `group`, `node`, `weight`（群 × 求積節点ごとに 1 行）。`Normal` では None |
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
score(data, *, method="EAP", group=None, item_names=None, key=None, n_draws=5) -> DataFrame
```

推定済みモデルで回答者を採点する（[ユーザーガイド 7 節](guide.ja.md#7-スコアリング)）。すべての手法で欠測反応は
尤度（WLE では $I(\theta)$ も）から除く。

| 引数 | 型 | 説明 |
|---|---|---|
| `data` | DataFrame / polars DataFrame / 2 次元配列 | 推定と同じ項目の反応データ（推定に使ったものでなくてよい） |
| `method` | `"EAP"` / `"MAP"` / `"ML"` / `"WLE"` / `"PV"` | 採点手法（既定 `"EAP"`）: EAP（推定時の求積格子上の事後平均・事後 SD; Bock & Mislevy, 1982）、MAP（連続正規事前分布 $N(\mu_g,\sigma_g^2)$ による事後最頻値と観測曲率 SE）、ML（最尤推定と観測情報量 SE。有限最大値のないパターンは $-\infty$/$+\infty$・SE NaN、全欠測は NaN）、WLE（Warm, 1989; Firth, 1993: $\partial\log L/\partial\theta + I'(\theta)/(2I(\theta))=0$ を解く。極端パターンでも有限、全欠測は NaN、SE は WLE 目的関数の観測曲率）、PV（501 点格子上の区分線形逆 CDF 補間による回答者ごと独立な `n_draws` 個の事後標本; Mislevy, 1991） |
| `group` | 配列（任意） | 多母集団モデルでは必須。行ごとの群ラベル |
| `item_names` | list[str]（任意） | `data` が配列のときの列名 |
| `key` | `jax.Array`（任意） | 明示的な JAX PRNG キー（`jax.random.key()` または `jax.random.PRNGKey()`）。`method="PV"` では必須、他の手法では `None` |
| `n_draws` | int | `method="PV"` で回答者ごとに抽出する plausible values の個数（既定 `5`、1 以上の整数） |

**戻り値**: 回答者 1 行ずつの `DataFrame`（`data` が DataFrame ならその index を引き継ぐ）。1 次元モデルでは
`"EAP"`, `"MAP"`, `"ML"`, `"WLE"` が `theta`, `se` 列、`"PV"` が `pv1`…`pv{n_draws}` 列。多次元モデルでは
宣言した因子順に `theta_<factor>`, `se_<factor>` 列（`"PV"` は `pv<m>_<factor>`）。

**例外**: 不正な `method`、`EmpiricalHistogram` または `DavidianCurve` で `method="MAP"` を指定した場合、`"PV"` で
`key` が未指定・不正、決定的手法に `key` を渡した場合、または `n_draws` が正の整数でない場合に `SpecificationError`。

#### `FitResult.item_fit`

```text
item_fit(data, *, mincell=1.0, item_names=None) -> DataFrame
```

Orlando–Thissen の S-X2 項目適合度（[ガイド 5 節](guide.ja.md#項目適合度s-x2)）。各項目について、合計得点
$1,\dots,S-1$ ごとの各カテゴリの観測度数とモデルによる期待度数を比べる。期待度数は、その項目を除いた
Lord–Wingersky 漸化式と、推定時の求積格子上の推定済み潜在分布の重みで計算する。度数の少ないセルは mirt の
`itemfit()` と同じ方法でまとめ（`mincell` は mirt の引数）、$df_j$ = 残ったセル数 − 行数 − その項目の自由パラメタ数 −
潜在分布の自由パラメタ数（mirt の約束）。

| 引数 | 説明 |
|---|---|
| `data` | 推定した項目への欠測のない反応（通常は推定に使ったデータ） |
| `mincell` | まとめる前のセルあたり最小期待度数（既定 `1.0`） |
| `item_names` | `data` が配列のときの列名 |

**戻り値**: `item`, `s_x2`, `df`, `rmsea`（$\sqrt{\max(0, X^2/df-1)/(N-1)}$）, `p` 列の `DataFrame`。まとめた結果
`df <= 0` になる項目は `df = 0` で統計量は NaN。

**例外**: 多次元・多母集団・`EmpiricalHistogram`・`DavidianCurve` のモデル、欠測を含むデータ、観測されないカテゴリがある
項目、正でない `mincell` の場合に `SpecificationError`。

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
`"format": "flexpyirt.FitResult"`、`"schema_version": 3`（version 1・2 のファイルも読める）、`"flexpyirt_version"` を記録する。多次元の結果は因子構造も保存する。有限な
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
model.fix_from(bank, *, items=None, parameterization="slope-intercept") -> Model
```

項目バンクの値で項目パラメタを固定する（FIPC）。潜在分布を推定するには `.with_latent(Normal.free())` を併用する。

| 引数 | 説明 |
|---|---|
| `bank` | `item` 列と、パラメタの列を持つ DataFrame。`"slope-intercept"` なら `a`, `d`（GRM/GPCM/PCM は `d1`, `d2`, …）、`"irt"` なら `a`, `b`（GRM/GPCM/PCM は `b1`, …）。Rasch・PCM 項目は `a` 列がなくてよい。漸近線 `g`（3PL/4PL）・`u`（4PL）はどちらのパラメタ化でもそのまま渡す（3PL 項目では `u` 列を省略できる） |
| `items` | 固定する項目（名前のリストか `str -> bool` の関数）。既定はバンクにあってモデルにもある項目すべて |
| `parameterization` | `"slope-intercept"` または `"irt"`（$d = -ab$、GPCM/PCM は $d_k = -a\sum_{c=1}^{k} b_c$ に変換） |

#### `Model.with_factors`

```text
model.with_factors(factors, *, correlated=True) -> Model
```

2 または 3 次の確認的因子負荷パターンを宣言する（[ガイド 14 節](guide.ja.md#14-確認的多次元モデル)）。
`factors` は因子名（順序つき、2 または 3 因子）から各因子に負荷する項目（項目名の列または `str -> bool` 関数）への
辞書。交差負荷を許し、各因子は 1 項目以上を持ち、すべてのモデル項目が 1 因子以上に負荷する必要がある。負荷しない因子の
傾きは `0` に固定され、項目パラメタ名は `a1`…`aD`, `d`（または `d1`…）になる。潜在分布は多変量正規分布（平均 `0` 固定、
分散 `1` 固定。全項目が `Rasch` / `PCM` の場合は分散も自由）で、`correlated=True` のとき既定で因子間相関を自由推定
（mirt の `COV = F1*F2` 相当）、`correlated=False` で `0` に固定する。多次元モデルでの多母集団（`with_groups()`）と
`with_latent()` は未対応（`SpecificationError`）。

#### `Model.with_latent`

```text
model.with_latent(dist) -> Model
```

単一群モデルの潜在分布を `Normal` または `EmpiricalHistogram`（1 次元モデルのみ）で指定する。多母集団モデルでは使えない（`with_groups()` で指定する）。

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
| `is_multidimensional` | `with_factors()` で確認的因子を宣言したか |
| `n_factors` | 潜在因子数（`1`, `2`, または `3`） |
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
Quadrature.default(n_factors=1, *, lim=(-6.0, 6.0))
Quadrature.gauss_hermite(n_points=21)
```

等間隔の求積格子（両端を含む）。2・3 次元では直積格子として使う。`Quadrature.default(n_factors)` は
mirt の既定と同じく、1 因子で 61 点、2 因子で各軸 31 点（$31^2 = 961$）、3 因子で各軸 15 点
（$15^3 = 3{,}375$）、区間 $[-6, 6]$ を返す。`nodes()` は 1 次元軸 `(n_points,)`、`grid(n_factors)` は
直積格子 `(n_points**n_factors, n_factors)` を返す。軸あたり点数は 3 以上、範囲は有限で `lower < upper`。

`Quadrature.gauss_hermite(n_points)` は `Normal` 潜在分布用に Gauss–Hermite 節点 $\sqrt2\,x_q$（次元あたり 3–200 点、
2・3 次元では直積則）を使い、推定した平均・共分散に合わせて重みを付け直す（DESIGN §15.44）。`base_log_weights(n_factors)` は
節点ごとの固定の対数重み（等間隔格子では 0）を返す。`EmpiricalHistogram` とは併用できない。

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
