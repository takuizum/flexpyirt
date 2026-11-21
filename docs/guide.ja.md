# ユーザーガイド

[English](guide.md) | 日本語

flexpyirt で IRT モデルを指定・推定・解釈する手順をまとめる。このページのコード例は上から順に実行でき、
`tests/test_docs.py` が毎回すべて実行して動作を確かめている。関数ごとの引数は
[API リファレンス](reference.ja.md) を参照。

1. [準備](#1-準備)
2. [データの形式](#2-データの形式)
3. [モデルを指定する](#3-モデルを指定する)
4. [推定する](#4-推定する)
5. [結果を読む](#5-結果を読む)
6. [標準誤差](#6-標準誤差)
7. [スコアリング](#7-スコアリング)
8. [潜在分布と識別性](#8-潜在分布と識別性)
9. [多母集団](#9-多母集団)
10. [固定項目パラメタ推定（FIPC）](#10-固定項目パラメタ推定fipc)
11. [ParameterTable を編集する](#11-parametertable-を編集する)
12. [求積格子の選び方](#12-求積格子の選び方)
13. [エラーと警告](#13-エラーと警告)

## 1. 準備

flexpyirt は 64 ビット浮動小数点で計算する。JAX の設定はユーザーのものなので、ライブラリは import 時に
変更しない。推定の前に有効にする（無効のまま `fit()` を呼ぶと `PrecisionError`）。

```python
import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pandas as pd

from flexpyirt import BAEM, GRM, Model, Normal, Quadrature, Rasch, TwoPL, fit
```

以下の例で使うデータをシミュレーションしておく。

```python
rng = np.random.default_rng(2026)


def simulate_2pl(a, d, theta, rng):
    """2PL の 0/1 反応を生成する。"""
    p = 1.0 / (1.0 + np.exp(-(np.outer(theta, a) + d)))
    return (rng.random(p.shape) < p).astype(float)


n_items = 12
items = [f"q{j:02d}" for j in range(1, n_items + 1)]
a_true = rng.lognormal(0.0, 0.3, n_items)
d_true = rng.normal(0.0, 1.0, n_items)
df = pd.DataFrame(simulate_2pl(a_true, d_true, rng.normal(0, 1, 1500), rng), columns=items)
```

## 2. データの形式

- **行が回答者、列が項目**。列名が項目名になる。
- pandas の DataFrame、polars の DataFrame、2 次元の numpy 配列を受け付ける。numpy 配列の場合は
  `item_names=` で列名を渡す（省略するとモデルの項目順とみなす）。
- 値はカテゴリ番号 `0, 1, ..., K-1`。2 値項目は 0/1。**欠測は NaN**（MAR を仮定して尤度から除く）。
- モデルにない列や、モデルにあってデータにない列はエラーになる。不要な列は先に落とす。
- 推定するパラメタを持つ項目は、すべてのカテゴリがデータに 1 回以上現れている必要がある
  （現れないカテゴリの閾値は発散するため。カテゴリをまとめるか、項目を固定する）。

```python
arr = df.to_numpy()
spec = Model.uniform(TwoPL(), items=items)
res_np = fit(spec, arr, item_names=items, se=None)
```

## 3. モデルを指定する

モデルは `Model` という変更不可能なオブジェクトで表す。`.fix()` や `.with_groups()` などのメソッドは
新しい `Model` を返し、元のオブジェクトは変わらない。

```python
spec = Model.uniform(TwoPL(), items=items)  # 全項目に同じ項目モデル
mixed = Model.from_items({"x1": TwoPL(), "x2": Rasch(), "s1": GRM(n_cat=4)})  # 項目ごとに指定
print(spec.item_names[:3], mixed.item_names)
```

### 項目モデル

| クラス | 反応確率 | 内部パラメタ | 表示用（`items` に追加） |
|---|---|---|---|
| `TwoPL()` | $P(y=1)=\sigma(a\theta+d)$ | `a`, `d` | `b = -d/a` |
| `Rasch()` | $P(y=1)=\sigma(\theta+d)$（$a=1$ 固定） | `a`（固定）, `d` | `b = -d` |
| `GRM(n_cat=K)` | $P(y\ge k)=\sigma(a\theta+d_k)$, $d_1>\dots>d_{K-1}$ | `a`, `d1`…`d{K-1}` | `bk = -dk/a` |

パラメタ化は mirt（`itemtype = "2PL"`, `"Rasch"`, `"graded"`）と同じ傾き・切片形式。
GRM のカテゴリは 0 始まり（mirt の `Science` のように 1 始まりのデータは 1 を引く）。

```python
n_cat = 4
cuts = np.array([-1.0, 0.2, 1.3])
theta = rng.normal(0, 1, 800)
latent_cont = np.outer(theta, np.ones(5)) + rng.logistic(0, 1, (800, 5))
grm_df = pd.DataFrame(
    np.searchsorted(cuts, latent_cont).astype(float), columns=[f"s{j}" for j in range(1, 6)]
)
grm = fit(Model.uniform(GRM(n_cat=n_cat), items=grm_df.columns), grm_df)
print(grm.items[["item", "a", "d1", "d2", "d3", "b1", "b2", "b3"]].round(3))
```

### パラメタを固定する

`.fix(item=..., <パラメタ>=値)` で、内部パラメタ（傾き・切片形式）の値に固定する。
固定したパラメタは推定されず、標準誤差も NaN になる。

```python
spec_fixed = Model.uniform(TwoPL(), items=items).fix(item="q01", a=1.0, d=0.0)
```

## 4. 推定する

```python
res = fit(
    spec,
    df,
    method=BAEM(tol=1e-4, max_iter=2000, m_step_newton_steps=3),  # 既定値
    quadrature=Quadrature.equispaced(61, lim=(-6, 6)),  # 既定値
    se="oakes",  # 既定値
)
```

- **`method`**: EM の設定。収束は「自由パラメタの最大絶対変化 < `tol`」（mirt の `TOL` と同じ定義）。
  `max_iter` に達すると `ConvergenceWarning`。M-step は Newton 法を `m_step_newton_steps` 回だけ進める
  （毎回の M-step で期待対数尤度が減らないので、EM の単調性は保たれる）。
- **`quadrature`**: 潜在変数の求積格子。既定は等間隔 61 点・$[-6, 6]$（mirt の既定と同じ）。
  広げるべき場合は [12 節](#12-求積格子の選び方)。
- **`se`**: `"oakes"`（既定）、`"hessian"`、`None`（計算しない）。

初回の推定には JAX のコンパイル時間がかかる。同じ形のデータを続けて推定すると 2 回目以降は速い。

## 5. 結果を読む

`fit()` は `FitResult` を返す。

```python
print(res)  # FitResult(loglik=..., n_params=..., n_obs=..., converged in ... iterations, se=oakes)
print(res.items.head(3))  # 項目ごとの推定値と SE
print(res.latent)  # 群ごとの潜在分布（mean, var と SE）
print(res.loglik, res.aic, res.bic, res.n_params, res.n_obs)
print(res.converged, res.n_iter)
print(res.history.tail(3))  # iter, loglik, max_change
```

| 属性 | 内容 |
|---|---|
| `items` | 1 項目 1 行（多母集団では群 × 項目）。`model`、内部パラメタ、`<param>_se`、表示用の `b` / `bk` |
| `latent` | 1 群 1 行。`group`, `mean`, `var`, `mean_se`, `var_se`（固定値の SE は NaN） |
| `parameters` | ParameterTable（[11 節](#11-parametertable-を編集する)）に `estimate` と `se` を足したもの |
| `loglik`, `aic`, `bic` | 周辺対数尤度と情報量規準（$-2\ell+2k$, $-2\ell+k\log N$） |
| `n_params`, `n_obs` | 自由パラメタ数、回答者数 |
| `converged`, `n_iter`, `history` | 収束したか、反復回数、反復ごとの対数尤度と最大変化 |
| `vcov`, `se_method` | 自由パラメタの共分散行列（`free_id` 順）と計算方法 |

`history.loglik` は反復ごとに減らない（EM の単調性）。減っていたらバグなので報告してほしい。

```python
assert (res.history["loglik"].diff().dropna() >= -1e-8).all()
```

## 6. 標準誤差

- **`se="oakes"`**（既定）: Oakes (1999) の恒等式で観測情報行列を求める。E-step の期待度数を通した
  2 階微分を JAX の自動微分で計算する。
- **`se="hessian"`**: 周辺対数尤度を直接 2 回微分する。

両者は数学的に同じ量で、数値的にも丸め誤差の範囲で一致する。`vcov` の行と列は自由パラメタの番号
（ParameterTable の `free_id`）の順。

```python
res_h = fit(spec, df, se="hessian")
print(np.max(np.abs(res.vcov - res_h.vcov)))  # ほぼ 0

params = res.parameters
labels = params[params["free"]].drop_duplicates("free_id").sort_values("free_id")
vcov = pd.DataFrame(
    res.vcov,
    index=labels["item"] + "." + labels["param"],
    columns=labels["item"] + "." + labels["param"],
)
print(vcov.iloc[:2, :2])
```

情報行列が正定値でない（推定値が境界にある、識別が弱いなど）と警告を出し、SE は NaN になる。
表示用の `b` の SE（デルタ法）は M4 で追加予定。

## 7. スコアリング

`FitResult.score()` は EAP（事後期待値）と事後標準偏差を返す。事前分布には推定した潜在分布を使う
（多母集団では回答者の群の分布）。欠測は尤度から除かれ、全項目が欠測の回答者は事前分布の平均・SD になる。

```python
scores = res.score(df)
print(scores.head(3))  # theta, se（index は df と同じ）

new = df.iloc[:2].copy()
new.iloc[0, :3] = np.nan  # 一部欠測でも採点できる
print(res.score(new))
```

採点に使うデータは、推定に使ったデータと別でもよい（列は同じ項目）。MAP / ML / WLE / plausible values は M4 で
追加予定。

## 8. 潜在分布と識別性

潜在分布は正規分布 `Normal` で、格子上で密度を正規化した重みを使う。

```python
Normal.standard()  # N(0, 1)、何も推定しない
Normal(free_var=True)  # N(0, σ²)、分散を推定（平均 0 固定）
Normal.free()  # 平均と分散を推定
Normal(mean=0.5, var=1.2)  # 固定値
```

**既定**: 単一群のモデルで潜在分布を指定しないとき、全項目が Rasch なら $N(0,\sigma^2)$（分散を推定）、
それ以外は $N(0,1)$ 固定。Rasch は傾きが 1 に固定されているので、尺度は分散が担う（mirt と同じ）。

```python
rasch = fit(Model.uniform(Rasch(), items=items), df)
print(rasch.latent[["mean", "var", "var_se"]])
```

潜在分布の平均・分散を推定するには、尺度を決める手がかりが必要になる。flexpyirt は推定の前に検査し、
足りなければ `IdentificationError` を出す。

- 分散を推定するには、値が 0 でない**固定された傾き**が 1 つ以上必要（Rasch 項目、`.fix(a=...)`、
  多母集団では傾きが群間で等しいアンカー項目）。
- 平均を推定するには、**固定された切片**が 1 つ以上必要（FIPC、多母集団ではアンカー項目）。

```python
from flexpyirt.errors import IdentificationError

try:
    fit(Model.uniform(TwoPL(), items=items).with_latent(Normal(free_var=True)), df)
except IdentificationError as e:
    print("IdentificationError:", e)
```

## 9. 多母集団

複数の集団（学年、受験年度、言語版など）の回答を同時に推定する。アンカー項目のパラメタを群間で等しくして
尺度をつなぎ、焦点群の潜在分布の平均・分散を推定する。

```python
n_per_group = 1200
theta_ref = rng.normal(0.0, 1.0, n_per_group)
theta_foc = rng.normal(0.5, 1.1, n_per_group)
d_foc = d_true.copy()
d_foc[6:] += rng.normal(0, 0.4, n_items - 6)  # 非アンカー項目は群で違ってよい
mg_df = pd.DataFrame(
    np.vstack(
        [simulate_2pl(a_true, d_true, theta_ref, rng), simulate_2pl(a_true, d_foc, theta_foc, rng)]
    ),
    columns=items,
)
group = np.repeat(["ref", "focal"], n_per_group)

anchors = items[:6]
# キーワードの順で群を定義し、最初の群が参照群になる。
mg_spec = (
    Model.uniform(TwoPL(), items=items)
    .with_groups(ref=Normal.standard(), focal=Normal.free())
    .equal_across_groups(items=anchors)  # 全パラメタを群間で等値
)
mg = fit(mg_spec, mg_df, group=group, quadrature=Quadrature.equispaced(81, lim=(-8, 8)))
print(mg.latent)
print(mg.items[mg.items["item"] == "q01"])  # アンカーは両群で同じ値
```

- `group=` は行ごとの群ラベル。`with_groups()` で宣言した群名と一致しなければエラー。
- 既定では、等値にしない項目のパラメタは**群ごとに別々に**推定される（mirt `multipleGroup` の既定と同じ）。
- `equal_across_groups(items=..., params=["a"])` のように一部のパラメタだけを等値にもできる。
  `items=` には名前のリストのほか、`lambda name: name.startswith("q0")` のような関数も渡せる。
- アンカーがない、または切片しか等値でないのに焦点群の分散を推定すると `IdentificationError`。
  焦点群の平均が 1 つのパラメタだけでつながっている場合は不安定になりやすいので警告を出す。
- 多母集団の `items` には `group` 列が付く。採点では `score(data, group=labels)` で群を渡す。

```python
mg_scores = mg.score(mg_df.iloc[[0, -1]], group=["ref", "focal"])
print(mg_scores)
```

## 10. 固定項目パラメタ推定（FIPC）

既に尺度化された項目バンクの値で一部の項目を固定し、新しい項目と潜在分布を推定する。固定項目が尺度を
決めるので、潜在分布の平均・分散を推定する（Kim, 2006 の MWU-MEM に相当）。

```python
theta_new = rng.normal(0.4, 1.15, 2000)
fipc_df = pd.DataFrame(simulate_2pl(a_true, d_true, theta_new, rng), columns=items)

bank = pd.DataFrame({"item": items[:6], "a": a_true[:6], "d": d_true[:6]})
fipc_spec = (
    Model.uniform(TwoPL(), items=items)
    .fix_from(bank)  # バンクにある項目を全群で固定
    .with_latent(Normal.free())  # 平均・分散を推定
)
fipc = fit(fipc_spec, fipc_df, quadrature=Quadrature.equispaced(81, lim=(-8, 8)))
print(fipc.latent[["mean", "var"]])  # 生成値は平均 0.4、分散 1.3225
```

項目バンクが IRT 表記（$a, b$）なら `parameterization="irt"` を指定する（$d=-ab$ に変換。GRM は `b1`, `b2`, … 列）。
Rasch 項目のバンクには `a` 列がなくてよい。

```python
bank_irt = pd.DataFrame({"item": items[:6], "a": a_true[:6], "b": -d_true[:6] / a_true[:6]})
fipc_spec_irt = (
    Model.uniform(TwoPL(), items=items)
    .fix_from(bank_irt, parameterization="irt")
    .with_latent(Normal.free())
)
```

## 11. ParameterTable を編集する

すべての制約は ParameterTable の列で表される（mirt の `pars = "values"` に相当）。表を取り出して pandas で
編集し、`fit(..., table=...)` に渡すと、表が `Model` の指定より優先される。

```python
table = spec.parameter_table(df)
print(table.head(4))
```

| 列 | 意味 |
|---|---|
| `group`, `item`, `param` | 行の識別子（単一群の群名は `"all"`、潜在分布の行は `item == "__latent__"`、`param` は `mean` / `var`） |
| `model` | 項目モデル名（編集しても無視される） |
| `value` | 初期値（固定なら固定値） |
| `free` | 推定するか |
| `lower`, `upper` | 範囲。初期値はこの範囲の内側でなければならない |
| `free_id` | 自由パラメタの番号。**同じ番号の行は同じ値に制約される**（等値制約）。新しく自由にした行は `-1` のままでよい（番号が振られる） |

```python
edited = table.copy()
row = (edited["item"] == "q05") & (edited["param"] == "a")
edited.loc[row, "value"] = 1.0
edited.loc[row, "free"] = False  # q05 の傾きを 1 に固定
row_d = (edited["item"] == "q05") & (edited["param"] == "d")
edited.loc[row_d, "value"] = 0.3  # 切片の初期値だけ変える
res_edit = fit(spec, df, table=edited, se=None)
print(res_edit.items.set_index("item").loc["q05", ["a", "d"]])
```

- 行の追加・削除や識別子の変更はできない（`SpecificationError`）。
- `free_id` の共有は**同じ項目の同じパラメタを群間で**等しくする用途に限られる。異なる項目をまたぐ等値制約
  （例: 全項目で共通の傾き）は未対応（M5 予定）。

## 12. 求積格子の選び方

潜在分布の重みは格子 $[\text{lower}, \text{upper}]$ の上で正規化するので、格子の端で分布が切れるとその分だけ
推定値が影響を受ける。既定の 61 点・$[-6, 6]$ は $N(0,1)$ 付近では十分だが、**推定する群の分散が大きい
（2 を超える）か平均が 0 から離れる**ときは、格子を広げて刻みを保つ。

```python
wide = Quadrature.equispaced(81, lim=(-8, 8))  # 刻み 0.2 のまま範囲を広げる
```

報告される `var` は格子で切る前の正規分布のパラメタで、格子上の分布の分散はそれよりわずかに小さい。
mirt は分散を閉形式のモーメントで更新するため、格子の端の影響が大きい条件では mirt と値がずれる。格子を
広げると両者は一致する（[VALIDATION.md](VALIDATION.ja.md)）。

## 13. エラーと警告

| 種類 | いつ出るか |
|---|---|
| `PrecisionError` | JAX の 64 ビット浮動小数点が無効なまま `fit()` を呼んだ |
| `SpecificationError` | 仕様とデータの矛盾（列の過不足、範囲外の値、観測されないカテゴリ、存在しない項目の固定、表の不正な編集、群ラベルの不一致） |
| `IdentificationError` | 尺度が識別できない（[8 節](#8-潜在分布と識別性)） |
| `ConvergenceWarning` | `max_iter` までに収束しなかった |
| `FlexpyirtWarning` | 情報行列が正定値でない（SE が NaN）、アンカーが 1 パラメタだけ |

すべての例外は `flexpyirt.errors.FlexpyirtError`、警告は `flexpyirt.errors.FlexpyirtWarning` を継承する。
`SpecificationError` と `IdentificationError` は `ValueError` でもある。

```python
from flexpyirt.errors import SpecificationError

try:
    fit(spec, df.assign(extra=0))
except SpecificationError as e:
    print("SpecificationError:", e)
```

## 参考文献

- Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of item parameters: Application of an EM algorithm. *Psychometrika*, 46, 443–459.
- Chalmers, R. P. (2012). mirt: A multidimensional item response theory package for the R environment. *Journal of Statistical Software*, 48(6).
- Kim, S. (2006). A comparative study of IRT fixed parameter calibration methods. *Journal of Educational Measurement*, 43, 355–381.
- Oakes, D. (1999). Direct calculation of the information matrix via the EM algorithm. *JRSS B*, 61, 479–482.
- Samejima, F. (1969). Estimation of latent ability using a response pattern of graded scores. *Psychometrika Monograph*, 17.
