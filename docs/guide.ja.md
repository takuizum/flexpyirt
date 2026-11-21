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
11. [事前分布と MAP 推定](#11-事前分布と-map-推定)
12. [ParameterTable を編集する](#12-parametertable-を編集する)
13. [求積格子の選び方](#13-求積格子の選び方)
14. [確認的多次元モデル](#14-確認的多次元モデル)
15. [エラーと警告](#15-エラーと警告)

## 1. 準備

flexpyirt は 64 ビット浮動小数点で計算する。JAX の設定はユーザーのものなので、ライブラリは import 時に
変更しない。推定の前に有効にする（無効のまま `fit()` を呼ぶと `PrecisionError`）。

```python
import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pandas as pd

from flexpyirt import (
    BAEM,
    GPCM,
    GRM,
    PCM,
    Beta,
    FitResult,
    FourPL,
    LogNormal,
    Model,
    Normal,
    NormalPrior,
    Quadrature,
    Rasch,
    ThreePL,
    TwoPL,
    fit,
)
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
| `ThreePL()` | $P(y=1)=g+(1-g)\sigma(a\theta+d)$（$u=1$ 固定） | `a`, `d`, `g`, `u`（固定） | `b = -d/a` |
| `FourPL()` | $P(y=1)=g+(u-g)\sigma(a\theta+d)$ | `a`, `d`, `g`, `u` | `b = -d/a` |
| `GRM(n_cat=K)` | $P(y\ge k)=\sigma(a\theta+d_k)$, $d_1>\dots>d_{K-1}$ | `a`, `d1`…`d{K-1}` | `bk = -dk/a` |
| `GPCM(n_cat=K)` | $P(y=k)\propto\exp(k\,a\theta+d_k)$, $d_0=0$ | `a`, `d1`…`d{K-1}` | `bk = -(dk - d{k-1})/a` |
| `PCM(n_cat=K)` | $P(y=k)\propto\exp(k\theta+d_k)$（$a=1$ 固定、$d_0=0$） | `a`（固定）, `d1`…`d{K-1}` | `bk = -(dk - d{k-1})` |

パラメタ化は mirt（`itemtype = "2PL"`, `"Rasch"`, `"3PL"`, `"4PL"`, `"graded"`, `"gpcm"`。多値に対する
`"Rasch"` は `PCM`）と同じ傾き・切片形式。3PL / 4PL の `g`, `u` は自然尺度（$0 < g < u \le 1$）で推定する
（mirt は logit 尺度で最適化するが、内点の最尤推定値は一致する）。多値項目のカテゴリは 0 始まり（mirt の
`Science` のように 1 始まりのデータは 1 を引く）。

```python
theta3 = rng.normal(0, 1, 2000)
a3 = np.array([1.6, 1.8, 1.7, 1.6, 1.9, 1.7, 1.8, 1.6, 1.9, 1.7])
d3 = np.array([-1.0, -0.7, -1.2, -0.8, -0.9, -0.6, -1.1, -0.8, -1.0, -0.7])
g3 = np.full(len(a3), 0.2)
p3 = g3 + (1.0 - g3) / (1.0 + np.exp(-(np.outer(theta3, a3) + d3)))
df_3pl = pd.DataFrame(
    (rng.random(p3.shape) < p3).astype(float), columns=[f"c{j}" for j in range(1, len(a3) + 1)]
)
res_3pl = fit(Model.uniform(ThreePL(), items=df_3pl.columns), df_3pl)
print(res_3pl.items[["item", "a", "d", "g", "b"]].round(3))
```

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

```python
gpcm = fit(Model.uniform(GPCM(n_cat=n_cat), items=grm_df.columns), grm_df)
print(gpcm.items[["item", "a", "d1", "d2", "d3", "b1", "b2", "b3"]].round(3))
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
    method=BAEM(tol=1e-4, max_iter=2000, m_step_newton_steps=3, squarem=False),  # 既定値
    quadrature=Quadrature.equispaced(61, lim=(-6, 6)),  # 既定値
    se="oakes",  # 既定値
)
```

- **`method`**: EM の設定。収束は「採択された連続する反復間での自由パラメタの最大絶対変化 < `tol`」
  （mirt の `TOL` と同じ定義）。`max_iter` に達すると `ConvergenceWarning`。M-step は Newton 法を
  `m_step_newton_steps` 回だけ進める（毎回の M-step で期待対数尤度が減らないので、EM の単調性は保たれる）。
- **`squarem=True`**: SQUAREM 加速（スキーム S3; Varadhan & Roland, 2008）を有効にする。1 回の SQUAREM 反復で
  EM 写像を 2 回評価してステップ方向に外挿し、安定化の EM ステップを 1 回行う（採択 1 回あたり EM 評価 3 回。
  単調性・実行可能性の安全装置が $\theta_2$ に戻した場合は 2 回）。収束の遅いモデル（分散を推定する Rasch / PCM、
  多母集団、3PL / 4PL、厳しい `tol`）で有効にする。同じ不動点に収束し、`history["loglik"]` は単調で、
  EM の評価回数が大きく減る。M3 では mirt との実行時間のベンチマークが済むまで、通常の EM（`squarem=False`）を既定とする。
- **`quadrature`**: 潜在変数の求積格子。既定は等間隔 61 点・$[-6, 6]$（mirt の既定と同じ）。
  広げるべき場合は [13 節](#13-求積格子の選び方)。
- **`se`**: `"oakes"`（既定）、`"hessian"`、`None`（計算しない）。

```python
res_sq = fit(spec, df, method=BAEM(tol=1e-6, squarem=True), se=None)
print(res_sq.n_iter, res_sq.n_em_evaluations, res_sq.n_safeguards)
print(res_sq.history.tail(3))  # iter, loglik, max_change, em_evaluations, safeguard
```

初回の推定には JAX のコンパイル時間がかかる。同じ形のデータを続けて推定すると 2 回目以降は速い。

## 5. 結果を読む

`fit()` は `FitResult` を返す。`res.summary()` で読みやすいテキスト要約が得られる（Jupyter
ノートブックでは `res` を評価すると `_repr_html_()` により同等の HTML 要約が表示される）。

```python
print(res)  # FitResult(loglik=..., n_params=..., n_obs=..., converged in ... iterations, se=oakes)
print(res.summary())  # モデル・適合度・潜在分布・項目表の読みやすい要約
print(res.items.head(3))  # 項目ごとの推定値と SE（b とデルタ法 b_se を含む）
print(res.latent)  # 群ごとの潜在分布（mean, var と SE）
print(res.loglik, res.aic, res.bic, res.n_params, res.n_obs)
print(res.converged, res.n_iter, res.n_em_evaluations, res.n_safeguards)
print(res.history.tail(3))  # iter, loglik, max_change, em_evaluations, safeguard
```

| 属性・メソッド | 内容 |
|---|---|
| `summary(digits=4)` | モデル・データ規模・群・収束情報・対数尤度・対数事後確率・AIC/BIC（MAP 時は注意文つき）・潜在分布表・項目表の読みやすい要約。Jupyter では `_repr_html_()` が HTML で表示する |
| `items` | 1 項目 1 行（多母集団では群 × 項目）。`model`、内部パラメタ、`<param>_se`、表示用の `b` / `bk` とデルタ法による `b_se` / `bk_se` |
| `latent` | 1 群 1 行。`group`, `mean`, `var`, `mean_se`, `var_se`（固定値の SE は NaN） |
| `parameters` | ParameterTable（[12 節](#12-parametertable-を編集する)）に `estimate` と `se` を足したもの |
| `loglik`, `log_posterior`, `aic`, `bic` | 周辺対数尤度、対数事後確率（事前分布なしでは `loglik` と同値）、情報量規準（$-2\ell+2k$, $-2\ell+k\log N$。MAP 時は `FlexpyirtWarning`） |
| `n_params`, `n_obs` | 自由パラメタ数、回答者数 |
| `converged`, `n_iter`, `n_em_evaluations`, `n_safeguards` | 収束したか、採択された反復回数、EM 写像の評価回数の合計（通常の EM では `n_iter` と同じ）、SQUAREM の安全装置が働いた回数 |
| `history` | 採択された反復ごとの `iter`, `loglik`, `log_posterior`, `max_change`, `em_evaluations`（累計）, `safeguard`（SQUAREM の安全装置が働いたか） |
| `vcov`, `se_method` | 自由パラメタの共分散行列（`free_id` 順）と計算方法 |
| `to_json(path=None, *, indent=2)` / `FitResult.from_json(source)` | 推定結果を厳格な JSON として保存（`path` 指定時はファイル書き出し）し、再推定なしで復元する |

`history.loglik` は反復ごとに減らない（EM の単調性）。減っていたらバグなので報告してほしい。

```python
assert (res.history["loglik"].diff().dropna() >= -1e-8).all()
```

### 保存と読み込み（JSON）

`to_json()` で推定結果をバージョン付きの素の JSON として保存し、`FitResult.from_json()` で再推定なしに
復元できる。`pickle` や `eval` は使わず、有限な float64 値は `repr` 厳密な JSON 数値、非有限値
（`NaN`, `±inf`）は明示的な文字列として符号化するため、出力は厳格な JSON でありビット単位で正確に往復する。
復元後の結果はすべての属性・`summary()`・`_repr_html_()`・`score()` で完全に一致する。`from_json()` を
呼ぶ前に JAX の 64 ビット浮動小数点を有効にしておく必要がある（無効なら `PrecisionError`）。

```python
import tempfile
from pathlib import Path

json_str = res.to_json()  # 厳格な JSON 文字列を返す（indent=None でコンパクト出力）
loaded_str = FitResult.from_json(json_str)

with tempfile.TemporaryDirectory() as tmp:
    out_path = Path(tmp) / "fit_result.json"
    res.to_json(out_path)  # UTF-8 JSON をファイルに書き出し、文字列も返す
    loaded = FitResult.from_json(out_path)

assert loaded.summary() == res.summary()
np.testing.assert_array_equal(loaded.score(df).to_numpy(), res.score(df).to_numpy())
```

### 項目適合度（S-X2）

`item_fit(data)` は各項目の Orlando と Thissen の S-X2 を計算する。合計得点ごとの各反応カテゴリの観測度数とモデルによる
期待度数を比べ、度数の少ないセルは mirt の `itemfit()` と同じ方法でまとめる（既定 `mincell=1`）。欠測のないデータ、
すべての反応カテゴリが観測されていること、`Normal` 潜在分布の 1 次元モデルが必要。多母集団モデルでは `group=` を
渡すと、（mirt と同じく）群ごとに S-X2 を計算し、先頭に `group` 列がつく。

```python
fit_table = res.item_fit(df)  # item, s_x2, df, rmsea, p (one row per item)
print(fit_table.round(3))
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
`FitResult.items` には、表示用の IRT 閾値（`b = -d/a`、GRM の `bk = -dk/a`、GPCM/PCM のステップ困難度
`bk = -(dk - d{k-1})/a`）に対するデルタ法標準誤差（`b_se` / `bk_se`）も含まれる。ヤコビアンは JAX の自動微分で
`vcov` から計算し、固定パラメタは分散に寄与しない（`a = 1` 固定の Rasch / PCM では `b_se = d_se` /
`b1_se = d1_se`）。`se=None` のとき、または閾値に入るパラメタがすべて固定のときは NaN になる。

## 7. スコアリング

`FitResult.score()` は 5 つの手法で回答者ごとの潜在能力 $\theta$ を推定する。

| `method` | 内容 | 極端パターン（全最低・全最高） | 全項目欠測の行 | 使い分け |
|---|---|---|---|---|
| `"EAP"`（既定） | 推定時の求積格子上の事後平均と事後 SD（Bock & Mislevy, 1982） | 有限 | 事前分布の平均・SD | 高速で個人スコアの MSE が小さい。標準の既定値 |
| `"MAP"` | 連続正規事前分布 $N(\mu_g,\sigma_g^2)$ による事後最頻値と観測曲率 SE | 有限 | 事前分布の最頻値 $\mu_g$ と SD $\sigma_g$ | 格子離散化を避けたベイズ最頻値推定 |
| `"ML"` | 最尤推定と観測情報量 SE | $-\infty$ / $+\infty$、SE は NaN | NaN、SE は NaN | 事前分布による縮小を避けて個人の反応のみで評価したいとき |
| `"WLE"` | 重み付き最尤推定 $\partial\log L/\partial\theta + I'(\theta)/(2I(\theta))=0$ と観測曲率 SE（Warm, 1989; Firth, 1993） | 有限 | NaN、SE は NaN | 事前分布なしで有限テストのバイアスを抑え、極端パターンも有限にしたいとき |
| `"PV"` | 事後分布から回答者ごとに独立に `n_draws` 個を抽出（Mislevy, 1991）、列 `pv1..pvM` | 有限 | 事前分布から抽出 | 測定誤差を伝播させる母集団統計・回帰などの二次解析 |

- ベイズ手法（`"EAP"`, `"MAP"`, `"PV"`）は回答者の群の推定済み潜在分布を事前分布に使い、`"ML"` と
  `"WLE"` は項目尤度のみを使う。
- どの手法でも欠測反応は尤度（`"WLE"` ではテスト情報量 $I(\theta)$ も）から除かれる。3PL / 4PL では、
  当て推量により尤度が $-\infty$ へ向かって単調増加する低得点パターンも `"ML"` で $-\infty$・SE NaN となる。
- **`method="PV"` は明示的な JAX PRNG の `key`（例: `jax.random.key(0)`）が必須**: flexpyirt は
  グローバルな乱数状態を持たないため、同じ `key` で完全に再現可能であり、同じ反応パターンの回答者にも互いに
  独立な標本を与える。

```python
scores_eap = res.score(df)  # 既定: method="EAP" -> 列 theta, se
scores_map = res.score(df, method="MAP")
scores_ml = res.score(df, method="ML")
scores_wle = res.score(df, method="WLE")
print(scores_eap.head(3))  # theta, se（index は df と同じ）
print(scores_wle.head(3))

# Plausible values は明示的な JAX PRNG の key が必須（key に対して決定的）
key = jax.random.key(0)
pvs = res.score(df.iloc[:5], method="PV", key=key, n_draws=5)
print(pvs)  # 列 pv1..pv5

new = df.iloc[:2].copy()
new.iloc[0, :3] = np.nan  # 一部欠測でも採点できる
print(res.score(new, method="MAP"))
```

採点に使うデータは、推定に使ったデータと別でもよい（列は同じ項目）。

## 8. 潜在分布と識別性

潜在分布は正規分布 `Normal` で、格子上で密度を正規化した重みを使う。

```python
Normal.standard()  # N(0, 1)、何も推定しない
Normal(free_var=True)  # N(0, σ²)、分散を推定（平均 0 固定）
Normal.free()  # 平均と分散を推定
Normal(mean=0.5, var=1.2)  # 固定値
```

**既定**: 単一群のモデルで潜在分布を指定しないとき、全項目が Rasch または PCM なら $N(0,\sigma^2)$
（分散を推定）、それ以外は $N(0,1)$ 固定。Rasch と PCM は傾きが 1 に固定されているので、尺度は分散が担う
（mirt と同じ）。

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

### 経験ヒストグラム（EH）

`EmpiricalHistogram()` は正規分布の代わりに求積節点上の自由な重みを使う（Woods, 2007）。歪んだ分布や多峰の分布に
向く。重みは毎 EM 反復で推定し直し、パラメタとして数える（`n_params`、AIC、BIC）。

```python
from flexpyirt import EmpiricalHistogram

eh = fit(Model.uniform(TwoPL(), items=items).with_latent(EmpiricalHistogram()), df)
print(eh.latent)  # group, mean, var, sd, skewness, kurtosis (excess)
print(eh.latent_weights.head())  # group, node, weight
```

- **識別**: 固定項目が尺度を決めない限り（FIPC 以外）、参照群のヒストグラムは毎反復で平均 0・分散 1 に制約する
  （Rasch / PCM は平均のみ）。FIPC ではヒストグラムは自由。多母集団では全群が EH を使い、焦点群はアンカー項目でつなぐ。
- `min_weight`（既定 `1e-10`）は各節点の重みの下限で、空の節点でも対数重みが有限になる。
- 標準誤差は、ヒストグラムを推定値に固定したときの項目パラメタの標準誤差（mirt と同じ）。
- 採点: EAP・ML・WLE・PV はヒストグラムを使う。MAP は連続な事前分布が必要なので `SpecificationError`。
- ヒストグラムの重みの多くは弱くしか決まらないため、厳しい収束基準では EM の収束が遅い。`BAEM(squarem=True)` が有効。

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

項目バンクが IRT 表記（$a, b$）なら `parameterization="irt"` を指定する（$d=-ab$ に変換。GRM は `b1`, `b2`, …
列を $d_k=-ab_k$、GPCM / PCM はステップ困難度 `b1`, `b2`, … を $d_k=-a\sum_{c=1}^{k}b_c$ で変換）。Rasch や
PCM 項目のバンクには `a` 列がなくてよい。

```python
bank_irt = pd.DataFrame({"item": items[:6], "a": a_true[:6], "b": -d_true[:6] / a_true[:6]})
fipc_spec_irt = (
    Model.uniform(TwoPL(), items=items)
    .fix_from(bank_irt, parameterization="irt")
    .with_latent(Normal.free())
)
```

## 11. 事前分布と MAP 推定

`Model.prior(*, param, dist, items=None)` で項目パラメタに事前分布を付けると、Bayes modal（MAP）推定を行う
（Mislevy, 1986; Swaminathan & Gifford, 1986）。自然尺度上の 3 種類の事前分布を用意している。

| クラス | 台（サポート） | 主な用途 |
|---|---|---|
| `NormalPrior(mean, sd)` | $(-\infty, \infty)$ | 切片 `d` / `dk`、または傾き `a` |
| `LogNormal(meanlog, sdlog)` | $(0, \infty)$ | 傾き `a`（$\log a \sim N(\mu, s^2)$） |
| `Beta(a, b)` | $(0, 1)$ | 3PL / 4PL の漸近線 `g`・`u` |

事前分布は**ヤコビアンなしの自然パラメタ尺度**で直接評価する（R mirt の対数事前目的関数 `LL.Priors` と一致）。
多母集団モデルでは `.prior()` が全群に適用され、群間で共有されるパラメタ（同じ `free_id`）の事前分布は **1 回だけ**
数えられる。固定されたパラメタの事前分布は無視される。潜在分布パラメタ（`"mean"`, `"var"`）への事前分布は対象外
（`SpecificationError`）。

```python
map_spec = (
    Model.uniform(ThreePL(), items=df_3pl.columns)
    .prior(param="a", dist=LogNormal(0.2, 0.3))
    .prior(param="d", dist=NormalPrior(0.0, 2.0))
    .prior(param="g", dist=Beta(5, 17))
)
map_res = fit(map_spec, df_3pl)
print(map_res.loglik, map_res.log_posterior)
print(map_res.items[["item", "a", "d", "g", "g_se"]].head(3).round(3))
assert (map_res.history["log_posterior"].diff().dropna() >= -1e-8).all()
```

- `res.loglik` は周辺対数尤度のみのままで、`res.log_posterior` が $\ell(\hat\eta) + \log\pi(\hat\eta)$ を返す
  （事前分布がないときは `loglik` と同値）。
- 通常の EM・SQUAREM とも `history["log_posterior"]` が単調非減少になる。
- 標準誤差（`se="oakes"`・`se="hessian"`）はどちらも観測情報行列に事前分布の負の Hessian
  $-\nabla^2_\eta\log\pi(\hat\eta)$ を加え、互いに一致する。
- 事前分布が有効なとき、`res.aic` と `res.bic` は MLE ではなく MAP 推定値での尤度を使うため `FlexpyirtWarning` を出す。

## 12. ParameterTable を編集する

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
| `prior` | 事前分布オブジェクト（`NormalPrior`, `LogNormal`, `Beta`, または `None`。固定なら無視） |
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

## 13. 求積格子の選び方

潜在分布の重みは格子 $[\text{lower}, \text{upper}]$ の上で正規化するので、格子の端で分布が切れるとその分だけ
推定値が影響を受ける。既定の 61 点・$[-6, 6]$ は $N(0,1)$ 付近では十分だが、**推定する群の分散が大きい
（2 を超える）か平均が 0 から離れる**ときは、格子を広げて刻みを保つ。

```python
wide = Quadrature.equispaced(81, lim=(-8, 8))  # 刻み 0.2 のまま範囲を広げる
```

報告される `var` は格子で切る前の正規分布のパラメタで、格子上の分布の分散はそれよりわずかに小さい。
mirt は分散を閉形式のモーメントで更新するため、格子の端の影響が大きい条件では mirt と値がずれる。格子を
広げると両者は一致する（[VALIDATION.md](VALIDATION.ja.md)）。

**Gauss–Hermite.** `Normal` 潜在分布では、`Quadrature.gauss_hermite(n)` が次元あたり `n` 個の Gauss–Hermite 節点を置き、
推定した平均・分散に合わせて重みを付け直す。$N(0,1)$ 付近では同じ点数の等間隔格子より精度が高いことが多いが、分散が
1 より十分大きいと精度が落ちる。`EmpiricalHistogram` には等間隔格子が必要。

```python
gh = Quadrature.gauss_hermite(41)
res_gh = fit(spec, df, quadrature=gh)
print(round(res_gh.loglik - res.loglik, 6))  # differs from the 61-point equispaced fit by < 1e-4
```

## 14. 確認的多次元モデル

`.with_factors({...})` で 2 または 3 次元の確認的モデルを指定できる。辞書の各項目で因子名と負荷する項目を指定する。
1 項目が複数因子に負荷すること（交差負荷）もでき、負荷しない因子の傾きは `0` に固定される。項目パラメタ名は
mirt と同じ傾き・切片形式 $z = \mathbf a^\top\boldsymbol\theta + d$ の `a1`…`aD`, `d`（GRM / GPCM / PCM は `d1`…）
になる。

潜在分布は多変量正規分布で、平均は `0` 固定、分散は `1` 固定（全項目が `Rasch` / `PCM` のときは分散も自由）、
因子間相関は既定で自由推定（mirt の `COV = F1*F2` 相当。直交因子にするには `correlated=False`）となる。
直積求積の既定は mirt の `select_quadpts` に合わせた 2 因子で各軸 31 点、3 因子で各軸 15 点（区間 $[-6, 6]$）。

```python
a_mirt = np.array(
    [
        [1.3, 0.0],
        [1.1, 0.0],
        [1.4, 0.0],
        [1.0, 0.0],
        [1.2, 0.0],
        [0.7, 0.8],  # 両因子に交差負荷する項目
        [0.0, 1.2],
        [0.0, 1.5],
        [0.0, 1.0],
        [0.0, 1.3],
    ]
)
d_mirt = np.linspace(-0.8, 0.8, a_mirt.shape[0])
theta_mirt = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.5], [0.5, 1.0]], size=1200)
p_mirt = 1.0 / (1.0 + np.exp(-(theta_mirt @ a_mirt.T + d_mirt)))
mirt_items = [f"m{j:02d}" for j in range(1, a_mirt.shape[0] + 1)]
mirt_df = pd.DataFrame((rng.random(p_mirt.shape) < p_mirt).astype(float), columns=mirt_items)

mirt_spec = Model.uniform(TwoPL(), items=mirt_items).with_factors(
    {"F1": mirt_items[:6], "F2": mirt_items[5:]}
)
res_mirt = fit(
    mirt_spec,
    mirt_df,
    method=BAEM(squarem=True),
    quadrature=Quadrature.equispaced(21, lim=(-6, 6)),
)
print(res_mirt.latent[["mean_1", "mean_2", "var_1", "var_2", "cor_12", "cor_12_se"]].round(3))
print(res_mirt.items[["item", "a1", "a2", "d", "a1_se", "a2_se"]].round(3))
print(res_mirt.score(mirt_df.iloc[:3], method="EAP").round(3))  # theta_F1, theta_F2, se_F1, se_F2
```

- **識別性と符号の向き**: 確認的モデルでは回転の不定性を除くために 0 固定（または既知値固定）の負荷が必要
  （全項目が全因子に自由負荷するような指定は `IdentificationError`）。各因子の符号の向きは、因子別の項目−残余相関から
  計算する厳密に正の初期傾きによって正の方向へ揃える。
- **スコアリング**: `"EAP"`・`"MAP"`（推定した多変量正規を事前分布とする）・`"ML"`・`"WLE"` が宣言した因子ごとの
  `theta_<factor>`, `se_<factor>` 列を返す。ML は推定値が発散する次元で $\pm\infty$ を返す。`"PV"` は `pv<m>_<factor>` 列を返す。
  多次元の多母集団モデルは未対応で `SpecificationError` を送出する。

## 15. エラーと警告

| 種類 | いつ出るか |
|---|---|
| `PrecisionError` | JAX の 64 ビット浮動小数点が無効なまま `fit()` を呼んだ |
| `SpecificationError` | 仕様とデータの矛盾（列の過不足、範囲外の値、観測されないカテゴリ、存在しない項目の固定、表や事前分布の不正な編集、群ラベルの不一致） |
| `IdentificationError` | 尺度が識別できない（[8 節](#8-潜在分布と識別性)） |
| `ConvergenceWarning` | `max_iter` までに収束しなかった、または SQUAREM が反復の 50% 超（5 回以上）で通常の EM に戻った |
| `HeywoodWarning` | 自由な項目パラメタが有限の境界から `1e-4` 以内（3PL / 4PL で `g` が 0 付近、`u` が 1 付近など）、または自由な傾きが $|a| > 20$ |
| `FlexpyirtWarning` | 情報行列が正定値でない（SE が NaN）、アンカーが 1 パラメタだけ、または MAP 推定で `aic` / `bic` を参照した |

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
- Bock, R. D., & Mislevy, R. J. (1982). Adaptive EAP estimation of ability in a microcomputer environment. *Applied Psychological Measurement*, 6(4), 431–444.
- Chalmers, R. P. (2012). mirt: A multidimensional item response theory package for the R environment. *Journal of Statistical Software*, 48(6).
- Firth, D. (1993). Bias reduction of maximum likelihood estimates. *Biometrika*, 80(1), 27–38.
- Kim, S. (2006). A comparative study of IRT fixed parameter calibration methods. *Journal of Educational Measurement*, 43, 355–381.
- Masters, G. N. (1982). A Rasch model for partial credit scoring. *Psychometrika*, 47, 149–174.
- Mislevy, R. J. (1986). Bayes modal estimation in item response models. *Psychometrika*, 51, 177–195.
- Mislevy, R. J. (1991). Randomization-based inference about latent variables from complex samples. *Psychometrika*, 56(2), 177–196.
- Muraki, E. (1992). A generalized partial credit model: Application of an EM algorithm. *Applied Psychological Measurement*, 16, 159–176.
- Oakes, D. (1999). Direct calculation of the information matrix via the EM algorithm. *JRSS B*, 61, 479–482.
- Samejima, F. (1969). Estimation of latent ability using a response pattern of graded scores. *Psychometrika Monograph*, 17.
- Swaminathan, H., & Gifford, J. A. (1986). Bayesian estimation in the three-parameter logistic model. *Psychometrika*, 51, 589–601.
- Varadhan, R., & Roland, C. (2008). Simple and globally convergent methods for accelerating the convergence of any EM algorithm. *Scand. J. Statist.*, 35, 335–353.
- Warm, T. A. (1989). Weighted likelihood estimation of ability in item response theory. *Psychometrika*, 54(3), 427–450.
