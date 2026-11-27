# flexpyirt

[English](index.md) | 日本語

JAX による Python の項目反応理論（IRT）推定ライブラリ。Bock–Aitkin EM による周辺最尤推定を中心に、
多母集団の同時推定や固定項目パラメタ推定（FIPC）などを備える。対応済みの機能はすべて R `mirt` と
数値照合している（[精度と検証](accuracy.ja.md)）。

> 開発中（0.x）。1.0.0 までは公開 API が変わることがある。

## インストール

Python 3.12 以上。

```bash
pip install flexpyirt  # または uv add flexpyirt
```

flexpyirt をインストールすると **flexpyctt** も入る。推定済みのモデルを使わない分析（ポリコリック相関、次元数、
DETECT、DIMTEST、古典的項目分析、観測得点による DIF）のための、対になるパッケージである。依存は NumPy、SciPy、pandas だけなので、JAX を入れたくない
環境では `pip install flexpyctt` で単独でインストールできる。

別のパッケージ [flexpyequate](https://github.com/takuizum/flexpyequate) は、得点分布の等化（線形等化、
等パーセンタイル等化、カーネル等化）を行う。flexpyirt とは独立で、flexpyirt をインストールしても入らない
（`pip install flexpyequate`）。

polars の DataFrame を使う場合は `flexpyirt[polars]`、図（`flexpyirt.plotting`）を描く場合は `flexpyirt[plot]`。GPU を使う場合は、先に CUDA 版の `jax` を入れる
（[JAX のインストール手順](https://docs.jax.dev/en/latest/installation.html)）。

flexpyirt は **64 bit 浮動小数点**で計算する。JAX の設定はユーザーのものなのでライブラリは変更しない。
推定の前に有効にすること（無効のままだと `fit()` が `PrecisionError` を出す）。

```python
import jax

jax.config.update("jax_enable_x64", True)
```

## 最初の推定

```python
import numpy as np
import pandas as pd

from flexpyirt import Model, TwoPL, fit

rng = np.random.default_rng(1)
a, d = rng.lognormal(0.0, 0.3, 10), rng.normal(0.0, 1.0, 10)
theta = rng.normal(0.0, 1.0, 1000)
p = 1.0 / (1.0 + np.exp(-(np.outer(theta, a) + d)))
df = pd.DataFrame((rng.random(p.shape) < p).astype(int), columns=[f"q{j:02d}" for j in range(10)])

res = fit(Model.uniform(TwoPL(), items=df.columns), df)
print(res.items.head())  # item, model, a, d, a_se, d_se, b (= -d/a), b_se
scores = res.score(df)  # 受検者ごとの EAP と事後 SD
```

## 次に読むもの

- [ユーザーガイド](guide.ja.md): モデルの指定、多母集団、FIPC、スコアリング、項目適合度、DIF、
  リンキング、推定オプション。
- [理論](theory.ja.md): flexpyirt の背後にあるモデルとアルゴリズムの要点（IRT モデル、周辺尤度、EM
  アルゴリズム、識別性、標準誤差、スコアリング、適合度、リンキング）。
- [API リファレンス](reference.ja.md): 公開クラス・関数の一覧。
- [精度と検証](accuracy.ja.md): R mirt との照合方法、許容誤差、mirt との既知の違い。
- ソースコード・Issue・リリース: [GitHub](https://github.com/takuizum/flexpyirt)。
