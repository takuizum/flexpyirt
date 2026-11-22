# はじめに

## インストール

Python 3.11 以上が必要です。

=== "uv"

    ```bash
    uv add flexpyirt
    ```

=== "pip"

    ```bash
    pip install flexpyirt
    ```

polars の DataFrame を入力に使う場合は `flexpyirt[polars]` を指定してください。

GPU で動かす場合は、先に [JAX のインストール手順](https://docs.jax.dev/en/latest/installation.html)
に従って CUDA 版の `jax` を入れてください。

## float64 を有効にする

flexpyirt は 64 bit 浮動小数点を前提にしています。JAX の既定は float32 なので、**JAX で配列を作る前に**
次を実行してください。

```python
import jax

jax.config.update("jax_enable_x64", True)
```

環境変数 `JAX_ENABLE_X64=1` でもかまいません。flexpyirt はユーザーのグローバル設定を書き換えません。
無効のまま `fit()` を呼ぶと `flexpyirt.errors.PrecisionError` になります。

## 最初の推定

```python
import numpy as np
import pandas as pd

from flexpyirt import Model, TwoPL, fit

# 例: 2PL に従う 1000 人 × 10 項目のシミュレーションデータ
rng = np.random.default_rng(1)
theta = rng.normal(size=1000)
a = rng.uniform(0.8, 2.0, 10)
d = rng.normal(size=10)
p = 1 / (1 + np.exp(-(theta[:, None] * a + d)))
df = pd.DataFrame(
    (rng.uniform(size=p.shape) < p).astype(float),
    columns=[f"q{j:02d}" for j in range(1, 11)],
)

spec = Model.uniform(TwoPL(), items=df.columns)
res = fit(spec, df)
res
# FitResult(loglik=-5545.6945, n_params=20, n_obs=1000, converged in 22 iterations, se=oakes)
```

### 結果を見る

```python
res.items  # item, model, a, d, a_se, d_se, b (= -d/a)
res.latent  # group, mean, var, mean_se, var_se
res.loglik, res.aic, res.bic
res.converged, res.n_iter
res.history  # iter, loglik, max_change
res.parameters  # 長形式のパラメタ表 + estimate, se
res.vcov  # 自由パラメタの共分散行列
```

項目パラメタは内部的に**傾き–切片**（$P(y=1\mid\theta)=\sigma(a\theta+d)$）で推定し、
`b` 列は表示用に $b=-d/a$ を計算したものです（`b` の SE は M4 で追加予定）。

### 能力値の推定

```python
res.score(df)  # theta（EAP）, se（事後 SD）
```

## データの形式

- 行が受検者、列が項目。pandas / polars の DataFrame、または numpy 配列（`item_names=` で列名を指定）。
- 二値項目は 0/1、多値項目（GRM）は $0,\dots,K-1$。
- 欠測は `NaN`（MAR を仮定）。
