# 単一群（2PL / Rasch）

## 2PL

$$
P(y_{ij}=1\mid\theta_i)=\sigma(a_j\theta_i+d_j),\qquad \theta_i\sim N(0,1)
$$

```python
from flexpyirt import Model, TwoPL, fit

res = fit(Model.uniform(TwoPL(), items=df.columns), df)
res.items
```

単一群の 2PL では潜在分布を $N(0,1)$ に固定して尺度を識別します。

## Rasch

$$
P(y_{ij}=1\mid\theta_i)=\sigma(\theta_i+d_j),\qquad \theta_i\sim N(0,\sigma^2)
$$

傾きは 1 に固定され、代わりに潜在分散 $\sigma^2$ を推定します（mirt の `itemtype="Rasch"` と同じ）。

```python
from flexpyirt import Rasch

res = fit(Model.uniform(Rasch(), items=df.columns), df)
res.latent  # var と var_se が推定される
```

## 項目ごとに違うモデル

```python
spec = Model.from_items({"q01": Rasch(), "q02": TwoPL(), "q03": TwoPL()})
```

## 推定オプション

```python
from flexpyirt import BAEM, Quadrature

res = fit(
    spec,
    df,
    method=BAEM(tol=1e-4, max_iter=2000, m_step_newton_steps=3),
    quadrature=Quadrature.equispaced(61, lim=(-6, 6)),  # 既定（mirt と同じ）
    se="oakes",  # "oakes" | "hessian" | None
)
```

- `tol` は 2 回の EM 反復間での自由パラメタの最大絶対変化（mirt の `TOL` と同じ定義）。
- 収束しなかった場合は `flexpyirt.errors.ConvergenceWarning` が出ます。`res.converged` でも確認できます。
- `se=None` にすると標準誤差の計算を省きます。
