# GRM と欠測

## 段階反応モデル（GRM）

Samejima (1969) の段階反応モデル。カテゴリは $0,\dots,K-1$ で符号化します。

$$
P(y\ge k\mid\theta)=\sigma(a\theta+d_k),\qquad d_1>\dots>d_{K-1}
$$

```python
from flexpyirt import GRM, Model, fit

res = fit(Model.uniform(GRM(n_cat=4), items=df.columns), df)
res.items  # a, d1, d2, d3, 各 SE, b1..b3 (= -d_k / a)
```

mirt の `itemtype="graded"` と同じパラメタ化です。二値項目と混ぜる場合は `Model.from_items` を使います。

## 欠測

データ中の `NaN` は欠測として扱います（MAR を仮定し、尤度から該当項目を除く）。
特別な指定は不要です。

```python
df.iloc[::7, 1] = float("nan")
res = fit(spec, df)
```
