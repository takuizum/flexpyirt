# 固定パラメタ推定（FIPC）

既知の項目パラメタ（項目バンク）で一部の項目を固定し、残りの項目と潜在分布の平均・分散を推定します。
新しいフォームの項目を既存の尺度に載せる（オンライン・キャリブレーション）ときに使います。

```python
import pandas as pd

from flexpyirt import Model, Normal, TwoPL, fit

bank = pd.DataFrame({"item": ["q01", "q02", "q03"], "a": [1.2, 0.9, 1.5], "b": [-0.3, 0.4, 1.1]})

spec = (
    Model.uniform(TwoPL(), items=df.columns)
    .fix_from(bank, parameterization="irt")  # a, b で与える（d = -a b に変換）
    .with_latent(Normal.free())  # 潜在分布の平均・分散を推定
)
res = fit(spec, df)
res.latent
```

- `parameterization="slope-intercept"`（既定）なら `a`, `d`（GRM は `d1, d2, ...`）の列で与えます。
- `items=` で固定する項目を絞れます（既定はバンクにあり、かつモデルにある全項目）。
- 個別に固定する場合は `.fix(item="q01", a=1.2, d=0.3)`（傾き–切片）。
