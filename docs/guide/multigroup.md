# 多母集団（部分アンカー）

複数の群（例: 参照群と焦点群、別フォーム）を同時に推定し、共通項目（アンカー）で尺度をつなぎます。

```python
from flexpyirt import Model, Normal, Quadrature, TwoPL, fit

anchors = ["q01", "q02", "q03", "q04", "q05"]

spec = (
    Model.uniform(TwoPL(), items=df.columns)
    .with_groups(ref=Normal.standard(), focal=Normal.free())  # 最初の群が参照群
    .equal_across_groups(items=anchors)  # アンカー項目は全パラメタ群間等値
)
res = fit(spec, df, group=labels)  # labels: 各行の群名（"ref" / "focal"）
```

```python
res.latent
#    group      mean       var   mean_se    var_se
# 0    ref  0.000000  1.000000       NaN       NaN
# 1  focal  0.414566  1.436587  0.095342  0.280893

res.items  # group 列つき。アンカー項目は両群で同じ値
res.score(df, group=labels)
```

- `with_groups()` のキーワードの順序が群の順序で、最初が参照群です。
- 一部のパラメタだけを等値にする場合は `equal_across_groups(items=..., params=["a"])`。

## 識別性のチェック

アンカーなしで焦点群の平均・分散を自由にするなど、識別できない仕様は推定の**前**に
`flexpyirt.errors.IdentificationError` になります。

## 格子についての注意

焦点群の平均が 0 から大きく離れる、または分散が大きい場合、既定の格子（61 点・$[-6,6]$）では
打ち切りの影響が出ることがあります。そのときは格子を広げてください。

```python
fit(spec, df, group=labels, quadrature=Quadrature.equispaced(81, lim=(-8, 8)))
```
