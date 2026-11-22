# パラメタ表の編集

mirt の `pars = "values"` に相当する上級者向けの機能です。仕様から既定のパラメタ表を作り、
直接編集して `fit()` に渡せます。

```python
table = spec.parameter_table(df)
table.head()
#   group item model param     value  free  lower  upper  free_id
# 0   all  q01   2PL     a  0.495881  True   -inf    inf        0
# 1   all  q01   2PL     d -0.867770  True   -inf    inf        1
# ...

mask = (table["item"] == "q05") & (table["param"] == "a")
table.loc[mask, ["value", "free"]] = [1.0, False]

res = fit(spec, df, table=table)  # 表が spec より優先
```

| 列 | 意味 |
|---|---|
| `value` | 初期値（`free=False` なら固定値） |
| `free` | 推定するか |
| `lower`, `upper` | 箱制約 |
| `free_id` | 同じ値を持つ行は等値制約（群間等値はこの仕組みで表現） |

表の誤り（未知の項目、範囲外の初期値など）は推定の前に `SpecificationError` になります。
