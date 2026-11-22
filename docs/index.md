# flexpyirt

**JAX による Bock–Aitkin EM の項目反応理論（IRT）推定ライブラリ。**

*Bock–Aitkin EM estimation for item response theory models, built on JAX.*

!!! warning "開発中（v0.0.x）"
    API は今後のマイルストーンで変わる可能性があります。このドキュメントに載せているのは、
    R `mirt` との数値照合テストが CI で通っている機能だけです。

## 特徴

- **周辺最尤推定（MML）を Bock–Aitkin EM で。** E-step・M-step・EM ループは `jax.jit` された純粋関数。
- **自動微分による標準誤差。** Oakes の恒等式、または周辺対数尤度の Hessian。
- **多母集団の同時推定。** 群ごとの潜在分布（平均・分散を自由推定）と、アンカー項目の群間等値制約。
- **固定パラメタ推定（FIPC）。** 項目バンクの値で一部の項目を固定し、潜在分布を推定。
- **R `mirt` との数値照合。** 各機能は mirt の参照値と ~1e-8 で一致することをテストで確認しています
  （[検証計画](VALIDATION.md)）。

## 対応状況（v0.0.1）

| 区分 | 対応済み（mirt 照合済み） | 予定 |
|---|---|---|
| 項目モデル | 2PL, Rasch, GRM | 3PL / 4PL, GPCM（M3） |
| 潜在分布 | Normal（固定・平均／分散自由） | Empirical Histogram（M3） |
| 推定 | BA-EM（MML）、多母集団、FIPC、欠測（MAR） | MAP（事前分布）、SQUAREM 加速（M3） |
| 標準誤差 | Oakes, Hessian | crossprod, sandwich |
| スコアリング | EAP | MAP, ML, WLE, plausible values（M4） |

詳しくは [ロードマップ](ROADMAP.md) を参照してください。

## 30 秒で試す

```python
import jax

jax.config.update("jax_enable_x64", True)  # float64 が必須

from flexpyirt import Model, TwoPL, fit

res = fit(Model.uniform(TwoPL(), items=df.columns), df)  # df: 0/1 の DataFrame
res.items  # 項目パラメタと SE
res.score(df)  # EAP
```

次は [はじめに](getting-started.md) へ。
