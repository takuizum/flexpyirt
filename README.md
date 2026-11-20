# flexpyirt

JAX による Bock–Aitkin EM の項目反応理論（IRT）推定ライブラリ — **開発中**。

目標とする機能（未実装のものを含む）:

- 多母集団の同時推定（部分アンカー）
- 固定パラメタ推定（FIPC）
- 潜在分布の指定（Normal / Empirical Histogram）
- 事前分布つきの Bayes modal（MAP）推定
- Oakes の恒等式による標準誤差（自動微分）

すべての機能は R `mirt` との数値照合テストが通ってから「対応済み」とする。

## ドキュメント
- [設計書](docs/DESIGN.md)
- [API スケッチ](docs/API.md)
- [検証計画](docs/VALIDATION.md)
- [ロードマップ](docs/ROADMAP.md)
- [Claude Code への引き継ぎ](CLAUDE.md)

## ステータス
| マイルストーン | 内容 | 状態 |
|---|---|---|
| M0 | 足場（パッケージ・CI） | 完了 |
| M1 | 単一群 2PL / Rasch の BA-EM、Oakes / Hessian SE、EAP | R mirt 照合（V1, V2, V10）済み |

M1 で使えるもの（すべて `tests/test_mirt_m1.py` で mirt と照合）:
- 2PL・Rasch（Rasch は潜在分散を自由推定）の周辺最尤推定
- 標準誤差（Oakes の恒等式／周辺対数尤度の Hessian、いずれも自動微分）
- EAP スコア

```python
import jax
jax.config.update("jax_enable_x64", True)
from flexpyirt import Model, Rasch, fit

res = fit(Model.uniform(Rasch(), items=df.columns), df)
res.items, res.latent, res.loglik
res.score(df)
```

## 開発
```bash
uv sync --all-extras --dev
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
```
