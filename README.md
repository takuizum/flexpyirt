# flexpyirt

[![CI](https://github.com/takuizum/flexpyirt/actions/workflows/ci.yml/badge.svg)](https://github.com/takuizum/flexpyirt/actions/workflows/ci.yml)
[![Docs](https://github.com/takuizum/flexpyirt/actions/workflows/docs.yml/badge.svg)](https://takuizum.github.io/flexpyirt/)
[![PyPI](https://img.shields.io/pypi/v/flexpyirt.svg)](https://pypi.org/project/flexpyirt/)
[![Python](https://img.shields.io/pypi/pyversions/flexpyirt.svg)](https://pypi.org/project/flexpyirt/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/takuizum/flexpyirt/blob/main/LICENSE)

JAX による Bock–Aitkin EM の項目反応理論（IRT）推定ライブラリ — **開発中（v0.0.x）**。

*Bock–Aitkin EM estimation for item response theory models, built on JAX. Every feature is
validated numerically against R `mirt` before it is listed as supported.*

**ドキュメント: <https://takuizum.github.io/flexpyirt/>**

## インストール
```bash
pip install flexpyirt  # または uv add flexpyirt
```
Python 3.11 以上。float64 が必須なので、使う前に `jax.config.update("jax_enable_x64", True)` を実行してください。

## 使い方
```python
import jax

jax.config.update("jax_enable_x64", True)
from flexpyirt import Model, Rasch, fit

res = fit(Model.uniform(Rasch(), items=df.columns), df)
res.items, res.latent, res.loglik
res.score(df)
```

## ステータス
| マイルストーン | 内容 | 状態 |
|---|---|---|
| M0 | 足場（パッケージ・CI） | 完了 |
| M1 | 単一群 2PL / Rasch の BA-EM、Oakes / Hessian SE、EAP | R mirt 照合（V1, V2, V10）済み |
| M2 | 多母集団（部分アンカー）、FIPC、GRM、欠測（MAR） | R mirt 照合（V3, V4, V5, V9）済み |

M1（すべて [`tests/test_mirt_m1.py`](https://github.com/takuizum/flexpyirt/blob/main/tests/test_mirt_m1.py) で mirt と照合）:
- 2PL・Rasch（Rasch は潜在分散を自由推定）の周辺最尤推定
- 標準誤差（Oakes の恒等式／周辺対数尤度の Hessian、いずれも自動微分）
- EAP スコア

M2 で追加（[`tests/test_mirt_m2.py`](https://github.com/takuizum/flexpyirt/blob/main/tests/test_mirt_m2.py) で mirt と照合）:
- 多母集団の同時推定（群間等値のアンカー項目、焦点群の平均・分散を自由推定）
- 固定項目パラメタ推定（FIPC）
- GRM（段階反応モデル）
- 欠測（MAR）を含むデータ

今後の予定（未実装）: Empirical Histogram、事前分布つき MAP 推定、3PL / 4PL / GPCM、SQUAREM 加速（M3）、
MAP / ML / WLE スコア（M4）。[ロードマップ](https://github.com/takuizum/flexpyirt/blob/main/docs/ROADMAP.md) を参照。

## 設計ドキュメント
- [設計書](https://github.com/takuizum/flexpyirt/blob/main/docs/DESIGN.md)
- [API 設計案](https://github.com/takuizum/flexpyirt/blob/main/docs/API.md)
- [検証計画](https://github.com/takuizum/flexpyirt/blob/main/docs/VALIDATION.md)
- [変更履歴](https://github.com/takuizum/flexpyirt/blob/main/CHANGELOG.md)

## 開発
```bash
uv sync --all-extras --dev
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
uv run --group docs mkdocs serve  # ドキュメントのプレビュー
```
リリース手順は [docs/dev/releasing.md](https://github.com/takuizum/flexpyirt/blob/main/docs/dev/releasing.md)。

## ライセンス
MIT
