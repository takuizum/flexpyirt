# flexpyirt

JAX による Bock–Aitkin EM の項目反応理論（IRT）推定ライブラリ — **設計段階**。

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
M0（足場）: パッケージ骨格・lint/型/テスト設定・CI 定義を追加済み（CI 未実行）。推定機能はまだありません。

## 開発
```bash
uv sync --all-extras --dev
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
```
