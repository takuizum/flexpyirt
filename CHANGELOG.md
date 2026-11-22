# 変更履歴

このプロジェクトは [Semantic Versioning](https://semver.org/lang/ja/) に従います（1.0 までは
マイナーバージョンでも互換性のない変更がありえます）。書式は
[Keep a Changelog](https://keepachangelog.com/ja/1.1.0/) に準じます。

## [Unreleased]

## [0.0.1] - 未定

最初の公開リリース。以下はすべて R `mirt` との数値照合テスト（V1–V5, V9, V10）が通っています。

### 追加
- 単一群の 2PL・Rasch（Rasch は潜在分散を自由推定）の Bock–Aitkin EM による周辺最尤推定。
- GRM（段階反応モデル）。
- 多母集団の同時推定（アンカー項目の群間等値、焦点群の平均・分散を自由推定）と識別性チェック。
- 固定項目パラメタ推定（FIPC、`Model.fix_from`）。
- 欠測（MAR）を含むデータ。
- 標準誤差: Oakes の恒等式、周辺対数尤度の Hessian（いずれも自動微分）。
- EAP スコア。
- パラメタ表（`Model.parameter_table`）の直接編集。

[Unreleased]: https://github.com/takuizum/flexpyirt/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/takuizum/flexpyirt/releases/tag/v0.0.1
