# ロードマップ

[English](ROADMAP.md) | 日本語

各マイルストーンは「受け入れ基準」をすべて満たして完了とする。順番を飛ばさない。

## M0 足場
- `uv` プロジェクト、`src/flexpyirt/` レイアウト、ruff / mypy / pytest の設定、GitHub Actions（Linux, Python 3.11–3.13）。
- `flexpyirt.errors`、x64 チェック関数、空の `Model` / `fit` / `FitResult` の型だけ。
- **受け入れ基準**: CI が通る。editable install（`uv run --isolated --no-project --with-editable .`）で import できる。環境管理は uv のみで pip は使わない。

## M1 単一群 2PL / Rasch の BA-EM
- Compile 層: パターン圧縮、ParameterTable、FreeMap（固定・等値の仕組みはここで作る）。
- Core: E-step、項目ごとの Newton M-step（vmap）、`lax.while_loop` の EM ループ、Normal(0,1) 固定。
- SE: Oakes と直接 Hessian。
- スコアリング: EAP。
- **受け入れ基準**: V1, V2, V10(Oakes) が許容誤差内。EM 単調性テストが通る。

## M2 多母集団・FIPC・GRM・欠測
- 群ごとの潜在分布（Normal 自由）、群間等値（`free_id` 共有）、識別性チェック。
- FIPC（固定項目 + 潜在分布推定）。
- GRM。欠測（MAR）。
- **受け入れ基準**: V3, V4, V5, V9 が許容誤差内。識別不能な仕様で `IdentificationError`。

## M3 EH・MAP・3PL/4PL・GPCM・SQUAREM
- [x] EmpiricalHistogram（`EmpiricalHistogram`、モーメント制約付き M-step による参照群の標準化、平方根尺度の SQUAREM。V6 で照合済み）。
- [x] 事前分布と MAP（`NormalPrior`, `LogNormal`, `Beta`, `Model.prior()`, `FitResult.log_posterior`、V7 で照合済み）。
- [x] 3PL / 4PL（`ThreePL`, `FourPL`, `HeywoodWarning`、V11 で照合済み）。
- [x] GPCM / PCM（`GPCM`, `PCM`、V8 で照合済み）。
- [x] SQUAREM（`BAEM(squarem=True)`、スキーム S3 と単調性・実行可能性の安全装置。全 mirt 照合スイートが SQUAREM あり・なしで通過）。
- [x] ベンチマーク（mirt との wall time。`benchmarks/` を参照）。
- **受け入れ基準**: V6, V7, V8, V11 が許容誤差内。SQUAREM ありでも照合が通る。

## M4 結果まわり・多次元の入口
- [x] `FitResult.summary()`、`FitResult._repr_html_()`、デルタ法による表示用 IRT パラメタ化の SE
  （`b_se` / `bk_se`、V12 で照合済み）。
- [x] `FitResult` の JSON 入出力（`FitResult.to_json`, `FitResult.from_json`）。
- [x] スコアリング（MAP, ML, WLE, plausible values。V13 で照合済み）。
- [x] 確認的 2–3 次元 MIRT（`Model.with_factors()`、直積格子、多変量正規の因子間相関、Oakes/Hessian SE、EAP・MAP スコアリング。V14 で照合済み）。
- **受け入れ基準**: mirt の `fscores` と EAP / MAP が一致。2 次元確認的モデルが mirt と一致。

## M5 以降（候補）
- [x] 項目をまたぐ等値制約の M-step（`Model.equal_across_items()`、項目・項目モデルをまたぐ `free_id` の共有、連結成分ごとの同時 Newton。V16 で照合済み）。
- Davidian 曲線・混合正規の潜在分布。
- 微分可能なリンキング（Haebara / Stocking–Lord とその一般化）をライブラリの上に実装。
- [x] 項目適合度: S-X2（`FitResult.item_fit()`、単一群 1 次元。V15 で照合済み）。
- [x] DIF: 多群の尤度比検定（`FitResult.dif()`、`add` / `drop` の 2 方式、Holm / BH 補正。V17 で照合済み）。
- M2（限定情報の適合度）。
