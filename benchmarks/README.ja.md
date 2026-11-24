# ベンチマーク（flexpyirt vs. R `mirt`）

[English](README.md) | 日本語

同一の決定的シミュレーションデータを用いて、**flexpyirt**（CPU 上の JAX）と R **`mirt`** の
wall time を比較するスイート。条件は `TOL = 1e-4`、`[-6, 6]` 上の等間隔 61 点求積。

最新のコミット済み結果: [results/2026-11-21-apple-m5.md](results/2026-11-21-apple-m5.md)
（[JSON](results/2026-11-21-apple-m5.json)）。他の処理を動かしていない状態で計測した。

## 実行方法

Python 環境は `uv` のみを使う（`pip` は使わない）。R 側の実行には R 4.4 以上と
`mirt`・`jsonlite` パッケージが必要。

### フル実行（Apple M5 で約 4 分、一般的なノート PC でも約 15 分以内）

```bash
uv run python benchmarks/data.py
uv run python benchmarks/run_flexpyirt.py
Rscript benchmarks/run_mirt.R
uv run python benchmarks/report.py
```

### クイックモード（1 分未満）

小さい部分集合（`2pl_n1000_j20` と `grm4_n2000_j20`）を warm 反復 1 回で実行する:

```bash
uv run python benchmarks/data.py --quick
uv run python benchmarks/run_flexpyirt.py --quick
Rscript benchmarks/run_mirt.R --quick
uv run python benchmarks/report.py --stem quick-check
```

R なしで flexpyirt のみ集計する場合は `report.py` に `--no-mirt` を渡す。特定のケースだけ
実行するには `data.py`・`run_flexpyirt.py`・`run_mirt.R` に `--cases <name>...` を指定する。

シミュレーション CSV と各エンジンの中間 JSON は `benchmarks/data/`（git 無視対象）に書かれ、
統合レポートが `benchmarks/results/<date>-<host-short>.{json,md}` に出力される。

## 計測内容

### データセット（`benchmarks/data.py`）

Python と R が完全に同一の反応行列を推定するよう、固定の NumPy シードで生成する:

| ケース | モデル | サイズ `(N, J)` | 備考 |
|---|---|---:|---|
| `2pl_n1000_j20` | 2PL | `(1000, 20)` | 単一群、$N(0, 1)$ |
| `2pl_n5000_j40` | 2PL | `(5000, 40)` | 単一群、$N(0, 1)$ |
| `2pl_n20000_j40` | 2PL | `(20000, 40)` | 単一群、$N(0, 1)$ |
| `grm4_n2000_j20` | GRM（$K=4$） | `(2000, 20)` | 単一群、$N(0, 1)$ |
| `3pl_n5000_j30` | 3PL | `(5000, 30)` | 単一群、内点の漸近線（$g \in [0.12, 0.25]$） |
| `twogroup_2pl_n6000_j30` | 2 群 2PL | `(2 × 3000, 30)` | アンカー 10 項目（`q01`–`q10`）、焦点群 $N(0.5, 1.2)$ 自由 |

### 推定器と計測方法

- **flexpyirt**（`benchmarks/run_flexpyirt.py`）:
  - `BAEM(tol=1e-4, max_iter=2000, squarem=False)` と `BAEM(tol=1e-4, max_iter=2000, squarem=True)`。
  - `se=None` と `se="oakes"`。
  - **cold 時間**: 新規 Python サブプロセス（`JAX_PLATFORMS=cpu`）での初回 `fit()` 呼び出し。
    他設定の JIT キャッシュを再利用せず、JAX/XLA の初回コンパイル時間を含む。
  - **warm 時間**: 同一プロセス内で初回に続けて 3 回実行した中央値。結果がホスト上の
    NumPy 配列として確定した後に `time.perf_counter()` で計測する。
- **R `mirt`**（`benchmarks/run_mirt.R`）:
  - **既定設定**（`TOL = 1e-4`、既定の加速 `"Ramsay"` と既定の最適化器 `"BFGS"` / `"nlminb"`）
    および **検証設定**（`optimizer = "NR"`, `accelerate = "none"`）。
  - `SE = FALSE` と `SE = TRUE, SE.type = "Oakes"`。
  - 同一の求積格子: `quadpts = 61`、`theta_lim = c(-6, 6)` を `technical = list(...)` 内に指定
    （`m@Model$Theta` で実格子を検証）。
  - `system.time()` による 3 回実行の中央値。
- **妥当性チェック**（`benchmarks/report.py`）:
  - 反復回数、flexpyirt の EM 写像評価回数・SQUAREM 安全装置作動回数、収束状態、および
    flexpyirt と `mirt` の周辺対数尤度差 $|\Delta \log L|$ を記録する。

## 注意点

- **コンパイル時間と warm 時間の差**: JAX は `(項目 family, パディング後のパターンバケット, se)` ごとに
  初回呼び出しでコンパイルが走る（CPU 上の cold の EM 推定で約 0.8–1.1 秒、大規模な Oakes SE では
  さらに増える）。
  小規模データの 1 回限りの推定では、コンパイル済み C++ で動く R `mirt` の方が cold 呼び出しより速い。
  ノートブックやシミュレーションのように繰り返し推定するセッションでは、2 回目以降は warm 時間のみとなる。
- **CPU 専用とスレッド数**: `run_flexpyirt.py` は `JAX_PLATFORMS=cpu` を固定する。XLA CPU は内部で
  複数スレッドを使うことがある一方、R `mirt` は `mirtCluster()` を設定しない限り既定で単一スレッドで
  動く。GPU 実行はここでは測っていない（[DESIGN §9](../docs/dev/DESIGN.ja.md#9-性能方針)）。
- **環境依存性**: 絶対時間や比率は CPU アーキテクチャ、メモリ帯域、OS、BLAS/XLA、各ライブラリの
  バージョンに強く依存する。コミット済みの表は 1 環境の計測例として扱い、性能判断には手元の環境で
  再実行すること。
- **大規模 `(N, J)` での Oakes SE**: `se="oakes"` は `jacfwd(grad)` による自動微分で $Q(\eta \mid \eta')$ を
  微分する。`N = 20000, J = 40`（ユニークパターン 19,859、バケットパディング 32,768）では Oakes SE が
  全体時間の大半（約 3.1–3.5 秒、EM 本体は約 0.3–0.4 秒）を占める。

## SQUAREM の既定値について

`BAEM(squarem=False)` の既定値は変更していない。
[2026-11-21 Apple M5 の結果](results/2026-11-21-apple-m5.md) では次のトレードオフが見られた:

- **warm 実行**: `squarem=True` は全 6 ケースで EM 評価回数を 33–69% 減らし、warm 時間を
  全ケースで短縮する（通常 EM 比で `se=None` は `0.38×–0.87×`、全体では `0.38×–0.90×`。2 群 2PL で 2.6 倍、
  3PL で 1.8 倍高速）。
  安全装置の介入は各ケース最大 1 回にとどまった。
- **cold 実行**: `squarem=True` は `se=None` で JIT コンパイル時間が約 0.05–0.4 秒増えるため、1 回限りの
  cold 呼び出しでは 6 ケースすべてで通常 EM の方が速い（2 群 2PL の差は 0.05 秒）。

既定値を切り替えるかどうかは、想定する使い方（1 回限りのスクリプトか、繰り返し実行や収束の遅いモデルか）
による判断になるため、メンテナーの決定に委ねる。
