# 検証計画（R mirt との数値照合）

## 方針
- 照合の参照値は R `mirt` で生成し、JSON で `tests/reference/` にコミットする。CI では R を実行しない。
- 生成スクリプトは `validation/r/<case>.R`。各スクリプトの先頭に `sessionInfo()` 相当（R・mirt のバージョン）を JSON に書き出す。
- 照合は**同じ条件**で行う: 同じ求積（等間隔 61 点・$[-6,6]$ など）、同じ収束基準（mirt 側は `TOL` を厳しく、例: 1e-8）、加速なし、同じ事前分布。
- データはシミュレーション（固定シード）と公開データ（例: mirt 同梱の `LSAT7`, `Science`, `SAT12`）の両方。

## 許容誤差（初期値、必要に応じて見直す）
| 量 | 許容（最大絶対差） |
|---|---|
| 対数尤度 | 1e-6（相対） |
| 項目パラメタ（内部尺度 $a, d$） | 1e-4 |
| 潜在分布パラメタ（平均・SD） | 1e-4 |
| 標準誤差 | 1e-3（相対） |
| EAP | 1e-5 |

## ケース一覧（マイルストーンと対応）
| ID | 内容 | mirt 側の呼び出し | MS |
|---|---|---|---|
| V1 | 2PL 単一群（LSAT7） | `mirt(LSAT7, 1, "2PL", SE=TRUE, SE.type="Oakes")` | M1 |
| V2 | Rasch 単一群 | `mirt(..., "Rasch")` | M1 |
| V3 | GRM 単一群（Science） | `mirt(Science, 1, "graded")` | M2 |
| V4 | 2PL 2 群・部分アンカー・焦点群自由 | `multipleGroup(..., invariance=c(anchors, "free_means","free_var"))` | M2 |
| V5 | FIPC（固定項目 + 潜在分布推定） | `pars='values'` で `est=FALSE` | M2 |
| V6 | EH（単一群・2 群） | `dentype="EH"` | M3 |
| V7 | 3PL + 事前分布（MAP） | `mirt.model` の `PRIOR = (1-30, g, expbeta, ...)` 相当 | M3 |
| V8 | GPCM | `"gpcm"` | M3 |
| V9 | 欠測あり（MAR） | NA を含むデータ | M2 |
| V10 | SE（Oakes / crossprod / sandwich） | `SE.type=` | M1–M3 |

## 照合以外の検証
- **パラメタ回復シミュレーション**: 条件（N, J, 群数, アンカー数）× 反復 100 で bias / RMSE を出し、`validation/recovery/` に結果と図を置く。
- **EM の単調性**: すべての照合ケースで、反復ごとの（MAP 時は事後）対数尤度が非減少であることをテストで確認。
- **SE の内部整合**: Oakes の SE と、収束点での周辺対数尤度の直接 `jax.hessian` による SE が一致すること（小規模ケース）。
- **不変性**: 項目の並べ替え、群ラベルの付け替え、パターン圧縮の有無で結果が変わらないこと。
