# 検証計画（R mirt との数値照合）

[English](VALIDATION.md) | 日本語

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

## mirt 側の設定と既知の差（M1 で判明）
- **M-step 最適化**: 参照値は `optimizer = "NR"` で生成する。mirt 既定の M-step 最適化（Rasch は `nlminb`、
  2PL は BFGS）は対数尤度の変化が小さくなると反復上限が縮み、M-step がほぼ動かなくなって `TOL` 判定が
  EM の収束前に通る（LSAT7 の Rasch で分散が約 2e-5 ずれ、`TOL` を変えても結果が変わらない）。
- **潜在分布の M-step**: mirt は分散（・平均）を閉形式のモーメントで更新する。これは離散化した周辺尤度
  （mirt 自身が報告する対数尤度）の最大化ではないため、Rasch の分散が flexpyirt（厳密な M-step, DESIGN §4.4）
  とわずかに異なる（シミュレーション N=2000 で分散差 7.9e-5、対数尤度は flexpyirt が 5.6e-7 高い）。
  閉形式更新で mirt の値が 1e-10 で再現できることを確認済み。許容誤差（1e-4）内なので照合は通常どおり行い、
  加えて「flexpyirt の対数尤度が mirt 以上」を検査する。
- **潜在分布パラメタの照合方針（決定）**: flexpyirt は厳密な M-step のみを持つ（DESIGN §4.4）。mirt との差は
  格子の打ち切りから生じるので、打ち切りが無視できない条件（V4 の焦点群など、平均が 0 から離れる・分散が
  2 を超える）では、**mirt・flexpyirt ともに格子を広げて（例: `theta_lim = c(-8, 8)`、点数も比例して増やす）**
  打ち切り差を許容誤差より十分小さくしてから照合する。既定格子（61 点・$[-6,6]$）での照合は、打ち切り差が
  1e-5 未満の条件に限る。いずれの場合も「flexpyirt の対数尤度が mirt 以上」を検査する。
- **`theta_lim` の渡し方（M2 で判明）**: `mirt()` / `multipleGroup()` では `technical = list(theta_lim = ...)` の中に
  入れる。トップレベル引数にすると黙って無視され、既定の $[-6,6]$ になる（`fscores()` はトップレベルで受け付ける）。
  参照値 JSON の `meta.grid_check` に実際に使われた格子の範囲を記録している。
- **対数尤度の比較**: mirt が報告する対数尤度は最終反復の E-step のもので、必ずしも最終推定値での値と一致しない。
  「flexpyirt ≥ mirt」の検査は、全パラメタを mirt の推定値に固定して flexpyirt で評価した対数尤度と比べる（M2 以降）。
- **M2 の照合結果**: V3（GRM）、V4（2 群・アンカー 5 項目・焦点群自由、81 点 $[-8,8]$）、V5（FIPC、同格子）、
  V9（15% MCAR）のいずれも、パラメタ・対数尤度・SE が mirt と ~1e-8 で一致。格子を両側で揃えれば、
  厳密な M-step と mirt の閉形式更新の差（打ち切り由来）は無視できることも確認された。
  照合テスト: `tests/test_mirt_m2.py`。参照値の生成: `validation/r/m2_reference.R`。
- **EAP の照合**: 推定値の差が EAP に伝播しないよう、全パラメタを mirt の推定値に固定して
  （ParameterTable の `free=False`）スコアリングだけを照合する。許容誤差 1e-5。
- 照合テスト: `tests/test_mirt_m1.py`（V1, V2, V10）。参照値の生成: `validation/r/m1_reference.R`。

## 照合以外の検証
- **パラメタ回復シミュレーション**: 条件（N, J, 群数, アンカー数）× 反復 100 で bias / RMSE を出し、`validation/recovery/` に結果と図を置く。
- **EM の単調性**: すべての照合ケースで、反復ごとの（MAP 時は事後）対数尤度が非減少であることをテストで確認。
- **SE の内部整合**: Oakes の SE と、収束点での周辺対数尤度の直接 `jax.hessian` による SE が一致すること（小規模ケース）。
- **不変性**: 項目の並べ替え、群ラベルの付け替え、パターン圧縮の有無で結果が変わらないこと。
