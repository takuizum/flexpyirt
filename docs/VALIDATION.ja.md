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
| MAP / ML / WLE（$\theta$） | 1e-4 |
| MAP / ML / WLE（SE） | 1e-3（相対） |

## ケース一覧（マイルストーンと対応）
| ID | 内容 | mirt 側の呼び出し | MS |
|---|---|---|---|
| V1 | 2PL 単一群（LSAT7） | `mirt(LSAT7, 1, "2PL", SE=TRUE, SE.type="Oakes")` | M1 |
| V2 | Rasch 単一群 | `mirt(..., "Rasch")` | M1 |
| V3 | GRM 単一群（Science） | `mirt(Science, 1, "graded")` | M2 |
| V4 | 2PL 2 群・部分アンカー・焦点群自由 | `multipleGroup(..., invariance=c(anchors, "free_means","free_var"))` | M2 |
| V5 | FIPC（固定項目 + 潜在分布推定） | `pars='values'` で `est=FALSE` | M2 |
| V6 | FIPC での EH（単一群・2 群） | `mirt(..., dentype = "EH", pars = <バンク固定>)` / `multipleGroup(...)` | M3 |
| V7 | 3PL + 事前分布（MAP） | `mirt.model` の `PRIOR = (1-30, g, expbeta, ...)` 相当 | M3 |
| V8 | GPCM・PCM（Science + シミュレーション） | `mirt(..., "gpcm")` / `mirt(..., "Rasch")`（多値） | M3 |
| V9 | 欠測あり（MAR） | NA を含むデータ | M2 |
| V10 | SE（Oakes / crossprod / sandwich） | `SE.type=` | M1–M3 |
| V11 | 3PL・4PL 単一群（シミュレーション） | `mirt(..., "3PL")` / `mirt(..., "4PL")` | M3 |
| V12 | 表示用 IRT 閾値とデルタ法 SE（2PL・Rasch・GRM・GPCM・PCM・2 群 2PL） | `coef(m, IRTpars = TRUE, printSE = TRUE)` | M4 |
| V13 | 能力値スコアリング: MAP・ML・WLE（2PL・GRM・GPCM・3PL・2 群・欠測） | `fscores(m, method = "MAP"/"ML"/"WLE", response.pattern = ..., full.scores.SE = TRUE)` | M4 |
| V15 | S-X2 項目適合度（GRM・GPCM・分散自由の PCM・3PL・FIPC の 2PL、`mincell` 1 と 3） | `itemfit(m, fit_stats = "S_X2")` | M5 |
| V16 | 項目をまたぐ等値制約（LSAT7 の共通傾き、2PL + graded のモデル間制約、アンカーつき 2 群 + 共通傾き） | `mirt(..., mirt.model("... CONSTRAIN = (1-5, a1)"))` / `multipleGroup(...)` | M5 |
| V17 | V4 のデータでの尤度比検定による DIF（`a`+`d` と `d` の `add`、完全不変モデルからの `a`+`d` の `drop`） | `DIF(mg, which.par = ..., scheme = ...)` | M5 |
| V18 | M2 / M2* 限定情報の適合度（LSAT7 2PL、シミュレーション 2PL、3PL、分散自由の Rasch、GRM $K=3$、GPCM $K=4$） | `M2(m, quadpts = 61, theta_lim = c(-6, 6))` | M5 |
| V14 | 確認的 2–3 次元 MIRT（2PL 単純構造・交差負荷・GRM・3 次元 2PL） | `mirt(..., mirt.model("F1 = ...\nF2 = ...\nCOV = F1*F2"), SE = TRUE, SE.type = "Oakes")` | M4 |
| V19 | Davidian 曲線潜在分布（2PL $k=3$・GRM $k=2$・平均分散自由の FIPC 2PL $k=3$） | `mirt(..., dentype = "Davidian-k")` および `mirt(..., customGroup = ..., SE.type = "Oakes")` | M5 |

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
- **3PL / 4PL の漸近線の尺度と SE の変換（V11、M3 で判明）**: mirt は $g, u$ を logit 尺度で最適化し、
  `coef(m, printSE = TRUE)` も `logit(g)` / `logit(u)` と logit 尺度の SE を返す（`rawug` を変えても同じ）。
  flexpyirt は自然尺度で推定するため（DESIGN §15.2）、`validation/r/m3_reference.R` では推定値を `plogis()`、
  SE をデルタ法 $SE_g = g(1-g)\,SE_{\mathrm{logit}(g)}$（$u$ も同様）で自然尺度に変換して照合し、変換前の
  logit 尺度の値も `raw_logit` に残している。内点の最尤解では観測情報行列がヤコビアンで厳密に変換される。
- **経験ヒストグラム（V6、M3）**: mirt の `dentype = "EH"` は毎サイクルでヒストグラムを標準化せずに推定し直すので、
  全項目が自由だと尺度が識別されず EM が漂流する（自由推定の 2PL は 50,000 サイクルで収束せず、ヒストグラムの分散は 1.5 まで
  ずれた）。flexpyirt は代わりに参照群を標準化する（DESIGN §15.34）が mirt にはこの方式がないため、V6 は固定項目が尺度を
  決めてどちらも標準化しない FIPC を使う: V6a（$J=20$、項目 1–10 を固定、$N=3000$、標準化した $\chi^2_4$ の特性）と
  V6b（3000 人 × 2 群、両群で同じバンクを固定、項目 11–20 を群間等値、焦点群の特性は $0.5-1.2\,\chi^2_4$）、
  $[-6,6]$ 上の 61 点。細かい格子では EH の尤度は多くのヒストグラム方向でほぼ平坦で最大点は尖っているため EM は這うように
  進み（mirt は `TOL = 1e-9` で 69,683 / 62,632 サイクル）、mirt の停止規則はヒストグラムを見ない。そのため最終的な重みを
  節点ごとには比較しない。mirt の解において: 対数尤度は mirt と 1.5e-15 で一致、flexpyirt の EM を 1 ステップ進めると
  項目は 1e-9（mirt の `TOL`）、ヒストグラムは 4e-7 動く（mirt は E-step 1 回分遅れたヒストグラムを報告する）、
  Hessian と Oakes の SE（ヒストグラム固定）は mirt の Richardson SE と 1.1e-7 で一致。flexpyirt の通常の推定（SQUAREM、
  `tol = 1e-6`）は 7,162 / 2,837 反復で収束し、mirt より高い対数尤度（1.6e-3 / 7.3e-4 上）で終わり、項目は mirt と 5e-5
  以内。mirt 1.46.1 の EH の Oakes SE は自身の Richardson SE より 0.1–1.3% 小さく（原因は未確認）、2 群 EH モデルでは
  Richardson SE の計算が失敗するので、V6b は Oakes == Hessian（4e-16）を確認する。テスト: `tests/test_mirt_v6_eh.py`。
  参照値の生成: `validation/r/m3_eh_reference.R`。
- **V11 のシミュレーション設計と mirt `NR` の初期値問題（M3）**: 事前分布なしの 3PL / 4PL 最尤推定では、
  項目が易しすぎ／難しすぎたり傾きが小さかったりして $N(0,1)$ の裾で漸近線が観測できないと、有限標本でも
  境界解（$g\to 0$ や $u\to 1$、情報行列が特異）が生じやすい。さらに 4PL では、mirt 既定の傾き初期値
  （`a = 0.851`）から logit 尺度で `optimizer = "NR"` を回すと初期反復で `logit(g) ≈ -15` に飛び、
  logit 尺度の勾配 $g(1-g)\,\partial Q/\partial g$ が 0 付近で消えるため対数尤度が真に低い境界点に捕捉される。
  許容誤差を緩めずに内点の収束した大域最尤解を照合するため、V11a は $J=15, N=5000$、
  $a\sim\mathrm{lognormal}(0.4,0.2)$、$d\sim U(-1.5,0.5)$、$g\sim U(0.12,0.25)$ とし、V11b は
  $J=12, N=20000$（パターン数は最大 $2^{12}=4096$）、$a\sim\mathrm{lognormal}(0.6,0.15)$、
  $d\sim U(-0.5,0.5)$、$g\sim U(0.10,0.22)$、$u\sim U(0.78,0.90)$ として mirt 側の初期値のみ $a=2.0$ とした
  （flexpyirt は既定の初期値のまま大域解に収束）。両ケースとも `HeywoodWarning` なしで収束し、パラメタは
  ~1e-8（3PL: 5.4e-9、4PL: 7.2e-8）、対数尤度は ~1e-15、Oakes SE は ~1e-7（3PL: 1.1e-8、4PL: 1.9e-7）、
  Hessian 対 mirt Richardson SE は ~6e-4（3PL: 6.8e-5、4PL: 5.6e-4。Richardson の数値微分誤差によるもので、
  flexpyirt 内の Oakes と Hessian は ~1e-11 で一致）、EAP は ~1e-15 で一致した。
  照合テスト: `tests/test_mirt_m3.py`。参照値の生成: `validation/r/m3_reference.R`。
- **GPCM・PCM のパラメタ対応と照合結果（V8、M3）**: mirt の `itemtype = "gpcm"`（および多値データに対する
  `itemtype = "Rasch"`。これは `a1 = 1` 固定・`COV_11` 自由推定の PCM に相当）は、`a1`、`ak0..ak{K-1}`
  （`0..K-1` 固定）、`d0`（`0` 固定）、`d1..d{K-1}`（自由）の列を返す。`validation/r/m3_gpcm_reference.R` は
  これらの固定値を確認したうえで、`a1` → `a`、`d1..d{K-1}` → `d1..d{K-1}` と対応付け、`ak*` と `d0` を除いて
  保存する。照合ケースは 3 件: V8a（`Science` を `0..3` に変換、GPCM、61 点 $[-6,6]$）、V8b（シミュレーション
  GPCM、$J=10, K=4, N=3000$、ステップ順が逆転した項目を含む、61 点 $[-6,6]$）、V8c（生成時の潜在分散 $1.5$ の
  シミュレーション PCM、mirt `itemtype = "Rasch"`、V4/V5 と同じく格子の打ち切りを無視できる 81 点 $[-8,8]$）。
  3 件とも許容誤差より十分に小さく一致した: 項目パラメタは ~1.4e-8（V8c の潜在分散は 6.1e-10）、対数尤度は
  ~3.7e-15、Oakes SE は ~1.3e-8、Hessian 対 mirt Richardson SE は ~1.1e-6（flexpyirt 内の Oakes と
  Hessian は ~1.1e-11）、EAP は ~8.0e-15。
  照合テスト: `tests/test_mirt_m3_gpcm.py`。参照値の生成: `validation/r/m3_gpcm_reference.R`。
- **SQUAREM の照合結果（M3）**: すべての mirt 照合テスト（`tests/test_mirt_m1.py`, `tests/test_mirt_m2.py`,
  `tests/test_mirt_m3.py`, `tests/test_mirt_m3_gpcm.py`, `tests/test_mirt_m3_priors.py`）を
  `BAEM(squarem=False)` と `BAEM(squarem=True)` の両方で実行し、同じ加速なし参照値と照合している。
  全 16 ケースが同一の不動点に許容誤差内で収束し（2PL/GRM/GPCM/PCM/3PL/4PL/FIPC/2 群/欠測/MAP の項目パラメタ最大差
  $\le 2.8\times 10^{-8}$、V2 シミュレーション Rasch の潜在分散差は通常 EM と同じ打ち切り由来の $7.9\times 10^{-5}$）、
  対数尤度（MAP 時は対数事後）は単調非減少で、EM 写像の評価回数を 1.3〜5.0 倍削減した（例:
  `v2_lsat7_rasch` 121 → 24、`v4_twogroup_2pl` 302 → 113、`v8_science_gpcm` 251 → 96、
  `v11_4pl` 531 → 272、`v7_2pl_map` 80 → 35、`v7_3pl_map` 64 → 37）。単体・Hypothesis 単調性・安全装置・評価回数削減テスト:
  `tests/test_m3_squarem.py`。
- **事前分布・MAP と R mirt 1.46 の検証（V7、M3）**: `mirt:::LL.Priors`、`mirt:::DerivativePriors`、
  `mirt:::BL.LL` の調査および R での数値停留・Hessian 検証により、目的関数の一致と mirt 1.46 側の 3 つの実装上の不具合を
  特定した:
  1. **目的関数（`mirt:::LL.Priors`）— DESIGN §4.7 と完全一致**: `norm(mu, sd)` は `dnorm(x, mu, sd, log = TRUE)`、
     `lnorm(mu, sd)` は `dlnorm(x, mu, sd, log = TRUE)`、`expbeta(a, b)` は
     `dbeta(plogis(logit_g), a, b, log = TRUE)` を**自然尺度・logit ヤコビアンなし**で評価する。厳密 MAP 推定値に
     全パラメタを固定して mirt で評価すると、`mirt@Fit$logLik` と `mirt@Fit$logPrior` は flexpyirt と $\le 10^{-14}$
     で一致する。
  2. **mirt 1.46 の `expbeta` 微分バグ**: `mirt:::DerivativePriors`（および C++ `computeDPars`）は、自然尺度の Beta
     微分 $(a-1)/g - (b-1)/(1-g)$ と $-(a-1)/g^2 - (b-1)/(1-g)^2$ に連鎖律係数 $g(1-g)$ を掛けずにそのまま
     **logit 尺度**の勾配・Hessian に加算している。このため `expbeta` では `optimizer = "NR"` が収束せず、勾配ベースの最適化
     （`BFGS`, `L-BFGS-B`, `nlminb`）は `logLik + logPrior` が真に低い非停留点（V7c では差 `4.89`、`max |Δg| = 0.095`）
     で停止し、mirt の `SE.type = "Oakes"` も `g_se` を約 10 分の 1 に過小評価する。
  3. **mirt 1.46 の `lnorm` Hessian 符号バグ**: `lnorm` の勾配は正確（そのため `NR` は ~1e-8 で厳密な MAP 推定値に収束する）
     だが、`DerivativePriors` / `computeDPars` の 2 階微分第 3 項の符号が逆（`+(log(a)-mu)/(a^2*s^2)` のところが
     `-(log(a)-mu)/(a^2*s^2)`）になっており、mirt の `Oakes` SE は真の事後 Hessian SE から約 5% ずれる（符号反転式とは
     `8.3e-10` で一致することを確認）。一方 `norm` 事前分布は勾配・Hessian とも正確で、`Oakes` SE は真の事後 Hessian SE と
     `9.6e-10` で一致する。
  4. **`SE.type = "Richardson"` は事前分布を含めない**: `mirt:::BL.LL` は `LL.Priors` を呼ばず周辺対数尤度のみを計算する
     ため、mirt の Richardson SE は**尤度のみ**の情報行列 $-\nabla^2\ell(\hat\eta_{\text{MAP}})$ に一致する（~1e-8 で確認）。

  以上を踏まえ、3 件の V7 ケース（$J=15, N=1000$、61 点 $[-6,6]$）を `tests/test_mirt_m3_priors.py` で照合する
  （参照値生成: `validation/r/m3_priors_reference.R`）:
  - **V7a（`v7_2pl_map`）**: 2PL、$a\sim\mathrm{NormalPrior}(1.2, 0.5)$、$d\sim\mathrm{NormalPrior}(0, 2)$。mirt
    （`NR`, `SE.type = "Oakes"`）と直接照合し、`loglik` 相対差 `3.0e-13`、`log_posterior` 相対差 `5.1e-15`、
    パラメタ `8.0e-9`、Oakes SE 対 mirt Oakes `3.4e-9`、Oakes 対 Hessian `1.2e-12`、EAP `6.4e-15`
    （mirt Richardson は $\hat\eta_{\text{MAP}}$ での flexpyirt 尤度のみ Hessian と一致することを確認）。
  - **V7b（`v7_2pl_lnorm_map`）**: 2PL、$a\sim\mathrm{LogNormal}(0.2, 0.3)$、$d\sim\mathrm{NormalPrior}(0, 2)$。
    `loglik`（`4.0e-13`）、`log_posterior`（`1.4e-14`）、パラメタ（`2.6e-9`）、EAP（`7.1e-15`）は mirt（`NR`）と直接照合し、
    Oakes / Hessian SE は R で求めた厳密な事後 Hessian と照合（`1.8e-9`、Oakes 対 Hessian `2.3e-12`）。
  - **V7c（`v7_3pl_map`）**: 3PL、$a\sim\mathrm{LogNormal}(0.2, 0.3)$、$d\sim\mathrm{NormalPrior}(0, 2)$、
    $g\sim\mathrm{Beta}(5, 17)$（mirt の `expbeta`）。R 上で mirt の厳密な目的関数 `logLik + logPrior` を最大化し
    （`max |grad| = 5.1e-13`、`mirt(..., pars = sv_fixed)` で `8.9e-15` まで相互検証、EAP は mirt の `fscores`）、
    `loglik` 相対差 `2.5e-13`、`log_posterior` 相対差 `8.1e-16`、パラメタ `2.9e-9`、Oakes / Hessian SE `5.7e-9`
    （Oakes 対 Hessian `5.7e-12`）、EAP `6.0e-15` で一致した。
  単体テスト（scipy `log_pdf` 照合、自動微分ガード、固定パラメタの事前分布無視、共有 `free_id` の重複なし、表編集、AIC/BIC 警告）:
  `tests/test_m3_priors.py`。
- **表示用 IRT パラメタ化とデルタ法 SE（V12、M4）**: mirt 1.46.1 の
  `coef(m, IRTpars = TRUE, printSE = TRUE)` は傾き・切片パラメタを $b = -d/a$（2PL / Rasch）、
  $b_k = -d_k/a$（GRM / `graded`）、ステップ困難度 $b_k = -(d_k - d_{k-1})/a$（$d_0 = 0$、GPCM / PCM）
  へ変換しており、flexpyirt の `FitResult.items` の定義と完全に一致する。一方で
  `mirt:::mirt2traditional` の `cls == "gpcm"` 分岐を調べたところ、mirt 1.46.1 には **GPCM の `b1_se` のみ**
  に影響する 1 つずれ（off-by-one）添字バグがあることが判明した:
  - `delta_index[[nfact + 1L]]` が `c(which.a, ncat + nfact + 2L)`（`d0` を含む `opar` 中の `a1` と `d1` の位置）で
    上書きされるのに対し、`fns` 内で返す `ret` は `d0` を除いているため `b1` は `ncat + nfact + 1L`、`b2` は
    `ncat + nfact + 2L` に置かれる。
  - その結果 `ret[index[2]]` は `b1 = -d1/a` ではなく `b2 = -(d2 - d1)/a` を取り出して `(a, d1)` のみで微分
    （`d2` 固定）するため、真の勾配 $(d_1/a^2,\, -1/a)$ ではなく $((d_2 - d_1)/a^2,\, 1/a)$ を使ってしまう
    （R 上で mirt の生 `b1_se` がこのバグ式と `< 1e-9` で一致し、`Science` では最大 268% ずれることを確認）。
  - mirt の `gpcm` 出力における点推定値 `b1..b{K-1}` と `b2_se..b{K-1}_se` は正確である（`vcov(m)` からの厳密計算と
    `< 1e-9` で一致）。PCM（`a = 1` 固定）では `a` の項が 0 で $|\partial b_2/\partial d_1| = 1 = |\partial b_1/\partial d_1|$
    となるためバグが相殺し、`b1_se..b3_se` すべて正確になる。

  以上を踏まえ、6 件のケースを `tests/test_mirt_m4_irtpars.py` で通常 EM と SQUAREM の両方で照合する
  （参照値生成: `validation/r/m4_irtpars_reference.R` → `tests/reference/v12_irtpars.json`、
  `optimizer = "NR"`, `accelerate = "none"`, `TOL = 1e-10`, `SE.type = "Oakes"`）:
  - **V12a（`lsat7_2pl`、2PL、61 点 $[-6,6]$）**: `loglik` 相対差 `3.4e-16`、`max |Δb| = 1.4e-8`、
    `b_se` 相対差 `1.5e-8`。
  - **V12b（`lsat7_rasch`、Rasch、61 点 $[-6,6]$）**: `loglik` 相対差 `3.1e-15`、`max |Δb| = 1.1e-7`、
    `b_se`（`= d_se`）相対差 `6.9e-8`。
  - **V12c（`science_grm`、GRM 0..3、61 点 $[-6,6]$）**: `loglik` 相対差 `3.7e-15`、
    `max |Δbk| = 2.7e-8`、`bk_se` 相対差 `1.1e-8`。
  - **V12d（`science_gpcm`、GPCM 0..3、61 点 $[-6,6]$）**: `loglik` 相対差 `3.1e-15`、
    `max |Δbk| = 1.5e-9`、`b2_se` / `b3_se` は mirt `IRTpars` と直接照合で相対差 `9.7e-10`、`b1_se` は
    R の `vcov(m)` による厳密デルタ法 SE と照合で相対差 `9.3e-10`。
  - **V12e（`science_pcm`、PCM 0..3、81 点 $[-8,8]$）**: `loglik` 相対差 `8.4e-16`、
    `max |Δbk| = 2.3e-9`、`bk_se`（3 列とも mirt `IRTpars` と直接照合）相対差 `3.4e-10`。
  - **V12f（`mg_2pl`、2 群 2PL、アンカー 4 項目、焦点群自由、81 点 $[-8,8]$）**: `loglik` 相対差 `1.1e-15`、
    `max |Δb| = 7.3e-9`、`b_se` 相対差 `1.5e-8`。
  単体テスト（解析式照合、固定パラメタ、`se=None`、非正定値 `vcov`、部分等値、混在モデル、`summary()`、`_repr_html_()`）:
  `tests/test_m4_result.py`。
- **EAP の照合**: 推定値の差が EAP に伝播しないよう、全パラメタを mirt の推定値に固定して
  （ParameterTable の `free=False`）スコアリングだけを照合する。許容誤差 1e-5。
- **能力値スコアリング: MAP・ML・WLE（V13、M4）**: mirt 1.46.1 の `mirt:::fscores.internal`、`mirt:::MAP`、
  `mirt:::ML`、`mirt:::WLE`、`mirt:::MAP.mirt`、`mirt:::WLE.mirt` を調査し、以下を確認した:
  - **範囲（境界）**: `theta_lim` は EAP の求積格子（および MAP / ML / WLE の EAP 初期値）に使われ、
    MAP・ML・WLE 本体は `[-max_theta, max_theta]`（既定 `max_theta = 20`）上で最適化する。
  - **目的関数と SE（既定 `expected.info = FALSE`）**: MAP は連続な正規潜在分布を事前分布とする
    $\log L(\theta\mid y) + \log\phi(\theta;\mu_g,\sigma_g^2)$、ML は $\log L(\theta\mid y)$、WLE は
    $\log L(\theta\mid y) + \tfrac{1}{2}\log I(\theta)$（$I(\theta)$ は観測項目の Fisher テスト情報量。
    Warm の方程式 $\partial\log L/\partial\theta + I'(\theta)/(2I(\theta)) = 0$ および Firth の Jeffreys
    事前分布と一致）を最大化する。3 手法とも mirt 既定の SE は最適点における負の目的関数の `nlm()` Hessian
    の逆平方根（観測曲率。`response.pattern` なしで `expected.info = TRUE` にすると $I(\hat\theta)^{-1/2}$
    へ切り替わり、短いテストや極端パターンで差が出る。mirt 1.46.1 では `response.pattern` 経由で
    `expected.info` が引き継がれない）。
  - **極端パターンと mirt 1.46.1 の不具合**: ML では全観測項目が最低／最高カテゴリなら `-Inf` / `+Inf`
    （SE は `NA`）、全欠測は `NA` / `NA`。mirt 1.46.1 で 2 つの境界での挙動を確認した: (a) 3PL の低得点パターンで
    $\log L(\theta)$ の上限が $-\infty$ にある場合、`nlm` の有限差分 Hessian が $\le 0$（`SE = NA`）に
    なると `mirt:::ML` が `SEest > 30` 判定をすり抜けて `-Inf` ではなく有限値 `F1 ≈ -17`（`SE = NA`）を
    返す（`v11_3pl` の 3,790 パターン中 31 件。`mirt:::ML` で SE が NaN になる分岐が `SEest > 30` →
    `±Inf` の変換を飛ばすことをソースで確認）。(b) 欠測データでは、最終項目が欠測の極端パターンの一部が
    `±Inf` ではなく `max_theta = 20`（`F1 ≈ 20`, `SE = NA`）に張り付く（挙動は観測したが、原因とした
    `fscores.internal` の欠測数の数え方は mirt の R ソースで該当箇所を確認できなかったため、説明は未確認と
    する）。どちらも V13 の照合には使っていない。flexpyirt はこれらを正しく検出し
    `±inf` と SE `NaN` を返す。これとは別に、3PL の二峰性パターンで厳密に凹な内部の局所最頻値 $\hat\theta$
    （$\mathrm{SE} \le 30$）の対数尤度より下側漸近線 $\lim_{\theta\to-\infty}\log L(\theta)$ が厳密に大きい場合、
    `mirt:::ML` の EAP 出発の局所 `nlm()` 探索は内部の局所最頻値で停止して有限値 `(F1, SE_F1)` を返すのに対し、
    flexpyirt は $f(\hat\theta)$ を $f(\pm 20)$ と比較して真の大域上限である `(-inf, NaN)`（本来の最尤推定）を
    意図的に返す。`v11_3pl` の全 3,790 ユニークパターン中 **25 パターン** でこの差異が生じる（V13d のシード付き
    抽出には含まれておらず、`tests/test_mirt_m4_scoring.py` でパターン index 1285 を直接検証している）。全欠測行については、mirt が全手法で `NA` に落とすのに対し、flexpyirt では
    ベイズ手法（EAP・MAP・PV）は事前分布、尤度のみの手法（ML・WLE）は `(NaN, NaN)` を返す。

  全パラメタを mirt の推定値に固定して 6 ケースを `tests/test_mirt_m4_scoring.py` で照合した
  （参照値生成: `validation/r/m4_scoring_reference.R` → `tests/reference/v13_fscores.json`、
  `optimizer = "NR"`, `accelerate = "none"`, `TOL = 1e-10`。許容誤差は $\theta$ 絶対 `1e-4`・SE 相対 `1e-3`、
  `-inf`/`+inf`/`NaN` は厳密一致。残る ~`1e-6`–`2e-4` の差は `nlm` の有限差分勾配・Hessian 由来）:
  - **V13a（`lsat7_2pl`、2PL、32 パターン、61 点 $[-6,6]$）**: `loglik` 相対差 `5.1e-16`、
    MAP $\theta$ `2.2e-6` / SE 相対 `1.8e-5`、ML $\theta$ `1.5e-6` / SE 相対 `4.1e-5`（`±inf` 2 件）、
    WLE $\theta$ `2.1e-6` / SE 相対 `5.1e-5`。
  - **V13b（`science_grm`、GRM 0..3、90 パターン、61 点 $[-6,6]$）**: `loglik` 相対差 `3.7e-15`、
    MAP $\theta$ `3.0e-6` / SE 相対 `3.6e-5`、ML $\theta$ `2.5e-6` / SE 相対 `8.7e-5`（`±inf` 2 件）、
    WLE $\theta$ `2.8e-6` / SE 相対 `1.5e-4`。
  - **V13c（`science_gpcm`、GPCM 0..3、90 パターン、61 点 $[-6,6]$）**: `loglik` 相対差 `2.8e-15`、
    MAP $\theta$ `2.2e-6` / SE 相対 `4.1e-5`、ML $\theta$ `2.0e-6` / SE 相対 `8.8e-5`（`±inf` 2 件）、
    WLE $\theta$ `2.5e-6` / SE 相対 `1.2e-4`。
  - **V13d（`v11_3pl`、3PL、112 パターン、61 点 $[-6,6]$）**: `loglik` 相対差 `1.5e-15`、
    MAP $\theta$ `3.8e-6` / SE 相対 `7.4e-5`、ML $\theta$ `5.8e-6` / SE 相対 `2.1e-4`（低得点発散を含む `±inf` 20 件）、
    WLE $\theta$ `3.2e-6` / SE 相対 `1.9e-4`。
  - **V13e（`twogroup_2pl`、2 群 2PL、ref/focal 計 104 パターン、81 点 $[-8,8]$）**:
    `loglik` 相対差 `1.9e-15`、MAP $\theta$ `2.9e-6` / SE 相対 `5.1e-5`、ML $\theta$ `2.6e-6` / SE 相対 `8.7e-5`
    （`±inf` 4 件）、WLE $\theta$ `3.3e-6` / SE 相対 `1.6e-4`。
  - **V13f（`missing_2pl`、15% MCAR 欠測 2PL、88 パターン、61 点 $[-6,6]$）**:
    `loglik` 相対差 `7.4e-16`、MAP $\theta$ `2.0e-6` / SE 相対 `4.6e-5`、ML $\theta$ `1.4e-6` / SE 相対 `1.1e-4`
    （`±inf` 8 件）、WLE $\theta$ `2.0e-6` / SE 相対 `1.5e-4`。
  Plausible values（`method="PV"`。$D > 1$ は `tests/test_mirt_m4_mirt.py`、DESIGN §15.42）は `tests/test_m4_scoring.py` で統計的に検証する（同一 `key` の再現性、
  同一パターンの回答者間での独立性、標本平均・標準偏差がモンテカルロ誤差内で EAP と事後 SD に一致すること）。
  あわせて平坦事前分布 MAP $\approx$ ML、全 7 項目モデルでの WLE 停留条件、極端パターン、多母集団、欠測、入力検証を
  単体テストで確認している。
- **S-X2 項目適合度（V15、M5）**: 既存の参照値を再利用した 6 ケース（Science の V3 GRM と V8 GPCM、潜在分散が自由な
  V8 PCM、V11 3PL、8 項目を固定した V5 FIPC 2PL、および `mincell = 3` の V3）。mirt は参照値の推定値で、各パラメタの
  `est` フラグを保ったまま（自由パラメタを自由度から引くため）`TOL = NaN` とし、推定と同じ格子で
  `itemfit(m, fit_stats = "S_X2")` を計算する（mirt 1.48、`validation/r/m5_itemfit_reference.R` →
  `tests/reference/v15_sx2.json`）。flexpyirt はデータを推定し直す: 自由度はすべてのケースで一致し、S-X2 は相対 `2.2e-8`、
  p 値は `1.3e-8` で一致（推定値自体の一致の水準）。単体テストで Lord–Wingersky 漸化式を全パターン列挙と比べ、期待度数の行が
  条件付き確率であることを確認する。テスト: `tests/test_mirt_m5_itemfit.py`。
- **確認的 2–3 次元 MIRT（V14, M4）**: 4 つのシミュレーションケースを `tests/test_mirt_m4_mirt.py` で照合した
  （通常 EM・SQUAREM の双方で通過。参照値生成: `validation/r/m4_mirt_reference.R` →
  `tests/reference/v14_mirt2d.json`、`accelerate = "none"`, `TOL = 1e-10`, `SE.type = "Oakes"`、
  求積格子は mirt の既定直積格子 $[-6, 6]$ 上で 2 次元 $31^2 = 961$ 点・3 次元 $15^3 = 3{,}375$ 点）。
  2 次元の 3 ケースは `optimizer = "NR"`、3 次元ケースは mirt の `NR` が 3,375 点上の 3 次元 `GroupPars` で
  数値 Hessian により極端に遅くなるため `optimizer = "BFGS"`（`control = list(reltol = 1e-13, maxit = 500L)`）を
  用いた。全 4 ケースで `flexpyirt.loglik >= fixed_at_mirt.loglik`、EM / SQUAREM の厳密な単調性
  （SQUAREM により EM 写像評価回数は 2.2–2.8 倍削減）、および Oakes と直接 Hessian の一致
  （相対 $\le 1.2\times 10^{-14}$）を確認した:
  - **V14a（`mirt2d_2pl_simple`、2 次元 2PL、$J=12, N=2000$、`F1=1-6`, `F2=7-12`, $\rho=0.5$、$31^2$ 点）**:
    `loglik` 相対 `2.1e-15`、項目パラメタ `3.3e-9`、`cor_12` `1.0e-8`、項目 Oakes SE 相対 `3.7e-9`、
    `cor_12_se` 相対 `1.1e-8`、Oakes 対 Hessian `7.5e-15`、EAP $\theta$ `7.3e-15` / SD `8.0e-15`。
  - **V14b（`mirt2d_2pl_cross`、交差負荷 `q07` つき 2 次元 2PL、$J=12, N=2500$, $\rho=0.45$、$31^2$ 点）**:
    `loglik` 相対 `1.8e-15`、項目パラメタ `4.0e-9`、`cor_12` `7.3e-9`、項目 Oakes SE 相対 `4.9e-9`、
    `cor_12_se` 相対 `2.9e-9`、Oakes 対 Hessian `5.6e-15`、EAP $\theta$ `6.9e-15` / SD `7.0e-15`。
  - **V14c（`mirt2d_grm`、2 次元 GRM $K=4$、$J=10, N=2000$、`F1=1-5`, `F2=6-10`, $\rho=0.55$、$31^2$ 点）**:
    `loglik` 相対 `1.7e-15`、項目パラメタ `2.0e-9`、`cor_12` `4.6e-9`、項目 Oakes SE 相対 `2.2e-9`、
    `cor_12_se` 相対 `3.0e-9`、Oakes 対 Hessian `4.4e-15`、EAP $\theta$ `8.0e-15` / SD `1.2e-14`。
  - **V14d（`mirt3d_2pl`、3 次元 2PL、$J=9, N=2000$、`F1=1-3`, `F2=4-6`, `F3=7-9`, $\rho=(0.45, 0.25, 0.50)$、$15^3$ 点）**:
    `loglik` 相対 `4.7e-15`、項目パラメタ `1.9e-8`、相関 `4.9e-8`、項目 Oakes SE 相対 `9.0e-7`、
    相関 Oakes SE 相対 `7.3e-5`（mirt の 3 次元有限差分 Oakes ステップ由来。flexpyirt の Oakes 対 Hessian は
    `1.2e-14`）、EAP $\theta$ `1.1e-14` / SD `1.1e-14`。
  - **MAP（4 ケースすべて）**: 全パラメタを mirt の V14 推定値に固定し、EAP のパターンと、それを 3 項目欠測にした
    20 パターン（ケースあたり 220 パターン。mirt 1.48、`validation/r/m4_mirt_map_reference.R` →
    `tests/reference/v14_mirt_map.json`）で `fscores(method = "MAP")` と比較: $\theta$ は `4.3e-6`、SE は相対 `3.8e-5`
    以内で一致（mirt の `nlm` の許容誤差の水準。許容誤差は `1e-4` / `1e-3`）。
  - **ML と WLE（4 ケースすべて）**: MAP と同じ設定に、全最低・全最高と 1 因子だけ極端なパターンを加えた（ケースあたり 223
    パターン。`validation/r/m4_mirt_mlwle_reference.R` → `tests/reference/v14_mirt_mlwle.json`）。WLE は 892 行すべてで一致
    （$\theta$ `5.0e-6`、SE 相対 `1.7e-4`）。ML は両者とも内点の 644 行で一致（$\theta$ `6.5e-6`、SE `1.2e-4`）。mirt が
    ある次元で `±Inf` を返す 242 行では flexpyirt も同じ符号の無限大を返し、mirt が発散する次元を ±20 近くで SE 0 として
    止める 22 行では flexpyirt は符号付きの無限大を返す（DESIGN §15.41）。
  単体テスト（カーネル・多変量正規重みと正定値性ガード・直積格子形状・2 次元 Rasch / 混在 / 事前分布 / 固定・
  Oakes == Hessian == 参照 Oakes・識別性と仕様エラー）: `tests/test_m4_mirt.py`。
- 照合テスト: `tests/test_mirt_m1.py`（V1, V2, V10）。参照値の生成: `validation/r/m1_reference.R`。
- **項目をまたぐ等値制約（V16、M5）**: mirt の `CONSTRAIN`（`optimizer = "NR"`、`TOL = 1e-10`）。V16a: LSAT7 の
  2PL で全項目共通の傾き（$N(0,1)$、$[-6,6]$ の 61 点）。V16b: シミュレーション $N=2500$、項目 1–6 が 2PL・7–10 が
  3 カテゴリの graded で、2PL と graded にまたがる共通傾き（1, 2, 7）、項目 3・4 の共通切片、8–10 の共通傾き、9・10 の
  共通 `d1`。V16c: 各 1500 人の 2 群、アンカー q01–q05 は群間等値、q06–q10 の傾きは各群の中で共通、焦点群の平均と分散を
  自由推定（$[-8,8]$ の 81 点）。いずれも mirt と一致（通常の EM・SQUAREM とも）: パラメタ $2.1\times10^{-8}$、
  対数尤度 $4\times10^{-11}$、Oakes SE（相対）$1.3\times10^{-8}$、Hessian と mirt Richardson SE $2.9\times10^{-8}$、
  flexpyirt の Oakes と Hessian $5\times10^{-15}$。自由パラメタ数は mirt の `nest` と一致。テスト:
  `tests/test_mirt_m5_equal_items.py`。参照値生成: `validation/r/m5_equal_items_reference.R`。
- **尤度比検定による DIF（V17、M5）**: V4 のデータで mirt の `DIF()`（`optimizer = "NR"`、`TOL = 1e-10`、基準モデルと
  すべての再推定で $[-8,8]$ の 81 点）。アンカーつきモデル（q01–q05 を等値、焦点群の平均と分散を自由）からの
  `scheme = "add"`（q06–q15 の `a1`+`d` と `d`）と、完全不変モデルからの `scheme = "drop"`（全項目の `a1`+`d`）。mirt は
  情報量規準を（どちらの方式でも）項目を自由にしたモデル − 制約したモデルで報告し、p 値は既定で Holm 補正
  （`p.adjust = "holm"`）。mirt の `p` は `1 - pchisq(...)` なので非常に大きな $X^2$ では厳密に 0（flexpyirt は生存関数）。
  一致（通常の EM・SQUAREM とも）: $X^2$ と AIC/SABIC/HQ/BIC の差 $2.2\times10^{-11}$、`p`・`adj_p` $3\times10^{-11}$。
  テスト: `tests/test_mirt_m5_dif.py`。参照値生成: `validation/r/m5_dif_reference.R`。
- **M2 / M2* 限定情報の適合度（V18、M5）**: $[-6,6]$ の 61 点上の 6 ケース（`validation/r/m5_m2_reference.R` →
  `tests/reference/v18_m2.json`、mirt 1.46.1）: `lsat7_2pl`（V1）、`sim_2pl`（V1）、`threepl`（V11 3PL）、
  `rasch_free_var`（V2 潜在分散自由の Rasch）、`grm_sim`（GRM $K=3, J=10, N=2000$）、`gpcm_sim`
  （GPCM $K=4, J=8, N=2000$）。mirt 1.46.1 の `M2()` との照合で、mirt 側の 4 つの相違・不具合を確認した（すべて R で検証済み）:
  1. **$\Delta_2$ から自由な潜在分布パラメタを除外**: mirt は `M2()` 内で `GroupPars` の `est` を `FALSE` に強制する一方、
     `df` からは引いている。flexpyirt は Maydeu-Olivares & Joe (2005, 2006) に従い既定ですべての自由パラメタを
     $\Delta_2$ に含める。内部の `_m2.m2_statistic(..., _latent_in_delta=False)` は mirt の `rasch_free_var` の値を再現する
     （mirt の推定値で `M2` 相対 `9.7e-16`、`SRMSR` `4.9e-17`、`TLI` `1.6e-9`）。既定（`True`）は潜在分布を含めた厳密な R 参照値と
     一致し（mirt の推定値で `M2` 相対 `1.0e-12`、再推定では $[-6,6]$ での $7.9\times10^{-5}$ の分散差により `9.5e-6`）、
     mirt の `M2` とは `0.035` 異なる。
  2. **3PL/4PL での `dP("dich")` の不具合**: `dich` の `mirt:::dP` は `(a1, d)` に $(u-g)\sigma(1-\sigma)$
     ではなく 2PL の微分 $P_0 P_1$ を使い、第 3・4 列にも $((1-\sigma)g(1-g),\, \sigma u(1-u))$ ではなく
     カテゴリ確率 $(P_0, P_1)$ をそのまま入れている（`cbind(Theta * PQ, PQ, P)`）。このため `threepl` の `M2` が
     4.8% ずれる（`64.82` 対 厳密値 `68.11`）。
  3. **GPCM の `dP("gpcm")` 傾き微分の不具合**: `gpcm` の `mirt:::dP` は `a1` の微分で `den = rowSums(num)` で割るのを
     忘れており（`rowSums(Theta[, i] * aknum[[i]])`）、`gpcm_sim` の `M2` が 10.8% ずれる（`5.27` 対 厳密値 `5.90`）。
  4. **多値帰無モデルのランク落ち**: $K_j > 2$ では帰無 $\Delta_2$ の項目 $j$ の $K_j - 1$ 本の切片列が厳密に線形従属
     （$\operatorname{rank}=J < \sum(K_j-1)$）になるが、mirt はランクを確認せず `qr.Q(qr(delta), complete = TRUE)` の
     `ncol(delta)` 列目以降を取り出すため、多値帰無 $M_2^*$ が項目順に依存する。flexpyirt は多値項目を含むとき `tli` と
     `cfi` を `NaN` とする（DESIGN §16.3）。

  一致（通常の EM・SQUAREM とも、全ケースで `df` は完全一致）:
  - **mirt と直接比較（`lsat7_2pl`, `sim_2pl`, `grm_sim`）**: `M2` 相対 $\le 3.8\times10^{-11}$、
    `p` $\le 7.8\times10^{-11}$、`RMSEA` $\le 4.4\times10^{-13}$、信頼区間端点 $\le 1.4\times10^{-8}$、
    `SRMSR` $\le 1.4\times10^{-12}$、2 値の `TLI` $\le 4.8\times10^{-10}$、`CFI` $\le 2.4\times10^{-10}$。
  - **厳密な R の $M_2^*$（`threepl`, `gpcm_sim`、`expected.item()` の中心差分微分 + mirt の `buildXi2els`）**:
    `M2` 相対 $\le 1.6\times10^{-9}$、`p` $\le 7.1\times10^{-10}$、`RMSEA` $\le 3.7\times10^{-11}$、
    信頼区間端点 $\le 6.0\times10^{-9}$、`threepl` の `TLI` `1.8e-10` / `CFI` `0`、`SRMSR` は mirt と直接比較で
    $\le 8.5\times10^{-13}$。
  テスト: `tests/test_mirt_m5_m2.py`（mirt 照合）、`tests/test_m5_m2.py`（単体テスト）。
- **Davidian 曲線潜在分布（V19、M5）**: `mirt` 1.46.1（`Theta_DC_den`, `Mstep.DC.optim`, `EM.group`,
  `LoadGroupPars`, `updatePrior`）および `dcurver` 0.9.3 の調査から、厳密な離散化 MMLE との 4 つの相違を確認した
  （DESIGN §16.4）:
  1. 組み込みの `mirt(..., dentype = "Davidian-k")` は毎 EM 反復で期待度数に Woods の線形補間
     `standardizeQuadrature` を適用してから未正規化の `dcurver::ddc` を `nlminb` で最適化するため、EM の単調性が
     崩れ（mirt は `"Davidian"` で単調性検査を無効化）、周辺対数尤度が真に低い点で終わる（V19a の 2PL $k=3$ で
     `21.47` 点、V19b の GRM $k=2$ で `139.32` 点低い）。V19a の終了点は mirt 自身の離散化尤度
     （$\sum\log\sum_q L_q\,\mathrm{ddc}(\theta_q)/\sum_q\mathrm{ddc}(\theta_q)$）に対しても停留点ではない: `PHI_1..3` の
     スコアは $(-0.54, -3.40, 56.7)$ で、mirt の項目パラメタのまま `PHI` だけを最大化すると `4.67` 改善する。V19b では
     mirt は停留点 $N(0,1)$（`PHI_1 = 1.5707963`、スコア $(3.6\times10^{-4}, 0)$、`PHI` の再最適化で改善なし）に
     停滞する: `PHI_2` のスコアは正弦優先座標の $\phi_1=\pi/2$ の極特異点のために 0 になり、`PHI_1` のスコアはその 1 次の
     方向が純粋な位置・尺度の変化で自由な項目パラメタに吸収されるために 0 になる（後者はどの座標系でも生じる。
     flexpyirt は摂動つき既定初期値で $\boldsymbol\phi=\mathbf 0$ を離れる。DESIGN §16.4）。
  2. 組み込みの `dentype = "Davidian-k"` は `SE = TRUE`（`"No standard error method currently supported
     for Davidian curves"`）と `MEAN_1` / `COV_11` の自由推定（`"Estimating Davidian mean-variance
     hyper-parameters is not currently supported"`）を禁止している（いずれも `mirt:::EM.group`）。
  3. `fscores()` は `use_dentype_estimate = TRUE` を明示しない限り $N(0,1)$ 事前分布を使う（`EAP` と plausible
     values のみ対応し `MAP` は拒否）。
  4. `createGroup()`（格子上で正規化した `dcurver::ddc` を呼び、`TOL = 1e-10`, `accelerate = "none"`,
     `SE.type = "Oakes"`）経由で厳密な標準化 Davidian MMLE を mirt 内で検証する際、`createGroup` 既定の
     `derivType = "Richardson"` は粗い 2 階差分幅（`mirt:::numerical_deriv` が `delta * 1000 = 0.01`、対角で
     `2*delta = 0.02` を用いる）により $k=3$ の Oakes SE を最大 25.6% 歪めるため `derivType = "central"`
     （`delta = 1e-5`）を用いた。また `mirt:::BL.LL` は `dentype == "custom"` で `pars2` ではなく `pars` を参照する
     （潜在密度が摂動されない）不具合があるため、Richardson Hessian SE は R の厳密な周辺対数尤度から直接算出した
     （`h = 1e-4`）。

  **V19 参照値が検証するもの。** `exact` ブロックは DESIGN §16.4 の標準化 Davidian MMLE を R で書き直したもので、
  座標系・閉形式モーメント $(\mu_Z,\sigma_Z^2)$・標準化 $z_q=\mu_Z+\sigma_Z(\theta_q-\mu)/\sigma$ は flexpyirt と同じ定義
  （R で別途記述）であり、密度値は `dcurver::ddc`、E-step・EM・Oakes SE・EAP は `mirt`（`createGroup()`）、Hessian SE は
  別途記述した R の周辺対数尤度による。したがってこのモデルに対する flexpyirt の推定機構（E-step・潜在 M-step・収束・
  Oakes/Hessian SE・スコアリング）と、`dcurver` に対する密度を検証する。閉形式モーメントは独立に検査する: R スクリプトは
  各推定解と組み込み解で $z\,\mathrm{ddc}(z)$ と $z^2\,\mathrm{ddc}(z)$ を `stats::integrate` で積分し（最大絶対誤差
  `1.2e-9`、`moments_check` に保存。誤差が `1e-8` を超えれば停止）、`tests/test_m5_davidian.py` は台形則で検査する。
  `mirt_builtin` ブロックは mirt のパラメタにおける密度・`Prior`・`logLik`・EAP を mirt 自身の Davidian 実装と照合する。
  **参照値の収束。** mirt の EM（BFGS M-step、`TOL = 1e-10`）は厳密な周辺対数尤度の勾配が最大 `1.7e-3`（V19c。V19a/b
  は `3.8e-5`・`5.8e-5`、`max_grad_r_mirt_em` に保存）の点で停止する。そこで R スクリプトは mirt の解を R の厳密な
  周辺対数尤度上の Newton 法（中心差分の勾配と Hessian）で max|grad| < `2e-5` まで磨き（各ケース 1 ステップ、最終
  `1.8e-6`〜`3.6e-6`、`max_grad_r`）、磨いた値で `TOL = NaN`（EM 反復なし）として `mirt` を再実行し、その点での
  Oakes SE・`logLik`・EAP を得る。この仕上げの前は V19c の項目・潜在パラメタの差は `8.5e-6`・`7.3e-6` で、flexpyirt の
  対数尤度が参照値より `1.8e-8` 高かった。すなわち残差は参照値側にあり、flexpyirt 側ではなかった。

  歪んだ標準化 $\chi^2_4$ 特性の 3 シミュレーションケース（$N = 3000$）を `tests/test_mirt_m5_davidian.py` で照合する
  （通常 EM・SQUAREM とも。参照値生成: `validation/r/m5_davidian_reference.R` → `tests/reference/v19_davidian_*.json`。
  値は通常 EM と SQUAREM の最大）:
  - **V19a（`v19_davidian_2pl_k3`、2PL、$J=20, k=3$、61 点 $[-6,6]$）**: 厳密 MMLE 対 R `mirt(customGroup)` / R
    Hessian: `loglik` 相対 `2.3e-16`、項目パラメタ `5.7e-8`、`phi_1..3` `1.3e-8`、Oakes SE 対 mirt Oakes `4.6e-6`
    （項目 `6.3e-7`）、Hessian SE 対 R Richardson `2.8e-6`（潜在 `3.8e-7`）、flexpyirt Oakes 対 Hessian `9.7e-16`、
    EAP `6.2e-15`（SQUAREM で EM 評価回数 `113 → 39`）。組み込み `mirt(..., dentype="Davidian-3")` のパラメタ
    （`mean = mu_z`, `var = var_z`）では `Prior`・`logLik`（相対 `1.4e-15`）・`fscores(..., use_dentype_estimate=TRUE)`
    （`8.2e-15`）が完全一致し、厳密 MMLE は対数尤度を `+21.47` 改善する。
  - **V19b（`v19_davidian_grm_k2`、GRM $K=4$、$J=10, k=2$、61 点 $[-6,6]$）**: 厳密 MMLE: `loglik` 相対 `1.9e-16`、
    項目パラメタ `4.9e-8`、`phi_1..2` `9.1e-9`、Oakes SE 対 mirt Oakes `3.6e-7`、Hessian SE 対 R Richardson `1.9e-6`、
    flexpyirt Oakes 対 Hessian `1.1e-15`、EAP `6.0e-15`（SQUAREM `111 → 33`）。組み込み `Davidian-2` のパラメタでは
    `logLik`（相対 `1.9e-16`）・EAP（`6.7e-15`）が完全一致し、厳密 MMLE は対数尤度を `+139.32` 改善する。
  - **V19c（`v19_davidian_fipc_k3`、FIPC 2PL、`q01–q10` 固定、$J=20, k=3$、`mean`/`var` 自由、移動・拡大した歪み特性
    $0.4 + 1.2\cdot\text{skewed}$、81 点 $[-8,8]$）**: 厳密 MMLE: `loglik` 相対 `1.6e-15`、項目パラメタ `7.5e-8`、
    潜在パラメタ（`mean, var, phi_1..3`）`1.4e-7`、Oakes SE 対 mirt Oakes `6.0e-6`、Hessian SE 対 R Richardson
    `1.8e-6`、flexpyirt Oakes 対 Hessian `2.9e-15`、EAP `5.3e-15`（SQUAREM `107 → 33`）。
  残る `~1e-6` の SE の差は mirt と R の有限差分微分（`delta = 1e-5`、`h = 1e-4`）によるもので、flexpyirt の SE は
  厳密な自動微分による。許容誤差は上の表のとおり（パラメタ `1e-4`、SE 相対 `1e-3`）。
  単体テスト（単位球面・モーメント・ゼロ角での $N(0,1)$ 一致・多項式零点の自動微分ガードと求積点上に零点を持つ推定・
  初期角の符号・`degree` 1 と 10 の推定・Rasch/FIPC・識別性と仕様エラー・スコアリング・JSON 往復と不正な
  `DavidianCurve` JSON）: `tests/test_m5_davidian.py`。

## 照合以外の検証
- **パラメタ回復シミュレーション**: 条件（N, J, 群数, アンカー数）× 反復 100 で bias / RMSE を出し、`validation/recovery/` に結果と図を置く。
- **EM の単調性**: すべての照合ケースで、反復ごとの（MAP 時は事後）対数尤度が非減少であることをテストで確認。
- **SE の内部整合**: Oakes の SE と、収束点での周辺対数尤度の直接 `jax.hessian` による SE が一致すること（小規模ケース）。
- **不変性**: 項目の並べ替え、群ラベルの付け替え、パターン圧縮の有無で結果が変わらないこと。
