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
| 尺度リンキング $(A, B)$ / 目的関数（同一の項目推定値） | 1e-9 / 1e-9（相対） |
| 尺度リンキングのデルタ法 SE / `vcov`（同一の推定値・共分散） | 1e-5（相対） |
| 尺度リンキング $(A, B)$ / 目的関数 / SE（推定 + リンキングの一連） | 1e-6 / 1e-5（相対） / 1e-3（相対） |
| ポリコリック相関（lavaan / psych と）、閾値、平滑化（`cor.smooth` と）。2 つの実装とも | 1e-6 / 1e-3、1e-12、1e-12 |
| V24 のケースでの flexpyctt（NumPy）と flexpyirt（JAX）のポリコリック相関、Owen の $T$ による $\Phi_2$ と Plackett の積分 | 1e-9（フラグは一致）、1e-14 |
| 同じ行列での EKC・MAP・観測固有値、正規の参照の平均 / 95 パーセンタイル | 1e-11、$\lvert z\rvert\le4.5$ / 6e-3 |
| LD $X^2$ / $G^2$、符号付き $V$・Q3（推定し直したモデル）、モデルに基づく平行分析の観測値 / シミュレーションの平均 / 95 パーセンタイル | 1e-5（相対）、1e-6、2e-4 / $\lvert z\rvert\le4.5$ / 0.02 |
| DETECT の条件付き共分散、指標（両方の半分）、Ward の併合の高さ | 1e-12 |
| EFA の含意する $\Lambda\Lambda^\top$ / 共通性、CAF・CFI・RMSEA、$\chi^2$、Hull の適合度 / $st$ | 1e-4、2e-4、1e-4（相対）、1e-5 / 1e-2（相対） |
| CD: 主因子法の $\Lambda\Lambda^\top$、同じ乱数による 1 因子の母集団、順位和の $p$、平均 RMSE（採用された $k$ / 棄却された追加の $k$） | 1e-12、厳密、1e-10（相対）、15% / 60%（相対） |
| DIMTEST（シミュレーション研究。テストではその小さい版）: .05 での第 1 種の誤り / 検出力 | 20 回中 $\le 0.15$ / 10 回中 $\ge 8$ |

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
| V15 | S-X2 項目適合度（GRM・GPCM・分散自由の PCM・3PL・FIPC の 2PL、`mincell` 1 と 3、2 群の 2PL と GRM を群ごと） | `itemfit(m, fit_stats = "S_X2")` | M5 |
| V16 | 項目をまたぐ等値制約（LSAT7 の共通傾き、2PL + graded のモデル間制約、アンカーつき 2 群 + 共通傾き） | `mirt(..., mirt.model("... CONSTRAIN = (1-5, a1)"))` / `multipleGroup(...)` | M5 |
| V17 | V4 のデータでの尤度比検定による DIF（`a`+`d` と `d` の `add`、完全不変モデルからの `a`+`d` の `drop`） | `DIF(mg, which.par = ..., scheme = ...)` | M5 |
| V18 | M2 / M2* 限定情報の適合度（LSAT7 2PL、シミュレーション 2PL、3PL、分散自由の Rasch、GRM $K=3$、GPCM $K=4$） | `M2(m, quadpts = 61, theta_lim = c(-6, 6))` | M5 |
| V14 | 確認的 2–3 次元 MIRT（2PL 単純構造・交差負荷・GRM・3 次元 2PL） | `mirt(..., mirt.model("F1 = ...\nF2 = ...\nCOV = F1*F2"), SE = TRUE, SE.type = "Oakes")` | M4 |
| V19 | Davidian 曲線潜在分布（2PL $k=3$・GRM $k=2$・平均分散自由の FIPC 2PL $k=3$） | `mirt(..., dentype = "Davidian-k")` および `mirt(..., customGroup = ..., SE.type = "Oakes")` | M5 |
| V20 | 尺度リンキング: Haebara・Stocking–Lord（非対称・対称）と Ogasawara SE（2PL・GRM・GPCM・3PL・非標準の新フォーム格子・2PL + GPCM 混在） | 個別 `mirt()` 推定 + `probtrace()` / `expected.item()` による素の R 実装 | M5 |
| V21 | 多母集団の確認的 2 次元 MIRT（単純構造・交差負荷。両因子にアンカー、焦点群の平均・分散を自由） | `COV = F1*F2` での `multipleGroup(..., invariance = c(anchors, "free_means", "free_var"))` | M4 |
| V22 | 混合正規潜在分布（2PL $k=2$ の二峰、小さい成分を持つ GRM $k=2$、GPCM $k=3$） | 像の格子での `multipleGroup(..., dentype = "mixture-k", invariance = c(<全項目>, "free_means", "free_var"))` と、標準化した尤度の R による独立な実装（Newton による仕上げ、Richardson ヘッセ行列 SE、EAP） | M5 |
| V23 | 名義反応モデル（Science、順序のない・負の採点係数を含むシミュレーション、名義 + 2PL の混在、大半が非単調な項目、2 因子の確認的モデル） | GPCM の点（またはより良い別の初期値）から開始する `pars` を与えた `mirt(..., "nominal")`、`itemfit(fit_stats = "S_X2")` | M5 |
| V24 | ポリコリック・テトラコリック相関（シミュレーションのテトラコリック、2〜5 カテゴリの混在、10% MCAR、空のセルが 1 つの 2 x 2 表、LSAT7、Science）と平滑化 | lavaan の `lavCor(..., ordered = , missing = )`、psych の `polychoric(..., smooth = FALSE, correct = 0)`、`cor.smooth()` | M5 |
| V25 | 次元数: V24 の行列での EKC、MAP（2 乗と 4 乗）、PCA・SMC の固有値。正規データの平行分析の参照（3 つの大きさ） | EFAtools の `EKC()`、`PARALLEL()`、psych の `vss()$map`、EFA.dimensions の `MAP()`、4 乗版 MAP の R による独立な実装 | M5 |
| V26 | 局所依存（LSAT7 の 2PL、Science の GRM、依存するペアと 5% MCAR を含むシミュレーションの 2PL）とモデルに基づく平行分析（LSAT7） | `residuals(m, type = "LD" / "LDG2" / "Q3")`、ltm の `unidimTest(data = , thetas = cbind(d, a1), IRT = FALSE)` | M5 |
| V26（M7） | 2 群（4% MCAR を含む 2PL、GRM、2 因子の 2PL）、`EmpiricalHistogram`、`DavidianCurve(2)`、`NormalMixture(2)`（V22a の推定値）、2 因子（交差負荷と 3% MCAR を含む 2PL、GRM）、3 因子（2PL）の局所依存 | `multipleGroup()`、`dentype = "EH"` / `"Davidian-2"`（`use_dentype_estimate = TRUE`。Q3 は `Theta = fscores(m, use_dentype_estimate = TRUE)`）、`mirt.model()` の推定への `residuals()`。混合分布は、混合分布の重みを事前分布に入れた単一群の mirt オブジェクト | M7 |
| V27 | DETECT: 確認的（シミュレーションの 2 次元で正しい・誤った分割、1 次元、Science）と探索的（3 クラスタのシミュレーション、分割を指定） | sirt の `conf.detect(use_sum_score = TRUE, smooth = FALSE)`、`expl.detect(use_sum_score = TRUE, estsample = )` | M5 |
| V28 | EFA（ULS、ML。Hull の上限までのすべての $k$）と Hull 法（CAF、CFI、RMSEA）。V24 の行列のうち変数が 6 つ以上のものとシミュレーションの行列 2 つ | EFAtools の `EFA(method = , rotation = "none")`、`HULL(method = , gof = )`、psych の `fa(fm = "minres" / "ml", rotate = "none")` | M5 |
| V29 | 比較データ法: 主因子法、書き出した乱数による 1 因子の母集団の生成、順位和検定、3 因子の連続・2 因子の順序・1 因子の 2 値のデータでの `CD()` | EFAtools の `.paf_iter`、`.simulate_cfm_empirical`、`CD()`、`wilcox.test(x, y, "less")` | M5 |
| V30 | DIMTEST: 第 1 種の誤りと検出力（2PL。$N$ 1000〜2000、$J$ 20〜30。1 次元と相関 0.7〜0.8 の 2 次元。各 200 反復） | なし（DIMPACK 以外に実装がない）。`packages/flexpyctt/validation/simulation/dimtest_study.py` | M5 |
| V31 | ジャックナイフ傾き指標（LSAT7 の 2PL、Science の GRM、依存するペア 2 組と 5% の MCAR を含むシミュレーションの 2PL、依存するペア 1 組を含むシミュレーションの GPCM）。Newton 法と 1 ステップの近似は推定し直した値と比べる | `residuals(m, type = "JSI", fold = FALSE, TOL = 1e-10)` | M5 |
| V32 | 次元縮約による双因子モデル（特殊因子 3 つと一般因子のみの項目 3 つを含む 2PL、テストレット 3 つの GRM $K=4$） | `bfactor(data, model, itemtype = , SE = TRUE, SE.type = "Oakes", quadpts = 31)`; `fscores(method = "EAP", quadpts = 15)` | M7 |
| V33 | 次元縮約による two-tier モデル（相関する主因子 2 つと特殊因子 3 つ（1 つは両方の主因子にまたがる）の 2PL、交差負荷項目 1 つとテストレット 2 つの GRM $K=4$） | `bfactor(data, model, model2 = mirt.model("F1 = ...\nF2 = ...\nCOV = F1*F2"), SE = TRUE, SE.type = "Oakes", quadpts = 15)`; `fscores(method = "EAP")` | M7 |
| V36 | 合計得点分布のカーネル（Poisson 二項分布: 確率ベクトル 5 例、$J$ 10〜300。多値: GRM $K=4$ と 2PL + GRM の混合） | `poibin::ppoibin(method = "RF" / "DFT-CF" / "RNA" / "NA")`、`poibin::dpoibin()`、`stats::dpois()`。mirt の `fscores(method = "EAPsum", full.scores = FALSE)` | M7 |

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
  - **V7b（`v7_2pl_lnorm_map`）**: 2PL、$a\sim\mathrm{LogNormalPrior}(0.2, 0.3)$、$d\sim\mathrm{NormalPrior}(0, 2)$。
    `loglik`（`4.0e-13`）、`log_posterior`（`1.4e-14`）、パラメタ（`2.6e-9`）、EAP（`7.1e-15`）は mirt（`NR`）と直接照合し、
    Oakes / Hessian SE は R で求めた厳密な事後 Hessian と照合（`1.8e-9`、Oakes 対 Hessian `2.3e-12`）。
  - **V7c（`v7_3pl_map`）**: 3PL、$a\sim\mathrm{LogNormalPrior}(0.2, 0.3)$、$d\sim\mathrm{NormalPrior}(0, 2)$、
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
- **多母集団の 2 次元 MIRT（V21、M4）**: シミュレーション 2 ケース（1,500 人 × 2 群。アンカー q01–q03 / q07–q09 の単純構造と、
  q07 が交差負荷しアンカーが q01–q03 / q08–q10 のもの）。焦点群の平均・分散は自由、相関は両群で自由、$[-6, 6]$ 上の次元あたり
  81 点（`validation/r/m4_mirt_multigroup_reference.R` → `tests/reference/v21_mirt_multigroup.json`、mirt 1.48、`NR`、
  `TOL = 1e-10`）。mirt は 1 つの因子の尾根（その因子の焦点群の平均・分散とその傾きの交換）に沿って最大点の手前で止まる:
  flexpyirt は mirt の推定値での mirt の対数尤度を正確に再現するが、自身の解はそれより 3.2e-5 高く、mirt の推定値でのスコアは
  その因子の焦点群の分散 / 平均で約 0.035 / 0.018（他は 1e-3 未満）、flexpyirt の解では 1e-6 未満。この差は次元あたり 31 点から
  81 点にしても縮まらなかった（issue #20 の 13）。そのため、その因子のブロック（その因子の項目の傾き・切片と焦点群の平均・分散）
  以外のパラメタと SE は通常の許容誤差（`1e-4`、SE 相対 `1e-3`）で比べ、ブロックは対数尤度（flexpyirt の方が高い）と停留性で
  確認する。Oakes と直接 Hessian の一致、EM と SQUAREM の単調性、スコアリングと JSON の往復保存も確認する。テスト:
  `tests/test_mirt_m4_multigroup.py`。
- **S-X2 項目適合度（V15、M5）**: 既存の参照値を再利用した 6 ケース（Science の V3 GRM と V8 GPCM、潜在分散が自由な
  V8 PCM、V11 3PL、8 項目を固定した V5 FIPC 2PL、および `mincell = 3` の V3）。mirt は参照値の推定値で、各パラメタの
  `est` フラグを保ったまま（自由パラメタを自由度から引くため）`TOL = NaN` とし、推定と同じ格子で
  `itemfit(m, fit_stats = "S_X2")` を計算する（mirt 1.48、`validation/r/m5_itemfit_reference.R` →
  `tests/reference/v15_sx2.json`）。flexpyirt はデータを推定し直す: 自由度はすべてのケースで一致し、S-X2 は相対 `2.2e-8`、
  p 値は `1.3e-8` で一致（推定値自体の一致の水準）。単体テストで Lord–Wingersky 漸化式を全パターン列挙と比べ、期待度数の行が
  条件付き確率であることを確認する。テスト: `tests/test_mirt_m5_itemfit.py`。多母集団（DESIGN §15.45）: mirt の群ごとの
  `itemfit()` を、V4 の 2 群 2PL では mirt の推定値で（`multipleGroup(TOL = NaN)` でも EM が 1 ステップ進むので、すべての
  パラメタを固定して推定してから各 `est` フラグを戻す）、Science（GRM）を行の偶奇で 2 群に分けすべての項目を等値・焦点群の
  平均と分散を自由にしたものでも計算する（`validation/r/m5_itemfit_multigroup_reference.R` →
  `tests/reference/v15_sx2_multigroup.json`）。Science では mirt の項目パラメタが約 2e-4 収束しきらない（flexpyirt の
  対数尤度のほうが高い）ので、flexpyirt を mirt の推定値で評価する。両ケースの両群で自由度は一致し、S-X2 は相対 `1e-6`
  以内。テスト: `tests/test_mirt_m5_itemfit_multigroup.py`。
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
- **微分可能な IRT 尺度リンキング（V20、M5）**: `mirt` は特性曲線法リンキングを持たないため、
  `validation/r/m5_linking_reference.R`（→ `tests/reference/v20_linking.json`）では共通項目を含む 2 つの
  フォームをそれぞれ `mirt` で推定し（`optimizer = "NR"`、`accelerate = "none"`、`TOL = 1e-10`、$N(0, 1)$、
  $[-6, 6]$ の等間隔 61 点、`SE.type = "Oakes"`。基準群 $\theta\sim N(0, 1)$、新群 $\theta\sim N(0.5, 1.2^2)$、
  真値 $A = 1.2, B = 0.5$。V20e の新フォームは固定した $N(0.3, 1.5)$ の下で推定し、対称版の逆方向の項が非標準の
  新フォーム格子を使う）、`mirt::probtrace()` / `mirt::expected.item()` と論文の定義式
  （Haebara, 1980; Stocking & Lord, 1983; Kim & Lee, 2006）に基づく独立した素の R 実装（`optim(BFGS)` + `nlm`
  と中心有限差分 Newton 仕上げ、`max |grad| <= 1.9e-11`）で参照値 $(A, B)$、目的関数値、および Ogasawara (2001) の
  デルタ法 SE を算出した。Haebara 法は 2 値項目の両カテゴリを含む全カテゴリで和をとる（Kim & Lee, 2006）。
  6 ケース × 4 方式（Stocking–Lord・Haebara の非対称・対称）について次の 3 通りで検証した:
  (i) mirt の項目推定値をそのまま入力した直接比較（推定誤差とリンキング誤差を分離）、(ii) mirt 自身の共通項目推定値と
  Oakes 共分散ブロックからのデルタ法 SE（SE の式を推定・共分散の差から分離）、(iii) flexpyirt で両フォームを推定して
  リンキングする一連の比較。さらに全ケース・全方式で Newton 法を 4 つの遠い初期値
  （$(A, B) = (0.25, -3), (4, 3), (0.5, 2.5), (3, -2)$）から走らせ、1e-9 以内で参照解に達することを確かめる。
  外部の R リンキングパッケージ（`plink`、`equateIRT` など）は使っていない。目的関数と SE は同じ公刊式の独立実装と
  照合しており、第三者ソフトウェアとは照合していない:
  - **V20a（`v20a_2pl`、2PL、各フォーム $J=20$、共通 10 項目、各群 $N=2000$）**: 直接比較 `max |Δ(A, B)| = 9.6e-13`、
    目的関数相対 `1.6e-13`、mirt の共分散での SE 相対 `2.2e-8`。一連の比較 `max |Δ(A, B)| = 1.9e-10`、目的関数相対
    `2.6e-9`、Ogasawara SE 相対 `2.2e-8`。
  - **V20b（`v20b_grm`、GRM $K=4$、各フォーム $J=16$、共通 8 項目、各群 $N=2000$）**: 直接比較 `max |Δ(A, B)| = 9.1e-13`、
    目的関数相対 `1.0e-13`、mirt の共分散での SE 相対 `8.3e-9`。一連の比較 `max |Δ(A, B)| = 9.5e-10`、目的関数相対
    `2.2e-9`、Ogasawara SE 相対 `8.1e-9`。
  - **V20c（`v20c_gpcm`、GPCM $K=4$、各フォーム $J=16$、共通 8 項目、各群 $N=2000$）**: 直接比較 `max |Δ(A, B)| = 2.5e-12`、
    目的関数相対 `9.5e-14`、mirt の共分散での SE 相対 `1.6e-8`。一連の比較 `max |Δ(A, B)| = 1.4e-9`、目的関数相対
    `7.1e-10`、Ogasawara SE 相対 `1.6e-8`。
  - **V20d（`v20d_3pl`、3PL、各フォーム $J=16$、共通 8 項目、各群 $N=4000$、内点 $g$）**: 直接比較
    `max |Δ(A, B)| = 3.6e-12`、目的関数相対 `1.2e-13`、mirt の共分散での SE 相対 `9.4e-8`。一連の比較
    `max |Δ(A, B)| = 3.0e-10`、目的関数相対 `5.2e-9`、Ogasawara SE 相対 `9.4e-8`。
  - **V20e（`v20e_2pl_srcgrid`、V20a のデータ、新フォームは固定 $N(0.3, 1.5)$ の下で推定）**: 直接比較
    `max |Δ(A, B)| = 9.1e-13`、目的関数相対 `1.4e-13`、mirt の共分散での SE 相対 `3.4e-9`。一連の比較
    `max |Δ(A, B)| = 4.2e-10`、目的関数相対 `7.0e-10`、Ogasawara SE 相対 `3.4e-9`。
  - **V20f（`v20f_mixed`、2PL + GPCM $K=4$ 混在、各フォーム $J=16$、共通 5 + 3 項目、各群 $N=2000$）**: 直接比較
    `max |Δ(A, B)| = 2.0e-11`、目的関数相対 `6.2e-14`、mirt の共分散での SE 相対 `2.1e-8`。一連の比較
    `max |Δ(A, B)| = 1.3e-10`、目的関数相対 `3.6e-10`、Ogasawara SE 相対 `2.1e-8`。
  したがってリンキングで照合済みの項目モデルは 2PL・3PL・GRM・GPCM と 2PL + GPCM 混在である。Rasch・4PL・PCM は
  同じカーネルを共有し、厳密回復の単体テストでのみ確認している。
  照合テスト: `tests/test_mirt_m5_linking.py`。単体テスト（全 7 種の 1 次元モデル、ICC・閾値の不変性、恒等変換と厳密回復、
  対称版の可逆性、独自格子、EH・多母集団、1 つの多母集団推定を `source` と `target` の両方に使うときの同時共分散、
  非収束時に SE を出さないこと、Rasch・PCM の傾き $1/A$、`transform_theta`、勾配の有限差分検証、Ogasawara SE の
  中心有限差分検証、および $N(\hat\eta, \Sigma)$ からのパラメタ抽出ごとに重み固定でリンキングを解き直すモンテカルロ
  線形化検証（推定のブートストラップではない））: `tests/test_m5_linking.py`。

- **混合正規潜在分布（V22、M5）**: flexpyirt の `NormalMixture`（DESIGN §16.7）は形状に mirt のクラスのパラメタ化を
  使い、混合を解析的に標準化する。したがって格子 $\theta_q$ 上では、像の格子 $z_q=M+\sqrt V\theta_q$ 上の mirt の
  `multipleGroup(dat, 1, itemtype = ..., dentype = "mixture-k", invariance = c(<全項目>, "free_means", "free_var"))`
  （項目は 1 組、クラス 1 は $N(0,1)$ に固定、クラス $2..k$ の `MEAN_1`・`COV_11`・`PI` が自由）に厳密に一致し、
  $a=a^{\text{mirt}}\sqrt V$、$d_k=d_k^{\text{mirt}}+s_k a^{\text{mirt}}M$（2PL/GRM では $s_k=1$、GPCM では $k$）である。
  そこで `validation/r/m5_mixture_reference.R`（→ `tests/reference/v22_mixture_*.json`）はケースごとに次を記録する:
  1. `exact`: flexpyirt の離散化尤度を R で独立に書き直したもの（2PL・GRM・GPCM のカテゴリ確率、成分ごとに正規化した
     重み、標準化）を、$[-6,6]$ 上の mirt の解から Newton 法（中心差分の勾配、Richardson ヘッセ行列）で
     max|grad| < 2e-5（最終 1.8e-6–3.6e-6）まで最大化した点と、Richardson ヘッセ行列 SE、EAP。
  2. `mirt`: 厳密な最大点の像の格子（非対称な `theta_lim`。V22a では `c(-7.76, 9.50)`、`optimizer = "NR"`、
     `TOL = 1e-10`）で組み込みの mirt を推定し直し、推定値と `fscores()` の EAP を flexpyirt の尺度へ写したもの。R での
     mirt 尺度の尤度は mirt の `logLik` を 1e-12 で（mirt の推定値はその停留点で max|grad| <= 3.6e-6）、
     `fscores()` の EAP を 1e-8 で再現する。
  mirt 1.46.1 について分かったこと:
  - `dentype = "mixture-k"` は既定でクラスごとの項目パラメタを推定する（混合 IRT モデル）。項目共通のモデルには
    `invariance = <全項目>` が要る（`ESTIMATION` から無害な再利用の警告が出る）。`multipleGroup` の第 3 位置引数は
    `group` なので `itemtype` は名前付きで渡す必要がある。
  - **mirt の混合モデルの Oakes SE は観測情報の SE ではない。** mirt 自身の尺度と格子で、mirt の離散化尤度の R
    Richardson ヘッセ行列による潜在分布パラメタ（`MEAN_1`・`COV_11`・`PI`）の SE は V22a で (0.159, 0.139, 0.135) だが、
    `SE.type = "Oakes"` は (0.0315, 0.0048, 0.0401) で、3 ケースを通じた比は 0.014–0.34。`SE.type = "complete"` は
    (0.0262, 0.0042, 0.0380) で、1/sqrt(N pi_1 pi_2) = 0.0380 はクラス既知のときの `PI` の完全データ SE である。項目の SE
    も小さすぎる（V22a の q01 `a1`: 0.0539 対 0.0898）。混合モデルでは他の観測情報 SE の種類は拒否される。そこで SE の
    基準は R の Richardson ヘッセ行列とし、mirt の尺度での Oakes・complete・R ヘッセ行列の SE を証拠として保存
    （`mirt.se_*_mirt_scale`）、flexpyirt の潜在分布パラメタの SE が mirt の尺度での R ヘッセ行列の SE と一致し
    （素の形状の座標は共通）、mirt の Oakes SE がその 0.6 倍未満であることをテストで確認する。
  - 混合モデルでは `fscores(method = "MAP")` は拒否される（flexpyirt も MAP を拒否する）。EAP の列名は
    `Class_1`, `SE_F1`。
  - クラスで拡大した mirt の EM は遅い（生成値から `TOL = 1e-10` で 3,092–4,555 (61 points, V22a/b) and 24,961 (121 points, V22c) 回の EM サイクル）。mirt のパラメタ化では
    flexpyirt の（節点レベルの）EM も、V22a で 4 通り中 2 通り、V22b で 4 通り中 3 通りの初期形状から劣った点
    （対数尤度で 0.73–53.2 低い）に止まった。標準化すると、すべての初期値から通常の EM で 168–661 反復（SQUAREM で
    27–93 反復）で最大値に達した（DESIGN §16.7）。
  - **格子の解像度（V22c）**: 成分が細く（標準化後 $\sigma_c\approx0.26$）3 カテゴリの項目が 20 あると、事後分布の幅が
    $[-6,6]$ 上 61 点の間隔 0.2 と同程度になる。61 点での最大値は 121 点・241 点（互いに 1e-9 で一致）と対数尤度で
    5.7e-3 異なり、mirt の固定格子と flexpyirt の動く標準化格子の最大値は対数尤度で 2.6e-4、パラメタで最大 6e-3 離れた。
    そこで V22c は $[-6,6]$ 上 121 点を使う（mirt は不動点が同じ `accelerate = "squarem"` を使ったが、それでも 24,961 回の EM
    サイクルを要した）。
  結果（通常の EM と SQUAREM の最大値。flexpyirt は既定の初期形状から `BAEM(tol=1e-9)` で推定。成分のラベルに
  意味はないため標準化した成分は平均で並べて比較。SE の検査は mirt のラベル付けの初期値から推定（尤度が同じゲージの
  選択））:
  - **V22a（`v22_mixture_2pl_k2`、2PL、$J=20$、$N=3000$、$0.65N(0,1)+0.35N(2.5,0.5^2)$、61 点）**: `loglik` 相対
    `4.5e-16`（exact・mirt とも）、項目パラメタ `3.5e-8` / `4.3e-8`、標準化した成分 `5.4e-8` / `5.5e-8`、mirt の
    `MEAN_1, COV_11, PI` に対する素の形状 `1.2e-8`。R ヘッセ行列に対する項目 SE 相対 `1.0e-5`（潜在分布 `3.9e-6`）、
    Oakes 対 Hessian `3.6e-14`。exact のパラメタでの EAP `4.7e-15`、mirt に対し `4.5e-9`（SE `3.1e-9`）。通常の EM 734 反復、
    SQUAREM 54 反復。
  - **V22b（`v22_mixture_grm_k2`、GRM $K=4$、$J=12$、$N=2500$、$0.8N(0,1)+0.2N(-2.2,0.4^2)$、61 点）**: `loglik` 相対
    `8.8e-16`、項目パラメタ `3.3e-8` / `1.7e-9`、成分 `9.7e-9` / `7.8e-9`、mirt に対する素の形状 `9.3e-9`。R ヘッセ行列に
    対する項目 SE 相対 `1.1e-5`（潜在分布 `3.2e-5`）、Oakes 対 Hessian `5.0e-15`。EAP `4.9e-15`、mirt に対し `2.6e-9`。
    通常の EM 227 反復、SQUAREM 34 反復（flexpyirt の既定の初期値は、素の成分 1 が小さい成分になるラベルの付け替えた解に
    収束する。尤度と標準化した成分は同じ）。
  - **V22c（`v22_mixture_gpcm_k3`、GPCM $K=3$、$J=20$、$N=4000$、$0.5N(0,1)+0.3N(2.3,0.45^2)+0.2N(-2.3,0.5^2)$、
    121 点）**: `loglik` 相対 `2.7e-16`、項目パラメタ `2.6e-8 / 2.6e-8`、成分 `9.6e-8 / 9.4e-8`、mirt に対する素の形状 `2.9e-7`。
    R ヘッセ行列に対する項目 SE 相対 `3.2e-5`（潜在分布 `1.9e-4`）、Oakes 対 Hessian `1.8e-13`。EAP `4.9e-15`、
    mirt に対し `1.9e-8 (SE 1.4e-8)`。通常の EM 416 反復、SQUAREM 55 反復。
  許容誤差は上の表のとおり。単体テスト（仕様の検証、素の初期形状、厳密な標準化、等間隔・Gauss–Hermite 格子での重み、
  NaN の防御、EM と SQUAREM の単調性、Oakes == Hessian、ラベルを付け替えた初期値、デルタ法の成分、Rasch/FIPC の
  識別性、入れ子の Normal、退化解の警告、3 成分、採点、拒否するメソッド、リンキングの重み、JSON）:
  `tests/test_m5_mixture.py`。mirt との照合: `tests/test_mirt_m5_mixture.py`。
- **名義反応モデル（V23、M5）**: mirt の `itemtype = "nominal"`（`optimizer = "NR"`、`TOL = 1e-10`、$[-6,6]$ の
  61 点。V23f は次元あたり 31 点）。V23a: `0..3` に符号化し直した `Science`。V23b: シミュレーション
  $J=10, K=4, N=3000$ で、内部の採点係数が入れ替わった項目（`q09`: $\mathrm{ak}_1=2.2 > \mathrm{ak}_2=0.7$）と
  負の採点係数をもつ項目（`q10`: $\mathrm{ak}_1=-0.4$）を含む。V23c: 名義 5 項目（$K=4$）と 2PL 5 項目、$N=2500$。
  V23e: シミュレーション $J=8, K=4, N=3000$ で、$\mathrm{ak}_k<0$ や $\mathrm{ak}_k>K-1$ を含む非単調な項目が 6 つ。
  V23f: 2 因子の確認的名義モデル（$J=8$、因子あたり 4 項目、$K=4$、$N=3000$、相関は自由推定）。
  **初期値**: mirt の名義モデルの既定初期値（`ak1 = ... = 0.5`、`d = 0`）では減衰なしの NR が `Science` で鞍点に止まる
  （対数尤度 $-1631.79$、Comfort の `a1` $\approx 0$。最尤解は $-1608.452$）ため、スクリプトは mirt の NR を GPCM の点
  （$\mathrm{ak}_k=k$、$d_k=\log(p_k/p_0)$）から開始する。さらに 1 次元の各ケースで mirt の別の初期値 5 通り（既定初期値の
  NR と BFGS、`GenRandomPars` による NR 3 通り）で推定して `alt_starts` に保存し、いずれかの方が高ければそこから NR を
  やり直す（`meta.reference_start`）。これは V23e で起こり、GPCM の点からの NR は局所最大（$-29561.14$）に止まり、
  既定初期値の BFGS は $-26107.48$ に達する。保存したどの初期値も参照値を上回らず、flexpyirt は常に自身の既定初期値から
  推定してすべての参照値（V23e を含む）に到達する。**一致**（通常の EM・SQUAREM とも。V23a–c, e の最大）: 対数尤度
  $3.6\times10^{-11}$（絶対。相対 $1.6\times10^{-15}$）、パラメタ $7.1\times10^{-8}$、Oakes SE（相対）$3.8\times10^{-8}$、
  Hessian と mirt Richardson SE $5.6\times10^{-7}$、flexpyirt の Oakes と Hessian $1.7\times10^{-13}$、EAP
  $8\times10^{-15}$、各ケース 40–44 パタンの MAP / ML / WLE で $\theta$ $3.7\times10^{-6}$・SE（相対）$1.9\times10^{-4}$、
  S-X2（`itemfit(fit_stats = "S_X2")`）は相対 $5.8\times10^{-9}$ で自由度も一致（Science: $8.235, 10.223, 6.246,
  12.483$、自由度 6, 9, 6, 9）。V23f: 対数尤度 $4.0\times10^{-11}$、パラメタ $7.0\times10^{-9}$、Oakes SE
  $2.5\times10^{-9}$、相関 $5.4\times10^{-12}$（SE は相対 $9.3\times10^{-11}$）、EAP は $10^{-5}$ 以内。**ML の既知の差**:
  mirt の `fscores(method = "ML")` は採点係数にかかわらず全項目カテゴリ 0 / 全項目最上位カテゴリのパタンに
  $-\infty$ / $+\infty$ を返し、V23e の真に $+\infty$ のパタン 1 つ（全項目最上位カテゴリではない）を `max_theta = 20` の
  境界に残す（$19.99989$、SE は NA）。flexpyirt は極端パタンをカテゴリ傾き $a\,\mathrm{ak}_c$ で判定し、テストではその
  判定を R でのパタン対数尤度の独立な走査（`fscores.ml_scan`: $[-20,20]$ 上 0.01 刻みの格子を `optimize()` で精密化）と
  照合し、内点の ML 推定値（V23b・V23c で 1 つ、V23e で 2 つ）をそれと比較する。多次元の名義モデルは 2 因子（V23f）で
  照合済み、3 因子は単体テストのみ。単体テスト（確率・勾配・GPCM / 2PL への帰着・初期値・識別性・`fix_from`・JSON・
  S-X2・スコアリング・事前分布・等値制約・多群と DIF・EH・2 次元・退化項目の `HeywoodWarning`）:
  `tests/test_m5_nominal.py`。テスト: `tests/test_mirt_m5_nominal.py`。参照値生成:
  `validation/r/m5_nominal_reference.R`。
- **ポリコリック相関（V24、M5）**: mirt との照合ではない。順序変数に対する lavaan の `lavCor`（`nlminb` に解析的な
  勾配とヘッセ行列を与えた two-step 最尤法、$|\rho|\le0.999$、既定の `zero.add = c(0.5, 0)`）を厳密な参照値とする。
  lavaan が非収束の警告を出したらスクリプトを止める。psych の `polychoric` は既定の許容誤差の `optimize()` と
  `mnormt::sadmvn` を使うので、1e-3 までの照合とし、欠測のないデータで、空のセルを補正したペアを除いて比べる
  （`global = TRUE` の psych は補正しない）。カテゴリ数が異なると psych はペアごとの閾値を使い、欠測があると結果が
  ずれる（MCAR のケースで 6e-3）ため、そのケースは lavaan とだけ比べる。さらに各ペアの two-step の勾配が推定値で
  0 になることも確かめる。2 変量正規分布の分布関数は Plackett の積分と単体テストで比べる。
- **次元数（V25、M5）**: 決定的な量は V24 の lavaan の行列から R で計算するので、両側が同じ入力を見る。
  EFA.dimensions と EFAtools を両方読み込むと EFA.dimensions の `PARALLEL()` が EFAtools のものを隠すため、
  スクリプトでは `EFAtools::` を明示する。EFA.dimensions の 4 乗版 MAP は $(\operatorname{tr}(R^{*4})-p)/(p(p-1))$
  （行列のべき乗。このデータで 1 を超える）なので、4 乗版は要素ごとの独立な実装と比べ、EFA.dimensions の値は記録
  するだけにする。正規データの参照は両側とも確率的なので、テストは固定のキーを使い、平均は $z$（反復から求めた標準
  誤差、独立な 2 つの平均なので $\sqrt2$ 倍）で、95 パーセンタイルは絶対誤差で比べる。
- **局所依存とモデルに基づく平行分析（V26、M5）**: mirt の推定は `quadpts = 61`、`TOL = 1e-10`、
  `optimizer = "NR"`。flexpyirt は `BAEM(tol=1e-9)` で推定し直すので、比較には推定の一致（対数尤度は相対 $10^{-9}$）
  も含まれる。ltm は独自の `Science` と `LSAT` のデータ（因子型）を持つので、スクリプトでは mirt のデータを
  `mirt::` で指定する。ltm の `unidimTest` は推定済みの ltm オブジェクトではなくデータとパラメタで呼ぶので、
  標準正規の推定での flexpyirt と同じく $\theta\sim N(0,1)$ からシミュレーションする。テトラコリック相関は polycor
  （`optimise()` の許容誤差）によるので、観測された統計量の一致はそれで決まる。
- **単一群・1 次元・正規分布以外の局所依存（V26、M7、issue #81）**:
  `validation/r/m7_local_dependence_reference.R`（mirt 1.48）が
  `tests/reference/v26_local_dependence_extensions.json` を書く。`tests/test_mirt_m7_local_dependence.py` は
  mirt の推定値を、すべてのパラメタを固定した flexpyirt の `FitResult` に入れる（`fit(..., table=)`。EH の重みは
  mirt の `Internals$Prior`）。比較するのは統計量だけになる（mirt の推定値での対数尤度は相対 $10^{-9}$ で一致、
  実測 $4\times10^{-15}$ 以下）。9 ケース、群とペアの組は 641。実測の最大差: $X^2$ $3\times10^{-12}$、$G^2$
  $8\times10^{-9}$（相対。許容誤差 $10^{-7}$）、$V$ $3\times10^{-15}$、Q3 $2\times10^{-15}$（絶対。許容誤差
  $10^{-8}$）。

  | ケース | mirt の呼び出し | 格子 | ペア |
  |---|---|---|---|
  | 2 群、2PL、4% MCAR | `multipleGroup(invariance = c(anchors, "free_means", "free_var"))` | 121 | 2 x 45 |
  | 2 群、GRM（4 カテゴリ） | 同じ | 121 | 2 x 15 |
  | 2 群、2 因子、2PL | 同じ（`mirt.model`） | $61^2$ | 2 x 45 |
  | EH、2PL、歪んだ特性値、3% MCAR | `mirt(dentype = "EH")`、`use_dentype_estimate = TRUE` | 61 | 66 |
  | 2 次の Davidian 曲線、同じデータ | `mirt(dentype = "Davidian-2")`、同じ | 61 | 66 |
  | 正規混合分布、V22a の推定値 | 混合分布の重みを `Internals$Prior` に入れる | 61 | 190 |
  | 2 因子、2PL、交差負荷、3% MCAR | `mirt(mirt.model(...))` | $31^2$ | 45 |
  | 2 因子、GRM（3 カテゴリ） | 同じ | $31^2$ | 28 |
  | 3 因子、2PL | 同じ | $15^3$ | 36 |

  mirt の既定との違い（すべてスクリプトで対処している）: (1) EH と Davidian の推定に対して、`residuals()` は
  `use_dentype_estimate = TRUE` を付けないと $N(0,1)$ の事前分布を使う。付けた場合、`type = "Q3"` は残差を
  格子点で評価してしまう（mirt 1.48）ので、Q3 は `Theta = fscores(m, use_dentype_estimate = TRUE)` で求める。
  (2) Q3 のスコアは `fscores()` から来る。その格子は推定の格子によらず 1 / 2 / 3 因子で各次元 121 / 61 / 31 点。
  単一群のケースは `Theta = fscores(m, quadpts = <推定の格子>)` を渡す。`Theta` を群ごとに渡せない
  `multipleGroup` のケースは `fscores()` の格子で推定する。mirt の既定の呼び出しは `q3_default` として保存した。
  推定した格子での flexpyirt との差は、$31^2$ 対 $61^2$ で $2\times10^{-9}$、$15^3$ 対 $31^3$ で
  $1.3\times10^{-4}$（M5 のケースの $4\times10^{-8}$ とも整合する。61 点対 121 点）。
  (3) mirt の混合分布のクラスには `residuals()` がない。混合分布のケースは密度を与えたときの統計量を確かめる
  もので、mirt の混合分布の推定との比較ではない（推定は V22）。(4) ある群で観測されないカテゴリがあると、mirt の
  `table()` は $K_i\times K_j$ より小さくなる。flexpyirt は表の大きさを保つ。これは単体テストだけで確かめている。
  エンドツーエンドの確認として、2 群の 2PL を flexpyirt で推定し直し、mirt（`TOL = 1e-6` で推定）と
  $4\times10^{-4}$（$X^2$、$G^2$）、$5\times10^{-6}$（$V$、Q3）で一致する。mirt は群をアルファベット順に
  並べるので、$N(0,1)$ の参照群は `focal` になる。テストも同じ群を標準化する。もう一方の群を標準化しても
  モデルは同じだが、離散化は同じにならない（格子は $\pm6$ で切れ、自由な群の SD は 1.34）。対数尤度は
  $1.9\times10^{-3}$、$X^2$ は $3\times10^{-3}$ ずれる。**照合しておらず、実装していないもの**: bifactor / two-tier モデル、単一群・
  1 次元・正規分布以外の `jsi()`（mirt は 1 因子のみ、Davidian 曲線の SE なし、混合分布は非対応、`multipleGroup`
  では群ごとに単一群の $N(0,1)$ で推定し直す別の統計量）。
- **DETECT（V27、M5）**: sirt の `conf.detect()` は `smooth = FALSE` で呼ぶ。既定の `smooth = TRUE` で
  `use_sum_score = TRUE` だと、標準化した総得点から生の項目得点を引くので、残りの得点の群が誤る（DESIGN §16.13）。
  `expl.detect()` は自分で `smooth = FALSE` にする。その戻り値の `detect.unweighted` は検証用の半分の指標だけを
  持つので、推定用の半分の指標はスクリプトで推定用の行に `conf.detect()` を当てて計算し直す。
- **EFA と Hull（V28、M5）**: 回転しない負荷量は、含意する $\Lambda\Lambda^\top$（列の符号と等しい固有値の順に
  よらない）で比べる。`HULL()` の上限 $J$ は乱数を使う平行分析から決まるので、スクリプトは `settings$n_fac_max` を
  記録し、テストはそれを `max_factors` として渡す。EFAtools の `optim()` による解は flexpyirt より収束が粗いので、
  テストでは EFAtools の $\psi=1-h^2$ での目的関数値を flexpyirt の値が上回らないことも確かめる。残る差は Heywood の解
  （独自性が境界にある）で最も大きく、Hull は真の因子数を超える $k$ でそれを含む。
- **比較データ法（V29、M5）**: CD は両側とも乱数を使う。生成器は、EFAtools の乱数を R で再現し（`set.seed` の
  あと、生成器と同じ順に `sample()` / `rnorm()` を呼ぶ）、書き出して flexpyirt に与えることで厳密に確かめる。負荷量の
  符号が生成器で決まる 1 因子では厳密に一致する。選択は答えがはっきりしたデータで確かめ、平均 RMSE は各側が別の
  母集団を引くので緩く比べる（採用された因子数では 15%。棄却された追加の因子では 60%。こちらは 1 つの母集団による
  ばらつきがずっと大きく、EFAtools の値 0.048 に対して 6 つのシードで 0.044〜0.072 になる）。JSON の `auto_unbox` は要素が 1 つの $p$ 値のベクトルをスカラーにするので、スクリプトは
  `as.list()` で書く。
- **DIMTEST（V30、M5）**: 参照値との照合ではなくシミュレーション研究で受け入れる（表は DESIGN §16.16）。200 反復で
  .05 での第 1 種の誤りは 0.5〜1.0%（保守的。ブートストラップは偏りを除くが、帰無仮説の下での $T$ の SD は 0.72 程度）、
  検出力は 0.99〜1.00。`uv run python packages/flexpyctt/validation/simulation/dimtest_study.py`（約 4 分）で再実行できる。テストは帰無
  20 反復と対立 10 反復を実行する。
- **双因子モデル（V32、M7）**: シミュレーション 2 例を `tests/test_mirt_m7_bifactor.py` で照合する（通常の EM と
  SQUAREM。生成スクリプト `validation/r/m7_bifactor_reference.R` → `tests/reference/v32_bifactor.json`、mirt 1.46.1 の
  `bfactor()`、`optimizer = "NR"`、`accelerate = "none"`、`TOL = 1e-10`、`SE.type = "Oakes"`、$[-6, 6]$ 上で次元あたり
  31 点、すなわち特殊因子ごとに $31^2$ 格子 1 つ）。mirt の `fscores()` には双因子用の処理がなく 4 次元の直積格子全体で
  積分するが、これは同じ 1 次元格子上の分解した事後分布と一致する。そこで EAP は全パラメタを mirt の推定値に固定し、
  次元あたり 15 点（mirt では $15^4$ 節点）で比べる。どちらの例でも mirt の推定値での縮約対数尤度は mirt の値と相対
  $10^{-15}$ で一致し、EM と SQUAREM は単調で、Oakes は Hessian と $3\times10^{-16}$（共分散の要素）で一致する:
  - **V32a（`bifactor_2pl`、2PL、$J=15, N=2000$、`s1 = q01-q04`、`s2 = q05-q08`、`s3 = q09-q12`、`q13-q15` は
    一般因子のみ）**: `loglik` 相対 `2.2e-16`、項目パラメタ `1.9e-7`（EM、498 反復。SQUAREM は `5.3e-9`、80 反復）、
    Oakes SE 相対 `3.5e-7`（SQUAREM `8.9e-9`）、EAP $\theta$ `4.9e-14` / SD `8.7e-15`。
  - **V32b（`bifactor_grm`、GRM $K=4$、$J=12, N=2000$、4 項目のテストレット 3 つ）**: `loglik` 相対 `8.9e-16`、項目
    パラメタ `1.9e-8`（EM、188 反復。SQUAREM は `2.5e-9`、29 反復）、Oakes SE 相対 `2.5e-8`（SQUAREM `3.7e-9`）、
    EAP $\theta$ `4.9e-14` / SD `7.9e-15`。
  - 単体テスト（`tests/test_m7_bifactor.py`）は、縮約した尤度・期待度数・EAP を $(1+S)$ 次元の直積格子全体での
    総当たり積分と比べ（欠測、GRM 項目、3PL 項目、一般因子のみの項目を含めて $10^{-12}$ 以内）、特殊因子 1 つの
    双因子モデルを検証済みの 2 次元の経路（`with_factors(..., correlated=False)`、対数尤度 $10^{-12}$）と比べる。
- **two-tier モデル（V33、M7）**: シミュレーション 2 例を `tests/test_mirt_m7_two_tier.py` で照合する（通常の EM と
  SQUAREM。生成スクリプト `validation/r/m7_two_tier_reference.R` → `tests/reference/v33_two_tier.json`、
  mirt 1.48 の `bfactor(data, model, model2 = )`、`accelerate = "none"`、`TOL = 1e-10`、`SE.type = "Oakes"`、
  $[-6, 6]$ 上の次元あたり 15 点、つまり特殊因子ごとに $15^3$ 格子 1 つ）。V14 の 3 次元の例と同じく、M ステップの
  最適化は `BFGS`（`control = list(reltol = 1e-13, maxit = 500L)`）を使う。mirt の `NR` は `GroupPars` に数値
  ヘッセ行列を使うので、3,375 節点では非常に遅い。主因子間の相関は mirt の `COV_21`（分散はどちらも 1 固定）。
  mirt の `fscores()` は全次元の直積格子で積分するが、これは同じ 1 次元格子での分解した事後分布と等しい。EAP は
  全パラメタを mirt の推定値に固定して比較する。どちらの例でも、mirt の推定値での縮約対数尤度は mirt の値と
  $2.1\times10^{-15}$（相対）で一致し、`flexpyirt.loglik >= fixed_at_mirt.loglik` が成り立ち、EM と SQUAREM は
  単調で、Oakes はヘッセ行列と $6.2\times10^{-17}$（共分散の要素）で一致する:
  - **V33a（`two_tier_2pl`、2PL、$J=16, N=2000$、`F1 = q01-q08`、`F2 = q09-q16`、$\rho=0.5$、`s1 = q01-q04`、
    `s2 = q09-q12`、`s3 = q05, q06, q13, q14`。4 項目は主因子のみ）**: `loglik` 相対 `2.1e-15`、項目パラメタ
    `3.0e-7`（EM、192 反復。SQUAREM は `3.1e-7`、33 反復）、`cor_12` `2.2e-8`（SQUAREM `1.3e-8`）、項目の Oakes SE
    相対 `4.4e-7`（SQUAREM `4.6e-7`）、`cor_12_se` 相対 `7.3e-6`、EAP $\theta$ `6.9e-15` / SD `4.0e-15`（mirt 側は
    $9^5$ 節点）。
  - **V33b（`two_tier_grm`、GRM $K=4$、$J=12, N=2000$、`F1 = g01-g06`、`F2 = g06-g12`（`g06` が交差負荷）、
    $\rho=0.4$、`s1 = g01-g04`、`s2 = g07-g10`）**: `loglik` 相対 `1.1e-15`、項目パラメタ `7.8e-9`（EM、192 反復。
    SQUAREM は `1.6e-8`、30 反復）、`cor_12` `6.3e-9`（SQUAREM `3.9e-9`）、項目の Oakes SE 相対 `1.0e-7`（SQUAREM
    `1.1e-7`）、`cor_12_se` 相対 `1.6e-6`、EAP $\theta$ `7.1e-15` / SD `3.4e-15`（mirt 側は $15^4$ 節点）。
  - 単体テスト（`tests/test_m7_two_tier.py`）では、縮約した尤度・期待度数・EAP を、全 $(P+S)$ 次元の直積格子での
    総当たり積分と比べる（$10^{-12}$ 以内。欠測、GRM 項目、交差負荷項目、特殊因子のない項目を含む）。主因子 1 つの
    two-tier モデルは双因子の経路と、特殊因子 1 つのモデルは照合済みの 3 次元の経路（`with_factors()` で特殊因子の
    相関を 0 に固定。対数尤度が $10^{-12}$ で一致）と比べる。
  .05 での第 1 種の誤りは 0.0〜2.0%（保守的。ブートストラップは偏りを除くが、帰無仮説の下での $T$ の SD は 0.75 程度）、
  検出力は 0.99〜1.00。`uv run python validation/simulation/dimtest_study.py`（約 4 分）で再実行できる。テストは帰無
  20 反復と対立 10 反復を実行する。- **合計得点分布のカーネル（V36、M7）**: `summed_score_distribution()`（DESIGN §18.1）を
  `tests/test_r_m7_sumscore.py` で照合する（fast tier。生成スクリプト `validation/r/m7_sumscore_reference.R` →
  `tests/reference/v36_sumscore.json`、poibin 1.6、mirt 1.48）。
  - **2 値項目（Poisson 二項分布）**: 確率ベクトル 5 例（`uniform_10`: $J=10$、$p\sim U(0.05, 0.95)$。
    `twopl_50`: $\theta=0.4$ での 2PL。`extreme_200`: $\theta=-3$ での 2PL。`rare_300`: $p\sim U(0.001, 0.02)$。
    `equal_40`: $p=0.3$）。絶対差の最大値: `"recursion"` と poibin の再帰式（累積分布）で `2.9e-15`、DFT-CF の
    確率関数で `5.8e-16`。`"fft"` と DFT-CF の確率関数で `6.1e-16`。`"normal"` と `method = "NA"` で `1.2e-15`。
    `"refined_normal"` と `"RNA"` で `1.3e-15`。`"poisson"` と `dpois` で `8.5e-16`（許容誤差は絶対 $10^{-13}$、
    `"poisson"` は相対 $10^{-11}$）。poibin の厳密値は累積分布か DFT-CF から来るので、確かめられるのは絶対精度だけ。
    裾の相対精度は、単体テストの総当たり列挙と二項分布の特別な場合で確かめる。
  - **多値項目**: GRM（$K=4$、$J=10$、$N=1500$）と、2PL 5 項目 + GRM 5 項目についての mirt の合計得点の表。
    参照値には mirt の 61 節点上のカテゴリ確率（`probtrace`）と重みを保存してあるので、試されるのはカーネルだけ。
    $N\sum_qw_qL(s\mid\theta_q)$ は mirt の `expected` 列と `3.1e-15`（相対）で一致し、カーネルから作った合計得点
    ごとの $\theta$ の EAP と事後 SD は mirt の値と `5.3e-15` / `3.1e-15` で一致する。
  - **S-$X^2$（V15）**: `_itemfit.lord_wingersky` はこのカーネルを呼ぶようになった。V15 の照合テストは、参照値と
    許容誤差を変えずに通る。
  - **単体テスト**（`tests/test_m7_sumscore.py`）: $K\in\{2,3,4\}$ が混在する場合の全反応パターンの列挙（相対
    $10^{-13}$、厳密な 2 方法とも絶対 $5\times10^{-16}$）、二項分布の特別な場合、$P=10^{-4000}$（2000 項目）での
    対数空間の再帰、全方法の `jit` / `vmap` / `grad`、勾配の恒等式
    $\partial P(S=s)/\partial p_j=P_{-j}(s-1)-P_{-j}(s)$。
  - **厳密な 2 方法の精度と速度**（61 節点、jit 済み、この検証用マシンの CPU、1 回あたりミリ秒）: $J=50, K=2$ で
    再帰 0.10、対数空間の再帰 2.7、FFT 0.51。$J=200, K=2$ で 1.0 / 49 / 6.8。$J=1000, K=2$ で 57 / 1200 / 174。
    $J=100, K=5$ で 2.9 / 178 / 9.7。FFT と再帰の絶対差の最大値はどの場合も $3\times10^{-16}$〜$1.1\times10^{-15}$
    だが、最小の確率は $10^{-31}$〜$10^{-308}$ なので、FFT はそこでは 1 桁も合わない。この実装では、厳密な方法の
    うち再帰のほうが精度も速度も上なので（特性関数の計算自体に $O(JS_{\max})$ かかる）、既定は再帰のままとし、
    FFT は比較用に提供する。

## 照合以外の検証
- **パラメタ回復シミュレーション**: 条件（N, J, 群数, アンカー数）× 反復 100 で bias / RMSE を出し、`validation/recovery/` に結果と図を置く。
- **EM の単調性**: すべての照合ケースで、反復ごとの（MAP 時は事後）対数尤度が非減少であることをテストで確認。
- **SE の内部整合**: Oakes の SE と、収束点での周辺対数尤度の直接 `jax.hessian` による SE が一致すること（小規模ケース）。
- **不変性**: 項目の並べ替え、群ラベルの付け替え、パターン圧縮の有無で結果が変わらないこと。
