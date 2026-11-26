# flexpyirt 設計書（DESIGN）

[English](DESIGN.md) | 日本語

> 状態: ドラフト v0.2（2026-11-20、M1 実装を反映）。実装前の設計。決定事項は「決定」、未確定は §12「未決事項」に置く。

## 1. 目的と非目的

### 目的
- **Bock–Aitkin EM（BA-EM, Bock & Aitkin, 1981）による周辺最尤推定（MMLE）** を中核にした、Python の IRT 推定ライブラリ。
- R `mirt` が実務で担っている以下を、Python らしい API で提供する:
  - 多母集団（multiple-group）同時推定（群間の等値制約・部分アンカー）
  - 固定パラメタ推定（FIPC: fixed item parameter calibration）
  - 潜在分布の柔軟な指定（Normal / Empirical Histogram / Davidian 曲線 / 混合正規）
  - 項目パラメタへの事前分布（Bayes modal = MAP 推定、EM で事後モードを求める）
  - 標準誤差（Oakes 1999 の恒等式を自動微分で）
- JAX による jit・vmap・GPU 対応、自動微分による拡張容易性（新しい項目モデルやリンキング基準を微分を手書きせず追加できる）。

### 非目的（少なくとも v1 では）
- R `mirt` の全機能の網羅。MHRM, MCEM, 探索的高次元 MIRT, CDM, CAT は対象外。
- R の `mirt.model` 文字列構文の再現。
- 速度で mirt（C++）に勝つこと自体。小規模問題では jit のコンパイル時間で負けてよい。売りは「正しさの検証済み」＋「微分可能・拡張可能」。

## 2. レイヤ構成

```
┌──────────────────────────────────────────────────────────────┐
│ ① Spec 層（ユーザーが書く・不変）                              │
│   Model, ItemSpec(TwoPL, GRM, ...), Group, LatentSpec(Normal, │
│   EmpiricalHistogram), Prior, 制約（fix / equal）              │
├──────────────────────────────────────────────────────────────┤
│ ② Compile 層（Python・検証）                                   │
│   Spec + Data → ParameterTable（mirt の pars='values' 相当）   │
│   → FreeMap（自由パラメタベクトル ⇄ 全パラメタの写像）          │
│   → 反応パターン圧縮・パディング・識別性チェック                 │
├──────────────────────────────────────────────────────────────┤
│ ③ Core 層（純粋 JAX・jit）                                     │
│   e_step, m_step_items, m_step_latent, em_loop(lax.while_loop) │
│   squarem, oakes_information, scoring                         │
├──────────────────────────────────────────────────────────────┤
│ ④ Result 層（Python）                                          │
│   FitResult: パラメタ表(DataFrame), SE, logLik, AIC/BIC,       │
│   収束履歴, score(), summary(), _repr_html_                    │
└──────────────────────────────────────────────────────────────┘
```

**決定**: ③ は名前・文字列・DataFrame を一切知らない。③ の入出力は pytree（`jax.Array` の入れ子と
`NamedTuple`／`flax.struct` ではなく自前の `@jax.tree_util.register_dataclass` な frozen dataclass）のみ。

## 3. 記法

| 記号 | 意味 |
|---|---|
| $g=1..G$ | 群（母集団）。$g=1$ を参照群とする |
| $j=1..J$ | 項目 |
| $p$ | 群内のユニークな反応パターン、$c_p$ はその度数 |
| $q=1..Q$ | 求積点 $\theta_q$ |
| $w_{gq}$ | 群 $g$ の潜在分布の求積重み（$\sum_q w_{gq}=1$） |
| $\xi_j^{(g)}$ | 群 $g$ における項目 $j$ のパラメタ（等値制約があれば群間で同一） |
| $\eta$ | 自由パラメタベクトル（FreeMap で $\xi,\ w$ のパラメタに展開） |

## 4. アルゴリズム

### 4.1 前処理: 反応パターン圧縮
- 群ごとに反応行列（欠測は `-1`）をユニーク化し、度数 $c_p$ を得る。標本ウェイトがあるとき、$c_p$ は
  パターン $p$ を持つ行のウェイトの和（§17.4）。
- jit の再コンパイルを抑えるため、パターン数を **2 のべき乗のバケットにパディング**（度数 0 の行で埋める）。
- 群ごとにパターン数が違うので、全群を連結した 1 本の配列 + `group_index` で持つ（群ごとの Python ループを jit 内に持ち込まない）。

### 4.2 E-step
各パターン・求積点で
$$\ell_{pq}=\sum_j m_{pj}\log P_j(y_{pj}\mid\theta_q;\xi_j^{(g)}),\qquad m_{pj}=\mathbf 1[y_{pj}\ \text{観測}]$$
$$h_{pq}=\operatorname{softmax}_q\big(\ell_{pq}+\log w_{g(p)q}\big)$$
期待度数（「人工データ」）:
$$\bar n_{gq}=\sum_{p\in g}c_p h_{pq},\qquad \bar r_{gjkq}=\sum_{p\in g}c_p h_{pq}\mathbf 1[y_{pj}=k]$$
周辺対数尤度: $\ell(\eta)=\sum_p c_p\operatorname{logsumexp}_q(\ell_{pq}+\log w_{g(p)q})$。

実装メモ: $\bar r$ は `one_hot(y)` と $c_p h_{pq}$ の einsum。メモリは $P\times J\times K\times Q$ を避け、
`(P,Q)` の重みと `(P,J,K)` の one-hot を einsum で縮約して `(G,J,K,Q)` を直接得る。

### 4.3 M-step（項目）
群 $g$・項目 $j$ の期待完全データ対数尤度に対数事前分布を足したもの:
$$Q_j(\xi_j)=\sum_g\sum_q\sum_k \bar r_{gjkq}\log P_{jk}(\theta_q;\xi_j^{(g)})\;+\;\log\pi(\xi_j)$$
- 群間等値制約は**同一項目内**に閉じるので、項目ごとの分解は保たれる（多母集団でも項目単位に解ける）。
- **決定（v1）**: 項目ごとの Newton–Raphson（自動微分で勾配・Hessian、$\dim\xi_j\le 5$ 程度）。
  同じ項目モデル（"item family"）の項目は配列にまとめて `vmap`。ステップは backtracking line search
  で $Q_j$ の増加を保証（一般化 EM として単調性を保つ）。M-step は完全収束させず、Newton 数ステップで打ち切ってよい。
- 固定パラメタは $\eta$ に含めない（定数として ParameterTable から渡す）。
- **項目をまたぐ等値制約**（例: 全項目共通の $a$）は分解を壊す。**決定（M5、§16.1）**: `free_id` の共有で連結された項目の成分ごとに、その自由パラメタ全体について同時にマスク付き Newton を解く。成分に属さない項目は従来どおり項目ごとの `vmap`。

### 4.4 M-step（潜在分布）
- **Normal**（群 $g$ の平均・分散が自由な場合）:
  **決定（M1）**: 求積重みは格子上の密度を正規化したもの
  $w_{gq}(\mu_g,\sigma_g^2)=\phi(\theta_q;\mu_g,\sigma_g^2)/\sum_{q'}\phi(\theta_{q'};\mu_g,\sigma_g^2)$ なので、
  M-step は離散化した期待対数尤度 $Q_g^{\text{lat}}=\sum_q\bar n_{gq}\log w_{gq}(\mu_g,\sigma_g^2)$ を
  項目と同じマスク付き Newton（§4.3）で最大化する。閉形式のモーメント
  $\mu_g=\frac{1}{N_g}\sum_q\bar n_{gq}\theta_q$、$\sigma_g^2=\frac{1}{N_g}\sum_q\bar n_{gq}(\theta_q-\mu_g)^2$
  は連続密度に対する最大化であり、離散化した周辺尤度の EM としては厳密ではない（単調性の保証が崩れ、
  Oakes の恒等式とも整合しない）。差は等間隔 61 点では無視できるほど小さいが、厳密さを優先する。
  パラメタは自然尺度（平均・**分散**）で持ち、$\sigma^2>0$ は line search の実行可能性判定で守る。
  （M1 で確認: mirt は閉形式で更新しており、Rasch の分散で最大 ~8e-5 の差が出る。VALIDATION.md 参照。）
  **決定（2026-11-20, レビュー済み）**: 厳密な M-step のみを持ち、mirt 互換の閉形式更新はオプションとしても提供しない。
  両者の差は格子の打ち切り（$V_w(\sigma^2)<\sigma^2$）から生じ、停留条件はそれぞれ
  厳密: $V_w(\hat\sigma^2)=\hat m$、閉形式: $\hat\sigma^2=\hat m$（$\hat m$ は期待度数の 2 次モーメント）。
  報告する `var` は格子で打ち切る前の正規分布のパラメタであり、格子上の分布の分散は $V_w(\hat\sigma^2)$ である。
  平均・分散が大きい群（M2 の焦点群など）では打ち切りの影響が平均にも及ぶので、格子幅の選択が結果に効く。
- **Empirical Histogram (EH)**（`EmpiricalHistogram(*, min_weight=1e-10)`；Woods, 2007；§15.34–§15.38）:
  各群の潜在分布を固定求積格子 $\theta_1<\dots<\theta_Q$（既定は $[-6,6]$ の等間隔 61 点）上の離散重み
  $w_{g1},\dots,w_{gQ}$ で表し、空節点でも $\log w_{gq}$ が常に有限になるよう設定可能な正の下限
  $w_{gq}\ge\tau$（$\tau=\texttt{min\_weight}>0$、$\tau Q<1$）を課す。EH の M-step は制約付き完全データ目的関数
  $\max_{w\in\Delta^{Q-1},\,w_q\ge\tau}\sum_q\bar n_{gq}\log w_q$ を厳密な KKT water-filling
  $w_{gq}=\max(\tau,\,c_g^*\bar n_{gq})$ で解く（どの節点も下限に当たらないときは $\bar n_{gq}/N_g$ と厳密に一致）。
  **決定（M3、§12-1 を解決）**: 参照群（$g=0$）を**毎 EM 反復で**平均 0・分散 1 に標準化する（平均は参照群の連結成分に
  固定切片がないとき、分散は非ゼロの固定傾きがないとき。**例外**: 固定傾きと固定切片を持つ FIPC では標準化せず全群の EH 重みを
  自由とし、傾き固定・切片自由の Rasch / PCM では平均のみ標準化する。固定切片があり非ゼロの固定傾きがない場合は、分散のみの制約が
  重みについて線形でないため `IdentificationError`）。標準化は参照群の M-step の一部として行い、
  $w_q\ge\tau$、$\sum_q w_q=1$、$\sum_q w_q\theta_q=0$、$\sum_q w_q\theta_q^2=1$ の制約下で $\max\sum_q\bar n_{0q}\log w_q$ を解く
  （§15.34）。したがって EM は厳密に単調で、項目パラメタの変換は行わない。（最初の実装は $\theta$ をアフィン変換して項目パラメタを
  逆変換し、全群の重みを CDF 補間で再格子化していた（mirt の `EHW` や Woods (2007) と同様の再格子化）。補間誤差によって反復間で
  対数尤度が減少し SQUAREM が停滞したため、リリース前に置き換えた。）
  多母集団モデルで EH を指定するときは
  **全群が EH を使う**（`EmpiricalHistogram` と `Normal` の混在は `SpecificationError`）。
- **Davidian 曲線**（`DavidianCurve(degree, *, mean=0.0, var=1.0, free_mean=False, free_var=False)`；
  Zhang & Davidian, 2001；Woods & Lin, 2009；§16.4）:
  単一群・1 次元の半ノンパラメトリック密度 $h_k(z;\boldsymbol\phi)=P_k(z;\boldsymbol\phi)^2\varphi(z)$ を、
  正規化確率論的 Hermite 多項式 $He_j(z)/\sqrt{j!}$ と単位球面上の係数 $\mathbf c(\boldsymbol\phi)\in S^k$
  （$k\in\{1,\dots,10\}$ 個の角度 $\boldsymbol\phi=(\phi_1,\dots,\phi_k)$ で表現し、$\boldsymbol\phi=\mathbf 0$ が非特異に
  $N(0,1)$ を与える）で展開する。$h_k$ の閉形式の連続モーメント $(\mu_Z(\boldsymbol\phi),\sigma_Z^2(\boldsymbol\phi))$ で
  解析的に標準化し、$\theta=\mu+\sigma(Z-\mu_Z)/\sigma_Z$ が任意の $\boldsymbol\phi$ で厳密に連続平均 $\mu$・分散 $\sigma^2$
  を持つようにする。求積重みは格子上で正規化した密度とし、潜在分布の M-step は
  $Q^{\text{lat}}=\sum_q \bar n_q\log w_q(\mu,\sigma^2,\boldsymbol\phi)$ を既存のマスク付き Newton（`_newton.py`）で最大化する。
  これにより EM の単調性（非減少）・SQUAREM の適用・Oakes の恒等式の厳密性を保つ。
- **混合正規**（`NormalMixture(n_components, *, mean=0.0, var=1.0, free_mean=False,
  free_var=False, means=None, variances=None, weights=None)`；Mislevy, 1984；§16.7）:
  単一群・1 次元で、$k\in\{2,\dots,10\}$ 個の正規成分の有限混合と共通の 1 組の項目パラメタ。形状は mirt の
  `dentype = "mixture-k"` と同じパラメタ化（素の成分 1 が $N(0,1)$、成分 $c\ge 2$ は素の平均・素の分散・ロジット
  $\log(\pi_c/\pi_1)$）とし、混合全体を厳密な連続モーメントで解析的に標準化して $\theta$ が平均 $\mu$・分散 $\sigma^2$
  を持つようにする（Davidian 曲線と同様）。各成分を格子上で正規化してから混合し、潜在分布の M-step は
  $Q^{\text{lat}}=\sum_q \bar n_q\log w_q$ を同じマスク付き Newton で最大化する。

### 4.5 識別性ルール（Compile 層で検査）
- 既定: 参照群は $N(0,1)$ 固定、焦点群の平均・分散は自由。焦点群を自由にするには、群間で全パラメタが等しいアンカー項目が必要
  （2PL なら最低 1 項目で理論上は識別、実務的には複数を推奨し、少ない場合は警告）。
- **FIPC**: 固定項目が尺度を決めるので、全群の平均・分散を自由にしてよい。固定項目が 0 なら参照群を固定。
  潜在分布を $N(0,1)$ に固定したまま固定項目以外を推定すると尺度がずれる（Kim, 2006）ので、
  FIPC では潜在分布（Normal の平均・分散、または EH の重み）を各 EM 反復で更新する方式（Kim の MWU-MEM に相当）を既定にする。
- **Empirical Histogram (EH)**: 全群が `EmpiricalHistogram` を使うとき、参照群の位置（平均）は連結成分に固定切片があればそれで、
  なければ §4.4 の平均 0 制約で識別する。尺度（分散）は連結成分に非ゼロの固定傾きがあればそれで、なければ分散 1 制約で
  識別する（したがって FIPC は全 EH 重みを自由のまま残し、Rasch / PCM は平均のみ標準化し、傾き・切片とも自由なモデルは両方標準化する）。
  焦点群の EH 重みは自由な位置と尺度を持つため、すべての焦点群はアンカー項目の傾き・切片（または固定項目）で識別済みの群と
  連結していなければ `IdentificationError`（`Normal` と同じグラフ検査および単一位置アンカーの警告）。
- **Rasch・PCM（決定, M1/M3）**: Rasch と PCM は $a_j=1$ 固定・$d_j$（$d_{jk}$）自由で、尺度は傾きで決まるので
  **潜在分布の分散を自由推定**し、平均は 0 に固定する（mirt の `itemtype="Rasch"` と同じ: 2 値データでは Rasch、
  多値データでは PCM になり、`a1` 固定、`MEAN_1` 固定、`COV_11` 自由）。潜在分布を明示しない単一群モデルでは、
  **全項目が Rasch または PCM のとき既定を $N(0,\sigma^2)$（分散自由）**、それ以外は $N(0,1)$ 固定とする。
- 一般則（M1 で検査）: 分散を自由にするには傾きを固定した項目（値が 0 でない）が 1 つ以上、
  平均を自由にするには切片を固定した項目が 1 つ以上必要。
- 識別できない指定は `IdentificationError` を fit 前に投げる。

### 4.6 収束と加速
- 収束判定: 連続して採択された反復点の間での自由パラメタ（自然尺度の $\eta$）の最大絶対変化 $<$ `tol`（既定 1e-4、
  mirt の `TOL` と同じ定義）。対数尤度・対数事後・最大変化・累積 EM 写像評価回数・SQUAREM 安全装置の発動履歴を保存し、
  対数事後（事前分布なしでは対数尤度）の単調性はテストで検査する。
  **決定（M1）**: 停止条件は最大絶対変化のみとする（対数尤度の変化量を条件に足すと mirt との対応が崩れるため）。
- **SQUAREM**（Varadhan & Roland, 2008, スキーム S3）を `BAEM(squarem=True)` で選択可能（M3 では mirt との
  wall-time ベンチマーク前のため既定 `False`、§15.13）。外挿候補が境界（事前分布の台との積集合）を出るか対数事後が
  非有限・$\theta_2$ 未満なら $\alpha$ を $-1$ 方向へ固定回数まで半減し、なお不適なら $\theta_2$ をそのまま採って
  対数事後の単調性を保つ（§15.11–§15.12, §15.17）。
- 通常の EM・SQUAREM とも全体がデバイス上の `lax.while_loop` で走る（Python の反復ループやホストコールバックなし）。
  `jax.debug.callback` による進捗表示は後回し。

### 4.7 MAP（Bayes modal）
- **決定（M3, §15.14–§15.17）**: 公開の不変事前分布 `NormalPrior(mean, sd)`, `LogNormalPrior(mean_log, sd_log)`,
  `BetaPrior(a, b)` を提供する（各々 `log_pdf` とパラメタ検証を持つ。潜在分布 `Normal` との衝突を避けるため `NormalPrior`）。
- 事前分布は**自然なパラメタ尺度**（$a, d, g, u, d_k$ など）で定義し、Jacobian 項は入れない（自然尺度での事後モード。
  R mirt 1.46 の `LL.Priors` でも `expbeta` は `dbeta(plogis(logit_g), a, b, log = TRUE)` を logit ヤコビアンなしで
  評価していることを確認済み。VALIDATION.ja.md 参照）。
- 内部の最適化を変換後の尺度で行っても、目的関数自体は自然尺度の事前分布で評価する。
- 固定パラメタは事前分布を無視し、群間で共有するパラメタ（同じ `free_id`）の事前分布は **1 回だけ**数える。
  潜在分布パラメタへの事前分布は対象外（`SpecificationError`）。
- 結果には `loglik`（尤度のみ）と `log_posterior`（事前分布なしでは `loglik` と同値）を分けて返し、`history` も
  `loglik`・`log_posterior` の両方を記録する。事前分布が有効なときの `aic`/`bic` は `FlexpyirtWarning` を出す。

### 4.8 標準誤差
- **決定**: 既定は Oakes (1999) の恒等式
  $$I(\hat\eta)=-\Big[\tfrac{\partial^2 Q_{\mathrm{post}}(\eta\mid\eta')}{\partial\eta\,\partial\eta^\top}+\tfrac{\partial^2 Q_{\mathrm{post}}(\eta\mid\eta')}{\partial\eta\,\partial\eta'^\top}\Big]_{\eta=\eta'=\hat\eta}$$
  で計算する。ここで $Q_{\mathrm{post}}(\eta\mid\eta')=Q(\eta\mid\eta')+\log\pi(\eta)$
  （「E-step の重みを $\eta'$ で計算し、完全データ尤度と対数事前分布を $\eta$ で評価する」2 引数関数）。
  微分はすべて JAX の自動微分で得つつ、1 回の E-step を再利用し、第 1 項のブロック構造と第 2 項の
  パターン分割カテゴリ共起構造を活用する（§15.20）。
- 代替: 基底ベクトルをチャンク分割した前向き×後ろ向き自動微分
  （`jax.linearize(jax.grad(...))`）による観測（事後）情報行列
  $-\nabla^2_\eta[\ell(\eta)+\log\pi(\eta)]$（Oakes の検証にも使う）。
- **決定（M7, §17.5）**: `se="crossprod"` はパターンごとのスコアのクロス積
  $B_{\mathrm{post}}=\sum_p c_p s_p s_p^\top-\nabla^2_\eta\log\pi(\hat\eta)$ の逆行列、`se="sandwich"` は
  上の観測（事後）情報行列 $H$ を使ったロバスト共分散 $H^{-1}B_{\mathrm{post}}H^{-1}$
  （Huber, 1967; White, 1982）。
- MAP 時は `oakes`・`hessian` とも情報行列に $-\nabla^2_\eta\log\pi(\hat\eta)$ を加え、数学的恒等性を保つ。
  固定パラメタは $\eta$ に含まれないので自動的に除外される。

### 4.9 求積
- **決定**: 1 次元の既定は等間隔 61 点、$[-6,6]$（EH と同じ格子を共有でき、Normal でも重みを密度から作る）。
- Normal 潜在分布では Gauss–Hermite も選べる（`Quadrature.gauss_hermite(n)`、§15.44）。
- **多次元直積格子（M4, §15.30）**: 2・3 次元モデルは 1 次元等間隔格子の直積（`Quadrature.grid(n_factors)` → `(Q, D)`）を使う。次元あたりの既定節点数は `Quadrature.default(n_factors)` により mirt の `select_quadpts` に合わせ、1 次元で 61 点、2 次元で 31 点（$31^2 = 961$）、3 次元で 15 点（$15^3 = 3{,}375$）、区間 $[-6, 6]$ とする。直積格子での 4 次元以上は対象外（非目的）。双因子モデルは特殊因子がいくつあっても $(1+1)$ 次元の格子に縮約する（§17.1）。

### 4.10 初期値
- $d_j$: 正答率の logit、$a_j$: 項目−合計の点双列相関から変換（mirt と同様の古典的初期値）。GRM の閾値はカテゴリ累積比率の logit。
- **決定（M1, 2PL/Rasch）**: 正答率 $p_j$（$[0.01,0.99]$ に丸める）と項目−残余得点の点双列相関 $r_j$ から
  $a_j^{(0)}=\operatorname{clip}(1.7\,r_j/\sqrt{1-r_j^2},\,0.25,\,3)$（Rasch は 1）、
  $d_j^{(0)}=\operatorname{logit}(p_j)\sqrt{1+(a_j^{(0)}/1.7)^2}$（周辺確率を近似的に合わせるため）。
  分散の初期値は 1。初期値は収束点に影響しない前提で、照合テストもそれを確認する。
- **決定（M3, 3PL/4PL）**: $[0.01,0.99]$ に丸めた正答率 $p_j$ から $g_j^{(0)}=\min(0.2,\,p_j/2)$、
  $u_j^{(0)}=\max(0.95,\,(1+p_j)/2)$（4PL）または $1$（3PL）とし、$g_j^{(0)}<p_j<u_j^{(0)}$ を保つ。
  $a_j^{(0)}$ は 2PL と同じ、$d_j^{(0)}$ は漸近線の幅で換算した正答率 $(p_j-g_j^{(0)})/(u_j^{(0)}-g_j^{(0)})$
  から 2PL と同じ式で求める（§15.4）。
- **決定（M3, GPCM/PCM）**: $a_j^{(0)}$ は 2PL と同じ項目−残余相関から（PCM は 1）、
  $d_{jk}^{(0)}=\sum_{c=1}^{k}\operatorname{logit}(p_{jc})\sqrt{1+(a_j^{(0)}/1.7)^2}$（ここで
  $p_{jc}=\operatorname{clip}(\hat P(y_j\ge c),\,0.01,\,0.99)$、§15.9）。$K=2$ では 2PL / Rasch の初期値と
  完全に一致する。
- ユーザー指定の初期値は ParameterTable の `value` 列で上書き可能。

## 5. データ構造

### 5.1 ParameterTable（Compile 層、ユーザーにも公開）
mirt の `pars='values'` に相当する長形式の表。**全ての制約はこの表の列で表現する。**

| 列 | 型 | 意味 |
|---|---|---|
| `group` | str | 群名 |
| `item` | str | 項目名（潜在分布パラメタは `"__latent__"`） |
| `param` | str | `a`, `d`, `g`, `u`, `d1..dK` など、内部パラメタ名 |
| `value` | float | 初期値（固定なら固定値） |
| `free` | bool | 推定するか |
| `lower`, `upper` | float | 範囲 |
| `prior` | `Prior \| None`（object） | 自然尺度の事前分布オブジェクト（なしは `None`、固定なら無視） |
| `free_id` | int \| -1 | 自由パラメタ番号。**等値制約は同じ `free_id` を共有することで表す** |

- ユーザーは `spec.parameter_table(data)` で表を取り出し、値・固定・範囲・事前分布・`free_id` を pandas で編集して
  `fit(..., table=edited)` に渡せる（mirt の pars 編集ワークフローを残す）。同じ `free_id` を共有する行は事前分布が一致している必要がある。
- `FreeMap` は `free_id` から「全パラメタ配列（群 × family × 項目 × パラメタ、パディング付き）」へのギャザー添字を持つ。$\eta\mapsto\xi$ は `xi = fixed_values.at[idx].set(eta[free_id])` の 1 行。

### 5.2 Core に渡す pytree
```python
@register_dataclass
@dataclass(frozen=True)
class CompiledProblem:
    y: Array  # (P, J) int, 欠測 -1, パディング行は度数 0
    counts: Array  # (P,)
    group: Array  # (P,) int
    nodes: Array  # (Q, D) float, D in {1, 2, 3}
    families: tuple[FamilyBlock, ...]  # 項目カーネルごとの塊（静的）
    latent: LatentBlock  # (G, L_D): L_1=2 (mean, var), L_2=5, L_3=9 (平均・分散・相関)
    priors: PriorBlock  # free_id ごとの事前分布パラメタ
    n_groups: int  # 静的
```
`families`（各カーネルの `n_factors`・`n_categories` を含む）と `n_groups` は静的（jit の static）、パラメタ値と配列は動的。

## 6. 拡張プロトコル

```python
class ItemKernel(Protocol):
    n_factors: int  # 1, 2, または 3
    param_names: tuple[str, ...]  # ("a", "d") または ("a1", ..., "aD", "d")
    n_categories: int
    slope_params: tuple[str, ...]
    intercept_params: tuple[str, ...]
    irt_names: tuple[str, ...]  # 表示用閾値（D > 1 では空）
    irt_dependency_indices: tuple[tuple[int, ...], ...]

    def with_factors(self, n_factors: int) -> ItemKernel: ...
    def irt_thresholds(self, params: Array) -> Array: ...
    def log_prob(self, theta: Array, params: Array) -> Array: ...  # theta (Q, D) -> (Q, K)
```
新しい項目カーネルは `(Q, D)` 節点に対する `log_prob` を書くだけで、$D \in \{1, 2, 3\}$ の E-step・M-step（自動微分 Newton）・Oakes/Hessian SE・EAP スコアリングがすべて動くことを設計目標にする。

## 7. 対応する項目モデル（段階的）
| モデル | 内部パラメタ | マイルストーン |
|---|---|---|
| Rasch / 2PL | $a, d$（$P=\sigma(a\theta+d)$） | M1 |
| 3PL / 4PL | $a, d, g, u$（自然尺度、$0<g<u\le 1$、§15.2） | M3 |
| GRM | $a, d_1>\dots>d_{K-1}$ | M2 |
| GPCM / PCM | $a, d_1..d_{K-1}$ | M3 |
| Nominal | $a, \mathrm{ak}_1..\mathrm{ak}_{K-2}, d_1..d_{K-1}$（$\mathrm{ak}_0=0, \mathrm{ak}_{K-1}=K-1, d_0=0$、§16.6） | M5 |

表示用の IRT パラメタ化（2PL / Rasch / 3PL / 4PL は $b=-d/a$、GRM は $b_k=-d_k/a$、GPCM / PCM は
ステップ困難度 $b_k=-(d_k-d_{k-1})/a$、$d_0=0$）は Result 層（`FitResult.items`）で計算し、SE
（`b_se` / `bk_se`）は JAX の前向き自動微分によるヤコビアンを用いたデルタ法で求める（§15.18）。

## 8. スコアリング
`FitResult.score(data, *, method="EAP"|"MAP"|"ML"|"WLE"|"PV"|"EAPsum", group=None, item_names=None, key=None, n_draws=5)`:
- **EAP**（Bock & Mislevy, 1982）: 推定済みモデルと同じ求積格子上で、回答者の群の推定済み潜在分布を事前分布とする事後平均と事後 SD。
- **MAP**: 群の推定済み正規潜在分布を連続事前分布として $\log L(\theta\mid y) + \log\phi(\theta;\mu_g,\sigma_g^2)$ を最大化するベイズ最頻値推定。SE は最頻値における観測事後情報量 $(-\partial^2/\partial\theta^2)^{-1/2}$（§15.22）。
- **ML**: $\log L(\theta\mid y)$ を最大化する最尤推定。SE は観測情報量 $(-\partial^2\log L/\partial\theta^2)^{-1/2}$。有限の最大値を持たないパターンは $-\infty$ / $+\infty$（全欠測は NaN）、SE は NaN（§15.23）。
- **WLE**（Warm, 1989; Firth, 1993）: 観測項目にわたる Fisher テスト情報量 $I(\theta)$ と JAX 自動微分による $J(\theta) = \partial I/\partial\theta$ を用いて $\partial\log L/\partial\theta + J(\theta)/(2I(\theta)) = 0$（等価に $\log L(\theta\mid y) + \tfrac{1}{2}\log I(\theta)$ の最大化）を解く重み付き最尤推定。極端パターンでも有限。SE は mirt の既定に合わせて WLE 目的関数の観測曲率から計算（§15.24）。
- **PV**（Mislevy, 1991）: 細かい等間隔格子上の事後分布から区分線形逆 CDF 補間により回答者ごとに独立に `n_draws` 個を抽出し、`pv1..pvM` 列の DataFrame を返す。明示的な `key: jax.Array` が必須（未指定は `SpecificationError`）で、`key` に対して決定的（§15.25）。
- **EAPsum**（Thissen, Pommerich, Billeaud & Williams, 1995）: 合計得点を与えたときの事後平均と事後 SD。回答者の群の合計得点の表を引く。欠測が 1 つでもある回答者は NaN。1 次元で潜在分布が `Normal` のモデルのみ（§18.2）。

EAPsum 以外の手法では欠測反応を尤度（および WLE の $I(\theta)$）から除く。EAPsum 以外のすべての手法がユニークパターン圧縮と 2 のべき乗パディングを用いてデータ値に依存しない形状で `jit` 実行される（§15.21）。

## 9. 性能方針
- パターン圧縮 + バケットパディングで再コンパイルを抑える。
- item family ごとの `vmap`。異種項目混在でも family 数（静的）だけのループ。
- 既定は CPU で十分速いことを目標。GPU は大規模（N ≥ 10^5, J ≥ 100）で効けばよい。
- ベンチマーク: R mirt（`TOL=1e-4`、既定設定および検証設定）との wall time 比較を
  [benchmarks/](../../benchmarks/README.ja.md) に置き、Apple M5 CPU での計測結果を
  [benchmarks/results/2026-11-21-apple-m5.md](../../benchmarks/results/2026-11-21-apple-m5.md) に
  コミットしている。この 1 環境の計測（$[-6, 6]$ 上の 61 点求積、CPU 専用。XLA CPU は複数スレッドを
  使う場合がある一方、mirt は既定で単一スレッド）の要約:
  - **warm 実行**: 全 6 ケースで mirt 既定より速い（通常 EM で `0.16×–0.60×`、`squarem=True` で
    `0.12×–0.46×`）。`TOL = 1e-4` で $|\Delta \log L| \le 1.5 \times 10^{-3}$。
  - **cold 実行**: コンパイルを含む初回の EM 推定は約 0.8–1.1 秒（`20000 × 40` の Oakes SE では
    さらに長い）かかるため、標準誤差なしの 1 回限りの推定では mirt の方が速い（6 ケースすべて）。Oakes SE
    つきでは 6 ケース中 4 ケースで flexpyirt の cold 推定の方が速い。
  - **大規模 `(N, J)` の Oakes SE**: `2PL (20000 × 40)` では `se="oakes"` が warm で約 3.1–3.5 秒
    （EM 本体は約 0.28–0.39 秒）と全体の大半を占める。
  - **SQUAREM のトレードオフ**: `BAEM(squarem=True)` は EM 評価回数を 33–69% 削減し、warm の EM 時間を
    全ケースで通常 EM 比 `0.38×–0.90×` に短縮する（2 群 2PL と 3PL で効果が大きい）一方、cold のコンパイル
    時間が約 0.05–0.4 秒（`20000 × 40` の Oakes SE ではさらに）増える。既定値（`squarem=False`）は据え置きとし、cold 遅延と warm／収束の遅いモデルの
    どちらを優先するかはメンテナーの判断に委ねる。

## 10. エラーと警告
- `SpecificationError`: 仕様の矛盾（存在しない項目名を固定、カテゴリ数不一致など）
- `IdentificationError`: 識別不能（アンカーなしで焦点群自由など）
- `ConvergenceWarning`: 最大反復到達、SQUAREM の安全装置多発
- `HeywoodWarning`: 境界解 — 推定後、自由な項目パラメタが有限の境界から `1e-4` 以内（$g\to 0$ や $u\to 1$ など）
  または自由な傾きが $|a|>20$ になった場合に発する（§15.6）。

## 11. テスト方針（詳細は VALIDATION.md）
- 単体: 各 `log_prob` の確率和 = 1、勾配の有限差分チェック、E-step の度数保存（$\sum_q\bar n_{gq}=N_g$）。
- 性質: hypothesis でランダムなパラメタ・データを生成し、EM の単調増加を検査。
- 回復: シミュレーションでのパラメタ回復（バイアス・RMSE）。
- 照合: R mirt の推定値・SE・対数尤度と許容誤差内で一致。

## 12. 未決事項
1. ~~**EH の識別の取り方**~~ → **決定（2026-11-21）**: 参照群を毎反復で平均 0・分散 1 に標準化する（FIPC では行わない）。モーメント制約付きの EH M-step として実装（§4.4、§15.34）。
2. 多母集団で群の数が多いとき（例: 学校 × 学年）の扱い。群ごとの潜在分布を vmap できる形にしておく。
3. ~~欠測の設計~~ → MAR 前提の除外のみ（§14.8、2026-11-20）。
4. ~~パッケージ名~~ → `flexpyirt` に決定（2026-11-20）。
5. ~~項目をまたぐ等値制約の M-step（§4.3）の実装方式~~ → 項目の連結成分ごとの同時 Newton（§16.1、2026-11-21）。

## 13. 実装上の決定（M1）
1. **M-step の Newton**: 勾配・Hessian はすべて JAX の自動微分（`jax.grad` / `jax.hessian`）で得て、手書きの解析微分は持たない。
   ループ（ステップ更新と line search）だけ自前で書く: 自由座標のマスク付き Newton。固定座標は Hessian を $-I$・勾配 0 に置き換えて動かさない。
   Newton 方向が上昇方向でない（Hessian が負定値でない）ときは勾配方向に切り替える。ステップ幅は
   $2^{-k}\ (k=0,\dots,29)$ を `vmap` で一括評価し、目的関数が減らず範囲内に収まる最大のものを採る
   （該当なしなら動かない）。jit 内に Python 分岐を持ち込まないための形。
2. **FreeMap**: family ごとに `free_index (J_f, P_f)`（固定は -1）と `fixed_values`。
   $\xi=\text{where}(\text{free\_index}\ge0,\ \eta[\text{free\_index}],\ \text{fixed})$。
   M-step 後は `eta.at[free_index].set(xi, mode="drop")` で戻す。M1 では `free_id` を異なる項目間で共有すると
   `SpecificationError`（§4.3、M5 で対応）。
3. **M1 の範囲外は明示的に拒否**: 多母集団（`group=`）、欠測（NaN）、等間隔以外の求積は M2 以降。
   E-step 自体は欠測マスクを持つ形で書くが、照合（V9）が済むまでデータ層で拒否する。
   全員が同じ反応の項目は MLE が発散するので `SpecificationError`。
   （多母集団と欠測は M2 で有効にした。§14。）
4. **依存**: pandas は必須依存（結果は DataFrame で返す）、polars は任意依存。
5. **SE**: $\eta$（自然尺度）に対する情報行列。Oakes は $Q(\eta\mid\eta')$ を
   `jax.jacfwd(jax.grad(Q, 0), argnums=(0, 1))` で微分。直接法は周辺対数尤度の `jax.hessian`。
   両者は恒等式として一致するので、内部整合テストは機械精度近く（相対 1e-6）で比較する。

## 14. 実装上の決定（M2）
1. **群ごとの項目パラメタ**: ParameterTable は群 × 項目 × パラメタの行を持ち、既定では群ごとに別の `free_id`
   （mirt `multipleGroup` の既定と同じ）。`equal_across_groups(items=..., params=None)` は指定項目の行を群間で
   同じ `free_id` にする。Core では family ごとに `free_index (G, J_f, P_f)`。
2. **項目内の局所パラメタ**: 項目 $j$ が群間で共有するパラメタを含めて解くため、項目ごとに局所ベクトル
   $u_j$（長さ $G\cdot P_f$ のパディング付き）を作り、`local_index (G, J_f, P_f)` で $\xi^{(g)}_j=u_j[\text{local}]$ と展開する。
   M-step は $\sum_g Q_{jg}(\xi_j^{(g)}(u_j))$ を $u_j$ についてマスク付き Newton で最大化し（§13.1）、`vmap` する。
   `free_id` は項目をまたがない（M5 まで）ので、局所ベクトルは項目内で閉じる。
3. **E-step の群**: $\ell_{pq}$ は全群ぶん `einsum("pjk,gjqk->pgq")` で計算してから各パターンの群を取り出す
   （$P\times J\times Q\times K$ の中間配列を作らない）。$\bar r$ は `(G, J_f, K, Q)`。
4. **群の指定**: `Model.with_groups(**latents)` のキーワード順で群を定義し、最初が参照群。`fit(..., group=labels)`
   の値はこの群名と一致しなければならない（過不足は `SpecificationError`）。`with_groups` なしで `group=` を
   渡すのもエラー。
5. **識別性（多群）**: 平均について、群を節点・切片の共有 `free_id` を辺とするグラフを作り、各連結成分に
   「平均が固定された群」または「固定された切片」が 1 つ以上あることを要求する（分散は傾きで同様）。
   必要条件であり十分条件ではない。自由な平均を持つ群がアンカー 1 項目だけで連結している場合は警告する。
6. **FIPC**: `Model.fix_from(bank, items=..., parameterization="slope_intercept" | "irt")` で項目パラメタを固定し、
   潜在分布は利用者が `with_latent(Normal.free())` で自由にする（既定は変えない。§4.5 の識別性検査で許可される）。
   IRT 表記（$b$）は $d=-ab$ で変換する（2PL のみ。GRM は $b_k$ → $d_k=-ab_k$）。
7. **GRM**（Samejima, 1969）: $P(y\ge k\mid\theta)=\sigma(a\theta+d_k)$、$d_1>\dots>d_{K-1}$、カテゴリは $0..K-1$。
   対数確率は端のカテゴリを $\log\sigma(-z_1)$、$\log\sigma(z_{K-1})$、中間を
   $\log\sigma(z_k)+\operatorname{log1mexp}(\log\sigma(z_{k+1})-\log\sigma(z_k))$ で安定に計算する
   （log1mexp は両分岐の入力を安全域に丸める二重 where で勾配の NaN を防ぐ）。順序制約は line search で
   有限値でない候補を棄却することで守る。family は `GRM(n_categories=K)` の $K$ ごと。データに現れないカテゴリが
   ある項目は閾値が発散するので `SpecificationError`。
8. **欠測（MAR）**: NaN を `-1` に符号化し、E-step で one-hot を全 0 にして尤度から除く（§4.2 の $m_{pj}$）。
   全員欠測の項目はエラー。初期値は観測値だけから計算する。not-reached は扱わない（§12-3 はこれで決着）。
9. **照合の格子**: 潜在分布の平均・分散が自由な照合（V4, V5）は、VALIDATION.md の方針どおり mirt・flexpyirt とも
   81 点・$[-8,8]$（刻み 0.2 は既定と同じ）で行う。

## 15. 実装上の決定（M3）
1. **3PL・4PL 項目モデル**: `ThreePL()`（$P(y=1\mid\theta)=g+(1-g)\sigma(a\theta+d)$、Birnbaum, 1968）と
   `FourPL()`（$P(y=1\mid\theta)=g+(u-g)\sigma(a\theta+d)$、Barton & Lord, 1981）は、内部パラメタ
   `(a, d, g, u)`（mirt と同じ名前）を持つ 1 つの共通カーネル `FourPLKernel` を共有する。これにより 3PL と
   4PL は 1 つの family にまとまる（Rasch と 2PL が `DichotomousKernel` を共有するのと同じ）。`ThreePL` は
   既定で `u = 1` を固定し、`FourPL` は `u` を自由推定する。
2. **自然尺度と有界アクティブセット Newton**: `g`, `u` は $\eta$ 上も自然尺度のままとし、自由パラメタの範囲は
   `0 < g < 1`, `0 < u < 1`（固定値としての `g = 0` や `u = 1` は許可し、$g < u$ を課す）。標準誤差もそのまま
   自然尺度で得られる（mirt の `coef(..., printSE=TRUE)` は `logit(g)` / `logit(u)` と logit 尺度の SE を
   返すため、照合時はデルタ法 $SE_g = g(1-g)\,SE_{\mathrm{logit}(g)}$ で変換する）。マスク付き Newton
   （`_newton.py`）では、箱 `(lower, upper)` の外に出る座標は他の座標のステップを縮める代わりに境界までの中点へ
   進め、境界から $10^{-8}$ 以内で勾配が外側を向く自由座標はそのステップで固定する（$\varepsilon$-active set;
   Bertsekas, 1982）。これにより、境界に張り付いた漸近線が `a`, `d` の収束を妨げず、目的関数の増加判定により
   EM の単調性も保たれる。
3. **$g=0$・$u=1$ での数値安定性**: 対数確率を
   $$\log P_1 = \operatorname{logaddexp}\big(\log g,\ \log(u-g)+\log\sigma(z)\big),\qquad
   \log P_0 = \operatorname{logaddexp}\big(\log(1-u),\ \log(u-g)+\log\sigma(-z)\big)$$
   （$z=a\theta+d$）で計算する。二重 `where`（$g=0$ または $u=1$ では $\log$ の引数を $1$ に差し替えたうえで
   本体分岐を選ぶ）により、`g = 0` や `u = 1` を固定しても 2PL と厳密に一致する対数確率と、全パラメタ
   （固定座標を含む）に対する有限な勾配・Hessian を返す。不正な漸近線（$u\le g$、$g<0$、$u>1$）は NaN を
   返して line search に棄却される（`validate_table` でも事前に拒否する）。
4. **初期値**: $[0.01,0.99]$ に丸めた正答率 $p_j$ から $g_j^{(0)}=\min(0.2,\,p_j/2)$、
   $u_j^{(0)}=\max(0.95,\,(1+p_j)/2)$（4PL）または $1$（3PL）として $g_j^{(0)}<p_j<u_j^{(0)}$ を満たすようにし、
   $a_j^{(0)}$ は 2PL と同じ項目−残余相関の式（§4.10）、$d_j^{(0)}$ は漸近線の幅で換算した正答率
   $(p_j-g_j^{(0)})/(u_j^{(0)}-g_j^{(0)})$ に 2PL の切片の式を適用する。
5. **表示と FIPC**: `FitResult.items` は引き続き `b = -d/a` を追加し、`g`, `u`, `g_se`, `u_se` は
   ParameterTable から自動的に現れる。`Model.fix_from()` は `"slope_intercept"`・`"irt"` のどちらでも
   `g`, `u` をそのまま渡し、3PL 項目では `u` 列を省略できる。
6. **HeywoodWarning**: 推定後、`fit()` はすべての自由な項目パラメタを検査し、有限の境界から `1e-4` 以内に
   到達したパラメタ（0 や 1 に近い `g`, `u`）または $|a|>20$ の自由な傾きがあれば、該当する項目名を列挙した
   `flexpyirt.errors.HeywoodWarning` を 1 回発する。
7. **識別性**: `g`, `u` は潜在変数の尺度も位置も担わない（`slope_params=("a",)`, `intercept_params=("d",)`）。
   したがって漸近線を固定しても潜在分布の平均や分散は識別されない。
8. **GPCM・PCM 項目モデル**: `GPCM(n_categories)`（Muraki, 1992）と `PCM(n_categories)`（Masters, 1982）は `n_categories` ごとに
   共通カーネル `GPCMKernel(n_categories)` を共有する。カテゴリは `0..K-1`（`n_categories >= 2` を検証）で、mirt の
   `itemtype="gpcm"` と同じ傾き・切片形式（採点係数 $\mathrm{ak}_k=k$ と $d_0=0$ を固定し、パラメタに含めない）:
   $$P(y=k\mid\theta)=\frac{\exp(k\,a\theta+d_k)}{\sum_{c=0}^{K-1}\exp(c\,a\theta+d_c)},\qquad d_0=0.$$
   内部パラメタは `(a, d1, ..., d{K-1})`。対数確率は `jax.nn.log_softmax` で計算する（数値安定で滑らかな微分を
   持ち、`d` に順序制約は不要）。`PCM(n_categories)` は既定で `a = 1` を固定し、単一群の既定潜在分布ルール（§4.5）を
   拡張して、全項目が `Rasch` または `PCM` のとき $N(0,\sigma^2)$（分散自由）とする（多値データに対する mirt の
   `itemtype="Rasch"` と同じ）。`n_categories = 2` では `GPCM(2)` は `TwoPL` に、`PCM(2)` は `Rasch` に厳密に一致する。
9. **GPCM/PCM の初期値・識別性・推定可能性**: $a_j^{(0)}$ は 2PL と同じ項目−残余相関の式（PCM は `1.0`）。
   各ステップの初期値を累積比率 $p_{jc}=\operatorname{clip}(\hat P(y_j\ge c),\,0.01,\,0.99)$ から
   $\delta_{jc}^{(0)}=\operatorname{logit}(p_{jc})\sqrt{1+(a_j^{(0)}/1.7)^2}$ とし、その累積和を
   $d_{jk}^{(0)}=\sum_{c=1}^{k}\delta_{jc}^{(0)}$ とする。識別性は `slope_params=("a",)`、
   `intercept_params=("d1", ..., "d{K-1}")`。自由パラメタを持つ項目では `0..K-1` の全カテゴリがデータに
   観測されている必要がある（GRM と同じ、`_check_estimable` で検査）。
10. **GPCM/PCM の表示と FIPC**: `FitResult.items` は GPCM / PCM 項目についてステップ困難度
    $b_k=-(d_k-d_{k-1})/a$（$d_0=0$）を `b1`…`b{K-1}` として表示し、2PL / Rasch / 3PL / 4PL（`b = -d/a`）と
    GRM（`bk = -dk/a`）の表示は同一モデルに混在しても変えない。`Model.fix_from(..., parameterization="irt")` は
    これを $d_k=-a\sum_{c=1}^{k}b_c$ で逆変換し、PCM 項目では `a` 列の省略を許す。
11. **SQUAREM スキーム S3（Varadhan & Roland, 2008）**: `lax.while_loop` 内の `_core.squarem` として実装。
    採択点 $\theta_0$ から EM 写像 $F$ を 2 回適用して $\theta_1=F(\theta_0)$、$\theta_2=F(\theta_1)$ を得て、
    $r=\theta_1-\theta_0$、$s=\theta_2-\theta_1$、$v=s-r$ からステップ長
    $\alpha=\min(-\|r\|_2/\|v\|_2,\,-1)$ を計算する（$\|v\|_2=0$ または非有限なら $\alpha=-1$）。
    $\delta=\alpha+1\le 0$ と置くと $\theta'(\alpha)=\theta_0-2\alpha r+\alpha^2 v$ は代数的に
    $\theta'(\alpha)=\theta_2-2\delta s+\delta^2 v$ と一致し、$\delta=0$ で桁落ちなく厳密に $\theta_2$ になるうえ、
    $-1$ への半減（$\alpha\leftarrow(\alpha-1)/2$）が単に $\delta\leftarrow\delta/2$ になる。
12. **SQUAREM の単調性・実行可能性の安全装置**: $\alpha<-1$ のとき、ループ前に `FamilyBlock.slot_lower/upper` と
    `LatentBlock.lower/upper` から一度だけ抽出した座標ごとの厳密境界 `(lower, upper)`（`_core._eta_bounds`）で
    $\theta'(\alpha)$ の実行可能性を検査する（M-step と完全に同一の境界: 潜在分散 $>0$、$0<g,u<1$、ユーザー編集の
    境界を含む）。境界内のときのみ `lax.cond` で `marginal_loglik` を評価し、有限かつ
    $\ge\max(\ell(\theta_2),\ell(\theta_0))$ なら採択する（GRM の閾値逆転や 4PL の $g\ge u$ もカーネルが NaN を
    返すため棄却される）。棄却された場合は内側の `lax.while_loop` で $\delta$ を最大 10 回半減する。採択された
    $\theta'$ には安定化 EM ステップ $\theta_{\text{new}}=F(\theta')$ を 1 回適用する（その反復の EM 評価回数 3）。
    すべて棄却された場合は $\theta_{\text{new}}=\theta_2$ をそのまま採る（EM 評価回数 2）。$F$ が単調なので、
    採択点では常に $\ell(\theta_{\text{new}})\ge\ell(\theta_0)$ が保たれる。
13. **SQUAREM の API 既定値・履歴・警告**: `BAEM(squarem: bool = False)` は、mirt との wall-time ベンチマーク
    （JIT コンパイル時間や問題規模ごとの得失を含む）を M3 後半で行うまで通常の EM を既定に保つ。停止条件は
    $\max_i|\eta^{(t+1)}_i-\eta^{(t)}_i|<\text{tol}$ のまま。`FitResult` に `n_em_evaluations`（$F$ の総評価回数、
    通常 EM では `n_iter` と同値）と `n_safeguards`（安全装置が $\alpha$ の縮小または $\theta_2$ への退避を行った
    反復数、通常 EM では 0）を追加し、`history` は `squarem` の有無にかかわらず統一スキーマ
    （`iter`, `loglik`, `max_change`, `em_evaluations`, `safeguard`）を持つ。`ConvergenceWarning` は `max_iter`
    到達時に安全装置・退避回数を含めて報告するほか、収束した SQUAREM でも $\theta_2$ への完全退避
    （`3 * n_iter - n_em_evaluations`）が 5 回以上かつ全反復の 50% を超えた場合に発する（部分縮小しつつ加速した
    反復では不要な警告を出さない）。
14. **事前分布と自然尺度 MAP 目的関数（M3, §4.7）**: `NormalPrior(mean, sd)`, `LogNormalPrior(mean_log, sd_log)`,
    `BetaPrior(a, b)` を `priors.py` の公開 frozen dataclass とし、パラメタ検証（`SpecificationError`）と `log_pdf(x)` を
    実装する。事前分布はヤコビアンなしの自然尺度で評価する（R mirt 1.46 の `LL.Priors` でも `expbeta` が
    `dbeta(plogis(logit_g), a, b, log = TRUE)` を logit ヤコビアンなしで評価することを確認済み。mirt 1.46 の
    `DerivativePriors` / `BL.LL` の不具合は VALIDATION.ja.md に記録）。`BetaPrior.log_pdf` の $\log B(a, b)$ は
    float64 で ~1e-7 の誤差を持つ `jax.scipy.special.betaln` ではなく `gammaln(a) + gammaln(b) - gammaln(a + b)` で
    計算し、完全な float64 精度を保つ。
15. **事前分布の指定・表編集・適用範囲**: `Model.prior(*, param, dist, items=None)` で選択した項目の `param` に全群
    共通の事前分布を付ける。ParameterTable に `prior` 列（`object` 型、未指定は `None`）を追加し、`fit(..., table=...)`
    は表で編集された事前分布を受け取って検証する（不正なオブジェクト、`__latent__` 行への指定、自由パラメタの初期値が
    事前分布の開区間台の外にある場合、境界が台と交わらない場合、同一 `free_id` の行で事前分布が矛盾する場合は
    `SpecificationError`）。固定パラメタは事前分布を無視し、群間で共有されるパラメタ（同一 `free_id`）は **1 回だけ**
    数える。潜在分布パラメタへの事前分布は対象外で `SpecificationError`。
16. **分岐なしの Core 事前分布表現**: `CompiledProblem` は自由パラメタごと（`free_id` ごとに 1 件）の
    `priors: PriorBlock`（`kind: (n_free,) int32`、`param1/param2: (n_free,) float64`）を持ち、`eval_log_prior` が
    全分岐で二重 `where` ガードを使って評価する（未選択分岐やパディング `PRIOR_NONE = 0` から NaN/inf 勾配が漏れない）。
    `m_step` では `slot_free_index` で項目局所スロットへ事前分布をギャザーして `q_item` に加算し、項目ごとの分解（§14.2）と
    共有パラメタの 1 回計上を保つ。`compile_problem` は自由スロットの `slot_lower`/`slot_upper` を事前分布の台
    （`LogNormalPrior` は `(0, inf)`、`BetaPrior` は `(0, 1)`）と交差させ、ParameterTable のモデル定義域を変えずに有界 Newton と
    SQUAREM が台を守るようにする。
17. **MAP の結果・履歴・SQUAREM 安全装置・事後 SE**: `FitResult.loglik` は周辺対数尤度のみとし、
    `FitResult.log_posterior` に $\ell(\hat\eta)+\log\pi(\hat\eta)$（事前分布なしでは `loglik` と同値）を返す。
    `history` は統一スキーマ `("iter", "loglik", "log_posterior", "max_change", "em_evaluations", "safeguard")` を持ち、
    EM・SQUAREM とも `log_posterior` が単調非減少で、SQUAREM 安全装置も `log_posterior` を比較する。自由パラメタに
    1 つ以上事前分布がある場合、`FitResult.aic` と `FitResult.bic` は `loglik` で計算しつつ `FlexpyirtWarning` を出す。
    `information_oakes` は $Q(\eta\mid\eta')$ に `log_prior(prob, e)` を足し、`information_hessian` は
    `log_posterior(prob, e)` を微分することで、両者とも厳密に $-\nabla^2_\eta\log\pi(\hat\eta)$ を加える。
18. **表示用 IRT 閾値のデルタ法 SE（M4, §7）**: 各 `ItemKernel` に `irt_names`（例: `("b",)` や
    `("b1", ..., "b{K-1}")`）、`irt_dependency_indices`（各閾値に入る内部パラメタの添字）、純粋 JAX 関数
    `irt_thresholds(params)`（2PL / Rasch / 3PL / 4PL は $b = -d/a$、GRM は $b_k = -d_k/a$、GPCM / PCM は
    `jnp.diff` による $b_k = -(d_k - d_{k-1})/a$、$d_0 = 0$）を持たせる。`FitResult.items` ではカーネルごとに
    キャッシュした jit 関数 `(irt_thresholds, jax.jacfwd(irt_thresholds))` で表示値とヤコビアン
    $J = \partial g/\partial\xi$ を手書き微分なしで同時に評価する。各閾値 $m$ について、その閾値に入る自由パラメタの
    `free_id` で `FitResult.vcov` を参照し、$\operatorname{Var}(g_m(\hat\xi)) \approx J_m\Sigma_\xi J_m^\top$
    （Casella & Berger, 2002; Oehlert, 1992）を計算する。固定パラメタは分散に寄与しない（$a=1$ 固定の
    Rasch / PCM では `b_se = d_se` / `b1_se = d1_se`、PCM のステップ差は $\operatorname{Cov}(d_{k-1}, d_k)$ を含む）。
    群間等値制約は同一の `free_id` を共有するため自動的に共有パラメタの共分散が使われる（傾き `a` のみ共有・切片 `d` は
    群別といった部分等値も含む）。`vcov` が `None` または非正定値（NaN）のとき、および閾値に入るパラメタがすべて固定
    （FIPC の固定項目や `a, d` 固定で `g` 自由の 3PL など）のとき、`b_se` / `bk_se` は NaN とする。
19. **結果の要約表示（`FitResult.summary()` と `FitResult._repr_html_()`）（M4, §2 ④）**:
    `FitResult.summary(*, digits=4)` はモデルと推定種別（`MMLE` / `MAP`）、データ規模とパラメタ数、群情報、
    収束状況・反復回数・EM 評価回数（0 でなければ SQUAREM 安全装置回数も）、SE 手法、求積格子、対数尤度・対数事後確率・
    AIC・BIC、潜在分布表、および項目パラメタ表（`b_se` / `bk_se` を含む）を読みやすいプレーンテキストで返す。事前分布が
    有効なときは、AIC / BIC が MLE ではなく MAP 推定値での周辺対数尤度を使う旨を本文に注意文として記載し、描画時に重複した
    `FlexpyirtWarning` を出さない。`FitResult._repr_html_()` は同じ内容を外部リソースなし・ライト／ダーク両対応の素の
    HTML/CSS で描画し、すべての名前を `html.escape()` でエスケープする。`__repr__` は短い 1 行要約のまま保つ。
20. **構造化 Oakes とチャンク分割 Hessian による高速化・省メモリ化（M3, §4.8, §9）**:
    初期の `information_oakes` は問題全体に対して `jax.jacfwd(jax.grad(q, argnums=0), argnums=(0, 1))(eta, eta)`
    を走らせており、全 $n_{\text{free}}$ 方向を同時に E-step へ通すため `(n_free, P, Q)` の巨大中間配列と
    方向ごとの密な $J$ 項目 einsum を生じていた。2PL・$N=20{,}000, J=40$（ユニークパターン $19{,}926$ を
    $P=32{,}768$ にパディング、$Q=61$、$n_{\text{free}}=80$）では EM 本体が ~0.43 s なのに対し、ウォーム時 ~2.66 s
    （コールド 5.46 s）・ピーク RSS 増分 ~6.11 GB を要していた。構造化実装は Oakes の恒等式を厳密に保ち、
    項目カーネル・潜在分布・事前分布の微分をすべて JAX 自動微分のみから求める:
    - **単一 E-step の再利用**: `_e_step_details` で期待度数 `counts` と事後重み $h_{pq}$ を一度だけ計算し、
      `e_step` と共有する。
    - **第 1 項のブロック構造**（`_oakes_own_hessian`）: $\partial^2 Q_{\mathrm{post}}/\partial\eta\,\partial\eta^\top$
      を項目局所スロットごとの `vmap(jax.hessian)`（`m_step` と同じ構成で固定境界座標を微分しない）、
      群ごとの潜在 `(mean, var)` に対する `vmap(jax.hessian)`、`prob.priors` に対する対角
      `vmap(jax.grad(jax.grad(...)))` に分解し、`(n_free, n_free)` へ scatter-add する。
    - **交差項の構造化**（`_oakes_cross_hessian`）:
      $Q(\eta\mid\eta')=\sum_p c_p\sum_q h_{pq}(\eta')\,\psi_{pq}(\eta)$・
      $h_{pq}(\eta')=\operatorname{softmax}_q(\psi_{pq}(\eta'))$ から
      $\partial^2 Q/\partial\eta\,\partial\eta'^\top|_{\eta'=\eta}=\sum_p c_p(\sum_q h_{pq}s_{pq}s_{pq}^\top-\bar s_p\bar s_p^\top)$
      （$s_{pq}=\nabla_\eta\psi_{pq}$、$\bar s_p=\sum_q h_{pq}s_{pq}$；Oakes, 1999; Louis, 1982）を得る。
      項目・潜在のスコア表を `jax.jacfwd` で一度求め、各 `(g, j, k, s)` で $\bar r$ 重み付き $q$ 平均を引く
      （$s_{pq}-\bar s_p$ は代数的に不変で、差の桁落ちを防ぐ）。群ごとにパターン分割 `lax.scan` で度数 0 の
      パディングチャンクを `lax.cond` でスキップしながら節点別カテゴリ共起行列
      $C_{g,q}=\sum_{p\in g}c_p h_{pq}x_p x_p^\top$ と $\sum_{p\in g}c_p\bar s_p\bar s_p^\top$ を蓄積し
      （`(Q, B, D)` 作業配列を ~32 MB に抑え、`(P, Q, n_free)` や `(P, D)` を実体化しない）、
      続けて $q$ 方向の `lax.scan` で $\sum_q L_{g,q}^\top C_{g,q} L_{g,q}$ を縮約する（`(G, Q, D, n_free)` も実体化しない）。
    - **直接 Hessian のチャンク分割**: `information_hessian` は全 $n_{\text{free}}$ 方向を一括微分する代わりに
      `jax.linearize(jax.grad(log_posterior), eta)` を基底ベクトル 8 本ずつ `lax.scan` で評価する。
    - **参考計測値**（同一の 2PL $N=20{,}000, J=40$、CPU、独立サブプロセスでのウォーム 5 回中央値、他ジョブ並行稼働下）:
      `information_oakes`: ウォーム **2.658 s → 0.0843 s**（約 31.5 倍高速）、コールド **5.464 s → 0.365 s**
      （約 15 倍高速）、ピーク RSS 増分 **6,110 MB → 9.6 MB**；
      `information_hessian`: ウォーム **0.768 s → 0.602 s**（約 1.28 倍高速）、コールド **1.215 s → 0.898 s**、
      ピーク RSS 増分 **4,991 MB → 612 MB**。
      V1–V11（2PL, Rasch, GRM, 2 群アンカー, FIPC, 欠測, 3PL, GPCM, PCM, MAP 事前分布）すべてで旧
      `jacfwd(grad(q))` 実装との差は $\max|\Delta I|/\max|I|\le 8.8\times 10^{-16}$（要素別相対誤差
      $\le 3.1\times 10^{-12}$、`tests/test_oakes.py`）。
    - **他の処理がない状態での計測**（メンテナー、同じ 2PL $20{,}000\times40$ データ、EM 約 0.41 s を含む
      ウォーム 5 回の中央値）: `fit(..., se="oakes")` 3.39 s → **0.50 s**（Oakes 部分は約 2.98 s → 約 0.09 s）、
      `se="hessian"` 1.16 s → 1.07 s。共分散行列は従来の実装と最大要素比 $8.5\times10^{-16}$ で一致。
21. **スコアリング API と共通の Core 構成（M4, §8）**: `FitResult.score(data, *, method="EAP", group=None, item_names=None, key=None, n_draws=5)` は `Literal["EAP", "MAP", "ML", "WLE", "PV"]` に対応する。
    EAP（`_core.eap`）の挙動と結果は変更しない。新規スコアリングカーネルは純粋 JAX の Core モジュール `_scoring.py` に置き、EAP と同じくユニークな `(group, pattern)` 圧縮と 2 のべき乗バケットパディングを行い、欠測反応（`y == -1`）をマスクで除外し、データ値に依存しない静的形状で `jit` 実行する。PV 以外に `key` を渡した場合、PV で `key` が未指定または不正な場合、不正な `method` や `n_draws` の場合は `SpecificationError` を送出する。
22. **MAP スコアリング（`method="MAP"`、M4, §8）**: 回答者の群の推定済み正規潜在分布 $(\mu_g, \sigma_g^2)$ を（求積格子ではなく）**連続**事前分布として $f_{\mathrm{MAP}}(\theta)=\log L(\theta\mid y) - (\theta-\mu_g)^2/(2\sigma_g^2)$ を最大化し、最頻値で $\mathrm{SE}=(-f''_{\mathrm{MAP}}(\hat\theta))^{-1/2}$ を計算する。最適化はユニークパターンに `vmap` した安全装置つき 1 次元 Newton 上昇法（`_newton_1d`、初期値は EAP）で行う: 微分 $f', f''$ はすべて JAX 自動微分から求め、$f''\ge -10^{-12}$（3PL / 4PL の非凹領域など）では勾配上昇方向へフォールバックし、ステップ幅 $2^{-k}\delta$（$k=0..29$、ステップ長上限 `10.0`）を `vmap` で一括評価して目的関数が減少することを防ぐ。全項目欠測の行は $\log L\equiv 0$ となるため連続事前分布の最頻値 $\mu_g$ と標準偏差 $\sigma_g$ を返す（全欠測で事前分布の平均・SD を返す既存の EAP と整合。mirt は全手法で全欠測行を `NA` に落とす点が異なる）。
23. **ML スコアリング（`method="ML"`、M4, §8）**: 同じ `vmap` 済み安全装置つき 1 次元 Newton（初期値 EAP）で $[-20, 20]$ 上（mirt の既定 `max_theta = 20` に対応）において $f_{\mathrm{ML}}(\theta)=\log L(\theta\mid y)$ を最大化し、観測情報量から $\mathrm{SE}=(-f''_{\mathrm{ML}}(\hat\theta))^{-1/2}$ を求める（mirt 既定の `expected.info = FALSE` に一致）。極端パターンの規約: 全欠測は `(NaN, NaN)`、全観測項目が低 $\theta$ 側の端カテゴリ（$a_j>0$ なら `0`、$a_j<0$ なら `K_j-1`）のときは `(-inf, NaN)`、高 $\theta$ 側の端カテゴリのときは `(+inf, NaN)`。非凹モデル（3PL / 4PL）で全端カテゴリ以外でも有限の内部最大値を持たないパターン（当て推量により低得点パターンで $\log L(\theta)$ が $\theta\to -\infty$ へ単調増加する場合など）も検出し、$|\hat\theta|$ が `±20` の上限に達した・$f''(\hat\theta)\ge 0$・$\mathrm{SE}>30$（`mirt:::ML` の閾値）・$f(\pm 20) > f(\hat\theta)$ のいずれかで `±inf` と SE NaN を返す。`±inf` の符号は、$f(\pm 20) > f(\hat\theta)$ が成り立つときにのみ優越する端点から決め、それ以外は $\operatorname{sign}(\hat\theta)$ に従う（`mirt:::ML` の `Inf * sign(est)` に一致）。特に、3PL の二峰性対数尤度で $-\infty$ の下側漸近線が厳密に凹な内部の局所最頻値を上回る場合、flexpyirt は真の大域上限である `(-inf, NaN)`（本来の最尤推定）を意図的に返すのに対し、mirt 1.46.1 の EAP 出発の局所 `nlm` 探索は内部の局所最頻値で停止するため両者は異なる（`v11_3pl` の 3,790 ユニークパターン中 25 件。この判定により、`nlm` 有限差分 Hessian が `SEest > 30` 判定をすり抜ける境界ケースもあわせて正しく扱える。VALIDATION.ja.md V13 参照）。
24. **WLE スコアリング（`method="WLE"`、M4, §8）**: Warm (1989) の重み付き最尤推定を、Jeffreys / Firth (1993) の罰則付き対数尤度 $f_{\mathrm{WLE}}(\theta)=\log L(\theta\mid y)+\tfrac{1}{2}\log I(\theta)$ の $[-20, 20]$ 上での最大化として同じ `vmap` 済み安全装置つき 1 次元 Newton（初期値 EAP）で解く。ここで $I(\theta)=\sum_{j:\,y_j\ge 0}\sum_k P_{jk}(\theta)\,(\partial\log P_{jk}/\partial\theta)^2$ は観測項目にわたる Fisher テスト情報量であり、各カーネルの `log_prob` に対する `jax.jacfwd` のみで計算する（手書き微分なし、全項目ファミリーで動作）。停留条件は $\partial\log L/\partial\theta + J(\theta)/(2I(\theta)) = 0$（$J(\theta)=\partial I/\partial\theta$）と恒等的に一致し、2PL・Rasch・GRM・GPCM・PCM・3PL・4PL のすべてで `mirt:::WLE.mirt` と一致する（指数型分布族では $J=\partial I/\partial\theta$ が Warm の $\sum P'P''/P$ と恒等的に一致し、3PL / 4PL / GRM では Firth の Jeffreys 事前分布バイアス補正および mirt と一致する）。極端パターン（全最低・全最高）でも有限の推定値を与え、全欠測パターンは $I(\theta)=0$ のため `(NaN, NaN)` を返す。SE は JAX 自動微分による $(-f''_{\mathrm{WLE}}(\hat\theta))^{-1/2}$ であり、mirt の既定 `fscores(..., method="WLE", full.scores.SE=TRUE)` と一致する（Warm の漸近式 $I(\hat\theta)^{-1/2}$ は mirt の `expected.info=TRUE` に対応し、短いテストや極端パターンで差が出る。VALIDATION.ja.md V13 参照）。
25. **Plausible values（`method="PV"`、M4, §8）**: 回答者ごとに事後分布 $p(\theta\mid y_i;\mu_{g(i)},\sigma_{g(i)}^2)$ から独立な `n_draws` 個（既定 `5`）の標本を抽出し、`pv1..pvM` 列の DataFrame を返す。求積格子による離散化を避けて連続な標本を得つつ `jit` 内で静的形状・高速動作を保つため、圧縮・2 のべき乗パディングしたユニークパターンごとに推定済みモデルの求積区間 `[quad.lower, quad.upper]` 上の細かい等間隔格子（`501` 点）で事後密度を評価し、台形則で $[0, 1]$ 上の区分線形 CDF を構成したうえで、`jax.random.uniform(key, (N_pad, n_draws))`（回答者数 `N` も 2 のべき乗 `N_pad` へパディングして `N` にスライス）に対し分岐なしの `searchsorted(..., side="right")` 線形補間で逆 CDF 抽出を行う。同じ反応パターンの回答者は事後 CDF を共有しつつ独立な一様乱数を用いるため独立な標本となり、全欠測行は群の事前分布から抽出される。明示的な JAX PRNG の `key`（`jax.random.key` または `jax.random.PRNGKey`、未指定・不正は `SpecificationError`）が必須で、`key` に対して完全に決定的。
26. **`FitResult` の JSON 入出力 API とバージョン付き形式（M4, API §7）**: `FitResult.to_json(path=None, *, indent=2) -> str` は推定結果を厳格な JSON 文字列に直列化し、`path`（`str` または `os.PathLike`）が指定された場合は UTF-8 ファイルにも書き出す。`FitResult.from_json(source) -> FitResult` はファイルパスまたは JSON 文字列から再推定なしで復元する。素の JSON のみを使い、`pickle` や `eval` は一切使わない。最上位オブジェクトには `"format": "flexpyirt.FitResult"`、`"schema_version"`（§15.46 の命名リファクタから 4、双因子モデル（§17.1）から 5、two-tier モデル（§17.2）から 6、標本ウェイト（§17.4）から 7。それより前の version のファイルも読める）、および書き出した版を記録する `"flexpyirt_version": flexpyirt.__version__` を持たせる。`FitResult` は `_model`・`_quadrature` とあわせて推定設定 `_method: BAEM` も保持し、メタデータ全体が往復保存できるようにする。`from_json` は入口で `check_x64()` を呼ぶ（スコアリングに float64 が必要なため、無効なら `PrecisionError`）。
27. **厳格な JSON と正確な float64 往復保存（M4）**: 書き出しは `json.dumps(..., allow_nan=False)` で厳格な RFC 8259 JSON のみを出力し、読み込みは `json.loads` に `parse_constant` を渡して生の `NaN`/`Infinity` リテラルを `SpecificationError` で拒否する。有限な IEEE-754 float64 値は通常の JSON 数値（Python の `repr` 厳密な最短 10 進表現）として書き出し、`-0.0` を含むすべての有限 float64 をビット単位で正確に往復保存する。浮動小数点型のフィールド（無制約の `lower`/`upper`、固定または未計算の `se`、非正定値の `vcov` など）に現れる非有限値（`NaN`, `+Inf`, `-Inf`）は明示的に文字列 `"NaN"`, `"Infinity"`, `"-Infinity"` として符号化する。これらを浮動小数点へ復号するのはスキーマ上 float 型のスロットのみであるため、項目名や群名などの文字列フィールドとは衝突しない。
28. **明示的レジストリと完全な状態保存（M4）**: 項目モデル（`"2PL"`, `"Rasch"`, `"3PL"`, `"4PL"`, `"GRM"`, `"GPCM"`, `"PCM"`、多値モデルでは `n_categories` を含む）、潜在分布（`"Normal"`）、事前分布（`"NormalPrior"`, `"LogNormalPrior"`, `"BetaPrior"`）、求積（`"equispaced"`）、推定手法（`"BAEM"`）は `_json.py` の明示的な名前レジストリで（逆）直列化する。未知の `format` タグ・スキーマ版・レジストリ名・不正な JSON・整合しない表や行列形状は明確なメッセージつきで `SpecificationError` を送出する。完全な `Model` 仕様（項目名と順序、項目モデル、`fixed`, `latent`, `groups`, `equal`, `priors`）と、`estimate`・`se` を含む検証済み `ParameterTable`（ユーザーによる `table=` 編集もそのまま保持）、`vcov`、`loglik`/`log_posterior`、収束情報と `history`、`quadrature`、`method`、`se_method`、`n_obs` をすべて保存する。`_eta` は `parameters` の検証済み自由行から直接復元し（`free_id` が `0..n_free-1` に連続であること、共有 `free_id` の `estimate` が一致すること、`vcov` 形状が整合することを検証）、推定値の重複を持たせない。復元後の `FitResult` はすべての公開属性・`items`・`latent`・`summary()`・`_repr_html_()`・`aic`/`bic`・全 5 手法の `score()` で完全に同一の結果を返す。
29. **確認的 2–3 次元 MIRT の仕様とカーネル（M4, §4.9, §6）**:
    `Model.with_factors(factors, *, correlated=True)` で 2 または 3 個の名前つき因子と負荷項目の確認的パターン（`Mapping[str, ItemSelector]`）を指定する。1 項目が複数因子に負荷すること（交差負荷）を許し、各因子には 1 項目以上が必要で、すべてのモデル項目がいずれか 1 因子以上に負荷しなければならない。負荷しない因子の傾きは 0 に固定する。`with_factors()` を呼ばない `Model.uniform(...)` / `Model.from_items(...)` は厳密に 1 次元（`n_factors=1`）のままビット単位で挙動を変えない。4 つのカーネル（`DichotomousKernel`, `FourPLKernel`, `GradedKernel`, `GPCMKernel`）はすべて静的な `n_factors: int = 1` と `with_factors(n_factors)` を持ち、線形予測子 $z = \mathbf a^\top\boldsymbol\theta + d$（$D=1$ は `params[0] * theta[:, 0]`、$D \in \{2, 3\}$ は `theta @ params[:D]`）を評価するため、既存の 7 項目モデル（`TwoPL`, `Rasch`, `ThreePL`, `FourPL`, `GRM`, `GPCM`, `PCM`）がすべて 2・3 次元でそのまま動作する。$D > 1$ では内部パラメタ名が mirt に合わせて `a1..aD`, `d`（GRM / GPCM / PCM は `d1..d{K-1}`）となり、1 次元用の表示閾値（`b` / `bk`）は省略する。$D > 1$ での `with_latent()` と `fix_from(..., parameterization="irt")` は `SpecificationError` を送出する（$D > 1$ の多母集団は §15.43）。
30. **直積求積と多変量正規潜在分布の厳密 M-step（M4, §4.4, §4.9）**:
    `Quadrature.default(n_factors)` は mirt の `select_quadpts` に合わせた既定値（1 次元 `61`、2 次元 `31` → $961$ 点、3 次元 `15` → $3{,}375$ 点、区間 $[-6, 6]$）を返し、`Quadrature.grid(n_factors)` が `(Q, D)` の直積格子を返す（`nodes()` は 1 次元の `(n_points,)` のまま）。`CompiledProblem.nodes` の形状は `(Q, D)`。$D \in \{2, 3\}$ の多変量正規潜在分布は `ParameterTable` と `LatentBlock` 上で自然尺度の `(mean_1..D, var_1..D, cor_ij...)`（$L_2 = 5$, $L_3 = 9$、$1 \le i < j \le D$）としてパラメタ化する: $\Sigma_{ii} = v_i > 0$（`lower = 0`）、$\Sigma_{ij} = \rho_{ij}\sqrt{v_i v_j}$ かつ $\rho_{ij} \in (-1, 1)$（`lower = -1, upper = 1`）。分散が 1 固定（2PL / GRM / 3PL / 4PL / GPCM の既定）では `cor_ij` が $\Sigma_{ij}$ と一致するため mirt の `COV_ji` およびその SE と直接一致し、全項目が `Rasch` または `PCM` の場合は `cor_ij` に加えて `var_1..var_D` も既定で自由推定とする。`mvn_log_weights(nodes, params)` は $D = 1$ では従来の `normal_log_weights` にそのまま振り分け、$D \in \{2, 3\}$ では相関行列の正定値性を Sylvester の判定法（主座小行列式 $> 0$）と二重 `where` で保護しつつ、閉形式二次形式（$D = 2$）または Cholesky + `solve_triangular`（$D = 3$）と `log_softmax` で格子上の正規化対数重みを計算する。正定値領域内部では滑らかで NaN のない微分を与え、非正定値候補は NaN を返して境界つき Newton の直線探索および SQUAREM の安全装置で棄却される。1 次元（§4.4）と同様、潜在 M-step は離散化後の厳密な $Q_g^{\text{lat}} = \sum_q \bar n_{gq}\log w_{gq}$ を最大化する（EM の単調性と Oakes 恒等式の厳密性を保つ）。
31. **$D > 1$ の識別性・符号不定性・初期値（M4, §4.5, §4.10）**:
    `_check_identification_mirt` が推定前に必要条件を検査する: 各因子に有効な負荷（自由または非ゼロ固定）が 1 つ以上あること、自由な `var_m` には非ゼロ固定の `a{m}` が 1 つ以上あること、自由な `mean_m` にはその因子に負荷する項目のうち切片が固定されたものが 1 つ以上あること、確認的な回転識別のため $c_m$ 個の自由な相関に関わる因子 $m$ は少なくとも $c_m$ 個の固定負荷（通常は 0 固定）を持ち、非ゼロ固定負荷のない 2 因子が同一の自由負荷集合を共有しないこと（Bollen, 1989; Reckase, 2009）。**符号不定性**: $\theta_m$ とすべての $a_{jm}$・$\mu_m$・$\rho_{mm'}$（$m' \neq m$）の符号反転で周辺尤度は不変（1 次元と同様）。逆転項目や自然尺度 SE を不自然な境界で歪めないよう傾き自体は非有界（`(-inf, inf)`）のまま保ち、各因子の向きは因子別項目−残余相関から $1/\sqrt{L_j}$（$L_j$ は項目 $j$ が負荷する因子数。切片・漸近線には $\|\mathbf a_j^{(0)}\|$ を使用）で計算する厳密に正の既定初期傾き $a_{jm}^{(0)} \in [0.25, 3.0]$ と、常に正定値な相関初期値 `0.0`（$R^{(0)} = I_D$）によって正の方向に揃える。編集済み表である因子の初期傾きがすべて 0 の場合は `SpecificationError` で弾く。
32. **多次元の結果・Oakes SE・EAP スコアリング（M4, §4.8, §8）**:
    Core 層の E-step・M-step・SQUAREM・構造化 Oakes（`_oakes_own_hessian`, `_oakes_cross_hessian`）・チャンク分割 Hessian はすべて `(Q, D)` 節点と `(G, L_D)` 潜在ブロックに対して静的形状・自動微分のみでそのまま動作する。`FitResult.items` は `a1..aD`, `d`（`d1..`）とその SE を表示し、`FitResult.latent` は `mean_1..D`, `var_1..D`, `cor_ij` とその SE を表示する。`FitResult.score(data, method="EAP")` は宣言した因子順に `theta_<factor>`, `se_<factor>` 列を返す。`"MAP"`（§15.39）、`"ML"`・`"WLE"`（§15.41）、`"PV"`（§15.42）も使える。
33. **多次元 `FitResult` の JSON 入出力（schema version 2、M4）**: JSON のモデルオブジェクトに `"factors"`
    （`null`、または宣言順の `{"name", "items"}` のリスト）と `"correlated_factors"` を保存し、`to_json()` /
    `from_json()` が 2–3 次元の結果（`items`・`latent`・`parameters`・`vcov`・`summary()`・EAP スコア）を厳密に
    往復保存する。読み込み時は因子構造を `Model.with_factors()` で組み立て直すので、宣言時と同じ検証がかかり、
    固定値・等値制約・事前分布は多次元のパラメタ名（`a1`..`aD`）で検査する。因子を持つモデルは `latent`・`groups` を
    持ってはならない。`schema_version` は 2。因子を保存する前に書かれた version 1 のファイルも読み込めるが、
    新しい 2 つのキーを含んではならない。`model_parameter_schema()` は `build_table()` と同じく `a1`..`aD` に
    傾きの範囲を繰り返す。
34. **経験ヒストグラム（`EmpiricalHistogram`）の仕様・KKT 下限・モーメント制約付き参照群 M-step（M3, §4.4, §4.5, §12-1 解決）**:
    `EmpiricalHistogram(*, min_weight: float = 1e-10)` を `flexpyirt` から公開する不変 dataclass とし、生成時に
    `0 < min_weight < 1`、コンパイル時に `min_weight * Q < 1` を検証する（`SpecificationError`）。
    `Model.with_groups()` で `EmpiricalHistogram` と `Normal` を混在させると `SpecificationError` を送出する
    （EH を使うときは全群が EH）。重みの初期値は求積格子上の標準正規密度をフロア付き単体へ射影したものとする。
    EH の M-step では、標準化しない群（焦点群と FIPC 時の参照群）について
    $\max_{w\in\Delta^{Q-1},\,w_q\ge\tau}\sum_q\bar n_{gq}\log w_q$（$\tau=\texttt{min\_weight}$）を KKT water-filling
    $w_{gq}=\max(\tau,\,c_g^*\bar n_{gq})$ で厳密に解く（`jnp.sort` で $O(Q\log Q)$。どの節点も下限に当たらなければ
    $\bar n_{gq}/N_g$ と厳密に一致）。標準化する参照群（連結成分に固定切片がなければ `standardize_mean`、非ゼロの固定傾きが
    なければ `standardize_var`；FIPC では両方偽、Rasch / PCM では平均のみ。平均なしの分散のみは `IdentificationError`）では、
    線形のモーメント制約 $\sum_q w_q\theta_q=0$ と（`standardize_var` のとき）$\sum_q w_q\theta_q^2=1$ を加える
    （`eh_constrained_weights`）。$\phi_q=(1,\theta_q,\theta_q^2)$ と乗数 $\lambda$ について KKT 解は
    $w_q=\max\{\tau,\,p_q/\lambda^\top\phi_q\}$（$p_q=\bar n_{0q}/N_0$）で、$\lambda$ は滑らかな凸の双対
    $G(\lambda)=\sum_q[p_q\log w_q(\lambda)-(\lambda^\top\phi_q)w_q(\lambda)]+\lambda^\top c$ を
    $\{\lambda^\top\phi_q>0\ \forall q\}$ 上で最小化する（Boyd & Vandenberghe, 2004, §5.2）。$\lambda=(1,0,0)$ から
    減衰 Newton（未知数 $K\le 3$、`lax.while_loop`、定義域を出ないバックトラック）で、制約残差 $\nabla G$ が $10^{-13}$ 未満に
    なるまで解く。これは識別された集合上で潜在 Q 関数を厳密に最大化するので、EM は厳密に単調で、参照群ヒストグラムは毎反復で
    平均 0・分散 1 を満たし、項目パラメタは変換しない。すべての $p_q>0$（E-step の期待度数は常にそう）なら最小点は内部にある。
    ある $p_q=0$ でその節点に質量が必要な場合は双対の下限が境界にあり Newton が途中で止まるので、最終的な参照群モーメントが
    目標から $10^{-8}$ を超えてずれていれば `fit()` が `ConvergenceWarning` を出す。
35. **EH のパラメタ数カウントと条件付き項目標準誤差（M3, §4.8）**:
    EH の重みは `FitResult.n_params`（および `aic`/`bic`）にパラメタとして数える: 参照群は
    $Q - 1 - \mathbf{1}[\text{standardize\_mean}] - \mathbf{1}[\text{standardize\_var}]$（平均・分散とも標準化なら
    $Q-3$、Rasch / PCM は $Q-2$、FIPC は $Q-1$）、各焦点群は $Q-1$。標準誤差については、既定 61 点格子上の
    $G(Q-1)$ 個の離散ヒストグラム重みすべてを含めた同時観測情報行列は数学的にランク落ちする（解析的な IRT 項目尤度が
    $\mathbb R^Q$ 内の低次元部分空間しか張らず数値ランクが約 $20\ll 60$ となり、境界・フロア節点では曲率が 0 または負になる）。
    そのため、`information_oakes` と `information_hessian` は**収束した EH 重み $\hat w$ を固定した条件での自由な項目パラメタの
    条件付き観測（事後）情報行列**を計算する。これにより `vcov` の正定値性と `oakes == hessian` の厳密な恒等性（相対差
    $<10^{-10}$）が保たれ、R `mirt` の `vcov(m)` の形状（`n_free_items × n_free_items`）および `SE.type = "Richardson"`
    （`mirt:::BL.LL` が `EHPrior` を固定）の定義と一致する。EH モデルの `ParameterTable` は項目行のみを持つ
    （EH 重みは `Model.parameter_table()` ではなく `fit()` に渡される `Quadrature` に依存するため、`__latent__` 行を持たない）。
36. **EH に対する平方根重み尺度での SQUAREM（M3, §4.6）**:
    `EmpiricalHistogram` で `BAEM(squarem=True)` を使う場合、SQUAREM は $z_{gq}=\sqrt{w_{gq}}$ として
    $(\eta_{\text{items}},\,z)$ の結合ベクトル上で外挿し、候補は $w_{gq}\propto z_{gq}^2$（$z$ の符号によらず常に単体上）の後、
    期待度数に使うのと同じ M-step 写像（KKT 下限。標準化する参照群では §15.34 のモーメント制約付き I 射影で、実行可能な入力は
    そのまま返す）で戻す。射影がモーメント制約を $10^{-10}$ を超えて満たさない候補は NaN にし、単調性安全装置
    （`log_posterior(cand) >= max(log_posterior(theta2), log_posterior(theta0))`、$\delta$ を最大 10 回半減し不適なら
    $\theta_2$ へ退避）で棄却させる。平方根尺度では多項分布の Fisher 情報が等方的（$4I$）になる。最初に決めた中心化対数重み
    （softmax）尺度では、尤度にほとんど影響しないほぼ空の節点がステップ長を支配する（$w\approx10^{-9}$ で $\log w$ が 1 単位
    ずつ動く）。V6a と自由推定の 2PL の 2 例（21 点・61 点）で、softmax 尺度では安全装置が反復の 99% 超で作動したのに対し、
    平方根尺度では 1% 未満で、21 点の例では 3.5 倍速く収束し、他の例でも同じ反復数でより高い対数尤度に達した。通常 EM・SQUAREM
    ともに収束判定は $\max(\max_i|\Delta\eta_i|,\;\max_{g,q}|\Delta w_{gq}|)<\text{tol}$（自然尺度の項目変化と確率尺度の重み変化）を用いる。
37. **EH のスコアリング（M3, §8）**:
    `EmpiricalHistogram` モデルに対する `FitResult.score()` は、`"EAP"`（推定済み求積格子上で群ごとの推定 EH 重みを
    事前分布とする事後平均・SD）、`"ML"`・`"WLE"`（尤度のみ、Normal と同一）、および `"PV"`（各群の EH 密度を `prob.nodes`
    から 501 点の細格子へ線形補間・再正規化して台形則逆 CDF 抽出）に対応する。`method="MAP"` は 2 階微分可能な連続事前分布を
    要するため EH モデルでは `SpecificationError` を送出する（なお R `mirt` 1.46.1 の `fscores(..., method="MAP")` は
    `EHW` 重みを暗黙に無視して $N(0,1)$ を使う）。
38. **EH の結果報告と JSON 入出力（M3, §2 ④, §15.26–§15.28）**:
    EH モデルでは、`FitResult.latent` が群ごとの 4 次までの要約モーメント — 列
    `("group", "mean", "var", "sd", "skewness", "kurtosis")`（`kurtosis` は Fisher の超過尖度 $\mu_4/\sigma^4 - 3$、
    正規分布で `0`）— を返し、新しい属性 `FitResult.latent_weights` が列 `("group", "node", "weight")` の DataFrame
    を返す（`Normal` モデルでは `None`）。モーメント表と重み表の双方を `FitResult.summary()` と `FitResult._repr_html_()`
    に含める。`_json.py` の潜在分布レジストリに `"EmpiricalHistogram"`（`{"type": "EmpiricalHistogram", "min_weight": ...}`）
    を登録し、EH 結果にはトップレベルキー `"latent_weights"`（形状 `(G, Q)`。読み込み時に有限・厳密に正・和が 1・`n_groups`
    および `quadrature.n_points` との整合を検証；`Normal` モデルでは省略かつ存在すれば拒否し、既存の schema_version 1 の
    ペイロードと完全互換）を保存する。`_n_eh_params` は読み込み時に `(model, parameters, quadrature)` から決定的に再計算する。
39. **多次元モデルの MAP スコアリング（M4, §8）**: $D \in \{2, 3\}$ の `score(method="MAP")` は、推定した多変量正規
    （$\Sigma_{ij}=\rho_{ij}\sqrt{v_iv_j}$）を連続な事前分布として
    $f(\boldsymbol\theta)=\log L(\boldsymbol\theta\mid y)-\tfrac12(\boldsymbol\theta-\boldsymbol\mu)^\top\boldsymbol\Sigma^{-1}(\boldsymbol\theta-\boldsymbol\mu)$
    を EAP から出発して最大化する。安全装置つき $D$ 次元 Newton（`_newton_nd`: 自動微分の Hessian が負定値なら Newton 方向、
    そうでなければ勾配。ステップのノルムを 10 で打ち切り、ステップ幅 $2^{-k}$（$k=0..29$）を同時に評価して $f$ が減少しない
    ようにする）を使う。SE は mirt の `fscores(method = "MAP")` と同じく
    $\big[(-\nabla^2 f(\hat{\boldsymbol\theta}))^{-1}\big]_{mm}^{1/2}$。全欠測行は事前分布の平均と $\sqrt{\Sigma_{mm}}$ を返す。
    $D > 1$ の `ML`・`WLE` は §15.41、`PV` は §15.42。
40. **S-X2 項目適合度（M5、`FitResult.item_fit`）**: `Normal` 潜在分布の単一群 1 次元モデルと欠測のないデータに対する
    Orlando と Thissen（2000, 2003）の S-X2。期待割合
    $E_j[s,k]=\sum_q P_{jk}(\theta_q)L_{-j}(s-k\mid\theta_q)w_q/\sum_q L(s\mid\theta_q)w_q$ を、推定時の求積格子と潜在分布の重みの上で
    Lord–Wingersky 漸化式（Lord & Wingersky, 1984）から合計得点 $1,\dots,S-1$ について求め、各得点の観測人数を掛ける。
    度数の少ないセルは `mirt:::collapseCells`（mirt 1.46–1.48）を 1 行ずつ移植した処理でまとめる（1 件だけの行をまとめる前に
    計算した行のフラグを後で使う点も含め、統計量と自由度が一致するようにヒューリスティックをそのまま再現する）。
    $df_j$ = 残ったセル数 − 行数 − 項目 $j$ の自由パラメタ数 − 潜在分布の自由パラメタ数で、mirt と同じ（mirt はすべての項目から
    潜在分布のパラメタ数を引く）。RMSEA と $\chi^2$ の p 値も mirt に合わせる。表の計算は推定後の NumPy 処理（形がデータに依存
    するので `jit` の外）で、カテゴリ確率は JAX の項目カーネルから得る。EH・欠測データは照合するまで
    `SpecificationError`（mirt は S-X2 で欠測を補完し、EH の重みを無視する）。多母集団は §15.45、確認的多次元モデル
    （直積格子）は §16.19。
41. **多次元モデルの ML・WLE スコアリング（M4, §8）**: $D \in \{2, 3\}$ で、`"ML"` は $\log L(\boldsymbol\theta\mid y)$ を、
    `"WLE"` は Jeffreys/Firth の罰則つき $\log L(\boldsymbol\theta\mid y)+\tfrac12\log\det\mathbf I(\boldsymbol\theta)$
    （`mirt:::WLE.mirt`; Warm, 1989; Wang, 2015。$\mathbf I$ は観測項目の期待テスト情報行列で自動微分から求める）を、
    範囲 $|\theta_m|<20$（mirt の `max_theta`）で、範囲つきの `_newton_nd` により EAP から出発して最大化する。SE は目的関数の
    負の Hessian の逆行列の対角の平方根。ML: 観測された反応がすべて最低（最高）の生カテゴリのパターンは全次元で
    $-\infty$（$+\infty$）を返す（mirt も最適化の前にそうする）。範囲の端に達するか SE が 30 を超えた次元は $\pm\infty$
    （SE は NaN）を返し、残りの次元の SE は負の Hessian のその次元のブロックの逆行列から求める。mirt はそのような次元を
    $\pm20$ 近くの有限値で SE 0 として止めることがある（`diag(hessian) > 0` による部分選択のため）が、flexpyirt は符号付きの
    無限大を返す。WLE は極端なパターンでも有限。全欠測行は NaN を返す。
42. **多次元モデルの plausible values（M4, §8）**: $D \in \{2, 3\}$ で、ユニークなパターンごとの事後分布
    $\log L(\boldsymbol\theta\mid y)-\tfrac12(\boldsymbol\theta-\boldsymbol\mu)^\top\boldsymbol\Sigma^{-1}(\boldsymbol\theta-\boldsymbol\mu)$
    （連続な多変量正規の事前分布）を、EAP ± 6 事後 SD（推定時の格子の範囲で切る）にわたる次元あたり 61 点（2 次元）または
    25 点（3 次元）のそのパターン専用の等間隔格子で評価する。各抽出は離散化した事後分布からセルを選び
    （`jax.random.categorical`）、セル内で一様なゆらぎを加えるので連続な値になる。ゆらぎにより分散は $\Delta_m^2/12$
    （2 次元で事後分散の約 0.2%、3 次元で約 0.6%）大きくなる。回答者ごとの鍵は `jax.random.split(key, N)` から作るので、
    同じパターンの回答者でも独立で、`key` に対して決定的。メモリを抑えるためパターンと回答者を `lax.map` で分割して処理する。
    列名は `pv<m>_<factor>`。統計的に検証する: 4,000 回抽出すると標本平均・SD・相関が格子上の事後分布とモンテカルロ誤差の
    範囲で一致する（mirt の既定の plausible values は正規近似なので参照にはしない）。
43. **多次元の多母集団モデル（M4, §4.5, §4.9）**: `with_factors()` と `with_groups()` は順序を問わず組み合わせられる
    （潜在分布は `Normal` のみ。`EmpiricalHistogram` は 1 次元のまま）。各群の `Normal` はすべての因子に適用され、その
    `mean` / `var` と `free_mean` / `free_var` がその群の `mean_1..D` / `var_1..D` を決め、`cor_ij` はすべての群で
    `with_factors(correlated=...)` に従う（mirt の `COV = F1*F2` での `multipleGroup(..., invariance = c(anchors, "free_means",
    "free_var"))` と同じ）。識別性は、§15.31 の回転の検査を群ごとに行い、尺度・位置のつながりを因子ごとに群をまたいで調べる:
    群は共有する自由な `a{m}`（尺度）や因子 $m$ に負荷する項目の共有する自由な切片（位置）でつながり、自由な `var_m` / `mean_m`
    は固定された非ゼロの `a{m}`・固定された切片・そのパラメタを固定した群のいずれかにつながっていなければならない（なければ
    群と因子を示して `IdentificationError`）。JSON の形式はすでに群と因子を保存する（§15.33）。因子を持つモデルは `Normal` の
    群を持てる。
44. **Gauss–Hermite 求積（`Quadrature.gauss_hermite(n)`、§4.9）**: $e^{-x^2}$ に対する Gauss–Hermite 則 $(x_q, w_q)$
    （Golub & Welsch, 1969; `numpy.polynomial.hermite.hermgauss`）から固定節点 $\theta_q=\sqrt2x_q$ と、節点ごとの固定の
    対数基準重み $b_q=\log(w_q/\sqrt\pi)+\tfrac12\theta_q^2$（直積則では次元について和）を作る。潜在分布の重みは
    $\operatorname{softmax}_q\big(b_q+\log\phi(\boldsymbol\theta_q;\boldsymbol\mu,\boldsymbol\Sigma)\big)$、つまり $N(0,1)$ の
    Gauss–Hermite 則を $\phi(\cdot;\boldsymbol\mu,\boldsymbol\Sigma)/\phi(\cdot;0,I)$ で重み付け直したもので、節点は動かず、
    自由な平均・分散・相関でも EM の構造（E-step、厳密な潜在 M-step、Oakes、SQUAREM）はそのまま。
    `CompiledProblem.node_log_base` が $b_q$（等間隔格子では 0）を持ち、潜在分布の対数重みを作るすべての箇所
    （`_normal_logw`）で加える。潜在分散が 1 より十分大きいと裾で重みの比が大きくなり精度が落ちる。`EmpiricalHistogram` は
    等間隔格子が必要（`SpecificationError`）。$D=1$ の PV は自前の細かい等間隔格子で抽出する（基準重み 0）。JSON は
    `{"type": "gauss_hermite", "n_points": n}` を保存する。`scipy.integrate.quad` と比べて検証（$N(0.3, 1.4)$ の周辺対数尤度、
    61 節点で相対誤差 $1.2\times10^{-10}$）し、非常に細かい等間隔格子での推定とも比べた（分散自由の Rasch、41 節点と
    $[-12,12]$ の 401 点: 推定値は $2\times10^{-10}$ 以内。2 次元 2PL、$31^2$ と $121^2$: $5\times10^{-7}$ 以内）。
    mirt は EM で Gauss–Hermite を使わない。
45. **多母集団モデルの S-X2（M5, §15.40）**: `item_fit(data, group=...)` は、mirt の `itemfit()` が `multipleGroup` の
    推定結果に対して行う（`MGC2SC`）のと同じく、群ごとに S-X2 を計算する。群 $g$ の回答者から観測表と $N_s$ を作り、
    期待割合には群 $g$ の項目パラメタと潜在分布の重み $w_{gq}$ を使い、$df_{gj}$ = 残ったセル数 − 行数 − 群 $g$ での
    項目 $j$ の自由パラメタ数 − 群 $g$ の潜在分布の自由パラメタ数（群間で共有するアンカーはどの群でも数え、平均 0・分散 1
    を固定した参照群は引かない）。RMSEA は群の $N_g$ を使う。各群ですべてのカテゴリが観測されている必要がある。結果の先頭に
    `group` 列がつく。mirt 1.48 と照合した（V15）: V4 の 2 群 2PL を mirt の推定値で、Science（GRM）をすべての項目を等値にして
    2 群に分けたものを mirt の推定値で（mirt はそこで項目パラメタが約 2e-4 収束しきらない）評価し、自由度は一致、S-X2 は
    相対 1e-6 以内。

46. **命名のリファクタ（スキーマ版 4）**: 公開名を mirt/R の引数名ではなく Python の慣習に合わせる。項目モデルは
    `n_categories`（旧 `n_cat`）、`Quadrature.equispaced`・`Quadrature.default` は `bounds=(lower, upper)`（旧 `lim`）を
    取り、既定の点数は `quadrature.DEFAULT_N_POINTS`。事前分布はすべて `Prior` で終わる（`NormalPrior`、
    `LogNormalPrior(mean_log, sd_log)`、`BetaPrior(a, b)`。旧 `LogNormal(meanlog, sdlog)`・`Beta`）。
    `FitResult.item_fit(min_cell=...)`（旧 `mincell`、mirt の名前）。文字列の選択肢は snake_case
    （`parameterization="slope_intercept"`、`link(method="stocking_lord")`）。`Model.is_eh` は
    `Model.is_empirical_histogram` にする。未リリースのため旧名は非推奨エイリアスを残さずに削除する。JSON の
    `schema_version` は 4 で、項目モデルは `n_categories` を保存し、事前分布は `"NormalPrior"`・`"LogNormalPrior"`
    （キー `mean_log`・`sd_log`）・`"BetaPrior"` で登録する。version 1–3 のファイルも読める: 検証の前に
    `_upgrade_legacy_names` が `n_cat`・`LogNormal`/`meanlog`/`sdlog`・`Beta` を version 4 の名前に置き換える。
    内部カーネルの数式記号（`G`, `Q`, `K_f` など）と `LinkResult.A`/`B`（Kolen & Brennan, 2014 のリンキング定数）は
    そのまま残し、mirt のセル併合の移植（`_itemfit.collapse_cells`）は行ごとに照合できるよう mirt の局所変数名を保つ。

## 16. 実装上の決定（M5）
1. **項目をまたぐ等値制約（`Model.equal_across_items`、`free_id` の共有）**:
   `Model.equal_across_items(*, items, params)` は、選んだ項目（2 つ以上）の `params` の各パラメタを各群の中で等値にする。
   自由な行に同じ `free_id` を与えて表す（mirt の `CONSTRAIN = (items, param)` に相当。多母集団では `multipleGroup` の
   `CONSTRAIN` と同じく群ごとに適用）。`equal_across_groups` と組み合わせられ、モデルのすべての等値制約は行の
   union–find でまとめるので、重なる制約は 1 つの id を共有する。固定された行は影響を受けない。`free_id` を共有する行は
   （モデルからでも編集した表からでも）異なる項目・異なる項目モデルにまたがってよい（例: 2PL と GRM の共通傾き）。
   ただし 1 つの項目の中では `free_id` は 1 つのパラメタにだけ対応し、行どうしの範囲と事前分布は一致しなければならない
   （従来どおり）。共有パラメタの既定の初期値は、各行の既定初期値の平均。
   **M-step**: 期待完全データ対数事後は $Q(\eta)=\sum_j Q_j(\xi_j(\eta))+\log\pi(\eta)$ なので、項目ごとの分解（§4.3）は
   項目どうしではなく、「`free_id` の共有を辺とする項目グラフの連結成分」どうしの間で成り立つ。Compile 層は項目の
   union–find で成分を求め、2 項目以上の成分を `CoupledBlock` にする（メンバーの family ごとに静的な family 番号・動的な
   項目位置・成分の自由パラメタベクトル $u_c$ への `(G, n, P_f)` の局所インデックス、および $u_c$ の `free_id` と範囲）。
   その M-step は $\sum_{j\in c}\sum_g Q_{jg}(\xi_j^{(g)}(u_c))+\sum_s\log\pi(u_{cs})$ を $u_c$ について同じマスク付き・範囲つき・
   単調な Newton（§13.1、§15.2。勾配と Hessian は自動微分、メンバー family ごとに項目を `vmap`）で最大化する。共有
   パラメタの事前分布は 1 回だけ数えられ、EM の単調性も保たれる。成分の項目は項目ごとの `vmap` ではマスクして
   （スロットを動かさず）、成分のステップをその後に実行する。成分のパラメタは前の反復値から読む（ほかのブロックと
   重ならない）。成分は静的な Python ループなので、小さな成分が多いモデルは成分ごとに Newton ブロックをトレースする。
   2PL の $J$ 項目共通傾きは $1+J$ パラメタの 1 ブロックで、密な Hessian でも想定する規模では軽い。
   **標準誤差と SQUAREM**: Oakes の第 1 項は各項目の Hessian ブロックを `free_id` で足し込み、交差項もスコアを
   `free_id` で集計するので、項目をまたぐ共有パラメタは変更なしで厳密に合算される（周辺対数事後を直接微分する
   `information_hessian` と SQUAREM の範囲も同様）。デルタ法 SE、`n_params`（`free_id` ごとに 1 回）、JSON
   （モデルの `equal_items` フィールド、スキーマバージョン 3）は表から従う。
   **識別**: §4.5・§14.5 の必要条件は変えない（例: 共通の自由傾きと自由な潜在分散は、固定された傾きがないので引き続き拒否）。
2. **多母集団の尤度比検定による DIF（`FitResult.dif`）**:
   `FitResult.dif(data, *, group, items, params=None, scheme="add", p_adjust="holm", item_names=None)` は、推定済みの多母集団モデルの
   群間で `items` の各項目の特異項目機能（DIF）を検定する（Thissen, Steinberg & Wainer, 1993。mirt の `DIF()` に相当）。
   `scheme="add"` では推定済みモデルを基準とし、各項目の `params`（既定: その項目の全パラメタ）を群間で等値にする。
   `scheme="drop"` では逆に群ごとに自由にする（群ごとに別の `free_id`）。変更したモデルを同じ `BAEM` 設定・求積で
   基準の推定値から推定し直し、$X^2=2(\ell_{\text{free}}-\ell_{\text{constrained}})$、$df=|k_{\text{base}}-k_{\text{new}}|$
   から $p=P(\chi^2_{df}>X^2)$ を求める。結果は項目ごとに 1 行で、`item`、`converged`、AIC・SABIC
   （$-2\ell+k\log((N+2)/24)$）・HQ（$-2\ell+2k\log\log N$）・BIC の差（項目を自由にしたモデル − 制約したモデル。mirt と同じ符号で、負なら DIF を支持）、`x2`、`df`、
   `p`、`adj_p`（検定した項目にわたる補正。既定は mirt の `p.adjust = "holm"` と同じ Holm (1979)、`p_adjust="BH"` で Benjamini & Hochberg (1995)、`"none"` で補正なし）。変更は
   パラメタ表で行うので識別性は通常の推定と同じく検査され（唯一のアンカーを自由にすると `IdentificationError`）、
   すでに等値（`"add"`）またはすでに自由（`"drop"`）な項目は `SpecificationError`。MAP 推定は拒否する（統計量は
   最尤推定値での周辺尤度を必要とするため）。DIF は Result 層の推定後の処理で、`jit` で動くのは再推定だけ。
3. **限定情報の適合度（`FitResult.m2`、$M_2$ / $M_2^*$）**:
   `FitResult.m2(data, *, item_names=None, ci=0.90)` は、Maydeu-Olivares & Joe (2005, 2006) に基づく
   項目得点の 1 次・2 次モーメント — $J$ 個の項目平均 $E[y_j]$ と $\binom{J}{2}$ 個の交差積 $E[y_iy_j]$
   （$i < j$、列優先のペア順で mirt の既定 `M2(type = "M2*")` と同じ。2 値項目では厳密に $M_2$）— による
   限定情報の適合度統計量を計算する。対象は単一母集団・欠測なし・`Normal` 潜在分布の MLE 推定のみ
   （多母集団・双因子 / two-tier・`EmpiricalHistogram`・欠測データ・MAP 推定は `SpecificationError`。直積格子上の
   確認的多次元モデルは §16.19）。
   標本周辺モーメント $\hat p_2$、推定時の求積格子と正規化した正規重みの上で積分したモデル周辺モーメント
   $\pi_2(\hat\eta)$、局所独立のもとでの $\sqrt{N}(\hat p_2-\pi_2)$ の漸近共分散 $\Xi_2$
   （条件付き 1 次・2 次モーメント $m_{1,jq}=\sum_k kP_{jk}(\theta_q)$、$m_{2,jq}=\sum_k k^2P_{jk}(\theta_q)$ から構成）、
   および自由パラメタベクトル $\eta$ に関する JAX 前進モード自動微分のヤコビ行列
   $\Delta_2=\partial\pi_2/\partial\eta^\top$（共有パラメタは 1 回だけ現れ、等値制約の列和と一致）を用いて、
   $$M_2 = N\,(\hat p_2-\pi_2)^\top C_2\,(\hat p_2-\pi_2),\qquad
     C_2 = \Delta_c(\Delta_c^\top\Xi_2\Delta_c)^{-1}\Delta_c^\top,\qquad
     df = s - q,$$
   とする。ここで $s = J + \binom{J}{2}$、$q = \text{n\_params}$（自由な潜在分布パラメタを含む全自由パラメタ数）、
   $\Delta_c$ は $\Delta_2$ の SVD から得る $\operatorname{col}(\Delta_2)$ の直交補空間の正規直交基底である。
   漸近 $\chi^2_{s-q}$ 分布が成り立つには $df = s - q > 0$ かつ列フルランク $\operatorname{rank}(\Delta_2) = q$
   （1 次・2 次モーメントからの局所識別可能性。Maydeu-Olivares & Joe, 2005, Theorem 1; 2006, Theorem 2）が
   必要である。$df \le 0$ または $\operatorname{rank}(\Delta_2) < \operatorname{ncol}(\Delta_2)$（例: 多値項目の自由
   パラメタ数が項目数を超える $P_j > J$ や傾きが 0 に固定された場合、$df > 0$ であっても起こりうる）のときは、
   1 次・2 次モーメントからは局所識別できない（または残差自由度がない）旨の `FlexpyirtWarning` を出し、
   `m2`・`p`・RMSEA・信頼区間・`tli`・`cfi` をすべて `NaN` とする（`df` と `srmsr` は有限のまま）。
   $(s - r)$ 次元の 2 次形式を $df = s - q$ と組み合わせることは決してしない。
   **$\Delta_2$ における自由な潜在分布パラメタ**: mirt 1.46.1 は自由な潜在平均・分散の列を $\Delta_2$ から除きつつ
   $df$ からは引いている。flexpyirt は Maydeu-Olivares & Joe (2005, 2006) に従い、既定ではすべての自由パラメタ
   （項目・潜在分布とも）を $\Delta_2$ に含める。内部関数 `_m2.m2_statistic` に非公開の `_latent_in_delta: bool = True`
   引数を持ち、検証テストで理論定義と `False` 指定時の mirt 一致の両方を確認できるようにしている（自由な潜在パラメタが
   ないモデルでは両者は一致する）。
   **補助適合度指標**:
   - $p = P(\chi^2_{df} > M_2)$（生存関数）。
   - $\mathrm{RMSEA} = \sqrt{\max(0, M_2/df - 1)/(N - 1)}$（$df \le 0$ または $N \le 1$ で `NaN`）。$(100\,\mathrm{ci})\%$ 信頼区間は
     非心 $\chi^2$ 分布の CDF を逆算して $\sqrt{\lambda/(N\,df)}$ で求める（MacCallum, Browne & Sugawara, 1996。
     mirt の `RMSEA.CI` と同じく、$P(\chi^2_{df}\le M_2)$ がすでに目標確率以下のときは 0 とし、上側端点の探索区間は
     $\max(N, 5M_2)$ で根が挟まれない場合に倍増して拡張し、上端が誤って 0 に潰れないようにする）。列名は `ci` に従う
     （例: `ci=0.90` で `rmsea_5`, `rmsea_95`）。
   - $\mathrm{SRMSR} = \sqrt{\binom{J}{2}^{-1}\sum_{i<j}(r_{ij} - \hat\rho_{ij})^2}$（観測とモデルから導く項目得点相関行列の差。
     モデル分散の対角要素は $E[y_j^2] - E[y_j]^2$）。
   - **帰無モデル・TLI・CFI**: 全項目が 2 値（全項目で $K_j = 2$）のとき、すべての傾きを 0 に固定した独立（帰無）モデルは
     各項目 1 個の自由切片を持ち、MLE は観測比率を再現する。このとき直交補空間上の 2 次形式は解析的に
     $M_{2,\text{null}} = N\sum_{i<j} r_{ij}^2$、$df_{\text{null}} = \binom{J}{2}$ へ簡約される（閉形式・厳密、mirt の
     `M2(null.mod)` と一致）。$M_{2,\text{null}} > M_2$ かつ $df > 0$ のとき、
     $\mathrm{TLI} = (M_{2,\text{null}}/df_{\text{null}} - M_2/df)/(M_{2,\text{null}}/df_{\text{null}} - 1)$（上限クリップなし）、
     $\mathrm{CFI} = \operatorname{clip}(1 - (M_2 - df)/(M_{2,\text{null}} - df_{\text{null}}), 0, 1)$ とし、
     $M_{2,\text{null}} \le M_2$ では `FlexpyirtWarning` を出して両者を `NaN` とする。
     多値項目（$K_j > 2$）を含むモデルでは `tli` と `cfi` を `NaN` とする。$M_2^*$ の項目得点モーメントでは、
     帰無モデルの $\Delta_{\text{null}}$ において各項目の $K_j - 1$ 本の切片列が厳密に線形従属となるため
     （$\operatorname{rank}(\Delta_{\text{null}}) = J < \sum_j(K_j - 1)$）、$df_{\text{null}} = s - \sum_j(K_j - 1)$ による
     帰無 $M_2^*$ は定義不能である（mirt 1.46.1 はランクを確認せず `qr.Q(qr(delta), complete = TRUE)` の `ncol(delta)`
     列目以降を取り出しており、項目順や丸め誤差に依存して任意の方向を捨てている）。
   モーメント積分と $\Delta_2$ は JAX、$\Xi_2$・SVD 直交補空間・各適合度指標は `jit` の外の NumPy/SciPy で計算する。
4. **Davidian 曲線潜在分布（`DavidianCurve`、§4.4、V19）**:
   `DavidianCurve(degree, *, mean=0.0, var=1.0, free_mean=False, free_var=False)`（および
   `DavidianCurve.free(degree, *, mean=0.0, var=1.0)`）は、次数 $k\in\{1,\dots,10\}$ の単一群 1 次元 Davidian 曲線密度を
   指定する（Zhang & Davidian, 2001；Woods & Lin, 2009）。多母集団（`Model.with_groups`）および多次元
   （`Model.with_factors`）での `DavidianCurve` は `SpecificationError` を投げる（mirt 1.46.1 は Davidian 曲線の自由平均・
   分散パラメタと標準誤差を禁止しているため）。
   **密度と解析的標準化**:
   $\psi_j(z)=He_j(z)/\sqrt{j!}$（$j=0,\dots,k$）を正規化確率論的 Hermite 多項式（$\varphi(z)=(2\pi)^{-1/2}e^{-z^2/2}$ に対して
   正規直交）とし、$P_k(z;\boldsymbol\phi)=\sum_{j=0}^k c_j(\boldsymbol\phi)\psi_j(z)$ により
   $h_k(z;\boldsymbol\phi)=P_k(z;\boldsymbol\phi)^2\varphi(z)$ とする。単位ベクトル $\mathbf c(\boldsymbol\phi)\in S^k$ には
   余弦優先の球面座標 $c_0=\prod_{m=1}^k\cos\phi_m$、$c_j=\sin\phi_j\prod_{m=j+1}^k\cos\phi_m$（$j=1,\dots,k$）を採用する。
   `dcurver` の正弦優先座標（$c_0=\sin\phi_1$）は $N(0,1)$ を $\partial\mathbf c/\partial\phi_{2:k}=\mathbf 0$ となる極特異点
   $\phi_1=\pi/2$ に置く。本座標系は $N(0,1)$ をヤコビアンが単位行列 $\partial c_j/\partial\phi_m|_{\mathbf 0}=\delta_{jm}$
   となる $\boldsymbol\phi=\mathbf 0$ に置き、かつ $\phi_k=0$ で次数 $k-1$ を厳密に内包する。両座標系はともに $S^k$ 全体を覆うため
   同一の密度族を張る（$P_k^2$ を変えない $\mathbf c\mapsto-\mathbf c$ を除いて識別される）。ただし座標系だけでは $N(0,1)$ は
   停留点でなくならない: 1 次の摂動 $\psi_1(z)\varphi(z)$ と $\psi_2(z)\varphi(z)$ は純粋な位置・尺度の変化であり、解析的標準化で
   除去される（mirt の未標準化パラメタ化では自由な項目パラメタが吸収する）ため、$\boldsymbol\phi=\mathbf 0$ における
   $\phi_1,\phi_2$ のスコアはどのデータでも恒等的に 0 になる（$k=1$ では $\phi=0$ が常に停留点）。mirt 1.46.1 が V19b
   （$k=2$）で $N(0,1)$ に停滞するのはこの理由と $\phi_2$ の極特異点の組み合わせによる。flexpyirt は後述の摂動つき既定初期値で
   これを脱する。
   **$\boldsymbol\phi=\mathbf 0$ での情報行列の特異性**: このため Fisher 情報行列は $\boldsymbol\phi=\mathbf 0$ で
   $(\phi_1,\phi_2)$ 方向に特異になる（$k=1$ では $\phi_1$ 方向）。真の潜在密度が正規に近いと $\hat{\boldsymbol\phi}$ は
   不安定に定まり、尤度が複数の局所最大を持ちうるうえ、角度の SE（とそれに基づく Wald 検定）は信頼できず、`Normal` に対する
   `DavidianCurve` の尤度比検定は通常の $\chi^2$ 参照分布に従わない（帰無仮説がモデルの 1 次の識別が失われる点にある）。
   次数の比較は `Normal` との尤度比検定ではなく AIC/BIC で行う（Woods & Lin, 2009 と同様）。
   Hermite の 3 項漸化式より $Z\sim h_k(\cdot;\boldsymbol\phi)$ の連続モーメントは閉形式で
   $$\mu_Z(\boldsymbol\phi)=2\sum_{j=0}^{k-1}\sqrt{j+1}\,c_jc_{j+1},\qquad
     \sigma_Z^2(\boldsymbol\phi)=\sum_{j=0}^k(2j+1)c_j^2+2\sum_{j=0}^{k-2}\sqrt{(j+1)(j+2)}\,c_jc_{j+2}-\mu_Z(\boldsymbol\phi)^2>0$$
   と与えられる。$\theta=\mu+\sigma(Z-\mu_Z)/\sigma_Z$ により解析的に標準化すると、密度
   $f_k(\theta;\mu,\sigma^2,\boldsymbol\phi)=(\sigma_Z/\sigma)h_k(z(\theta);\boldsymbol\phi)$
   （$z(\theta)=\mu_Z(\boldsymbol\phi)+\sigma_Z(\boldsymbol\phi)(\theta-\mu)/\sigma$）は任意の $\boldsymbol\phi\in\mathbb R^k$ で
   厳密な連続平均 $\mu$・分散 $\sigma^2$ を持つ（また $\mu=\mu_Z$、$\sigma^2=\sigma_Z^2$ と置けば `dcurver::ddc` に厳密一致する）。
   **格子重み・M-step・mirt との比較**:
   求積対数重みは $\log w_q=\operatorname{log\_softmax}_q(\log(P_k(z_q)^2)-\tfrac12 z_q^2)$ で格子上に正規化する
   （多項式の厳密な零点 $P_k(z_q)=0$ は二重 `where` でガードする）。角度 $\phi_1,\dots,\phi_k$ は最適化中 `(-inf, inf)` の
   無制約とし、係数 $c_j$ の符号反転を Newton や SQUAREM が妨げられないようにする。`ParameterTable` は
   `item="__latent__"`、`model="Davidian"` の下に `mean`・`var`・`phi_1..phi_k` を持ち、識別性検査は §4.5
   （`Normal` と同じ平均・分散ルール、潜在パラメタへの事前分布は `SpecificationError`）に従う。解析的標準化により
   $\boldsymbol\phi=\mathbf 0$ で $\phi_1,\phi_2$ の 1 次微分が相殺するため、既定初期値には合計得点の経験的歪度の符号 `sign` に
   合わせた微小摂動（$k=1$ で `phi_1 = 0.1 * sign`、$k\ge 2$ で `phi_1 = -0.1 * sign, phi_2 = 0.1, phi_{3..k} = 0`）を与える。
   これらの初期値では、標準化した初期密度の 3 次標準化モーメントが `sign` と同符号になる: 主要項は $k=1$ で
   $\approx 4\phi_1^3$（$\phi_1=0.1$ で `+0.0039`）、$k\ge 2$ で $\approx -8.3\,\phi_1\phi_2$（$(\phi_1,\phi_2)=(-0.1,0.1)$ で
   `+0.083`）であり、$\phi_2=0.1$ は初期値を $\boldsymbol\phi=\mathbf 0$ の停留点から外す（`tests/test_m5_davidian.py` で検査）。
   潜在分布の M-step は $Q^{\text{lat}}=\sum_q \bar n_q\log w_q$ を既存のマスク付き Newton（`_newton.py`）で最大化し、
   EM の単調性（非減少）・特別処理なしの SQUAREM・相対 $\sim 10^{-12}$ での Oakes == Hessian SE を保つ（対照的に mirt 1.46.1 は
   期待度数に Woods の線形補間 `standardizeQuadrature` をかけてから未正規化の `ddc` を `nlminb` で最適化するため、EM 単調性が
   崩れ、`SE = TRUE` と自由平均・分散が禁止される。V19a では mirt 自身の離散化尤度に対しても停留点でない点（`PHI_3` の
   スコア $\approx 56.7$）で停止し、V19b では $N(0,1)$ の停留点に停滞する。いずれも対数尤度は真に低い。VALIDATION.ja.md V19 参照）。
   **スコアリング・出力・JSON**:
   `FitResult.latent` は `mean`・`var`・`phi_1..phi_k` とその SE を返し、`latent_weights` はパラメトリック分布のため `None`。
   `score()` は `"EAP"`（求積格子上）と `"PV"`（501 点細密格子上）で推定した Davidian 密度を用い、`"ML"` と `"WLE"` は尤度のみ、
   `"MAP"` は $P_k(z)^2\varphi(z)$ が多峰になったり実根で密度 0 になったりするため `SpecificationError`（mirt の
   `use_dentype_estimate` の制限と同じ）。`item_fit()` は検証されるまで `DavidianCurve` を拒否する。`_json.py` に
   `"DavidianCurve"` を登録し、JSON ペイロードの形は変わらないため `schema_version` は 3 のままとする。
5. **微分可能な IRT 尺度リンキング（`link`、`LinkResult`、V20）**:
   別々に推定した 2 つの 1 次元尺度を共通項目を介して結ぶ特性曲線法（§1）。`source`（新フォーム）の尺度を
   `target`（基準フォーム）の尺度へ写す線形変換 $\theta^* = A\theta + B$（$A > 0$）の係数 $(A, B)$ を求める
   （Kolen & Brennan, 2014）。$\theta = (\theta^* - B)/A$ より、傾き–切片パラメタ化での項目パラメタは
   $a^* = a / A$ および以下で変換される。
   - `DichotomousKernel`（2PL/Rasch）・`FourPLKernel`（3PL/4PL）: $d^* = d - aB/A$（漸近線 $g, u$ は不変）。
   - `GradedKernel`（GRM）: $d_k^* = d_k - aB/A$（$k = 1, \dots, K-1$。順序 $d_1 > \dots > d_{K-1}$ を保存）。
   - `GPCMKernel`（GPCM/PCM）: $d_k^* = d_k - k\,aB/A$（$k = 1, \dots, K-1$、$d_0 = 0$）。
     $k a^* \theta^* + d_k^* = k a\theta + d_k$ とステップ困難度 $b_k^* = A b_k + B$ を保存する。
   - `NominalKernel`（名義反応モデル。V40）: スコアリング係数は不変で $\mathrm{ak}_k^* = \mathrm{ak}_k$
     （$k = 1, \dots, K-2$。$\mathrm{ak}_0 = 0$ と $\mathrm{ak}_{K-1} = K-1$ は固定のまま）、切片は
     $d_k^* = d_k - \mathrm{ak}_k\,aB/A$（$k = 1, \dots, K-1$、$d_0 = 0$）。
     $\mathrm{ak}_k a^* \theta^* + d_k^* = \mathrm{ak}_k a\theta + d_k$ を保存する。GPCM の変換で $k$ を
     $\mathrm{ak}_k$ に置き換えたもので、$1/A$ は $a$ だけが吸収するので mirt の識別制約
     （$\mathrm{ak}_0 = 0$、$\mathrm{ak}_{K-1} = K-1$、$d_0 = 0$）の中に収まる。
   4PL の上方漸近線 $u$ は $g$ と同じく不変である（どちらも確率であり $\theta$ 上の位置ではない）。変わるのは
   $a$ と $d$ だけである（V40）。
   任意の 1 次元 `ItemKernel` で $P_j(y=k \mid \theta^*; \xi_j^*) = P_j(y=k \mid (\theta^* - B)/A; \xi_j)$ が成り立つため、
   目的関数は `kernel.log_prob` から直接計算し、手書きの導関数を持たない。
   - **Haebara 法**（Haebara, 1980; 多値拡張 Kim & Lee, 2006）:
     $H_{\mathrm{fwd}}(A,B) = \sum_q w_q^{(t)} \sum_{j\in\mathcal{C}} \sum_{k=0}^{K_j-1} [P_{jk}^{\mathrm{tgt}}(\theta_q^{(t)}) - P_{jk}^{\mathrm{src}}((\theta_q^{(t)}-B)/A)]^2$。
     和は 2 値項目の両カテゴリを含む全カテゴリにわたる（Kim & Lee, 2006 の流儀）。そのため 2 値項目の寄与は
     $2[P^{\mathrm{tgt}} - P^{\mathrm{src}}]^2$ で Haebara (1980) の原式の 2 倍になる。共通項目がすべて 2 値なら解は変わらず
     報告される目的関数値が 2 倍になり、混在形式のアンカーでは 2 値項目は多値項目に対して Kim & Lee の重みを持つ。
   - **Stocking–Lord 法**（Stocking & Lord, 1983; 多値拡張 Kim & Lee, 2006）:
     $\mathrm{SL}_{\mathrm{fwd}}(A,B) = \sum_q w_q^{(t)} [\mathrm{TCC}^{\mathrm{tgt}}(\theta_q^{(t)}) - \mathrm{TCC}^{\mathrm{src}}((\theta_q^{(t)}-B)/A)]^2$。
     ただし $\mathrm{TCC}(\theta) = \sum_{j\in\mathcal{C}}\sum_{k=0}^{K_j-1} k\,P_{jk}(\theta)$。
     名義項目も他の項目と同じくカテゴリ番号 $k$ で採点する（mirt の `expected.item()` と同じ。$\mathrm{ak}_k$ では
     ない）。変換が厳密に成り立てば曲線が一致するときに限り 0 になる損失である点は変わらないが、名義項目の得点は
     カテゴリの符号化に依存するので、順序のないカテゴリでは得点を使わない Haebara 法が自然な選択である。
   - **対称版**（`symmetric=True`; Kim & Lee, 2006）: 逆方向（$P_{jk}^{\mathrm{src}}(\theta_q^{(s)})$ と
     $P_{jk}^{\mathrm{tgt}}(A\theta_q^{(s)} + B)$、$\mathrm{TCC}$ も同様）の二乗差を加える。既定では $(\theta^{(t)}, w^{(t)})$ は
     `target` 群の推定済み潜在分布の求積格子・重み、$(\theta^{(s)}, w^{(s)})$ は `source` 群のものを使うため、対称版は
     `source` と `target` を入れ替えると厳密に $(1/A, -B/A)$ になる。`nodes` / `weights` を明示すると格子を上書きする
     （重みは和が 1 になるよう正規化し、`nodes` のみ指定時は一様重み）。
   **最適化**: Mean/Mean 初期値（$A_0 = \sum |a_j^{\mathrm{src}}| / \sum |a_j^{\mathrm{tgt}}|$、
   $B_0 = \bar b^{\mathrm{tgt}} - A_0 \bar b^{\mathrm{src}}$）から出発し、`LinkingProblem` pytree 上の jit 済み・
   形状静的な 2 次元範囲付き単調 Newton 最小化器（`_linking.solve_linking`、`lax.while_loop`）で `jax.grad` と
   `jax.hessian` を用いて解く。$A > 0$ は境界の中点への射影で保ち、Hessian が正定値でないときは
   Levenberg–Marquardt 減衰にフォールバックし、ステップのユークリッドノルムを `_MAX_STEP = 1` で打ち切り
   （悪い初期値から曲線が重ならない平坦な領域へ飛ばないため）、40 段階の並列バックトラッキングと最適点近傍での
   1-ULP 勾配仕上げを行う。収束判定は $\|\nabla F\|_\infty \le$ `tol`（絶対値、既定 $10^{-10}$。重みの和が 1 で
   $F$ は共通項目数とともに大きくなるが、スケール調整はしない）。`tol` に達する前に $F$ を減らすステップがなくなると
   `converged = False` で終了し `ConvergenceWarning` を出す。Mean/Mean は初期値にすぎず、傾きの絶対値を合計し、
   全共通項目の表示用閾値をすべてまとめて平均するため、多値項目は項目単位の平均より重みが大きい。名義項目には
   表示用閾値がないので、位置 $-d_{K-1}/((K-1)a)$（最初と最後のカテゴリ曲線の交点）1 つを $\bar b$ に入れる。
   これは $Ab + B$ で変換される。
   **標準誤差（Ogasawara, 2001）**: 最適点 $\hat c = (\hat A, \hat B)$ の停留条件
   $g(c, \eta_s, \eta_t) = \nabla_c F(c, \eta_s, \eta_t) = 0$ に陰関数定理を適用し、共通項目に現れる一意な自由パラメタ
   （固定パラメタや共有 `free_id` をそのまま反映）に対して `jax.jacfwd(g, 0)` と `jax.jacrev(g, 1/2)` から
   $J_s = -H_{cc}^{-1} H_{c,\eta_s}$、$J_t = -H_{cc}^{-1} H_{c,\eta_t}$ を得て、
   $\mathrm{Cov}(\hat A, \hat B) = J_s \Sigma_s J_s^\top + J_t \Sigma_t J_t^\top + J_s \Sigma_{st} J_t^\top + J_t \Sigma_{st}^\top J_s^\top$
   を計算する。別々の `FitResult` は独立な推定とみなす（$\Sigma_{st} = 0$）。同じ `FitResult` を `source` と `target` の
   両方に渡したとき（1 つの多母集団推定の 2 群）は、その同時 `vcov` の交差ブロックを $\Sigma_{st}$ とするため、両群で
   共有される等値制約アンカーでは $(A, B) = (1, 0)$、共分散 0 になる。重なるデータで推定した別々の `FitResult` は
   検出しない。**求積重みは固定とする。** 既定の重みは推定済み潜在分布（群の平均・分散が自由なときや EH では推定値）
   だが、正しく指定されたモデルの下では 1 次まで厳密である: $\partial g/\partial w_q$ は曲線残差と曲線の導関数の積の
   重み付き和で、母集団の解では残差が 0 なので $H_{c,w} = 0$ になる。項目パラメタのドリフトや DIF（母集団の残差が
   0 でない）の下では重みの標本変動を無視する。SE は収束した解でのみ計算する。収束しない場合、自由な共通項目パラメタが
   ある側の `vcov` がない場合、$H_{cc}$ が正定値でない場合は `vcov = None`、SE は `NaN` とする。
   **変換後の項目と得点**: `LinkResult.items` は `source` の全項目を基準尺度で表す。傾き固定のモデル（Rasch・PCM）は
   $a^* = 1/A$（1 ではない）となり、`Model.fix_from` はその傾きで固定する。`transform_theta` は `theta` と全 `pv*` 列を
   $A\theta + B$、`se` を $A\,\mathrm{se}$ で写し、$(A, B)$ を既知として扱う（リンキング誤差は加えない）。
   **Python シェル層での検証**: 多次元結果、群指定のない多母集団結果、共通項目でカーネルまたはカテゴリ数が一致しない場合、
   および不正な `nodes` / `weights` は `SpecificationError` を送出する。
6. **名義反応モデル（`Nominal(n_categories)`、`NominalKernel(n_categories, n_factors=1)`、V23）**:
   Thissen, Cai & Bock (2010) の全体傾き × 採点関数パラメタ化による Bock (1972) の名義反応モデルで、R mirt の
   `itemtype = "nominal"`（Chalmers, 2012）に対応する:
   $$P(y=k\mid\boldsymbol\theta)=\frac{\exp\big(\mathrm{ak}_k\,\mathbf a^\top\boldsymbol\theta+d_k\big)}{\sum_{c=0}^{K-1}\exp\big(\mathrm{ak}_c\,\mathbf a^\top\boldsymbol\theta+d_c\big)},\qquad k=0,\dots,K-1,\quad \mathrm{ak}_0=0,\ \mathrm{ak}_{K-1}=K-1,\ d_0=0.$$
   カテゴリは `0..K-1`（`n_categories >= 2` を検証）で、$D\in\{1,2,3\}$ に対応する（`_linear_predictor(theta, params[:D], D)`
   に採点ベクトル $\mathrm{ak}$ を掛け、`jax.nn.log_softmax` で数値安定に評価する）。
   - **内部パラメタと範囲**: mirt の `coef(m)` / `pars = "values"` は `a1`（`a1..aD`）、`ak0..ak{K-1}`、
     `d0..d{K-1}` を返し、`ak0 = 0`・`ak{K-1} = K - 1`・`d0 = 0` を固定（`est = FALSE`）、範囲はすべて
     `(-Inf, Inf)` とする。`GPCMKernel`（§15.8）と同様、これら 3 つの固定識別定数は `ParameterTable` の行ではなく
     カーネル内部の定数として持つ（`_check_identification` が固定傾きや固定切片と誤認しないようにするため）。したがって
     `param_names` は $D=1$ で `("a", "ak1", ..., "ak{K-2}", "d1", ..., "d{K-1}")`、$D\in\{2,3\}$ で
     `("a1", ..., "aD", "ak1", ..., "ak{K-2}", "d1", ..., "d{K-1}")` となり、既定ですべて自由推定・非有界
     `(-inf, inf)` とする。$\mathrm{ak}_k=k$（$k=1,\dots,K-2$）に固定すると `Nominal(K)` は `GPCM(K)` に厳密に
     一致する（`Nominal(2)` は自由な `ak` を持たず、`GPCM(2)` / `TwoPL` と一致する）。
   - **初期値・識別性・推定可能性**: mirt の既定初期値（`a1 = 0.851`、`ak1 = ... = ak{K-2} = 0.5`、
     `d1 = ... = d{K-1} = 0`）は反復 0 で中間カテゴリ $1,\dots,K-2$ の予測子が同一になり、減衰なしの
     `optimizer = "NR"` では同時 Hessian が不定になって `a1 = 0` に捕捉されることがある（`Science` など）。
     flexpyirt では `Nominal(n_categories)` を GPCM の初期値から開始する: $a_j^{(0)}$ は項目−残余相関（§4.10。$D>1$ では
     因子別項目−残余相関、§15.31）、採点係数は互いに異なる $\mathrm{ak}_{jk}^{(0)}=k$（$k=1,\dots,K-2$）、切片は GPCM の
     初期切片 $d_{jk}^{(0)}$（`_gpcm_intercept_starts`、§15.9）。これにより反復 0 でのカテゴリ対称性を破りつつ、真の
     $\mathrm{ak}_k$ が非単調や負の項目も順序制約なしの最尤解へ収束する。§4.5 / §15.31 の識別性検査では
     `slope_params` を `("a",)`（`("a1", ..., "aD")`）、`intercept_params` を `("d1", ..., "d{K-1}")` とする
     （`ak1..ak{K-2}` はカテゴリ間の相対形状を担い、$\mathrm{ak}_k=k$ を固定しても GPCM になるだけで自由な潜在分散や
     潜在平均を識別しない）。自由パラメタを持つ項目では `0..K-1` の全カテゴリが観測されている必要がある
     （`_check_estimable`）。
   - **表示用閾値・FIPC・JSON・ML スコアリング**: mirt の `coef(m, IRTpars = TRUE)` は名義反応項目について $\theta$
     尺度の閾値 $b_k$ ではなく Bock (1972) のゼロ和パラメタ $a_k^*=a\,\mathrm{ak}_{k-1}-\overline{a\,\mathrm{ak}}$
     （`a1..aK`）と $c_k^*=d_{k-1}-\bar d$（`c1..cK`）を返し、隣接カテゴリの交点
     $-(d_k-d_{k-1})/[a(\mathrm{ak}_k-\mathrm{ak}_{k-1})]$ も $\mathrm{ak}_k=\mathrm{ak}_{k-1}$ で特異になる。そのため
     `NominalKernel` は `irt_names = ()`・`irt_dependency_indices = ()`（`FitResult.items` に `b` / `bk` 列を追加しない）
     とし、`Model.fix_from()` は `Nominal` 項目に対して `parameterization="slope_intercept"` を受け付け、
     `parameterization="irt"` は `SpecificationError` とする。`_json.py` の多値項目レジストリに `"Nominal"` を追加する
     （スキーマバージョン 3 および旧バージョン 1–2 の読み込み互換を維持）。1 次元 ML スコアリング
     （`_scoring._pattern_flags`）では、各カテゴリの有効傾き $s_c=a\,\mathrm{ak}_c$ から最低・最高 $\theta$ カテゴリ
     （$a\neq 0$ のもとで $s_{y_j}=\min_c s_c$ および $s_{y_j}=\max_c s_c$）を判定し、非単調なカテゴリ
     （$\mathrm{ak}_k<0$ や $\mathrm{ak}_k>K-1$）を正しく扱いつつ $0\le\mathrm{ak}_k\le K-1$ では $0$ と $K-1$ に
     帰着させる。これは mirt 1.46.1 と意図的に異なる。mirt の `fscores(method = "ML")` は $\mathrm{ak}$ にかかわらず
     全項目カテゴリ 0 / 全項目最上位カテゴリのパタンに $\mp\infty$ を返すが、$\mathrm{ak}_1<0$ では全 0 パタンの
     ML 推定値は内点にあり、flexpyirt はその値を返す（VALIDATION.md V23）。
   - **退化項目**: `fit()` は、傾きが自由な `Nominal` 項目（$K\ge 3$）について、傾きのノルムが $10^{-3}$ 未満
     （項目が情報を持たず $\mathrm{ak}_1..\mathrm{ak}_{K-2}$ が識別されない。mirt の NR が既定初期値から陥る鞍点）か、
     $|\mathrm{ak}_k|$ または $|d_k|$ が 50 を超える（カテゴリの発散）ときに `HeywoodWarning` を出す。
   - **照合（V23）**: mirt の名義モデルの既定初期値では NR が `Science` で鞍点に止まるため、
     `validation/r/m5_nominal_reference.R` は mirt の NR を GPCM の点から開始し、mirt の別の初期値 5 通りのうち最良のものが
     より高ければそこから NR をやり直す（大半が非単調な V23e では GPCM の点からの NR は局所最大に止まる）。flexpyirt は
     自身の既定初期値から推定し、1 次元（`Science`、順序のない・負の $\mathrm{ak}$ を含むシミュレーション、名義 + 2PL、
     大半が非単調な項目）と 2 因子の確認的モデル（V23f）のすべてで VALIDATION.md の許容誤差内で mirt の解に到達する。
     3 因子の名義モデルは単体テストのみ。mirt は V23e の真に $+\infty$ のパタン 1 つを `max_theta = 20` の境界に残す。
     ML の極端パタンは R での独立な対数尤度走査と照合する。

7. **混合正規潜在分布（`NormalMixture`、§4.4、V22）**:
   `NormalMixture(n_components, *, mean=0.0, var=1.0, free_mean=False, free_var=False,
   means=None, variances=None, weights=None)`（および `NormalMixture.free(n_components, ...)`）は、
   $k\in\{2,\dots,10\}$ 個の正規成分の有限混合（Mislevy, 1984）である単一群 1 次元の潜在密度を指定し、
   **項目パラメタは共通の 1 組**とする（潜在密度が非正規な 1 つの母集団）。クラスごとに項目パラメタを持つ混合 IRT
   モデル（Rost, 1990；Bolt, Cohen & Wollack, 2001）ではない。後者は mirt の `dentype = "mixture-k"` の既定だが対象外。
   多母集団（`Model.with_groups`）および多次元（`Model.with_factors`）での `NormalMixture` は `SpecificationError`
   を投げる。
   **素の（ゲージ）パラメタ化**: $\boldsymbol\lambda=(0,\lambda_2,\dots,\lambda_k)$、$\pi=\operatorname{softmax}(\boldsymbol\lambda)$ として
   $$h(z)=\sum_{c=1}^k\pi_c\,\phi(z;\tilde\mu_c,\tilde\sigma_c^2),\qquad
     \tilde\mu_1=0,\ \tilde\sigma_1^2=1$$
   とし、$c=2,\dots,k$ について `rmean_c` $=\tilde\mu_c$、`rvar_c` $=\tilde\sigma_c^2>0$、`logit_c`
   $=\lambda_c=\log(\pi_c/\pi_1)$ を自由にする。これは `multipleGroup(..., dentype = "mixture-k", invariance =
   c(<全項目>, "free_means", "free_var"))` におけるクラス `MIXTURE_c` の `MEAN_1`・`COV_11`・`PI` そのものである
   （クラス 1 は $N(0,1)$・`PI = 0` に固定され、項目はクラス間で等値。mirt は混合係数をクラス 1 との対数比で保持する）。
   素の成分 1 は $z$ の位置・尺度のゲージを固定するだけで、母集団の「基準クラス」ではない。
   **解析的標準化**: 素の混合は厳密な連続モーメント $M=\sum_c\pi_c\tilde\mu_c$、
   $V=\sum_c\pi_c(\tilde\sigma_c^2+\tilde\mu_c^2)-M^2>0$ を持つ。Davidian 曲線（§16.4）と同様に
   $\theta=\mu+\sigma(Z-M)/\sqrt V$ とすると、$\theta$ 尺度で成分 $c$ は $N(\mu_c,\sigma_c^2)$、
   $$\mu_c=\mu+\sigma\,\frac{\tilde\mu_c-M}{\sqrt V},\qquad \sigma_c^2=\sigma^2\,\frac{\tilde\sigma_c^2}{V}$$
   となり、任意の形状で $\sum_c\pi_c\mu_c=\mu$、$\sum_c\pi_c(\sigma_c^2+\mu_c^2)-\mu^2=\sigma^2$ が厳密に成り立つ。
   `mean` $=\mu$ と `var` $=\sigma^2$ は既定で 0 と 1 に固定し、自由にするときは §4.5 の `Normal` の識別規則に従う
   （Rasch/PCM では `free_var=True`、FIPC では `NormalMixture.free(k)`）。形状は $3(k-1)$ 個の自由パラメタを加える。
   **標準化する理由（決定）**: mirt 自身のパラメタ化はクラス 1 を $N(0,1)$ に固定し、項目パラメタがクラス 1 の尺度を
   担う。項目の初期値は全体が $N(0,1)$ である前提で求める（§4.10）ので二峰の母集団のどの成分とも合わず、一般的な
   初期値からの EM は尺度の合わない領域を通り、劣った停留点に止まりうる（例えば自由な成分がほぼすべての質量を吸収し
   $\pi_1\to 0$ となって尺度が識別されない点）。V22a のデータでは、mirt のパラメタ化での同じ EM（下記の節点レベルの
   M-step）は 4 通りの初期形状のうち 2 通り（成分 2 を $+1.5$ または $-1.5$ に置く）で最大値より対数尤度で 12.7–47.6
   低い点に止まり、残り 2 通りでも通常の EM で 3,848–4,505 反復を要した。標準化すると 4 通りすべてから最大値に達し、
   通常の EM で 602–661 反復（SQUAREM で 28–93 反復）だった。mirt 自身は生成値から始めて `TOL = 1e-10` で 4,555 回の
   EM サイクルを要した（VALIDATION.md V22）。
   項目パラメタは全体を標準化した尺度にあり、`Normal`・`EmpiricalHistogram`・`DavidianCurve` の推定と同じ約束なので
   それらと比較できる。
   mirt との同値性: $\mu=0,\sigma^2=1$ のとき、格子 $\theta_q$ 上の flexpyirt のモデルは格子 $z_q=M+\sqrt V\theta_q$ 上の
   mirt のモデルに一致し、$a=a^{\text{mirt}}\sqrt V$、$d_k=d_k^{\text{mirt}}+s_k\,a^{\text{mirt}}M$
   （2PL/3PL/4PL/GRM では $s_k=1$、GPCM/PCM では $s_k=k$）である。正規化した各成分の密度はアフィン変換で不変だからである。
   **格子の重み**: mirt の `dentype = "mixture-k"` がクラスごとに正規化した密度を積み重ねるのと同様に、各成分を
   格子上で正規化してから混合する:
   $$\log w_q=\log\sum_{c=1}^k\pi_c\tilde w_{cq},\qquad
     \log\tilde w_{cq}=\operatorname{log\_softmax}_q\Big(b_q-\frac{(\theta_q-\mu_c)^2}{2\sigma_c^2}\Big)$$
   （$b_q$ は節点ごとの固定の対数基底重みで、等間隔格子では 0）。各成分は同じ格子上で $(\mu_c,\sigma_c^2)$ の `Normal` が
   受け取る重みそのものになるので、Gauss–Hermite 格子（§15.44）も `Normal` と同様に使え、$\pi_c$ は成分 $c$ の格子上の
   質量に厳密に一致する。分散が正でなければ NaN を返す（直線探索と SQUAREM が退ける）。
   **M-step**: 完全データは他の潜在分布と同じく（パタン, 節点）である。潜在部分
   $Q^{\text{lat}}=\sum_q\bar n_q\log w_q(\mu,\sigma^2,\tilde{\boldsymbol\mu},\tilde{\boldsymbol\sigma}^2,\boldsymbol\lambda)$
   を既存のマスク付き・有界・単調な Newton（`_newton.py`、勾配とヘッセ行列は自動微分）で同時に最大化するので、EM は
   単調で、SQUAREM はそのまま動き、Oakes の恒等式（§4.8）と観測情報ヘッセ行列に新しいコードは要らない。mirt はクラスを
   追加の欠測データとして扱う（クラス比率を閉形式で $[10^{-4},1-10^{-4}]$ に切り詰め、クラスごとに平均・分散を更新）。
   両 EM の不動点は同じだが、クラスによる拡大は欠測情報を増やし収束が遅い。
   **初期値**: `means`・`variances`・`weights`（長さ $k$。既定は $(0,1.5,-1.5,3,-3,\dots)$、すべて 1、$1/k$）は
   任意のアフィン尺度で初期形状を与え、$\tilde\mu_c=(m_c-m_1)/\sqrt{v_1}$、$\tilde\sigma_c^2=v_c/v_1$、
   $\lambda_c=\log(\pi_c/\pi_1)$ で素のゲージへ写す。成分がすべて同一の点は停留点である（成分を分ける 1 次の効果は
   位置・尺度の変化で、標準化が取り除く。Davidian 曲線の $\boldsymbol\phi=\mathbf 0$ と同様）ため、既定値は互いに異なる。
   **ラベルスイッチングと退化解**: 尤度は成分の並べ替え（どの物理的成分が素の成分 1 になるかを含む）について不変で、
   局所最大が複数ありうる（McLachlan & Peel, 2000）。EM は決定的かつ単調なので推定中にラベルは入れ替わらないが、
   利用者は複数の初期形状からの推定を比べるべきである。推定後、`fit()` は、成分の重みが $10^{-3}$ 未満（位置と尺度が
   識別されず情報行列がほぼ特異）、成分が格子より細い（$\sigma_c$ が最小の節点間隔の半分未満で格子が分解できない）、
   成分の平均が格子の外、2 つの成分がほぼ一致（$|\mu_c-\mu_{c'}|<10^{-3}$ かつ $|\sigma_c^2-\sigma_{c'}^2|<10^{-3}$、
   潰れた混合）のいずれかで `HeywoodWarning` を出す。
   **表示・スコアリング・その他のメソッド**: `ParameterTable` は `item="__latent__"`、`model="Mixture"` の下に
   `mean`・`var`・`rmean_c`・`rvar_c`・`logit_c` を持つ（潜在分布パラメタへの事前分布は他の潜在分布と同様に
   `SpecificationError`）。`FitResult.latent` は標準化した成分 `pi_c`・`mean_c`・`var_c`（$c=1,\dots,k$）とその
   デルタ法 SE（上の写像の JAX 前進モード Jacobian）を加え、`latent_weights` は `None` のまま。`score()` は
   `"EAP"`（求積格子）と `"PV"`（501 点細密格子、各成分をそこで再正規化）で推定した混合を用い、`"ML"` と `"WLE"` は
   尤度のみ、`"MAP"` は混合の事前分布が多峰になりうるため `SpecificationError` とする（mirt 1.46.1 も混合モデルの MAP
   を拒否する）。`item_fit()` と `m2()` は検証されるまで `NormalMixture` を拒否し、`dif()` は多母集団の推定を要し、
   `link()` は既定で推定した混合の格子重みを使う。`_json.py` に `"NormalMixture"` を登録し、ペイロードの形は
   変わらないため `schema_version` は 3 のまま。
   **照合（V22）**: `validation/r/m5_mixture_reference.R` は標準化した離散化尤度を R で独立に書き直し、mirt の解を
   その厳密な最大値まで磨き（Richardson ヘッセ行列による SE と EAP も R で計算）、さらに組み込みの
   `mirt(dentype = "mixture-k")` を像の格子 $z_q=M+\sqrt V\theta_q$（非対称な `theta_lim`）で推定し直して、推定値と EAP を
   flexpyirt の尺度へ写す。2PL $k=2$（二峰）、GRM $k=2$（小さい成分）、GPCM $k=3$（格子 121 点。VALIDATION.md V22）。SE は R の
   Richardson ヘッセ行列と照合する。mirt 1.46.1 の混合モデルでの `SE.type = "Oakes"` は観測情報になっていない
   （潜在分布パラメタの SE が数倍小さく、クラス所属を既知とした完全データの SE に近い）ためである。
8. **次元性診断（計画。2026-11-24 決定、未実装）**:
   **範囲**: 2 つの問いを分けて扱う。両方をまとめた多数決は意味を持たないためである。
   (A) *次元はいくつか*: 整数 $k$ を返す（平行分析、MAP、EKC、Hull、CD）。
   (B) *本質的一次元性（Stout, 1990）は認められるか*: 判定・$p$ 値・程度の指標を返す（DIMTEST、DETECT、推定済み
   モデルの局所依存統計量、$M_2$ / RMSEA、モデルベース平行分析、第 1/第 2 固有値比）。単独で信頼できる手法はない
   （Auerswald & Moshagen, 2019）ため、利用者向けの入口はアンサンブルにする。
   **範囲外**: 探索的 MIRT（§4 の直積格子の求積は探索的モデルに必要な次元数に伸びない）。したがって (A) は診断で
   あり推定にはつなげない。学習済み予測器（Factor Forest; Goretzko & Bühner, 2020）は学習済みモデルを同梱し、数式から
   照合できないため採らない。カイザー基準は参考表示のみで集計に数えない。EGA（Golino & Epskamp, 2017）と
   バイファクター指標（ECV、$\omega_h$、PUC; Rodriguez, Reise & Haviland, 2016）は後回し。
   **EFA（2026-11-24 改訂）**: 当初は `factor_analyzer` を任意依存として呼ぶ計画だったが、その最新版（0.5.1、
   2024 年 2 月）は scikit-learn 1.8 以降で `fit()` が失敗する（削除された `force_all_finite` 引数を `check_array` に
   渡す）。そのため ULS と ML の EFA は SciPy だけで flexpyirt に実装し（§16.14）、同じ行列で EFAtools と
   `psych::fa` と照合する。`efa` extra は設けない。
   **段階**（前の段階の照合が通ってから次へ。ROADMAP M5）:
   D1 ポリコリック・テトラコリック相関（two-step、Olsson, 1979。ペアごとに `vmap`）と非正定値行列の平滑化。
   `psych::polychoric` と `lavaan::lavCor` と照合。
   D2 固有値、平行分析（Horn, 1965。ポリコリック、PCA と FA の固有値、Glorfeld, 1995 のパーセンタイル。
   Garrido, Abad & Ponsoda, 2013）、改訂 MAP（Velicer, Eaton & Fava, 2000）、EKC（Braeken & van Assen, 2017）。
   `psych::fa.parallel`、`psych::vss`、`EFAtools::EKC` と照合（決定的な部分は厳密に、シミュレーションの参照固有値は
   R の生成データを渡すか Monte Carlo 誤差の許容幅で）。
   D3 推定済み 1 次元モデルの局所依存: Q3（Yen, 1984）、Q3*（Christensen, Makransky & Horton, 2017）、
   LD-$X^2$（Chen & Thissen, 1997）、JSI（Edwards, Houts & Cai, 2018）を `mirt::residuals` と照合。モデルベース平行分析
   （Drasgow & Lissak, 1983）を `ltm::unidimTest` と照合。
   D4 DETECT の確認的・探索的版（Zhang & Stout, 1999; Zhang, 2007）と ASSI・$R$。`sirt::conf.detect` /
   `expl.detect` と照合。分割の探索は外側の層で行う。
   D5 EFA（§16.14）と Hull（Lorenzo-Seva, Timmerman & Kiers, 2011）を EFAtools の `EFA` / `HULL` と `psych::fa` と
   照合。比較データ法（Ruscio & Roche, 2012）を `EFAtools::CD` と照合。
   D6 ブートストラップでバイアス補正する DIMTEST（Stout, Froelich & Gao, 2001; Froelich & Habing, 2008）。CRAN に
   参照実装がないため、シミュレーション研究（1 次元データで名目水準どおりの第 1 種の誤り、2 次元データでの検出力）を
   VALIDATION.md に記録して受け入れる。
   **アンサンブル**（`dimensionality()`、D3 の後）: 各手法は共通の結果（問い A か B か、推奨 $k$ または判定、統計量、
   用いた閾値とその出典）を返す。要約には全手法を並べ、(A) の合意は $k$ の最頻値・中央値と一致数（同数は決めずに報告）、
   (B) は本質的一次元性を支持する手法の数とする。重みはシミュレーション研究の裏付けができるまで付けない。閾値
   （DETECT の 0.2 / 0.4 / 1、Q3* の 0.2、RMSEA）は引数にし、出典を docstring に書く。ポリコリック行列は 1 回だけ計算して
   共有し、シミュレーションは JAX の PRNG キーを使う。アンサンブル自体はシミュレーション研究（1 次元・2 次元・
   バイファクター構造のデータ）で確かめる。
9. **ポリコリック・テトラコリック相関（`polychoric()`、§16.8 の D1、V24。以下は JAX 版。NumPy 版は §16.18）**:
   `flexpyirt.polychoric(data, *, smooth=True, zero_cell_correction=0.5, tol=1e-10, max_iter=100)` を
   `correlation.py`（シェル）と `_polychoric.py`（jit カーネル）に置き、変更不可の `PolychoricResult` を返す。
   **データ**: 任意の数値符号でよい。各項目を観測された値の昇順で $0..K_j-1$ に符号化し直すので、観測されない符号は
   カテゴリにならない（psych・lavaan でも空のカテゴリは確率 0 のセルを足すだけなので結果は同じ）。観測されたカテゴリが
   1 つの項目、両方に回答した回答者がいないペア、その回答者の中で一方の項目が一定のペアは `SpecificationError`。
   **閾値**: $\hat\tau_{jk}=\Phi^{-1}(\hat P(Y_j\le k))$ を項目へのすべての観測された回答（利用可能なケース）から求め、
   その項目を含むすべてのペアで共有する（two-step）。ペアの分割表は両方に回答した回答者から作る。lavaan の
   `missing = "pairwise"`、psych の `global = TRUE` と同じ。psych は項目のカテゴリ数が異なると、ペアごとの閾値
   （`global = FALSE`）に切り替えるため、照合は欠測のないデータに限る。
   **2 変量正規分布**: Genz (2004) の `BVNU`。$r$ によらず 20 点の Gauss–Legendre 則を使い（6 点・12 点の則は安価な
   だけ）、$|r|\ge0.925$ では Drezner & Wesolowsky (1990) の展開を使う。分岐はすべて評価して `jnp.where` で選び、
   無限の端点は厳密に扱う（式の中では $\pm40$ で代用）。`custom_jvp` で解析的な微分
   $\partial_r\Phi_2=\phi_2$、$\partial_h\Phi_2=\phi(h)\Phi((k-rh)/\sqrt{1-r^2})$ を与えるので、Newton に使う 2 階微分も
   厳密である。Plackett の積分を適応求積で計算した値と絶対誤差 $10^{-14}$ で一致する。**ペアごと**: セル確率
   $\pi_{ab}=\Phi_2(\tau_a,\tau'_b)-\Phi_2(\tau_{a-1},\tau'_b)-\Phi_2(\tau_a,\tau'_{b-1})+\Phi_2(\tau_{a-1},\tau'_{b-1})$
   を $+\infty$ で $K_{\max}$ までパディングした格子で計算し（パディングのセルは度数も確率も 0）、
   $\ell=\sum n_{ab}\log\pi_{ab}$ を $z=\operatorname{atanh}\rho$ について Newton 法（凹でないときは勾配ステップ、
   ステップ半減、$|z|\le 8$）で最大化する。全ペアを `vmap` + `lax.while_loop` で処理する。初期値は符号の Pearson
   相関（クリップ）の atanh。射影勾配が `tol` $\times n$ 以下、またはステップが $10^{-14}$ 未満で収束とする。
   **境界**: $\ell$ が $|\rho|=1$ に向けて単調に増える場合（例: 空のセルがある 2 x 2 表）、$z$ での勾配は指数的に
   小さくなり、反復は早く止まる。そこで反復の後に、$\hat z$ の側の $\pm8$ での $\ell$ を $\ell(\hat z)$ と比べ、
   それ以上なら推定値を $\pm\tanh 8$ とし、`at_boundary` を立てて `HeywoodWarning` を出す。**空のセル**: lavaan の
   既定（`zero.add = c(0.5, 0)`、`zero.keep.margins = TRUE`）。空のセルがちょうど 1 つの 2 x 2 表で、空のセルを含む
   対角に $+c$、もう一方に $-c$ を加え、周辺は変えない。それより大きい表は補正しない。次元性の手法は内点の推定値を
   要するので、既定の $c=0.5$ は lavaan に合わせた。$c=0$ でそのままの最尤推定。**平滑化**: psych の `cor.smooth` と
   同じ（最小固有値がマシンイプシロン未満なら、$10^{-12}$ 未満の固有値を $10^{-10}$ にし、和を $p$ にそろえ、行列を
   組み直し、対角を 1 にそろえる）。`errors` に新設した `NotPositiveDefiniteWarning` を出す。$\hat\rho$ の標準誤差は
   返さない（照合しておらず、次元性の手法も使わない）。
   **照合（V24）**: `packages/flexpyctt/validation/r/m5_polychoric_reference.R`（lavaan 0.7-3、psych 2.6.9）。シミュレーションの
   テトラコリック（8 項目）、2 因子で 2〜5 カテゴリの混在、その 10% MCAR（pairwise）、空のセルが 1 つの 2 x 2 表を
   含む極端な 2 値項目、LSAT7、Science。lavaan と $3.1\times10^{-8}$ 以内（その `nlminb` の許容誤差。flexpyirt の
   ペアごとの勾配は $10^{-7}n$ まで 0）、閾値は $10^{-12}$ 以内、psych とは補正したペアを除く欠測のないデータで
   $2\times10^{-5}$ 以内（その `optimize()` の許容誤差）、`cor.smooth` とは $10^{-12}$ 以内で一致する。
10. **平行分析・MAP・EKC（`parallel_analysis()`、`velicer_map()`、`empirical_kaiser()`、§16.8 の D2、V25）**:
    `dimensionality.py`（シェル。$p\times p$ の線形代数は NumPy、正規の参照データは JAX の `vmap`。現在は
    flexpyctt にあり、乱数は NumPy、`key` の代わりに `rng` を取る。§16.18）に置く。3 つとも
    問い (A) に答え、`n_factors` をもつ変更不可の結果を返す。行列の入力（DataFrame、配列、`PolychoricResult`）は
    正方・有限・対称・対角 1・$p\ge3$ であること。**数え方**: 参照を厳密に上回る固有値を先頭から数え、最初に下回った
    ところで止める（psych・EFAtools と同じ逐次の規則）。**SMC の固有値**: 対角を $h_j=1-1/(R^{-1})_{jj}$ にした $R$
    （Humphreys & Ilgen, 1969）。psych・EFAtools の EFA の共通性を使う版は EFA 連携（D5）を待つ。**平行分析の参照**:
    `"permutation"`（既定）は JAX の PRNG キーで各項目を独立に並べ替え（入れ子の `vmap` で `(R, p, N)` の置換）、
    周辺分布と欠測を保ったまま、反復ごとに照合済みの `polychoric()`（またはペアごとの Pearson）で相関を求め直す
    （警告は抑える）。並べ替えたペアが推定できない反復（欠測による）は飛ばして数え、`FlexpyirtWarning` を出す。
    使える反復が 2 つ未満なら `SpecificationError`。`"normal"` は同じ $N\times p$ の独立な $N(0,1)$ データ（欠測のある
    行も数える）の Pearson 相関で、`vmap` を分割して計算する。分位点は NumPy の既定（線形補間、R の type 7）。既定の
    分位点 0.95（Glorfeld, 1995）、PCA の固有値、ポリコリック相関での並べ替えの参照は Garrido, Abad & Ponsoda (2013)
    に従う。**MAP**: 主成分の負荷量 $V_m\Lambda_m^{1/2}$（負の固有値は 0 に切る）で $m=0,\dots,p-1$ の $f_m$ を
    求める。4 乗は要素ごと（O'Connor, 2000）で、推奨される改訂版（Velicer, Eaton & Fava, 2000）なので既定にした。
    偏分散が $10^{-10}$ 以下になったところで NaN として打ち切り、最小値は NaN を除いて探す。EFA.dimensions の 4 乗の
    統計量は行列のべき乗である（記録するが従わない）。**EKC**: EFAtools と同じ Braeken & van Assen (2017)（残りの
    分散の項に観測固有値を使う）。`PolychoricResult` を渡して `n_obs` を省くと、ペアごとの $n$ の最小値を使う
    （欠測があるときに保守的）。
    **照合（V25）**: `packages/flexpyctt/validation/r/m5_dimensionality_reference.R` は V24 の lavaan のポリコリック行列 6 つを EFAtools
    1.1.0（`EKC`）、psych 2.6.9（`vss()$map`）、EFA.dimensions 0.1.9.3（`MAP`）、両方の MAP のべきの R による独立な
    実装、PCA・SMC の固有値には `eigen()` に与える。選ばれる因子数も含めすべて $10^{-11}$ 以内で一致する。正規の
    参照固有値（3 つの大きさ x PCA・SMC、各 20000 データセット）は EFAtools の `PARALLEL` と Monte Carlo 誤差の範囲で
    一致する（固定のキーで平均は $|z|\le4.5$、95 パーセンタイルは $6\times10^{-3}$）。並べ替えの経路は性質（独立な
    置換、2 因子の回復、キーによる再現性）で確かめる。
11. **局所依存とモデルに基づく平行分析（`FitResult.local_dependence()`、`model_parallel_analysis()`、§16.8 の D3、
    V26）**: どちらも単一群・1 次元・`Normal` 潜在分布のモデルを要する（`FitResult._require_unidimensional_normal`）。
    `FitResult._traces(prob, nodes)` は `CompiledProblem.nodes` を差し替えて（カーネルが読むのはノードだけ）任意の
    点で全項目のカテゴリ確率を計算するので、推定した格子・EAP スコア・シミュレーションの $\theta$ に同じコードを使う。
    **Q3**: `score()` の EAP $\hat\theta$ による残差 $x_{ij}-\sum_k kP_{jk}(\hat\theta_i)$（欠測可）の、ペアごとの完全な
    ケースでの Pearson 相関（mirt のペアごとの `na.omit` と同じ）。**Q3\***: Q3 から非対角の Q3 の平均を引いたもの
    （Christensen, Makransky & Horton, 2017）。したがって `max_q3_star` は彼らの $Q3_{\max}-\overline{Q3}$。
    **LD**（Chen & Thissen, 1997、mirt の `residuals(type = "LD")` と同じ）: ペアごとに、両方に回答した回答者の
    $K_i\times K_j$ の観測表、推定した格子の重みによる期待表 $n_{ij}\sum_q P_{ia}P_{jb}w_q$（mirt の 61 点格子での
    正規化した正規密度と同じ）、$X^2$、$G^2$（$O>0$ のセル）、$\chi^2_{(K_i-1)(K_j-1)}$ による $p$、Goodman–Kruskal の
    $\gamma$ による $V=\operatorname{sign}(\gamma_O-\gamma_E)\sqrt{X^2/(n\min(K_i-1,K_j-1))}$（同点なら符号 $+$、$\gamma$ が
    定義できなければ $V$ は NaN）。**モデルに基づく平行分析**（Drasgow & Lissak, 1983）: 統計量は観測された平滑化済み
    ポリコリック行列の第 2 固有値。`_simulate_responses` は $\theta\sim N(\hat\mu,\hat\sigma^2)$ を引き、累積カテゴリ
    確率に対して一様乱数を逆変換して回答を作り、データの欠測をそのまま写す。$B$ 回の反復を `polychoric()` に通し
    （警告は抑え、使えない反復は `FlexpyirtWarning` を出して飛ばす）、$p=(1+\#\{T_b\ge T\})/(B+1)$。
    `eigen_type="max"`（既定）は ltm の `unidimTest` と同じく対角全体を非対角の最大の相関にする。`"smc"` と `"pca"` は
    §16.10 を使う。ltm は推定済みオブジェクトを渡すと $\theta$ を $N(\mathrm{EAP}_i,\mathrm{SE}_i^2)$ から引く。
    flexpyirt は推定した潜在分布（参照となる生成モデル）を使う。ltm もパラメタを渡したときはそうする。
    パラメタは同じデータから推定するので、モデルが正しいとき $p$ 値は保守的になる（シミュレーションの 2PL の
    データ 30 個、$B=50$ で $p$ の平均 0.66、0.05 未満は 30 個中 1 個）。docstring とガイドにそう書く。
    **JSI**（Edwards, Houts & Cai, 2018）: 下の 17 項。
    **照合（V26）**: `validation/r/m5_local_dependence_reference.R`（mirt 1.46.1、ltm 1.2.0）。LSAT7 の 2PL、Science
    の GRM、局所依存のペアを 1 つと 5% MCAR を含むシミュレーションの 2PL。flexpyirt はモデルを推定し直し
    （対数尤度は相対 $10^{-9}$ で一致）、`residuals()` と $10^{-5}$（相対、$X^2$、$G^2$）、$10^{-6}$（$V$、Q3。実測
    $4\times10^{-8}$ 以下）で一致する。LSAT7 に mirt の推定値（`thetas = cbind(d, a1)`、$B=4000$）で ltm の
    `unidimTest` を実行し、観測固有値は $2\times10^{-4}$（polycor の `optimise()` の許容誤差。実測 $2.4\times10^{-5}$）、
    シミュレーションの第 2 固有値の平均は $|z|\le4.5$（2 つのキーで 2000 反復なら実測 $|z|<1$）、その
    95 パーセンタイルは 0.02 で一致する。
    **単一群・1 次元・正規分布以外への拡張（M7、issue #81、V26）**: `local_dependence()` は `group=` を受け取り
    （`item_fit()` と同じ）、推定したモデルのコンパイル済み問題の上で計算する。これで潜在分布と格子の種類に
    よらず同じコードになる。重みは `_latent_log_weights`（`(G, Q)`。EH のモデルではその重み）、カテゴリ確率は
    `_item_log_probs`（`(G, J_f, Q, K)`）、EAP は各受験者の群を事前分布にした `_core.eap`。(a) **多母集団**:
    すべての統計量を群ごとに、その群の受験者・項目パラメタ・潜在分布の重みで計算する。mirt の `residuals()` が
    `multipleGroup` の結果に対してすること（群ごとの単一群オブジェクトで繰り返す）と同じ。`pairs` の先頭に
    `group` 列が付き、`q3` の行インデックスは `(group, item)`、`q3_mean` は群をインデックスにした Series、
    Q3\* はその群の平均を引き、`max_q3_star` は全群での最大。`data` に受験者がいない群は飛ばす。表は、ある群で
    観測されないカテゴリがあっても常に $K_i\times K_j$（mirt の `table()` はそのカテゴリを落とし、そこで失敗する）。
    (b) **`EmpiricalHistogram`、`DavidianCurve`、`NormalMixture`**: $E_{ab}$ は推定した格子の重みを、Q3 は
    推定した分布のもとでの EAP を使う。mirt の `dentype = "EH"` / `"Davidian-k"` の既定は違い、どちらにも
    $N(0,1)$ の事前分布を使う。推定した密度を使うのは `residuals(type = "LD", use_dentype_estimate = TRUE)` と、
    Q3 では `Theta = fscores(m, use_dentype_estimate = TRUE)` を渡したとき（mirt 1.48 の `type = "Q3",
    use_dentype_estimate = TRUE` は残差をスコアではなく格子点で評価してしまい、使えない）。正規の事前分布は
    推定したモデルと矛盾するので、flexpyirt は推定した密度を使う。一致する mirt の呼び出しは docstring に書く。
    (c) **確認的な 2〜3 次元モデル**: 直積格子と推定した多変量正規の重みで $E_{ab}=n\sum_qP_{ia}
    (\boldsymbol\theta_q)P_{jb}(\boldsymbol\theta_q)w_q$、ベクトルの EAP で
    $d_{ij}=x_{ij}-E(X_j\mid\hat{\boldsymbol\theta}_i)$。`_traces` はカーネルを `(N, D)` の点で評価する。EAP は
    推定に使った格子で計算する。mirt の `fscores()` は推定の格子によらず独自の既定（1 / 2 / 3 因子で各次元
    121 / 61 / 31 点）を使うので、格子が同じなら両者は一致する。$df$ はどの場合も $(K_i-1)(K_j-1)$ のまま（mirt と同じ）で、
    推定したパラメタを考慮しない。統計量は診断用（Chen & Thissen, 1997）。**拡張しないもの**: bifactor /
    two-tier モデル（bifactor の PR の後）、`jsi()`、`model_parallel_analysis()`、`dimensionality(fit=...)`。
    これらは `_require_unidimensional_normal` のまま。mirt の JSI は 1 因子が必要で、Davidian 曲線の SE がなく
    （`"No standard error method currently supported"`）、混合分布に対応せず、`multipleGroup` では傾きを多母集団の
    結果から取りながら各群を単独の $N(0,1)$ の単一群モデルとして推定し直す（推定したモデルの再推定ではない）。
    拡張を照合できる参照値がない。**照合（V26、M7）**: `validation/r/m7_local_dependence_reference.R`
    （mirt 1.48）、9 ケース（1 因子と 2 因子の 2 群、EH、Davidian、混合分布、2 因子と 3 因子）。mirt の推定値を、
    すべてのパラメタを固定した `FitResult` に入れて比べる。$X^2$ / $G^2$ は相対 $10^{-7}$（実測
    $8\times10^{-9}$ 以下）、$V$ / Q3 は $10^{-8}$（実測 $3\times10^{-15}$ 以下）。詳細と mirt の既定との違いは
    VALIDATION.md。
12. **`dimensionality()` のアンサンブル（§16.8）**: 1 つのポリコリック行列（`fit` があれば、まずデータを推定した
    項目にそろえる）から問い (A) の `parallel_analysis()`（既定値）、`velicer_map()`（4 乗）、`empirical_kaiser()` を
    計算する。キーはシミュレーションを使う手法ごとに分割する。問い (B) は `fit` があるときだけで、
    `model_parallel_analysis()`（$p\ge\alpha=0.05$）、最大の Q3\* $<0.2$、`m2()` の RMSEA $\le0.05$
    （Maydeu-Olivares & Joe, 2014）。`m2()` の `SpecificationError`（欠測など）は隠さず、使えない行
    （`supports_unidimensionality=None`）として示す。合意 (A) は最頻値（同数ならすべて）、中央値、一致数、(B) は使えた
    手法のうち支持する数。重みは付けず、2 つの問いはまとめない（§16.8）。閾値は引数にし、出典を docstring に書く。
    Q3\* の 0.2 は目安であると明記する。`DimensionalityResult.summary()` はすべての手法とその詳細、2 つの合意を
    表示する。モジュールと関数が同じ名前 `dimensionality` を持つ（`fit` と同じ）ので、`flexpyirt.dimensionality` は
    関数を指す。テストではモジュールを `importlib.import_module` で取得する。`tests/test_m5_ensemble.py` で、各行を
    個々の関数と同じキーで計算し直して確かめる。
13. **DETECT（`detect()`、`exploratory_detect()`、§16.8 の D4、V27）**: `_detect.py`（NumPy。現在は flexpyctt に
    ある。§16.18）。
    **条件付き共分散**: Zhang & Stout (1999)。各ペアについて、総得点 $S$ が等しい群と残りの得点 $S-X_i-X_j$ が等しい
    群のそれぞれで最尤の共分散を求め、群の割合で平均し、さらに 2 つを平均する（総得点による推定は負に、残りの得点に
    よる推定は正に偏る）。群は `np.unique` と `np.bincount` でペアごとに 1 回で作る。生の項目得点を使い（多値項目も
    可）、欠測のないデータを要する（そうでなければ得点が定義できない）。そのため `dimensionality()` は欠測があると
    DETECT を使えないと示す。**指標**（sirt の `detect.index`）: DETECT $=100\,\overline{\delta\,\mathrm{ccov}}$、ASSI、
    RATIO、MADCOV100、MCOV100 の、重みなしと $\sqrt{n_{ij}}$ で重み付けした版（欠測がなければ同じ。sirt の形に合わせて
    残す）。sirt の既定の `conf.detect` の平滑化した（カーネルによる）条件付き共分散は実装しない。**探索的**: 行を半分に
    分け（JAX のキーで `floor(N/2)` 行を推定用に、または指定）、推定用の半分の行列から作る
    $d=\max(\mathrm{ccov})-\mathrm{ccov}$（最大値は sirt の `create.ccov` と同じく 0 の対角を含む）に Ward 法を当てる。
    R の `hclust(method = "ward.D")` は Lance–Williams の Ward の更新を $d$ そのものに当て、SciPy の `"ward"` は 2 乗距離に
    当てるので、SciPy には $\sqrt d$ を渡す（併合の高さは 2 乗すると R と同じ）。`cutree` はクラスタを最初に現れた順に
    番号付けするので、SciPy の `fcluster` の出力も同じく付け直す。$k=2..$`max_clusters` の各分割を両方の半分で評価し、
    `n_clusters` は推定用の半分での DETECT を最大にする（sirt と同じ選び方）。Zhang & Stout の遺伝的アルゴリズムによる
    分割の探索は実装しない。**閾値**（docstring、ガイド）: DETECT $<0.2$ / $0.4$ / $1$、ASSI $0.25$、RATIO $0.36$
    （Jang & Roussos, 2007; Zhang, 2007。§16.8 の 0.1 / 0.5 / 1 はここで訂正した）。**アンサンブル**:
    `dimensionality()` は選んだ分割の検証用の半分での DETECT を問い (B) の行として加える（`detect_threshold` $=0.2$
    未満なら支持）。`fit` がなくても使える。キーは 3 つ（平行分析、モデルに基づく平行分析、DETECT）に分ける。
    **照合（V27）**: `packages/flexpyctt/validation/r/m5_detect_reference.R`（sirt 4.2.133）。シミュレーションの 2 次元（正しい分割と
    誤った分割）、1 次元、Science（多値）のデータで `conf.detect(use_sum_score = TRUE, bias_corr = TRUE, smooth = FALSE)`、
    3 クラスタのデータで `expl.detect(use_sum_score = TRUE, estsample = )`。条件付き共分散、5 つの指標（両方の
    重み付け）、分割、両方の半分での指標は $10^{-12}$（実測 $10^{-14}$）で、Ward の併合の高さは $10^{-12}$ で一致する。
    既定の `smooth = TRUE` のままだと、sirt の `conf.detect(use_sum_score = TRUE)` は総得点を標準化してから生の項目得点を
    引いて残りの得点とするので、残りの得点の群が誤る（`sim_2d` で DETECT 1.60 が 6.73 になる）。精度のページに記録し、
    従わない。
14. **EFA と Hull 法（`efa()`、`hull()`、§16.8 の D5、V28）**: `_efa.py`（NumPy / SciPy。現在は flexpyctt に
    ある。§16.18）。
    **推定**: 独自性 $\Psi$ から回転しない解を求め、SciPy の L-BFGS-B（`ftol=1e-15`、`gtol=1e-11`）で
    $\psi_i=1-\mathrm{SMC}_i$ から $[0.005,\max(1,\mathrm{SMC})]$ の範囲（psych・EFAtools と同じ境界）で最小化する。
    ULS: $F=\tfrac12\lVert R-\Psi-\Lambda\Lambda^\top\rVert_F^2$。$\Lambda$ は $R-\Psi$ の先頭の固有ベクトルを
    $\sqrt{\max(\lambda,0)}$ でスケールしたもので、勾配は $\operatorname{diag}(\Lambda\Lambda^\top+\Psi-R)$（psych の
    `FAgr.minres` で、この $F$ の厳密な勾配）。psych の minres、EFAtools の ULS と同じ（psych の `fit.residuals` は
    非対角の残差だけを足すが、内点では最小点が同じ）。ML: $\Psi^{-1/2}R\Psi^{-1/2}$ の固有値による Jöreskog の
    集約した不一致度で、勾配は $\operatorname{diag}(\Lambda\Lambda^\top+\Psi-R)/\psi^2$（psych の `FAfn` / `FAgr`）。
    列は和が非負になるよう符号をそろえ、回転はしない。独自性が下限にあれば Heywood ケース（`efa()` は
    `HeywoodWarning`）。許容誤差は R の `optim()` の既定より厳しいので、flexpyirt の目的関数値は EFAtools の解での値を
    上回らない（テストで確認）。**適合度**（EFAtools の `.gof`）: CAF $=1-$（対角を 1 にした残差行列の KMO）。
    $\chi^2$ は Bartlett の乗数 $N-1-(2p+5)/6-2k/3$ を使い、その $p$ 値も求める。CFI と RMSEA は補正しない
    $F(N-1)$ と独立モデルの $-\log|R|(N-1)$（$df_0=p(p-1)/2$）から求める。$df=((p-k)^2-(p+k))/2$。
    **Hull**（EFAtools の `HULL` / `.hull_calc`）: $k=0..J$ の行で、適合度は CAF、CFI、$1-$RMSEA のいずれか
    （$k=0$ は $1-$KMO($R$)、0、$1-$RMSEA$_0$ と $df_0$）。適合度が増えない解を除き、両隣を結ぶ弦以下の内点が
    なくなるまで除き、残った内点で scree 値 $st$ を求めて最大のものを選ぶ（凸包上の点が 3 未満なら、警告を出して
    適合度が最大の解）。既定の $J$ は、正規データの SMC 固有値と平均の基準による平行分析（EFAtools の `efa_parallel`
    の既定。ここでは JAX のキーで 1000 反復）の結果に 1 を足し、$df>0$ となる最大の $k$ で上を抑え、
    $\min(3,\text{上限})$ 以上にする。`max_factors` を与えればその $J$ をそのまま使う。変数は 6 つ以上。表には解ごとの
    `heywood` と `converged` を記録し、選ばれた解が不適切なら EFAtools と同じく `HeywoodWarning` を出す。
    **アンサンブル**: 項目が 6 つ以上なら、`dimensionality()` は問い (A) に Hull（CAF、ULS、キーの 4 つ目の分割による
    既定の $J$）を加える。
    **照合（V28）**: `packages/flexpyctt/validation/r/m5_efa_hull_reference.R`（EFAtools 1.1.0、psych 2.6.9）。V24 のポリコリック行列のうち
    変数が 6 つ以上のものと、シミュレーションの Pearson 行列 2 つ（3 因子、$N=500$。弱い 2 因子、$N=300$）。
    $k\le J$ のすべてで ULS と ML について、含意する $\Lambda\Lambda^\top$ と共通性は $10^{-4}$（実測 $7\times10^{-5}$
    以下で、最大は Heywood の解）、CAF / CFI / RMSEA は $2\times10^{-4}$、$\chi^2$ は相対 $10^{-4}$、$df$ は厳密に一致。
    3 因子の行列での psych の `fa()` とも $10^{-4}$ で一致する。Hull は行列 x 推定法 x 適合度のすべてで EFAtools が
    記録した $J$ を使い、選ばれる $k$ と凸包上の点は同じ、適合度は $10^{-5}$（実測 $1.4\times10^{-6}$ 以下）、$st$ は
    相対 1%（小さな差の比なので。実測は絶対 $3.5\times10^{-3}$ 以下）で一致する。比較データ法は §16.15。
15. **比較データ法（`comparison_data()`、§16.8 の D5b、V29。現在は flexpyctt にあり、`key` の代わりに `rng` を
    取る。§16.18）**: `_comparison_data.py`（NumPy。乱数を引数で渡すので、それを与えれば生成は決定的）。$k=1..K$（$K$ は $df$ が正になる最大の $k$、または
    `max_factors`）について、各変数の観測値を `n_population` 回復元抽出して並べ、標準正規の $F$ $(N,k)$ と
    $E$ $(N,p)$ を引き、母集団を生成する（Ruscio & Kaczetow, 2008。EFAtools の `.simulate_cfm_empirical` と同じ）。
    中間の $R^*$ を主因子法（EFAtools の `.paf_iter`。共通性の初期値は 1、固有値は絶対値、最大の変化に対する
    基準 $10^{-3}$、`max_iter` 50）で分解し、負荷量を $[-1,1]$ に切り、最初の負荷量が負なら符号を反転して、得点
    $F\Lambda^\top+E\,\mathrm{diag}(\sqrt{\max(1-h^2,0)})$ を作る。そのあと列ごとに順に、その列で行を並べ替え（安定な
    整列）、その列を並べた復元抽出値で置き換える。$R^*$ は $R$ から始め、下三角の残差の RMSE が（1 から）改善する
    限り残差 $R-R_{\text{rep}}$ だけ動かし、改善しなければ最良の $R^*$ からステップを半分にし、5 回続けて失敗したら
    止める。母集団からデータと同じ大きさの標本を `n_samples` 個引き、Pearson 相関の固有値と観測固有値の RMSE を
    記録する。$k$ 因子での RMSE が $k-1$ 因子より小さいと片側の Mann–Whitney 検定（連続性と同順位の補正を含む正規
    近似で、$n\ge50$ での R の `wilcox.test` と同じ）が $\alpha=0.30$（Ruscio & Roche の推奨）で言う限り $k$ を
    残し、最後に残した $k$ を結果とする。Pearson 相関のみ（Ruscio & Roche、EFAtools と同じ）、欠測のないデータ。
    主因子法の負荷量は列の和が非負になるよう符号をそろえる。EFAtools は LAPACK の符号をそのまま使うが、効くのは
    $k\ge2$ のときだけで、母集団の分布は変わらない。**アンサンブル**: 欠測がなければ `dimensionality()` は問い (A) に
    CD を加える（キーの 5 つ目の分割）。**照合（V29）**: `packages/flexpyctt/validation/r/m5_comparison_data_reference.R`（EFAtools 1.1.0）。
    $k=1..3$ の主因子法の負荷量は $\Lambda\Lambda^\top$ で $10^{-12}$、EFAtools 自身の乱数（同じシードで再現して
    書き出す）による 1 因子の母集団は完全に同じ、EFAtools の RMSE の列に対する順位和の $p$ 値は相対 $10^{-10}$ で
    一致する。3 因子、2 因子の順序、1 因子の 2 値のデータでの `CD()` とは、それぞれ 2 つのキーで同じ提案、平均 RMSE は
    15% 以内（母集団の乱数が違うため）で一致する。
16. **DIMTEST（`dimtest()`、§16.8 の D6、V30）**: `_dimtest.py`（NumPy。乱数は引数で渡す。現在は flexpyctt に
    あり、`key` の代わりに `rng` を取る。§16.18）。
    欠測のない 2 値データ、項目は 4 つ以上。**統計量**（Nandakumar & Stout, 1993）: 回答者を PT の得点で群に分け、
    `min_group`（20）未満の群と $S_k\le0$ の群を除く。$T_L=K^{-1/2}\sum_k(\hat\sigma_k^2-\hat\sigma_{U,k}^2)/S_k$。
    $\hat\sigma_k^2$ は AT の得点の（$J_k$ で割る）分散、$\hat\sigma_{U,k}^2=\sum_i\hat p_{ik}(1-\hat p_{ik})$、
    $S_k^2=(\hat\mu_{4,k}-\hat\sigma_k^4+\hat\delta_{4,k})/J_k$、$\hat\delta_{4,k}=\sum_i\hat p_{ik}(1-\hat p_{ik})
    (1-2\hat p_{ik})^2$。**バイアス補正**（Stout, Froelich & Gao, 2001）: 検定用の標本のカーネル平滑化した IRF（能力は
    総得点の順位の $\Phi^{-1}(r/(N+1))$ で同順位は一様乱数で分け、$[-4,4]$ の 161 点上で帯域幅 $1.1N^{-1/5}$ の
    ガウスカーネル。Ramsay, 1991）から、$\theta\sim N(0,1)$ と線形補間した IRF で検定用の標本と同じ大きさの 1 次元
    データを $B$（100）個作り、$T=(T_L-\bar T_G)/\sqrt{1+1/B}$、$p=1-\Phi(T)$。**AT の選択**: `assessment_items` を
    与えれば標本全体で検定する。与えなければ無作為な 3 分の 1（`training_fraction`）で AT を選び、残りで検定する
    （Froelich & Habing, 2008 は別の部分標本を使う）。訓練用の条件付き共分散（§16.13）の Ward 法で 2〜6 群に分けた
    ときのクラスタのうち、要素が 2 から `max_assessment_size`（項目の半分）までで、{AT, 残り} への分割の DETECT が
    最大のもの。Froelich & Habing の条件付き共分散による AT の選択の考え方に沿うが、彼らの ATFIND そのものではなく、
    そう明記する。**アンサンブル**: 欠測のない 2 値データなら `dimensionality()` は問い (B) に DIMTEST（$p\ge\alpha$）を
    加える（キーの 6 つ目の分割）。そうでなければ使えない行とする。**照合（V30）**: 単独のプログラム DIMPACK 以外に
    実装がないので、`packages/flexpyctt/validation/simulation/dimtest_study.py`（結果は `dimtest_study.json`）のシミュレーション研究で
    受け入れる。2PL のデータ（傾き $U(0.8,1.8)$、切片 $N(0,0.8)$、項目を半分ずつ相関 $\rho$ の次元に置く）、各条件
    200 反復、すべて既定値:

    | データ | $N$ | $J$ | $\rho$ | $T$ の平均 | $T$ の SD | .05 で棄却 | .10 で棄却 |
    |---|---|---|---|---|---|---|---|
    | 1 次元 | 1000 | 20 | 1 | -0.07 | 0.72 | 0.005 | 0.020 |
    | 1 次元 | 2000 | 20 | 1 | -0.12 | 0.71 | 0.010 | 0.030 |
    | 1 次元 | 2000 | 30 | 1 | -0.03 | 0.73 | 0.010 | 0.025 |
    | 2 次元 | 1000 | 20 | 0.7 | 4.85 | 0.97 | 0.990 | 0.995 |
    | 2 次元 | 2000 | 20 | 0.7 | 6.94 | 0.88 | 1.000 | 1.000 |
    | 2 次元 | 2000 | 30 | 0.8 | 6.50 | 0.86 | 1.000 | 1.000 |

    （2026-11-25 に flexpyctt の NumPy の生成器で再実行した値。§16.18。移す前の JAX のキーでは、.05 での棄却率は
    0.020、0.005、0.000 と 0.990、1.000、1.000 だった。）

    検定は保守的である。$\bar T_G$ は $T_L$ の偏りを除く（$T$ の平均は 0 に近い）が、帰無仮説の下での $T_L$ の標準
    偏差は 1 ではなく 0.72 程度なので、第 1 種の誤りは名目水準を下回る。相関 0.7〜0.8 の 2 次元に対する検出力は 0.99〜1。
    したがって小さい $p$ は多次元性の強い証拠であり、大きい $p$ は一次元性の弱い証拠にとどまる（docstring とガイドに
    書く）。$T_G$ のブートストラップ SD で標準化し直せば較正できるが、公表された統計量から離れるので行わない。各部分は
    直接の計算と単体テストで比べ、`packages/flexpyctt/tests/test_dimtest.py` で研究の小さい版（帰無 20 反復、対立 10 反復）を実行する。
17. **ジャックナイフ傾き指標（`FitResult.jsi()`、V31）**: `_jsi.py`。**定義**（Edwards, Houts & Cai, 2018。mirt の
    `residuals(type = "JSI")` と同じ）: ジャックナイフは回答者ではなく項目について行う（11 項の以前の版は「部分群」と
    書いていたので直した）。各項目 $j$ について項目 $j$ を除いてモデルを推定し直し、他のすべての項目 $i$ について
    $\mathrm{JSI}_{ji}=(\hat a_i-\hat a_i^{(-j)})/\mathrm{SE}(\hat a_i^{(-j)})$ を求める。SE は推定し直したモデルの
    もの。行列は対称でなく、mirt の `fold = TRUE` は転置を足す（`pairs` の `folded` 列）。mirt はカットオフを示さず、
    flexpyirt も示さない。docstring では、他より目立つ値を探すよう書く。**項目を除く = マスク**: 項目 $j$ の反応を
    欠測にする（`y[:, j] = -1`、one-hot の行がすべて 0）。項目 $j$ は $\ell$ から抜け、配列の shape は変わらない。
    除く項目の番号は trace される引数なので、1 回の jit コンパイルですべての $j$ をまかなえる（§5）。残すパラメタは、
    他のいずれかの行（他の項目と潜在分布）が使う自由パラメタ id なので、項目間の等値制約と事前分布はそのまま
    引き継がれる。項目 $j$ だけの id は推定値のまま置いて無視する。自由な傾きを持たない項目（Rasch、PCM、固定した
    傾き）の列は NaN。単一群・1 次元・`Normal` のモデルのみ（mirt も 1 因子を要求する）。項目は 3 つ以上。
    **方法**（どれも全項目モデルの推定値 $\hat\eta$ から始める）: `"refit"`（既定）は、残す id について
    $\ell_{(-j)}+\log\pi$ を Newton–Raphson で最大化する。ステップ半減を入れ、情報行列が正定値でなければリッジを
    足す。最大のステップが `tol`（$10^{-8}$）未満になれば止め、`max_iter` を超えたら `ConvergenceWarning`。SE は最後に
    評価した観測情報（`information_hessian`。どの点でも Oakes のものと等しい）から求める。ウォームスタートの Newton 法は
    5 ステップ程度で収束する（EM なら数十回）。`"newton"` は同じ反復を `newton_steps` 回だけ行い、SE は最後に評価した
    情報行列（1 ステップなら出発点）から求める。`"one_step"`（オーナーが示した Julia の案）は、全項目モデルの $V$
    （`FitResult.vcov`）を使って $\hat\eta^{(-j)}\approx\hat\eta+V\nabla\ell_{(-j)}(\hat\eta)$ とし、分母にも
    全項目モデルの SE を使う。$J$ 個の勾配は 1 回の `vmap` で求まるので、計算量は E ステップ $J$ 回分ほど。EM 1 ステップ
    は採らない。欠測情報の割合を $f$ とすると、1 ステップでは本来の変化の $1-f$ しか動かない（予備実験で JSI が 3 割ほど
    縮んだ）。**近似の精度**（推定し直した値との差の絶対値の最大。局所依存のペアを 1 組含む 2PL / GRM）:

    | データ | $J$ | refit | `"newton"` 1 ステップ | 2 ステップ | `"one_step"` |
    |---|---|---|---|---|---|
    | 2PL、$N=1000$ | 15 | 0.7 秒 | 0.012（0.20 秒） | 0.010 | 0.021（0.02 秒） |
    | 2PL、$N=2000$ | 30 | 4.6 秒 | 0.002（1.2 秒） | <0.001 | 0.002（0.06 秒） |
    | 2PL、$N=3000$ | 60 | 45 秒 | 0.001（15 秒） | <0.001 | <0.001（0.2 秒） |
    | GRM(4)、$N=2000$ | 30 | 17 秒 | 0.002（5.1 秒） | <0.001 | 0.002（0.06 秒） |
    | LSAT7 の 2PL | 5 | | 2.0 | 0.88 | 1.3 |
    | Science の GRM | 4 | | 2.7 | 2.9 | 2.5 |
    | GPCM(4)、$N=800$ | 8 | | 0.11 | 0.45 | 0.39 |

    項目が少ないと 1 つ除くことは小さな摂動でなく、Taylor 近似は外れる（Newton 法の 2 次収束の段階に入る前は、2 ステップ
    のほうが 1 ステップより悪いこともある）。15 項目以上なら `"one_step"` は推定し直した値と 0.02 以内で、計算時間は
    数百分の一。10 項目未満（`_JSI_APPROXIMATION_MIN_ITEMS`）では近似は警告を出し、refit を勧める。
    **照合（V31）**: `validation/r/m5_jsi_reference.R`（mirt 1.46.1。推定もその中の推定し直しも TOL $=10^{-10}$）。
    LSAT7 の 2PL、Science の GRM、局所依存のペア 2 組と 5% の MCAR を含むシミュレーションの 2PL、局所依存のペア 1 組を
    含むシミュレーションの GPCM。推定し直した値は $10^{-5}$ で一致する（観測値 $\le1.2\times10^{-7}$）。テストでは
    ほかに、推定し直した値を項目を除いたデータへの `fit()` と、近似を 15 項目のデータで推定し直した値と（`"newton"`
    1 ステップ 0.03、2 ステップ 0.02、`"one_step"` 0.05）、vmap した勾配を 1 つずつの勾配と比べる。
18. **パッケージの分割: `flexpyctt` と `flexpyequate`（2026-11-25 決定、issue #87）**: 推定済みの IRT モデルを使わない道具を、JAX に依存
    しない別の配布物としてこのリポジトリから公開する。flexpyirt は BA-EM による推定と、その結果を使う道具に絞る。

    | 配布物 | 範囲 | 依存 |
    |---|---|---|
    | `flexpyirt` | 推定。`FitResult` を使う道具（`local_dependence()`、`jsi()`、`model_parallel_analysis()`、`link()`、IRT の等化）。`dimensionality()` のレポート | JAX、`flexpyctt` |
    | `flexpyctt` | モデルを使わない項目・テスト分析: `polychoric()`、`parallel_analysis()`、`velicer_map()`、`empirical_kaiser()`、`efa()`、`hull()`、`comparison_data()`、`detect()`、`exploratory_detect()`、`dimtest()`。今後、古典的項目分析と Mantel–Haenszel DIF | NumPy、SciPy、pandas |
    | `flexpyequate` | 得点分布の等化のみ: 平均等化、線形等化、等パーセンタイル等化、アンカーテストを使う方法、対数線形モデルによる事前平滑化、カーネル等化（設計文書は別にある: `packages/flexpyequate/docs/DESIGN.ja.md`） | NumPy、SciPy、pandas |

    **配置**: uv ワークスペース。flexpyirt はリポジトリのルートに置いたままにする（`pyproject.toml`、`src/flexpyirt/`、
    `tests/`）。issue の `packages/flexpyirt/` への移動は見送る（ディレクトリを動かすと、開いているすべてのブランチと
    衝突するため）。メンバーは `packages/*`。`packages/flexpyctt/` は自分の `pyproject.toml`、`README.md`（PyPI の
    ページ。リンクは絶対 URL）、`LICENSE`、`src/flexpyctt/`、`tests/`（`tests/reference/` を含む）、`validation/` を持つ。
    flexpyirt は flexpyctt に依存する（`[tool.uv.sources] flexpyctt = { workspace = true }`）ので、
    `pip install flexpyirt` だけでこの 2 つの道具はすべて入る。`packages/flexpyequate/` も同じ構成である
    （`docs/DESIGN.md` が加わる。R スクリプトはルートの `validation/r/equate_*.R`）。**flexpyequate は独立している**:
    JAX も flexpyirt も flexpyctt も import せず、その 2 つも flexpyequate に依存せず、再公開もしない
    （`pip install flexpyirt` では入らない）。ルートでは `dev` の依存グループにだけ入れてあり、`uv sync` でテスト用に
    インストールされる。flexpyirt の sdist には含めない。項目パラメータを使う等化は flexpyirt に残す。
    **バージョンは 1 つ**: 3 つとも同じタグ `vX.Y.Z` を hatch-vcs で
    読む（メンバー側は `raw-options.root = "../.."`）。リリースのワークフローは `uv build --package` でそれぞれをビルド
    し、flexpyctt を flexpyirt より先に公開する。flexpyequate は自分の PyPI のジョブを持ち、どちらも待たない。flexpyirt は flexpyctt の非公開ヘルパーをいくつか使う（後述）ので、
    2 つは同じバージョンで使う前提。依存にはまだバージョンの範囲を付けていない（未決: ビルドフックで
    `flexpyctt==<同じバージョン>` に固定する）。

    **flexpyctt は JAX を import しない**。64 ビットの設定も要らない（NumPy は float64）。CI のジョブが flexpyctt
    だけをインストールし、`jax` が無いことを確かめたうえでテストを実行する。別のジョブが flexpyequate について同じことを
    行う（`jax`、`flexpyirt`、`flexpyctt` が無いことを確かめる）。
    **ツールの設定**: flexpyirt と flexpyctt の ruff と mypy は、ルートの `pyproject.toml` で設定する。flexpyequate は
    同じ規則の `[tool.ruff]` と `[tool.mypy]` を自分で持つ（ruff はルートから実行してもそのディレクトリにはその設定を
    使う。mypy は `packages/flexpyequate/` で実行する）。flexpyequate のテストはルートと同じ `tests` という名前の
    パッケージで、同じ名前のパッケージ 2 つを 1 回の mypy では扱えないためである。**ドキュメントサイト**: flexpyctt の
    道具はユーザーガイドと API リファレンスで説明する（flexpyirt が再公開している）。flexpyequate は 1 ページ
    （`docs/equating.md`）で、`scripts/mkdocs_hooks.py` がそこにパッケージの README（PyPI のページ）を入れる。文章は
    1 か所にしかない。

    **ポリコリック相関の 2 つの実装。** `flexpyctt.polychoric()` は §16.9 の推定量を NumPy / SciPy で新しく実装した
    もの（定義、引数、既定値、警告、`PolychoricResult` は同じ）:
    - $\Phi_2$ は Owen (1956) の $T$ 関数（`scipy.special.owens_t`）で計算する:
      $\Phi_2(h,k;\rho)=\tfrac12\Phi(h)+\tfrac12\Phi(k)-T(h,a_h)-T(k,a_k)-\beta$、
      $a_h=(k-\rho h)/(h\sqrt{1-\rho^2})$、$a_k=(h-\rho k)/(k\sqrt{1-\rho^2})$、$\beta$ は $hk>0$ または
      （$hk=0$ かつ $h+k\ge0$）のとき 0、それ以外は $\tfrac12$。$\Phi_2(0,0;\rho)=\tfrac14+\arcsin\rho/(2\pi)$。
      無限大の限界は正確に扱う。JAX 版の Genz (2004) の求積とは独立な計算なので、2 つの一致は意味のある確認になる。
    - $z=\operatorname{atanh}\rho$ についての Newton 法を、すべてのペアについて同時に行う（配列演算。ペアについての
      Python ループは無い）。微分は解析的: $\partial_\rho\pi_{ab}=\Delta\phi_2$（Plackett, 1954）、
      $\partial_\rho\phi_2=\phi_2\,[\rho/s+(hk(1+\rho^2)-\rho(h^2+k^2))/s^2]$、$s=1-\rho^2$。ステップの規則（凹なら
      Newton、そうでなければ勾配方向に 0.5。ステップは $\pm2$ に制限。ステップ半減）、境界 $|z|\le8$、収束判定、境界解の
      判定は §16.9 と同じ。

    `flexpyirt.polychoric()` は JAX 版のまま（`_polychoric.py`、`correlation.py`）公開し、動作は変えない。返す
    `PolychoricResult` は同じクラス（flexpyctt で定義）なので、どちらの結果も `velicer_map()`、`efa()` などに渡せる。
    移した道具はすべて NumPy 版を使い、`model_parallel_analysis()` と `dimensionality()` も NumPy 版を使う。
    **照合**: NumPy 版は V24 を同じ許容誤差で通る（lavaan $10^{-6}$、閾値 $10^{-12}$、psych $10^{-3}$）。$\Phi_2$ は
    Plackett の積分と $10^{-14}$ で一致する。flexpyirt のテストで、V24 のすべてのケースについて 2 つの版が $10^{-9}$ で
    一致し、フラグが等しいことを確かめる。

    **乱数。** flexpyctt の関数は、JAX 版が `key` を取っていたところで `rng`（`numpy.random.Generator` または整数の
    シード）を取る。`key` が必須だった関数では `rng` も必須。乱数列は JAX のものと異なるので、参照データを使う結果は
    モンテカルロ誤差の範囲で変わる。決定的な参照値（V24、V25、V27–V29）は変わらない。flexpyirt に残る関数は `key` の
    まま: `model_parallel_analysis()` はそれでシミュレーションし、`dimensionality()` はこれまでどおり 6 つのキーに
    分けてから、flexpyctt の呼び出しごとに生成器に変える（`dimensionality._rng_from_key`: キーのデータをシードにした
    `numpy.random.default_rng`）。

    **エラーと警告。** flexpyctt は `FlexpycttError`、`SpecificationError`、`FlexpycttWarning`、
    `ConvergenceWarning`、`HeywoodWarning`、`NotPositiveDefiniteWarning` を定義する（`flexpyctt.errors`）。
    `flexpyirt.errors` は同じクラスオブジェクトを再公開し、`FlexpyirtError` / `FlexpyirtWarning` は flexpyctt の基底
    クラスの別名とする。これで `except flexpyirt.errors.SpecificationError`（または `FlexpyirtError`）は flexpyctt の
    関数が送出したものも捕まえる。`IdentificationError` と `PrecisionError` は flexpyirt に残す。

    **再公開（決定: する）。** `flexpyirt` は移した関数と結果クラスをすべて再公開する
    （`flexpyirt.parallel_analysis is flexpyctt.parallel_analysis`）。`develop` に対して書いたコードの import は
    そのまま動き、API の変更は `key=` → `rng=` だけ。例外は `polychoric` で、`flexpyirt.polychoric` は JAX 版。
    ドキュメントでは、移した道具を flexpyctt のものとして説明する。

    **置き場所。** `flexpyctt.correlation`（`polychoric`、`PolychoricResult`、`smooth_correlation`）、
    `flexpyctt._polychoric`（カーネル）、`flexpyctt.dimensionality`（9 つの関数と結果クラス）、`flexpyctt._efa`、
    `_detect`、`_dimtest`、`_comparison_data`（そのまま移動）。`flexpyirt.dimensionality` には `dimensionality()`、
    `model_parallel_analysis()`、`LocalDependenceResult`、`JSIResult`、`ModelParallelAnalysisResult`、
    `DimensionalityResult` が残り、flexpyctt の非公開ヘルパー `_coerce_ordinal`、`_observed_correlation`、
    `_eigenvalues`、`_descending_eigenvalues` を import する。**テストと参照値**: 移した道具のテストと `v24`、`v25`、
    `v27`、`v28`、`v29` は `packages/flexpyctt/tests/`（JAX なし）、それらの R スクリプトと DIMTEST のシミュレーションは
    `packages/flexpyctt/validation/` に置く。flexpyirt のテストには、JAX 版ポリコリックのテスト（`v24` は flexpyctt の
    ディレクトリから読む）、2 つの版の一致テスト、V26、アンサンブルのテストが残る。

19. **確認的多次元モデルの $M_2$ / $M_2^*$ と S-X2（M6、#77、V39）**:
    `FitResult.m2()`（§16.3）と `FitResult.item_fit()`（§15.40）は、`Model.with_factors()` で宣言した単一母集団のモデル
    （$D\in\{2,3\}$、直積格子、§15.29–§15.30）を受け付ける。どちらの統計量も、項目カテゴリ確率の積を潜在分布で積分した
    ものなので、定義は $D$ に依存しない。ノードの添字 $q$ は推定時の直積格子の $Q=n^D$ 個のノード $\boldsymbol\theta_q$ を
    動き、重みは推定時と同じ正規化した多変量正規の重み
    $w_q(\boldsymbol\rho)=\phi_D(\boldsymbol\theta_q;\mathbf 0,\mathbf R)/\sum_{q'}\phi_D(\boldsymbol\theta_{q'};\mathbf 0,\mathbf R)$
    （§15.30）、$P_{jk}(\boldsymbol\theta)$ は同じ項目カーネル（線形予測子に $\boldsymbol a_j^\top\boldsymbol\theta$。負荷しない
    因子の識別力は 0 に固定）から得る。
    **$M_2^*$。** $m_{1,jq}=\sum_k kP_{jk}(\boldsymbol\theta_q)$、$m_{2,jq}=\sum_k k^2P_{jk}(\boldsymbol\theta_q)$ として
    $$\pi_j=\sum_q m_{1,jq}w_q,\qquad \pi_{ij}=\sum_q m_{1,iq}m_{1,jq}w_q\quad(i<j)$$
    であり、$\Xi_2$ は §16.3 と同じ和から作る（局所独立はベクトル $\boldsymbol\theta$ 全体を与えたもとで成り立つので、
    2 つの項目が同じ因子に負荷するかどうかによらず、$i\ne j$ で $E[y_iy_j\mid\boldsymbol\theta]=m_{1,i}m_{1,j}$）。
    ヤコビアン $\Delta_2=\partial\pi_2/\partial\eta^\top$ には、項目の列
    $$\frac{\partial\pi_j}{\partial\xi_j}=\sum_q\frac{\partial m_{1,jq}}{\partial\xi_j}w_q,\qquad
      \frac{\partial\pi_{ij}}{\partial\xi_i}=\sum_q\frac{\partial m_{1,iq}}{\partial\xi_i}m_{1,jq}w_q$$
    に加えて、自由な因子間相関 $\rho_{ab}$（$a<b$）ごとに 1 列がある:
    $$\frac{\partial\pi_{ij}}{\partial\rho_{ab}}=\sum_q m_{1,iq}m_{1,jq}\frac{\partial w_q}{\partial\rho_{ab}},\qquad
      \frac{\partial w_q}{\partial\rho_{ab}}=w_q\Big(g_{ab,q}-\sum_{q'}w_{q'}g_{ab,q'}\Big),\qquad
      g_{ab,q}=\big[\mathbf R^{-1}\boldsymbol\theta_q\boldsymbol\theta_q^\top\mathbf R^{-1}-\mathbf R^{-1}\big]_{ab}$$
    （$\pi_j$ も同様）。コードでは、すべての列を周辺モーメントの $\eta$ についての `jax.jacfwd` 1 回で得る（潜在分布の
    パラメタは `_latent_log_weights` を通じて入る）ので、パラメタの種類ごとの式は書かない。上の式は、R の参照値が独立に
    計算し直しているものである。$df=J+\binom J2-q$ で、$q$ = `n_params` は自由な相関を含む。RMSEA とその区間、SRMSR、
    （2 値モデルの）独立モデルに対する TLI / CFI は変わらない（帰無モデルに潜在構造はない）。**mirt との違い**: 自由な
    平均・分散の場合（§16.3）と同じく、mirt の `M2()` は $\Delta_2$ を作るときに `GroupPars` の推定フラグを `FALSE` に
    するので、自由な相関は $df$ では引かれるのに $\Delta_2$ に列を持たない。flexpyirt はすべての自由パラメタを $\Delta_2$
    に入れる（Maydeu-Olivares & Joe, 2005, 2006）ため、自由な相関のあるモデルでは $M_2$ が mirt よりわずかに小さい
    （V39 では相対 $3\times10^{-7}$ 〜 $7\times10^{-4}$）。非公開の `_latent_in_delta=False` で mirt の値を再現できる。
    **周辺モーメントからの識別**はこれまでどおり調べる（$\Delta_2$ の階数、$df>0$）。たとえば、小さいテストでは
    3 カテゴリの GRM 項目の自由パラメタ数がその項目を含む行数を超えることがあり、1 項目だけで測る因子は識別されない。
    **S-X2。** 合計得点は mirt と同じく全項目の合計（因子ごとではない）。Lord–Wingersky 漸化式を直積格子のすべての
    ノードで回し、
    $$L_j(s\mid\boldsymbol\theta_q)=\sum_{k=0}^{K_j-1}L_{j-1}(s-k\mid\boldsymbol\theta_q)P_{jk}(\boldsymbol\theta_q),$$
    期待割合はすべての因子について積分する:
    $$E_j[s,k]=\frac{\sum_qP_{jk}(\boldsymbol\theta_q)L_{-j}(s-k\mid\boldsymbol\theta_q)w_q}{\sum_qL(s\mid\boldsymbol\theta_q)w_q}.$$
    `_itemfit.lord_wingersky` と `s_x2_expected_tables` は `(Q, K_j)` の配列のリストと重みベクトルを受け取り、ノードの
    座標は見ないので、平らにした格子でそのまま使う（計算量は $O(J^2SQ)$、$Q=15^3$・9 項目で約 1 秒）。セルの併合
    （`mirt:::collapseCells`）と $df_j$ = 残ったセル数 − 行数 − 項目 $j$ の自由パラメタ数 − 潜在分布の自由パラメタ数
    （自由な相関。交差負荷のある項目は自由な識別力が 1 つ多い）は §15.40 と同じ。mirt の
    `itemfit(fit_stats = "S_X2")` は `thetaComb(seq(-6, 6, length.out = quadpts), nfact)` の上で `mirt_dmvnorm` の重みを
    使って同じ計算をする。既定の `quadpts`（`select_quadpts(nfact)`: 2 因子で 31、3 因子で 15）と範囲は `mirt()` と
    `Quadrature.default()` の既定と同じだが、mirt の `itemfit()` と `M2()` は推定時の格子を読まないので、別の
    `quadpts` や `theta_lim` で推定したときは R 側で改めて渡す必要がある（flexpyirt は常に推定時の格子を使う）。
    **未対応**（いずれも `SpecificationError`）: 双因子モデルと two-tier モデル（次元縮約した E-step には同時格子がなく、
    §17 のブロック構造に沿った計算が必要）、多母集団の多次元モデル（未照合。多母集団の $M_2$ は #76）。
    1 次元の結果はビット単位で変わらない（$D=1$ の同じコードパス）。
    **照合（V39）**: V14 の 4 モデルで mirt 1.48 と照合した。VALIDATION.md を参照。

## 17. 実装上の決定（M7）

1. **Gibbons–Hedeker の次元縮約による双因子モデル（`Model.with_bifactor`、V32）**:
   **指定。** `Model.with_bifactor(*, specific, general="general")` は、全項目が負荷する一般因子
   1 つと $S\ge1$ 個の特殊因子（`specific: Mapping[str, ItemSelector]`）を宣言する。各項目が負荷
   する特殊因子は高々 1 つで、どこにも挙げない項目は一般因子のみに負荷する。特殊因子はそれぞれ
   2 項目以上を要する（1 項目の特殊因子の傾きは識別されない）。表現は §15.29 の確認的モデルを
   再利用し、`factors = ((general, 全項目), (s_1, 項目), …)`、傾きは $D=1+S$ 個の `a1..aD`
   （`a1` が一般因子、負荷しない傾きは 0 に固定）で、mirt の `bfactor()` の列順と一致する。新しい
   フィールド `Model.bifactor = True` が縮約 E ステップを選ぶ。因子は直交で、潜在パラメータの行は
   `mean_1..D`（0 に固定）と `var_1..D`（1 に固定）のみで `cor_ij` 行はない
   （`Model.latent_parameter_names()`）。項目カーネルは `MAX_KERNEL_FACTORS = 32` 次元まで受け付ける
   （直積格子は `with_factors()` と `Quadrature.grid()` で 3 次元までのまま）。未対応（いずれも
   `SpecificationError`）: 多母集団、`with_latent()`、`with_factors()` との併用、Rasch / PCM 項目
   （一般因子・特殊因子の分散を自由にする必要がある）、編集したテーブルでの潜在パラメータの自由
   推定。two-tier モデル（相関する主因子）は下の 2 を参照。
   **尤度**（Gibbons & Hedeker, 1992; Cai, 2010）。項目 $j$ は $(\theta_0,\theta_{s(j)})$ にしか
   依存せず因子は独立なので、1 次元格子 $\{\theta_g\}_{g=1}^{n}$（一般）と
   $\{\theta_t\}_{t=1}^{n}$（各特殊）で
   $$L_p=\sum_g w^{(0)}_g\,\prod_{s=1}^{S}\Big[\sum_t w^{(s)}_t\prod_{j\in s}P_j(y_{pj}\mid\theta_g,\theta_t)\Big]
   \prod_{j\in s_\varnothing}P_j(y_{pj}\mid\theta_g)$$
   となる。$s_\varnothing$ は一般因子のみの項目の集合、
   $w^{(m)}=\mathrm{softmax}(b+\log\phi(\cdot;\mu_m,\sigma^2_m))$ は因子ごとの正規化した 1 次元重み
   （§4.4、§15.44）。計算量はパターンあたり $O(n^{1+S})$ ではなく $O(S\,n^2)$。**実装。**
   一般因子のみの項目は「特殊」重みが一様な追加ブロック $s_\varnothing$ に入れ、すべてのブロックを
   同じに扱う（その対数確率は $\theta_t$ に依存しないので内側の和は厳密）。ブロック $s$ は形
   $(n^2, D)$ の埋め込み節点を持ち（0 列目が $\theta_g$、$s$ 列目が $\theta_t$、他は 0）、既存の
   カーネルがそのまま $P_j$ を評価する（他の特殊因子への傾きは 0 に固定）。
   $\ell^{(s)}_{pgt}=\sum_{j\in s}\log P_j$、
   $L^{(s)}_{pg}=\operatorname{logsumexp}_t(\ell^{(s)}_{pgt}+\log w^{(s)}_t)$、
   $\lambda_{pg}=\log w^{(0)}_g+\sum_s L^{(s)}_{pg}$ とおくと、周辺対数尤度は
   $\sum_pc_p\operatorname{logsumexp}_g\lambda_{pg}$、ブロック $s$ の事後確率は
   $h^{(s)}_{pgt}=\exp(\lambda_{pg}-\log L_p)\exp(\ell^{(s)}_{pgt}+\log w^{(s)}_t-L^{(s)}_{pg})$。
   項目 $j$ の期待度数はそのブロックの $(n^2)$ 格子上
   $\bar r_{jkgt}=\sum_pc_ph^{(s(j))}_{pgt}\mathbf 1[y_{pj}=k]$ なので、項目ごとの Newton M ステップ
   （§4.3）、項目間の等値制約（§16.1）、事前分布は、項目のブロック節点を使う以外は変わらない。
   $\bar n$ は一般因子の周辺と各ブロックの特殊因子の周辺 $(G, 1+B, n)$ を持つ（後の潜在 M ステップ
   用）。`CompiledProblem.reduction`（`BifactorBlock`: ブロック節点、1 次元節点と基底重み、
   各ファミリーのブロック番号）が Core の関数を切り替え、`CompiledProblem.nodes` は一般因子の
   格子を $D$ 列に埋め込んだものを持つ。
   **標準誤差。** `"hessian"` は縮約した周辺対数尤度を微分する（チャンク化した forward-over-reverse、
   §15.20）。`"oakes"` は Oakes の恒等式を $I(\hat\eta)=-\,\mathrm d F(\eta)/\mathrm d\eta$、
   $F(\eta)=\nabla_\eta Q_{\mathrm{post}}(\eta\mid\eta')\big|_{\eta'=\eta}$（同じ点で計算した
   E ステップの重みでの完全データ勾配。両方の引数で微分するとちょうど Oakes の 2 項になる）の形で、
   縮約 E ステップを通したチャンク化 forward-mode 自動微分で計算する。§15.20 の構造化した交差項は
   全節点上の同時事後分布 1 つを前提とするので使わない。**スコアリング。** EAP は分解した事後分布で
   厳密に計算する: $\hat\theta_{p0}=\sum_g\theta_g h_{pg}$、
   $\hat\theta_{ps}=\sum_{g,t}\theta_t h^{(s)}_{pgt}$（事後 SD も同様）。MAP・ML・WLE・PV、項目
   適合度、$M_2$、局所依存、JSI、DIF、リンキングは双因子モデルでは `SpecificationError`（多次元
   なので）。**求積。** 1 つの `Quadrature` が全因子の 1 次元格子を与え、既定は mirt の
   `select_quadpts(2) = 31` 点、$[-6, 6]$（mirt の `bfactor()` は
   `select_quadpts(nfact - nspec + 1)` を使う）。**JSON**（schema version 5）: モデルに
   `"bifactor": true|false` を保存する。版 1–4 はこのキーを含んではならず、`false` として読む。
   読み込み時、双因子構造は `with_bifactor()` で組み立て直す。

2. **two-tier モデル（`Model.with_two_tier`、V33）**（Cai, 2010）:
   **指定。** `Model.with_two_tier(*, primary, specific, correlated=True)` は $P\in\{1,2\}$ 個の
   主因子（primary）と $S\ge1$ 個の特殊因子を宣言する（どちらも `Mapping[str, ItemSelector]`）。
   各項目は主因子の少なくとも 1 つに負荷し（主因子どうしのクロスローディングは可）、特殊因子には
   1 つまで負荷する。各特殊因子には 2 項目以上が必要。因子の順は主因子が先で、傾き `a1..aP` が
   主因子、`a(P+1)..aD` が特殊因子（$D=P+S$）。mirt の `bfactor(data, model, model2 = )` の列順と
   同じ。主因子どうしは相関してよく（`correlated=True` で `cor_ij`、$i<j\le P$ を自由推定）、
   特殊因子は他のすべてと直交する。潜在分布の行は `mean_1..D`（0 固定）、`var_1..D`（1 固定）と、
   主因子の組だけの `cor_ij`。`Model.n_primary_factors = P` がこの構造を示す（他のモデルでは 0）。
   双因子モデルは $P=1$ で主因子が全項目に負荷する場合にあたり、どちらも Core の同じ縮約経路を
   使う（`Model.is_reduced`、`Model.n_primary`）。$P=3$ はブロックごとに 4 次元格子が必要なので、
   積格子の 3 次元までという制限（§4.9）に合わせて受け付けない。未対応（いずれも
   `SpecificationError`）: 双因子モデルと同じもの（多母集団、`with_latent()`、Rasch / PCM 項目、
   平均・分散の自由推定）と、`with_factors()` / `with_bifactor()` との併用。
   **尤度。** 主因子の積格子 $\{\boldsymbol\theta_g\}_{g=1}^{n^P}$ と、その正規化した多変量正規の
   重み $w^{(0)}_g(\boldsymbol\rho)$（§15.30）を使って
   $$L_p=\sum_g w^{(0)}_g\,\prod_{s=1}^{S}\Big[\sum_t w^{(s)}_t\prod_{j\in s}P_j(y_{pj}\mid\boldsymbol\theta_g,\theta_t)\Big]
   \prod_{j\in s_\varnothing}P_j(y_{pj}\mid\boldsymbol\theta_g)$$
   となる。§17.1 の一般因子の節点を主因子の節点に置き換えたもので、計算量はパターンあたり
   $O(n^{P+S})$ ではなく $O(S\,n^{P+1})$。**実装。** `BifactorBlock` を一般化する: 主因子の格子
   `(Q_p, P)`（$Q_p=n^P$）とその基底重み、形 $(B, Q_p n, D)$ のブロック節点（列 $0..P-1$ が主因子の
   節点、列 $P+b$ が特殊因子の節点、主因子の添字が上位）、static な `n_primary` を持つ。§17.1 の
   式は、$g$ が $Q_p$ 個の主因子節点を動くと読み替えればすべて成り立つ。$\bar n$ は平坦に
   `(G, Q_p + B n)` で持つ（主因子の周辺、続いて各ブロックの特殊因子の周辺）。
   **潜在分布の M ステップ。** $Q$ の潜在部分は
   $Q^{\text{lat}}=\sum_g\bar n^{(0)}_g\log w^{(0)}_g(\boldsymbol\rho)+\sum_{s,t}\bar n^{(s)}_t\log w^{(s)}_t$
   で、自由な相関に依存するのは第 1 項だけ。これを §4.4 のマスク付き Newton で主因子の多変量正規
   パラメタについて最大化する（離散化した尤度に対して厳密なので EM は単調のまま。正定値でない
   候補は NaN になり、直線探索で棄却される）。双因子モデルには自由な潜在パラメタがないので、
   これまでどおり何もしない。**識別。** §15.31 の規則を主因子に適用する（相関を自由にするには
   固定した傾きが必要。因子構造による 0 固定の傾きも数える）。**標準誤差・スコアリング。** §17.1 と
   同じ: 縮約した対数尤度のヘッセ行列、E ステップを通した全微分としての Oakes（相関を含む）、
   分解した事後分布による EAP（主因子の平均は主因子の格子上でとる）。**求積。** 次元あたり
   `Quadrature.default(P+1)` 点: $P=1$ で 31、$P=2$ で 15（mirt の
   `select_quadpts(nfact - nspec + 1)`）。**JSON**（schema version 6）: モデルに
   `"n_primary_factors"`（two-tier 以外は 0）を保存する。版 1–5 はこのキーを含んではならない。
   読み込み時、two-tier 構造は `with_two_tier()` で組み立て直す。
3. **双因子・two-tier モデルの局所依存（`FitResult.local_dependence`、V38、issue #81）**:
   §16.11 の統計量を項目 1–2 の縮約グリッド上で計算し、$n^{P+S}$ の積グリッドは作らない。
   記法は上と同じ: 主因子のノード $\boldsymbol\theta_g$ と重み $w^{(0)}_g$（$g=1..Q_p$、$Q_p=n^P$）、
   特殊因子のノード $\theta_t$ と重み $w^{(s)}_t$、$s(j)$ は項目 $j$ の特殊因子（持たないとき
   $\varnothing$）、$P_{ja}(\boldsymbol\theta_g,\theta_t)$ はブロックグリッド上のカテゴリ確率。
   **期待度数表。** 1 項目について特殊因子を積分すると、主因子を与えたときのトレースになる:
   $$\bar P_{ja}(\boldsymbol\theta_g)=\sum_t w^{(s(j))}_t\,P_{ja}(\boldsymbol\theta_g,\theta_t)$$
   （$s(j)=\varnothing$ なら $\bar P_{ja}=P_{ja}$。そのブロックの一様重みでも同じ値になる）。
   特殊因子は互いに、また主因子とも独立なので、両方に回答した $n_{ij}$ 人についてのモデル上の表は
   $$E_{ab}=n_{ij}\sum_g w^{(0)}_g\sum_t w^{(s)}_t\,P_{ia}(\boldsymbol\theta_g,\theta_t)
   P_{jb}(\boldsymbol\theta_g,\theta_t)\qquad s(i)=s(j)=s,$$
   $$E_{ab}=n_{ij}\sum_g w^{(0)}_g\,\bar P_{ia}(\boldsymbol\theta_g)\,\bar P_{jb}(\boldsymbol\theta_g)
   \qquad s(i)\ne s(j)\ \text{または}\ s(i)=s(j)=\varnothing .$$
   どちらも、重み $w^{(0)}_g\prod_s w^{(s)}_{t_s}$ の完全な積グリッド上の和を並べ替えただけの
   厳密な式である（どちらの項目も負荷しない因子の和は 1）。したがって 1 次元グリッドが同じなら、
   積グリッドを作る mirt の `residuals(type = "LD")`（`thetaComb(theta, nfact)`）と一致する。
   計算量は 1 ペアあたり $O(n^{P+S}K_iK_j)$ ではなく $O(Q_p n K_iK_j)$。$X^2$、$G^2$、
   $df=(K_i-1)(K_j-1)$、$p$ 値、符号付き Cramér の $V$ は §16.11 と同じく $O_{ab}$ と $E_{ab}$
   から求める。
   **Q3。** $d_{ij}=x_{ij}-E(X_j\mid\hat{\boldsymbol\theta}_i)$ を、因子分解した事後分布
   （`_bifactor_eap`）による全 $D=P+S$ 因子のベクトル EAP で評価する。カーネルは縮約を外して
   （`CompiledProblem.reduction = None`。`_family_nodes` がスコアを返す）`(N, D)` のスコアで
   評価する。Q3\* は §16.11 と同じ。
   **実装。** `_local_dependence.ld_pairs` を分割した: `ld_pairs_from_tables(y,
   pair_probabilities)` はペアの $K_i\times K_j$ 同時確率を返す関数から統計量を計算し、
   `ld_pairs`（共通グリッド）と `reduced_pair_probabilities`（上の 2 式。ブロックグリッド上の
   トレース `(Q_p, n, K_j)`、`_bifactor_log_weights` の重み、各項目のブロックから）がその関数を
   与える。双因子・two-tier モデルは単一群のみ（項目 1–2）なので、群についての処理は既存のまま。
   **解釈。** 同じ特殊因子内のペアは、その因子を**超える**依存について検定される。`specific` に
   入れなかったテストレットは、その中の大きな正の $V$ として現れる（シミュレーションデータの単体テストで確認）。
   **検証（V38）。** 双因子 2PL（一般因子のみの項目と欠測を含む）、双因子 GRM、two-tier 2PL
   （`bfactor(model2 = )`）の mirt の推定値を、全パラメータ固定の `FitResult` に入れ、
   `residuals(type = "LD" / "LDG2" / "Q3")` と比較する（Q3 は `Theta = fscores(m, quadpts =
   <適合時の値>)`。指定しないと `fscores()` は独自のグリッドを使う）。単体テストでは因子分解した
   表を積グリッドの総当たり計算と比較する。**拡張しないもの。** `jsi()` はこれらのモデルで
   引き続き `SpecificationError` を出す。統計量は 1 項目 1 傾きについて定義されており
   （Edwards, Houts & Cai, 2018）、ここでは 1 項目が主因子と特殊因子の傾きを持ち、mirt の
   `type = "JSI"` も 1 因子を要求するため、定義も参照値もない。

4. **標本ウェイト（`fit(..., weights=...)`, V34）**: 受験者ごとのウェイト $w_i\ge0$ に対し、`fit()` は重み付き
   周辺対数尤度（擬似尤度。Skinner, 1989。mirt の `survey.weights`）
   $$\ell_w(\eta)=\sum_i w_i\log L_i(\eta)=\sum_p c_p\log L_p(\eta),\qquad c_p=\sum_{i:\,y_i=p}w_i$$
   （MAP では対数事前分布を加える）を最大化する。§4.2 の尤度で、パターンの度数 $c_p$ を、そのパターンを持つ行の
   ウェイトの和（群ごと）に置き換えたものなので、Core は変更しない。E-step、項目と潜在分布の M-step、SQUAREM、
   多群、FIPC、すべての潜在分布、2 つの情報行列は、もともと実数の `CompiledProblem.counts` で動く。変わるのは
   Compile 層だけ（`coerce_weights`、`np.bincount` で和をとる `compress_patterns(..., weights)`）。
   - **引数**: `weights` は `data` の行ごとに 1 つの数を持つ array-like（NumPy 配列、リスト、pandas / polars の
     Series）。列名は**受け付けない**。`data` はモデルの項目だけを持つ決まりで、`group=` も array-like のみなので、
     `weights=` も同じ規則にする。長さや形が違う、数値でない、NaN・無限大・負の値がある、合計が 0、宣言した群の
     ウェイトがすべて 0、の場合は `SpecificationError`。
   - **再スケールしない**: ウェイトは与えられたまま使う（mirt と同じ）。したがって `loglik`、`log_posterior`、AIC、
     情報行列はウェイトの尺度に従う。合計が行数になるウェイトなら、ウェイトなしの推定と比べやすい。事前分布が
     あるとき、事前分布の絶対的な重みは $w$ の尺度によらないので、ウェイトを定数倍すると MAP 推定値は変わる
     （MLE は変わらない）。
   - **ウェイト 0**: $w_i=0$ の行は検証の直後、初期値と推定可能性の検査（§14.5）の前に取り除く。したがって何にも
     影響しない（ウェイト 0 の行にしか現れないカテゴリは「観測されない」扱い）。
   - **整数ウェイト**は度数ウェイトである。行 $i$ を $w_i$ 回繰り返したデータへの推定と一致する（尤度・推定値・
     情報行列が同じ。単体テストで確認）。
   - **「N」の意味**: `n_obs` はウェイトが正の行の数。BIC は $\log N$ の代わりに $W=\sum_i w_i$ の $\log W$ を使う。
     mirt と同じで（mirt の $N$ はパターン度数の和）、BIC は mirt と一致し、整数ウェイトでは行の複製とも一致する。
     AIC は $N$ を使わない。$W$ は非公開の `FitResult._sum_weights`（ウェイトなしでは `None`）に保持し、
     `summary()` に表示する。**初期値**（§4.10）は、ウェイトが正の行からウェイトなしで計算する。初期値は EM の
     経路を変えるだけで収束先を変えず、ウェイトなしの積率で十分よい出発点になる。識別性の検査と Heywood・退化解の
     警告は $N$ を使わない。
   - **標準誤差**: `se="oakes"`（既定）と `se="hessian"` は $-\nabla^2\ell_w(\hat\eta)$ の逆行列を返す。度数
     ウェイトなら通常の情報行列である。標本ウェイトでは**正しい分散推定量ではない**。標本設計に整合する分散は
     サンドイッチ $H^{-1}\big(\sum_i w_i^2 s_i s_i^\top\big)H^{-1}$（Binder, 1983）で、ロバスト標準誤差として
     別に実装する。決定: 既定は `"oakes"` のままとし（値は `survey.weights` を付けた mirt の
     `SE.type = "Oakes"` と一致するので照合できる）、`weights` があり `se` が `None` でないとき `fit()` は
     `FlexpyirtWarning` を出す。
   - **推定の後**: スコアリング（`score()`）は推定値だけに依存し、変わらない。`link()` は推定値と `vcov` を使う
     （その SE は上の注意を引き継ぐ）。データを使う診断にはウェイト付きの検証済みの形がない（統計量と参照分布が
     ウェイトなしの度数を前提とする）ので、`item_fit()`、`m2()`、`local_dependence()`、`jsi()`、`dif()`、
     `model_parallel_analysis()`、`dimensionality(fit=...)` はウェイト付きの推定結果に対して
     `SpecificationError` を出す（`FitResult._require_unweighted`）。
   - **JSON**: `schema_version` 7 で、省略可能なトップレベルのキー `sum_weights` を加える（ウェイト付きの推定に
     だけ書く。正の有限な数）。version 6 以下のファイルにこのキーがあれば拒否し、version 6 以下のファイルは
     これまでどおり読める。ウェイトそのものは保存しない（データと同じ）。
   - **照合（V34）**: `validation/r/m7_weights_reference.R`（mirt 1.48、`survey.weights`、TOL $=10^{-10}$、
     Oakes）。合計が $N$ でない非整数ウェイトと 5 つのウェイト 0 を持つ 2PL、合計を $N$ に正規化したウェイトの
     Science GRM、アンカー・focal 群の潜在分布の推定・5% の欠測を持つ 2 群、潜在分布を推定する FIPC、双因子 2PL（`bfactor()`）。双因子モデルと two-tier モデル（§17.1、§17.2）に
     特別な扱いは要らない。次元縮約した E-step も `counts` で動く。two-tier は行の複製との一致だけで確かめる。推定値、
     対数尤度、AIC、BIC、Oakes の SE が共通の許容誤差内で一致する（VALIDATION.ja.md V34）。
5. **ロバスト標準誤差: クロス積とサンドイッチ（`fit(se="crossprod" | "sandwich")`, V35, #85）**:
   **定義。** 応答パターン $p$（度数 $c_p$）の周辺尤度を
   $L_p(\eta)=\sum_q w_{g(p)q}(\eta)\prod_j P_j(y_{pj}\mid\theta_q;\eta)$、そのスコアを
   $s_p=\nabla_\eta\log L_p(\hat\eta)$ として
   $$B=\sum_p c_p\,s_p s_p^\top,\qquad B_{\mathrm{post}}=B-\nabla^2_\eta\log\pi(\hat\eta),$$
   $$V_{\mathrm{crossprod}}=B_{\mathrm{post}}^{-1},\qquad
     V_{\mathrm{sandwich}}=H^{-1}B_{\mathrm{post}}H^{-1}$$
   とする。$H=-\nabla^2_\eta[\ell(\hat\eta)+\log\pi(\hat\eta)]$ は §4.8 の観測（事後）情報行列で、
   Oakes の恒等式（`information_oakes`、`information_hessian` と同一）で計算する。$B$ は Fisher 情報量の
   勾配外積推定（Berndt, Hall, Hall & Hausman, 1974）、サンドイッチはモデル誤特定のもとでの最尤推定量の
   共分散（Huber, 1967; White, 1982）。独立な単位は受検者で、$c_p$ 回観測されたパターンは同じ項を
   $c_p$ 個寄与するので、圧縮したデータと行を複製したデータで $B$ は一致する。スコアは中心化しない
   （最尤推定値では $\sum_p c_p s_p=0$。mirt も中心化しない）。
   **事前分布。** 事前分布の曲率は、$H$ に加えるのと同じように（§15.17）クロス積にも加える。これで
   `oakes`/`hessian` と `crossprod` は同じ事後情報量の推定になり、情報等式 $B=-\nabla^2\ell$ が成り立てば
   サンドイッチは既存の MAP の共分散 $H^{-1}$ に一致する。mirt と同じ規約（`SE.simple` の
   `info <- Igrad + Iprior` が `"crossprod"`、同じ行列が `"sandwich"` の中央）なので照合できる。尤度の
   スコアだけを使う $H^{-1}BH^{-1}$（罰則付き推定量の反復標本分散。対数凹の事前分布では正しいモデルでも $H^{-1}$ より小さい）は
   提供しない。固定パラメタは $\eta$ に含まれない。群や項目をまたいで共有されるパラメタ（同じ
   `free_id`）のスコアは 1 成分で、出現箇所の和になる。
   **多母集団と潜在分布のパラメタ。** $\eta$ は全群の項目・潜在分布パラメタを持つ。群 $g$ のパターンは
   他の群にしか現れないパラメタについてスコアが 0 なので、特別な処理なしで $B$ は各群のクロス積の和になる。
   **経験ヒストグラム。** `oakes`/`hessian` と同じく（§15.35）、ヒストグラムの重みは推定値に固定する。
   $s_p$ と $H$ は自由な項目パラメタだけの微分で、`vcov` の形は `n_free_items × n_free_items` のまま。
   したがってサンドイッチは、推定されたヒストグラムを与えたときの項目パラメタについてロバストであり、
   ヒストグラムの推定に対してロバストなわけではない。
   **対象外。** 潜在分布が `DavidianCurve`・`NormalMixture` のとき、どちらのオプションも
   `SpecificationError` を出す。mirt はこれらを計算しないので（"No standard error method currently
   supported for Davidian curves"、"Only Oakes and complete SE.types current supported for mixture
   models"）、照合する参照値がない（原則 1）。双因子・two-tier モデル（§17.1、§17.2）も
   `SpecificationError` を出す。これらの尤度は次元縮約（`CompiledProblem.reduction`）で評価するが、下の
   スコアのカーネルは全体のグリッド上の表を使うためである。mirt は `bfactor()` でどちらの種類も計算する
   ので、これは参照値がないのではなく実装が未対応なだけで、あとで照合できる。
   標本ウェイト（項目 4）つきの推定では、どちらの選択肢も `SpecificationError` を出す。ウェイトを足した
   パターン度数から作る $B$ は標本設計に基づく推定量（$\sum_i w_i^2 s_i s_i^\top$）ではなく、その版と
   クラスタロバスト版（スコアをクラスタ内で合計する）は未実装である（#85）。
   **計算**（`_robust_se.py`。`_core.py` を変更しないよう Core の別モジュールにした）。
   `_fold_pattern_scores` はノード上の表 $\log P_{gjqk}(\eta)$ と $\log w_{gq}(\eta)$ を 1 回だけ評価し、
   `jax.vjp` で線形化する。パターンは `lax.scan` でチャンクごとに処理する。チャンクの各パターンの対数尤度を
   `vmap(jax.grad)` で表について微分し、保存した pullback で各余接ベクトルを $\eta$ に戻して `(B, n_free)` の
   スコアを得て、$\sum c_p s_p s_p^\top$ に足し込む。`(P, n_free)` も `(P, G, J, Q, K)` も作らない。
   チャンクサイズは `chunk × G × Q × (1 + Σ_f J_f K_f (P_f + 1))` が $8\times10^6$ 要素（約 64 MB）以下に
   なる最大値（上限 1024）で、度数 0 のパディング行だけのチャンクは `lax.cond` で飛ばす。微分はすべて
   自動微分。単体テストで `score_sum`（$\sum_p c_p s_p$）を `jax.grad(marginal_loglik)` と、クロス積を
   パターン対数尤度の密な `jax.jacfwd` と照合する。`fit()` はサンドイッチを $H$ の逆行列から NumPy で
   組み立てる（既存の正定値チェックを使う。$H$ が正定値でなければ `FlexpyirtWarning` を出して標準誤差は
   NaN。`crossprod` では同じチェックを $B_{\mathrm{post}}$ に行う。異なるパターン数が自由パラメタ数より
   少ないと特異になる）。
   **`vcov` を使う箇所。** `FitResult.vcov`、`se` 列、デルタ法の表示用 SE（`b_se`・`bk_se`、潜在分布の
   `sd`/`cor` の SE。§15.18）、`link()`（§16.5）は `vcov` を読むので、変更なしで選んだ共分散を使う。
   `FitResult.jsi(method="one_step")` は `vcov` を Newton ステップの観測情報行列の逆行列として使うが、
   $V_{\mathrm{sandwich}}$・$V_{\mathrm{crossprod}}$ はそれに当たらないので、`oakes`・`hessian` 以外の当てはめでは
   `SpecificationError` を出す。
   **オプション名と JSON。** `se` に `"sandwich"`・`"crossprod"` を追加する（mirt の `SE.type` の綴り。
   §15.46 の命名規則に従えば `"cross_product"` で、どちらにするかはオーナーが決める）。JSON の
   `se_method` には同じ文字列を保存する。値の追加だけで保存名は変わらないので `schema_version` は
   上げない（§16.4・§16.7 で潜在分布の種類を追加したときと同じ）。mirt の `"sandwich.Louis"` は $H$ を
   Oakes ではなく Louis (1982) の式で計算するだけの同じ推定量で、どちらの式も観測情報行列を与えるので、
   別のオプションにはしない。

## 18. 合計得点分布のカーネル、情報関数と信頼性（M7）

1. **Poisson 二項分布のカーネルとしての Lord–Wingersky 再帰（`summed_score_distribution`、V36）**:
   **計算するもの。** $\theta$ を与えると、項目得点 $X_j\in\{0,\dots,K_j-1\}$ はカテゴリ確率
   $P_{jk}(\theta)$ をもつ独立な変数になる。カーネルは合計得点 $S=\sum_jX_j$ の分布
   （$s=0,\dots,S_{\max}=\sum_j(K_j-1)$）を返す。2 値項目なら $S$ は独立で同一でない Bernoulli 変数の
   和、つまり成功確率 $p_j(\theta)$ の **Poisson 二項分布**である（González, Wiberg & von Davier,
   2016）。多値項目なら、独立なカテゴリ得点の和の分布になる。Lord–Wingersky (1984) の再帰（多値項目は
   Thissen, Pommerich, Billeaud & Williams, 1995）は、この分布を直接たたみ込みで求める算法で、
   Poisson 二項分布の文献でいう再帰式「RF」（Hong, 2013）と同じものである。
   **API（決定、オーナー 2026-11-25）。** `summed_score_distribution(probs, *, method="recursion",
   log=False)`。`probs` の形は `(..., J, K)`: $J$ 項目のカテゴリ確率で、各行の和は 1、カテゴリ数が
   $K$ より少ない項目は 0 で埋める。先頭の軸（求積節点、受検者など）はバッチ軸。結果の形は
   `(..., J(K-1)+1)` で、その項目集合の最大得点を超える得点の確率は 0 になる。関数は純粋な JAX の
   カーネルで、どの方法でも `jit`・`vmap`・自動微分が通る。値の検査はしない（`jit` の中から呼ばれると
   検査できないため）。検査するのは形と `method` だけ（`SpecificationError`）。
   **方法。**
   - `"recursion"`（既定、厳密）: $L_j(s)=\sum_kL_{j-1}(s-k)P_{jk}$、$L_0(0)=1$ を、固定長ベクトル上の
     項目についての `lax.scan` で計算する。バッチ要素あたり $O(JKS_{\max})$。すべての項が非負なので
     桁落ちがなく、各確率の相対誤差は裾も含めて数 ulp の $J$ 倍程度。`log=True` では対数空間で再帰する:
     $\log L_j(s)=\operatorname{logsumexp}_k[\log L_{j-1}(s-k)+\log P_{jk}]$。通常の再帰では 0 に
     アンダーフローする、倍精度の最小値（$\approx10^{-308}$）より小さい裾の確率も保てる（長いテスト、
     極端な $\theta$）。
   - `"fft"`（丸め誤差の範囲で厳密）: 特性関数の離散フーリエ変換（DFT-CF; Hong, 2013）。
     $\omega_l=2\pi l/(S_{\max}+1)$ で $\varphi(\omega_l)=\prod_j\sum_kP_{jk}e^{i\omega_lk}$ を求め、
     $P(S=s)=\frac1{S_{\max}+1}\sum_l\varphi(\omega_l)e^{-i\omega_ls}$ をバッチ化した FFT 1 回で計算する。
     誤差は確率の大きさによらず**絶対誤差**で約 $10^{-16}$。$\sim10^{-15}$ より小さい確率には正しい桁が
     ない（丸めで負になった値は 0 に切り上げる）。裾が効く用途（S-$X^2$、極端な得点の対数尤度）には
     使ってはならない。`log=True` は結果の対数をとるだけ。
   - `"normal"`・`"refined_normal"`・`"poisson"`（近似。2 値項目 $K=2$ のみ。それ以外の $K$ は
     `SpecificationError`。R の参照があるのが 2 値の場合だけのため）: $\mu=\sum_jp_j$、
     $\sigma^2=\sum_jp_j(1-p_j)$、$\gamma=\sigma^{-3}\sum_jp_j(1-p_j)(1-2p_j)$ として、連続修正つきの
     正規近似は $P(S\le s)\approx\Phi(x)$、$x=(s+0.5-\mu)/\sigma$。精密化正規近似は
     $P(S\le s)\approx G(x)$、$G(x)=\Phi(x)+\gamma(1-x^2)\phi(x)/6$ を $[0,1]$ に切り詰めたもの
     （Volkova, 1996; Hong, 2013）。$s$ の確率は隣り合う値の差なので、和がちょうど 1 になるとは
     限らない。Poisson 近似は $P(S=s)\approx e^{-\mu}\mu^s/s!$（Le Cam, 1960）で、全変動距離の誤差は
     高々 $\sum_jp_j^2$。すべての $p_j$ が小さいときにだけ役に立つ。近似の計算量は $O(J+S_{\max})$ で、
     任意で使える高速な経路という位置づけ。flexpyirt の内部では使わない。
   **再利用。** `_itemfit.lord_wingersky`（S-$X^2$）は同じカーネルを呼ぶようにする。V15 の参照値は
   変わらない。合計得点 EAP と IRT 観測得点等化もこれを使う（#82）。
   **精度と速度の実測**は VALIDATION.md（V36）に記録する。

2. **情報関数・期待得点・信頼性・合計得点 EAP（#82、V37）**:
   **範囲。** 1 次元モデル（単一群・多母集団）。多次元モデルではどの関数も `SpecificationError`
   （情報行列と方向別の情報量は別の変更で扱う）。
   **API（決定、オーナー 2026-11-25）。** `link()` や `dimensionality()` と同じく、推定結果を第 1 引数に
   とるモジュール関数にする: `item_information(res, theta, *, group=None)`、
   `test_information(res, theta, *, group=None)`、`expected_score(res, theta, *, group=None)`、
   `reliability(res, data=None, *, group=None, item_names=None)`、`summed_score_table(res)`、および
   `FitResult.score(data, method="EAPsum")`。$\theta$ の関数 3 つは `theta` をインデックスにした
   DataFrame を返す（`.plot()` でそのまま曲線になる）。多母集団モデルでは `group=`（群名）が必須
   （項目が群で異なりうるため）。`reliability()` と `summed_score_table()` は、多母集団モデルでは
   `item_fit()` と同じく先頭に `group` 列を付けて群ごとの行（ブロック）を返す。`reliability()` の
   `group=` は `score()` と同じく `data` の各行の群ラベル。
   **項目情報量・テスト情報量。** すべての項目族について、項目カーネルのカテゴリ対数確率を JAX で
   自動微分して求める（導関数は手書きしない。§15.24 の WLE と同じ式）:
   $$I_j(\theta)=\sum_{k=0}^{K_j-1}P_{jk}(\theta)\Big(\frac{\partial\log P_{jk}(\theta)}{\partial\theta}\Big)^2,\qquad
   I(\theta)=\sum_jI_j(\theta),\qquad \mathrm{SE}(\theta)=I(\theta)^{-1/2}$$
   （Birnbaum, 1968; Samejima, 1969）。mirt の `iteminfo()` は、代数的に等しい
   $\sum_k[(P'_{jk})^2/P_{jk}-P''_{jk}]$（$\sum_kP''_{jk}=0$）を手書きの導関数で計算する。
   `item_information` は項目ごとに 1 列、`test_information` は `information` と `se`（$\theta$ の
   有効推定量の条件付き測定標準誤差）の 2 列。
   **期待得点。** 項目ごとの $E_j(\theta)=\sum_kk\,P_{jk}(\theta)$ と、テスト特性曲線
   $T(\theta)=\sum_jE_j(\theta)$（列 `total`。`total` という名前の項目があれば
   `SpecificationError`）。カテゴリの得点はデータのコードと同じ $0,\dots,K_j-1$ で、観測された最小
   カテゴリが 0 のときの mirt の `expected.item()` / `expected.test()` に一致する。
   **周辺信頼性。** 潜在分布が $N(\mu_g,\sigma_g^2)$、テスト情報量が $I_g$ の群 $g$ について
   $$\bar\rho_g=\int\frac{\sigma_g^2I_g(\theta)}{\sigma_g^2I_g(\theta)+1}\,\phi(\theta;\mu_g,\sigma_g^2)\,d\theta .$$
   誤差分散 $1/I(\theta)$ の得点の条件付き信頼性 $\sigma^2/(\sigma^2+I(\theta)^{-1})$ を潜在分布で
   平均したもの（Green, Bock, Humphreys, Linn & Reckase, 1984; Thissen & Wainer, 2001）。mirt の
   `marginal_rxx()` が計算するのは $\int\frac{I(\theta)}{I(\theta)+\sigma^2}\phi(\theta;0,1)\,d\theta$
   で、$\mu=0$、$\sigma^2=1$（既定の識別をした単一群モデルはすべて）なら同じ値になるが、それ以外の
   $(\mu,\sigma^2)$ では推定された密度で積分しておらず、潜在尺度の線形変換に対して不変でもない。
   flexpyirt の定義は不変である: $\theta^*=(\theta-\mu)/\sigma$ とすると情報量は
   $I^*(\theta^*)=\sigma^2I(\theta)$ で、$\bar\rho$ は $\int I^*/(I^*+1)\,\phi(\theta^*)\,d\theta^*$ になる。
   そこで $\sigma^2\ne1$ や $\mu\ne0$（標準でない固定の `Normal`、多母集団モデルの焦点群）の参照値は、
   **標準化した等価モデル**（$a^*=a\sigma$、$d^*=d+a\mu$、潜在分布 $N(0,1)$）に対する mirt の
   `marginal_rxx()` と、mirt の `testinfo()` を使って上の式を直接 `integrate()` した値にする。
   積分は推定に使った求積ではなく専用の格子で評価する: $\mu_g\pm8\sigma_g$ 上の等間隔 801 点、重みは
   $\propto\phi$（矩形則）。被積分関数は解析的で、正規の重みは両端で消えるので、この則は幾何級数的に
   収束する（打ち切り $2\Phi(-8)\approx10^{-15}$）。結果はモデルの推定に使った求積に依存しない。mirt は
   R の既定の許容誤差（`rel.tol` $\approx1.2\times10^{-4}$）で `integrate()` を呼ぶので、参照値には
   `rel.tol = 1e-12` での再計算も保存する。
   **経験的信頼性。** `data` の受検者の EAP 得点 $\hat\theta_i$ と事後 SD $\mathrm{SE}_i$
   （`score(method="EAP")`、推定に使った格子上）から
   $$\hat\rho=\frac{s^2_{\hat\theta}}{s^2_{\hat\theta}+\overline{\mathrm{SE}^2}},\qquad
   s^2_{\hat\theta}=\frac1{N-1}\sum_i(\hat\theta_i-\bar{\hat\theta})^2,\quad
   \overline{\mathrm{SE}^2}=\frac1N\sum_i\mathrm{SE}_i^2 .$$
   mirt の `empirical_rxx()`（既定の `T_as_X = FALSE`）と同じ。EAP 得点の分散が真の得点の分散を、
   事後分散の平均が誤差分散を推定し、その和が $\sigma^2$ を推定する（Thissen & Wainer, 2001）。
   多母集団モデルでは群ごとに、その群の受検者から計算する（mirt は `multipleGroup` の全受検者を
   まとめて計算する。群ごとの値は、その群の行に対する `empirical_rxx()` に等しい）。欠測のある行も
   通常どおり採点して含める（mirt と同じ）。受検者が 2 人未満の群は NaN。`reliability(res)` は列
   `marginal` を返し、`data` を渡すと `empirical` が加わる。
   **合計得点 EAP**（Thissen, Pommerich, Billeaud & Williams, 1995）。推定に使った求積節点 $\theta_q$ と
   潜在重み $w_{gq}$（EM が積分に使う重み）、および `summed_score_distribution`（項目 1、厳密な再帰）で
   求めた $L_g(s\mid\theta_q)$ から
   $$P_g(s)=\sum_qL_g(s\mid\theta_q)w_{gq},\qquad
   \hat\theta_g(s)=\frac{\sum_q\theta_qL_g(s\mid\theta_q)w_{gq}}{P_g(s)},\qquad
   \mathrm{SD}_g(s)=\Big(\frac{\sum_q(\theta_q-\hat\theta_g(s))^2L_g(s\mid\theta_q)w_{gq}}{P_g(s)}\Big)^{1/2}.$$
   `summed_score_table(res)` は $s=0,\dots,\sum_j(K_j-1)$ について `score`・`theta`・`se`・
   `probability`（$P_g(s)$、モデルが含意する合計得点分布）を返す。mirt の
   `fscores(method = "EAPsum", full.scores = FALSE)` は同じ表で、`expected` $=N\,P(s)$。観測度数と
   $X^2$ はデータが必要なので再現しない。$P_g(s)$ が 0 にアンダーフローした得点の `theta` と `se` は
   NaN。`score(data, method="EAPsum")` は、各受検者の合計得点をその受検者の群の表で引く。
   **欠測のあるパターン**: 1 つでも欠測のある受検者の `theta` と `se` は NaN にする。観測された項目だけの
   合計得点は表の尺度に乗らないため。mirt の `fscores(method = "EAPsum", full.scores = TRUE)` も
   同じ（`NA`）。観測された項目だけの部分テストで採点する方式（欠測パターンごとに表を作る）は
   入れていない。
   **潜在分布。** 情報量と期待得点は潜在分布を使わないので、1 次元モデルならどれでも使える。信頼性・
   合計得点の表・EAPsum 採点は、`Normal` の潜在分布（平均・分散は任意、単一群と多母集団）でのみ照合
   している。`EmpiricalHistogram`・`DavidianCurve`・`NormalMixture` は `SpecificationError`
   （推定された非正規の密度に対応するものが mirt の `marginal_rxx()` になく、mirt の EAPsum は既定で
   正規の事前分布を使うため）。
   **`FitResult` に新しく保存するものはない。** `schema_version` は変えない。
   **一致の実測値**は VALIDATION.md（V37）に記録する。

## 19. 多次元尺度リンキング（M6）

1. **確認的多次元モデルのリンキング（`link(..., transformation=)`、V40、#78）**:
   `link()`（§16.5）は、`Model.with_factors` で宣言した同じ因子数 $D \in \{2, 3\}$ の 2 つの推定結果も受け付ける
   （直積格子。双因子・two-tier は拒否する）。因子は位置で対応させ、`source` の第 $m$ 因子を `target` の第 $m$ 因子に写す。
   **変換。** `source` の尺度を $\boldsymbol\theta^* = \mathbf A\boldsymbol\theta + \boldsymbol\beta$
   （$\mathbf A$ は正則な $D \times D$ 行列）で `target` の尺度へ写す（Oshima, Davey & Lee, 2000; Li & Lissitz, 2000）。
   $\boldsymbol\theta = \mathbf A^{-1}(\boldsymbol\theta^* - \boldsymbol\beta)$ を線形予測子
   $\mathbf a^\top\boldsymbol\theta + d$ に代入すると、どのカーネルでも
   $$\mathbf a^* = \mathbf A^{-\top}\mathbf a, \qquad d_k^* = d_k - m_k\,\mathbf a^{*\top}\boldsymbol\beta$$
   となる。$m_k$ はカテゴリ $k$ における線形予測子の係数で、1（2PL/Rasch、3PL/4PL、GRM）、$k$（GPCM/PCM）、
   $\mathrm{ak}_k$（名義、$\mathrm{ak}_{K-1} = K-1$）。漸近線と名義モデルのスコアリング係数は不変で、
   $P_{jk}(\boldsymbol\theta^*; \xi^*) = P_{jk}(\mathbf A^{-1}(\boldsymbol\theta^* - \boldsymbol\beta); \xi)$
   が成り立つ。$D = 1$ では §16.5 に一致する。潜在分布 $N(\boldsymbol\mu, \boldsymbol\Sigma)$ は
   $N(\mathbf A\boldsymbol\mu + \boldsymbol\beta, \mathbf A\boldsymbol\Sigma\mathbf A^\top)$ になる。
   **$\mathbf A$ の 3 つの族**（`transformation`、静的引数。自由パラメタ $c$ は $p$ 個）:
   - `"diagonal"`（既定。$p = 2D$）: $\mathbf A = \operatorname{diag}(A_1, \dots, A_D)$、$A_m > 0$、
     $c = (A_1..A_D, \beta_1..\beta_D)$。因子ごとに尺度と原点を変える。*同じ*確認的構造（各推定で平均 0・分散 1）で
     推定した 2 つの結果のあいだの変換はこれであり、0 に固定した傾きは 0 のまま、因子間相関も変わらない。
     変換後の項目が `source` のモデル構造を保つのはこの族だけである。
   - `"orthogonal"`（$p = 1 + D(D-1)/2 + D$; Li & Lissitz, 2000）: $\mathbf A = k\,\mathbf R$。$k > 0$ は中心拡大、
     $\mathbf R = \exp(\mathbf S(\boldsymbol\phi))$ は回転で、$\mathbf S$ は $S_{ij} = -\phi_{ij}$、
     $S_{ji} = \phi_{ij}$（$i < j$、辞書式順）の歪対称行列、$c = (k, \boldsymbol\phi, \boldsymbol\beta)$。$D = 2$ では
     $\mathbf R = \begin{pmatrix}\cos\phi & -\sin\phi\\ \sin\phi & \cos\phi\end{pmatrix}$。到達するのは
     $\det\mathbf R = +1$ の回転だけで、鏡映は含まない。
   - `"oblique"`（$p = D^2 + D$; Oshima, Davey & Lee, 2000）: $\det\mathbf A > 0$ の任意の $\mathbf A$、
     $c = (\operatorname{vec}(\mathbf A)$ 行優先$, \boldsymbol\beta)$。
   回転と斜交変換は `source` 項目の構造的な 0 の傾きを 0 でない値にする。因子の識別のしかたが異なるフォームのための変換である。
   **目的関数**は `target` の直積格子 $(\boldsymbol\theta_q^{(t)}, w_q^{(t)})$（§15.30 の推定済み多変量正規重み、または
   形状 $(Q, D)$ の `nodes` と `weights`）の上で
   $$H_{\mathrm{fwd}}(c) = \sum_q w_q^{(t)} \sum_{j\in\mathcal C}\sum_{k=0}^{K_j-1}\big[P_{jk}^{\mathrm{tgt}}(\boldsymbol\theta_q^{(t)}) - P_{jk}^{\mathrm{src}}(\mathbf A^{-1}(\boldsymbol\theta_q^{(t)} - \boldsymbol\beta))\big]^2,$$
   $$\mathrm{SL}_{\mathrm{fwd}}(c) = \sum_q w_q^{(t)} \big[\mathrm{TCC}^{\mathrm{tgt}}(\boldsymbol\theta_q^{(t)}) - \mathrm{TCC}^{\mathrm{src}}(\mathbf A^{-1}(\boldsymbol\theta_q^{(t)} - \boldsymbol\beta))\big]^2$$
   とする。Oshima, Davey & Lee (2000) の項目特性関数法・テスト特性関数法を Kim & Lee (2006) のカテゴリ形式で書いたもので、
   $\mathrm{TCC}(\boldsymbol\theta) = \sum_j\sum_k k\,P_{jk}(\boldsymbol\theta)$。`symmetric=True` は `source` の格子上の
   逆方向の項（$\boldsymbol\theta_q^{(s)}$ での `source` 曲線と $\mathbf A\boldsymbol\theta_q^{(s)} + \boldsymbol\beta$ での
   `target` 曲線）を加える。Stocking–Lord 法は 1 つのスカラー曲面を合わせるので、$\mathbf A$ は共通項目の傾きの方向の
   多様さを通じてしか識別されない。Haebara 法はすべての項目曲面を使う。目的関数の Hessian が正定値でなければ標準誤差は出さない。
   **最適化。** `_linking.solve_multidim_linking` は §16.5 の Newton 反復を $p$ 次元にしたもの（Hessian が正定値なら
   厳密なステップ、そうでなければ Levenberg–Marquardt、ステップのノルムは 1 で打ち切り、40 段階の並列ステップ幅、勾配仕上げ）。
   有効領域の外（$A_m \le 0$、$k \le 0$、$\det\mathbf A \le 0$）では目的関数が NaN になりステップ幅探索で棄却されるので、
   反復は初期値と同じ連結成分にとどまる。**初期値**: 共通項目の傾き行列（$J_c \times D$）を $X_s, X_t$ とすると
   $\mathbf a_s = \mathbf A^\top\mathbf a_t$ から最小二乗解 $\mathbf A_0 = X_t^{+} X_s$ を得る（斜交。Oshima, Davey & Lee, 2000
   の「直接法」）。直交の初期値はその極分解（$\mathbf A_0 = U\Sigma V^\top \Rightarrow \mathbf R_0 = UV^\top$、
   $k_0 = \operatorname{mean}\Sigma$、$\boldsymbol\phi_0$ は行列対数から。$\det UV^\top < 0$ なら $\mathbf R_0 = I$）、
   対角の初期値は $A_{m,0} = \sum_j |a^{\mathrm{src}}_{jm}| / \sum_j |a^{\mathrm{tgt}}_{jm}|$。$\mathbf A_0$ が無効なら対角の
   初期値に戻す。$\boldsymbol\beta_0$ は共通項目の全切片についての
   $d^{\mathrm{tgt}}_{jk} - d^{\mathrm{src}}_{jk} = -m_k\,\mathbf a_j^{\mathrm{tgt}\top}\boldsymbol\beta$ の最小二乗解。
   **標準誤差。** §16.5 と同じく $J_s = -H_{cc}^{-1}H_{c,\eta_s}$、$J_t = -H_{cc}^{-1}H_{c,\eta_t}$、
   $\operatorname{Cov}(\hat c) = J_s\Sigma_sJ_s^\top + J_t\Sigma_tJ_t^\top$（同じ推定の 2 群なら交差項を加える）。報告する定数
   $\kappa(c) = (\operatorname{vec}(\mathbf A)$ 行優先$, \boldsymbol\beta)$ は $c$ の滑らかな関数なので、
   $G = \partial\kappa/\partial c^\top$（`jax.jacfwd`）を使って
   $\operatorname{Cov}(\hat\kappa) = G\operatorname{Cov}(\hat c)G^\top$ とする。この $(D^2 + D)$ 次の行列の階数は高々 $p$ で、
   対角の $\mathbf A$ の構造的な 0 は分散 0、直交の $\mathbf A$ の要素は 1 次の範囲で線形従属である。
   **結果。** $D > 1$ では `LinkResult.A` は $D \times D$ 配列 $\mathbf A$、`B` は長さ $D$ の配列 $\boldsymbol\beta$、
   `A_se` / `B_se` は同じ形の配列、`vcov` は $(\operatorname{vec}(\mathbf A), \boldsymbol\beta)$ の $(D^2 + D)$ 次の共分散行列。
   $D = 1$ では従来どおり float と $2 \times 2$ 行列。`LinkResult.transformation` に族を記録する（$D = 1$ では 3 つが一致するので
   常に `"diagonal"`）。`items` は変換後の傾き `a1..aD` と切片を持つ。`transform_theta` は `theta_<因子>` 列、
   `pv<m>_<因子>` 列、形状 $(\dots, D)$ の配列を $\mathbf A\boldsymbol\theta + \boldsymbol\beta$ で写す。`se_<因子>` 列は
   対角の $\mathbf A$ では $A_m\,\mathrm{se}_m$、それ以外では NaN にする（回転後の標準誤差には事後共分散が必要だが、
   得点表はそれを持たない）。

## 20. IRT 真値等化・観測得点等化と素点-尺度得点変換表（#106）

1. **範囲と API（決定、オーナー 2026-11-26: #106 で提案した API と範囲）**:
   1 次元モデル、全項目モデル、単一群・多母集団。
   - `irt_equate(new, old, *, method="true_score", new_items=None, old_items=None, link=None,
     new_group=None, old_group=None, population="new", synthetic_weight=None)` →
     `IRTEquatingResult` は新フォーム $X$ のすべての合計得点 $x=0,\dots,X_{\max}$ を旧フォーム $Y$ の
     合計得点の尺度に変換する。`new` と `old` は $\theta$ 尺度が一致している `FitResult`（FIPC、同時推定、
     または同じ推定結果を `new_items` / `old_items` で 2 つのフォームに分けたもの）か、`link=` に
     `new` の尺度を `old` の尺度へ写す `LinkResult`（$\theta_Y = A\theta_X + B$、§16.5）を渡す。
     共通尺度は常に旧フォームの尺度。`link` は 1 次元で、その source / target の群が
     `new_group` / `old_group` と一致しなければならない。
   - `score_distribution(res, *, items=None, group=None)` → DataFrame `score`, `probability`:
     ある群の推定母集団におけるフォームのモデル上の合計得点分布。
   - `scale_table(res, *, slope, intercept, method="true_score", items=None, group=None,
     lowest=None, highest=None, n_digits=0)` → DataFrame `score`, `theta`, `theta_se`,
     `scale_score`, `scale_score_se`, `scale_score_rounded`。
   **#106 の案からの変更。** `synthetic_weight` の既定値は `None` で、`population="synthetic"` の
   ときだけ指定できる（必須）。`population="new"` / `"old"` は重み 1 / 0。一方の引数が他方を黙って
   上書きすることがない。
   **範囲外**（後続 issue）: 等化得点の標準誤差（`vcov` を通したデルタ法）、モデル上の分布のカーネル平滑化、
   非線形の尺度変換、多次元モデル。

2. **共通尺度上のフォームの曲線**: フォームは 1 つの推定結果の 1 つの群の、カーネルパラメタ $\xi_j$ を
   持つ項目の集まり。リンクがあるとき、新フォームの項目は自分の尺度の $\theta_X=(\theta-B)/A$ で評価する。
   つまり $P^{X}_{jk}(\theta)=P_{jk}((\theta-B)/A;\xi_j)$ で、パラメタを変換しなくても全カーネルで
   §16.5 の変換後の項目そのものになる。テスト特性曲線（TCC）は
   $$\tau(\theta)=\sum_j\sum_{k=0}^{K_j-1}k\,P_{jk}(\theta)$$
   （カテゴリは `expected_score()` と同じく $0,\dots,K_j-1$ で得点化）。漸近値
   $\tau_{\mathrm{lo}}=\lim_{\theta\to-\infty}\tau$、$\tau_{\mathrm{hi}}=\lim_{\theta\to\infty}\tau$
   （3PL/4PL 項目の下側・上側漸近線 $g_j$, $u_j$ の和、他の項目は 0 と $K_j-1$）は $\theta_X=\mp10^{8}$ で
   評価する。傾きが約 $10^{-6}$ 未満でない限り、倍精度ではどのカーネルもそこで極限に達している。
   真値の方法は TCC が単調非減少であることを要する。$[-20,20]$（自尺度）の 801 点で確かめ、
   $10^{-10}(1+X_{\max})$ を超えて減少すれば `SpecificationError`（負の傾き、期待得点が単調でない名義項目）。

3. **真値等化（Lord, 1980; Kolen & Brennan, 2014, §6.5）**:
   $\tau^X_{\mathrm{lo}}<x<\tau^X_{\mathrm{hi}}$ について $\tau^X(\theta_x)=x$ を解き、$e_Y(x)=\tau^Y(\theta_x)$。
   **解法**: 全得点を 1 つの `jit` した `lax.while_loop` でまとめて解く安全化 Newton 法（Newton–二分法）。
   区間は共通尺度で $[-10,10]$ から始め、すべての根を含むまで（Python 側で $|\theta|=10^{6}$ まで）倍にする。
   $\tau'$ は `jax.jvp`。区間を出る Newton ステップや非正の導関数は二分に置き換える。
   すべての $|\tau^X(\theta)-x|\le10^{-13}(1+X_{\max})$ か区間幅が $10^{-14}(1+|\theta|)$ 未満で停止
   （最大 300 反復。収束しなければ `ConvergenceWarning`）。
   **TCC の値域外の得点**（Kolen & Brennan, 2014, §6.5, 式 6.27 と最高点の扱い）: 方程式を満たす
   $\theta$ がないので、点 $(0,0)$ と $(\tau^X_{\mathrm{lo}},\tau^Y_{\mathrm{lo}})$、上側では
   $(\tau^X_{\mathrm{hi}},\tau^Y_{\mathrm{hi}})$ と $(X_{\max},Y_{\max})$ の間の線形補間で等化する:
   $$e_Y(x)=x\,\frac{\tau^Y_{\mathrm{lo}}}{\tau^X_{\mathrm{lo}}}\ (0\le x\le\tau^X_{\mathrm{lo}}),\qquad
   e_Y(x)=\tau^Y_{\mathrm{hi}}+(x-\tau^X_{\mathrm{hi}})\frac{Y_{\max}-\tau^Y_{\mathrm{hi}}}{X_{\max}-\tau^X_{\mathrm{hi}}}\ (\tau^X_{\mathrm{hi}}\le x\le X_{\max}).$$
   $\tau^X_{\mathrm{lo}}=0$（下側漸近線なし）のときは $x=0$ だけが前者に入り 0 に写る。
   $\tau^X_{\mathrm{hi}}=X_{\max}$ のときは $x=X_{\max}$ だけが $Y_{\max}$ に写る。`theta` 列は
   $x\le\tau^X_{\mathrm{lo}}$ で $-\infty$、$x\ge\tau^X_{\mathrm{hi}}$ で $+\infty$（$\theta_x$ の極限。漸近値を
   厳密に越える $x$ には解がない）。

4. **観測得点等化（Lord & Wingersky, 1984; Kolen & Brennan, 2014, §6.6）**:
   **母集団。** 新母集団は `new` の `new_group` の推定潜在分布（`link()` と同じ節点と重み:
   Normal、EH、Davidian、混合。`_fitted_grid`）で、リンクがあれば節点を $A\theta+B$ で共通尺度へ写す。
   旧母集団は `old` の `old_group` のもの。合成母集団は $w_1\cdot\text{new}+(1-w_1)\cdot\text{old}$、
   $w_1$ は 1、0 または `synthetic_weight`（Braun & Holland, 1982）。
   **分布。** 各フォーム・各母集団で $f(x)=\sum_qL(x\mid\theta_q)w_q$（§18.1 の Lord–Wingersky 尤度。
   既存カーネル `summed_score_distribution`、`method="recursion"`）。合成母集団では
   $f_s(x)=w_1f_1(x)+(1-w_1)f_2(x)$、$g_s(y)$ も同様。
   **等パーセンタイルの計算（#106 のオーナー判断 2 への決定: 案 (b)）**: flexpyirt は整数得点用の
   パーセンタイル順位関数を自前で持ち、flexpyequate（開発依存のみ）の
   `flexpyequate.equate(method="equipercentile")` と照合して検査する。累積分布を $F$, $G$ として
   $P(x)=F(x-1)+f(x)/2$、
   $$e_Y(x)=Q^{-1}(P(x)),\quad Q^{-1}_U(p)=\frac{p-G(y_U-1)}{g(y_U)}+y_U-\tfrac12,\ y_U=\min\{y:G(y)>p\},$$
   $Q^{-1}_L(p)=\frac{p-G(y_L)}{g(y_L+1)}+y_L+\tfrac12$、$y_L=\max\{y:G(y)<p\}$、
   $Q^{-1}=(Q^{-1}_U+Q^{-1}_L)/2$（Kolen & Brennan, 2014, 2 章。ある $g(y)=0$ でない限り両者は一致）。
   flexpyequate、`equate::equate(type = "equipercentile")`、equateIRT の OSE と同じ計算。
   flexpyequate への依存（案 (a)）は採らなかった。NumPy 約 30 行のために 2 つ目の配布物を引き込むことに
   なり、逆向きの依存は flexpyequate の設計がもともと禁じている。

5. **素点-尺度得点変換表**: `method="true_score"`: $\theta(x)$ は 3 と同じく推定結果自身の尺度で求める
   （TCC の値域外は $\pm\infty$）。`theta_se` $=I(\theta)^{-1/2}$（フォームの項目のテスト情報量、§18.2。
   $\theta$ が無限大のところは NaN）。
   `method="EAPsum"`: §18.2 の合計得点 EAP と事後 SD を、フォームの項目について群の推定潜在分布の下で
   計算する（潜在分布の種類は問わない。Normal で全項目なら `summed_score_table()` と一致）。そのうえで
   `scale_score` $=\mathrm{clip}(\text{slope}\cdot\theta+\text{intercept},\text{lowest},\text{highest})$
   （slope $>0$。指定のない側は切らないので、無限大の $\theta$ は $\pm\infty$ になる）、
   `scale_score_se` $=\text{slope}\cdot$`theta_se`（切らない）、`scale_score_rounded` は NumPy の
   `round(scale_score, n_digits)`（偶数丸め。実際上ちょうど .5 にはならない）。

6. **層**: `_equating.py`（Core、純粋 JAX: フォームのカテゴリ確率、TCC とその導関数、情報量、
   Newton–二分法、格子上の合計得点分布と EAP。NumPy: 等パーセンタイル関数）と `equating.py`（シェル:
   検証、推定結果からのフォーム構築、Kolen–Brennan の境界の規則、DataFrame、`IRTEquatingResult`）。
   V51 で照合。

## 参考文献
- Auerswald, M., & Moshagen, M. (2019). How to determine the number of factors to retain in exploratory factor analysis: A comparison of extraction methods under realistic conditions. *Psychological Methods*, 24(4), 468–491.
- Barton, M. A., & Lord, F. M. (1981). An upper asymptote for the three-parameter logistic item-response model. *ETS Research Report Series*, RR-81-20.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *JRSS B*, 57(1), 289–300.
- Berndt, E. R., Hall, B. H., Hall, R. E., & Hausman, J. A. (1974). Estimation and inference in nonlinear structural models. *Annals of Economic and Social Measurement*, 3(4), 653–665.
- Bertsekas, D. P. (1982). Projected Newton methods for optimization problems with simple constraints. *SIAM Journal on Control and Optimization*, 20(2), 221–246.
- Binder, D. A. (1983). On the variances of asymptotically normal estimators from complex surveys. *International Statistical Review*, 51(3), 279–292.
- Birnbaum, A. (1968). Some latent trait models and their use in inferring an examinee's ability. In F. M. Lord & M. R. Novick, *Statistical Theories of Mental Test Scores* (pp. 397–479). Addison-Wesley.
- Bock, R. D. (1972). Estimating item parameters and latent ability when responses are scored in two or more nominal categories. *Psychometrika*, 37(1), 29–51.
- Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of item parameters: Application of an EM algorithm. *Psychometrika*, 46, 443–459.
- Bock, R. D., & Lieberman, M. (1970). Fitting a response model for n dichotomously scored items. *Psychometrika*, 35, 179–197.
- Bock, R. D., & Mislevy, R. J. (1982). Adaptive EAP estimation of ability in a microcomputer environment. *Applied Psychological Measurement*, 6(4), 431–444.
- Bollen, K. A. (1989). *Structural Equations with Latent Variables*. Wiley.
- Bolt, D. M., Cohen, A. S., & Wollack, J. A. (2001). A mixture item response model for multiple-choice data. *Journal of Educational and Behavioral Statistics*, 26(4), 381–409.
- Braeken, J., & van Assen, M. A. L. M. (2017). An empirical Kaiser criterion. *Psychological Methods*, 22(3), 450–466.
- Braun, H. I., & Holland, P. W. (1982). Observed-score test equating: A mathematical analysis of some ETS equating procedures. In P. W. Holland & D. B. Rubin (Eds.), *Test Equating* (pp. 9–49). Academic Press.
- Cai, L. (2010). A two-tier full-information item factor analysis model with applications. *Psychometrika*, 75(4), 581–612.
- Casella, G., & Berger, R. L. (2002). *Statistical Inference* (2nd ed.). Duxbury.
- Chalmers, R. P. (2012). mirt: A multidimensional item response theory package for the R environment. *Journal of Statistical Software*, 48(6).
- Chen, W.-H., & Thissen, D. (1997). Local dependence indexes for item pairs using item response theory. *Journal of Educational and Behavioral Statistics*, 22(3), 265–289.
- Christensen, K. B., Makransky, G., & Horton, M. (2017). Critical values for Yen's Q3: Identification of local dependence in the Rasch model using residual correlations. *Applied Psychological Measurement*, 41(3), 178–194.
- Drasgow, F., & Lissak, R. I. (1983). Modified parallel analysis: A procedure for examining the latent dimensionality of dichotomously scored item responses. *Journal of Applied Psychology*, 68(3), 363–373.
- Drezner, Z., & Wesolowsky, G. O. (1990). On the computation of the bivariate normal integral. *Journal of Statistical Computation and Simulation*, 35(1–2), 101–107.
- Edwards, M. C., Houts, C. R., & Cai, L. (2018). A diagnostic procedure to detect departures from local independence in item response theory models. *Psychological Methods*, 23(1), 138–149.
- Firth, D. (1993). Bias reduction of maximum likelihood estimates. *Biometrika*, 80(1), 27–38.
- Froelich, A. G., & Habing, B. (2008). Conditional covariance-based subtest selection for DIMTEST. *Applied Psychological Measurement*, 32(2), 138–155.
- Garrido, L. E., Abad, F. J., & Ponsoda, V. (2013). A new look at Horn's parallel analysis with ordinal variables. *Psychological Methods*, 18(4), 454–474.
- Genz, A. (2004). Numerical computation of rectangular bivariate and trivariate normal and t probabilities. *Statistics and Computing*, 14(3), 251–260.
- Gibbons, R. D., & Hedeker, D. R. (1992). Full-information item bi-factor analysis. *Psychometrika*, 57(3), 423–436.
- Glorfeld, L. W. (1995). An improvement on Horn's parallel analysis methodology for selecting the correct number of factors to retain. *Educational and Psychological Measurement*, 55(3), 377–393.
- Golino, H. F., & Epskamp, S. (2017). Exploratory graph analysis: A new approach for estimating the number of dimensions in psychological research. *PLoS ONE*, 12(6), e0174035.
- Golub, G. H., & Welsch, J. H. (1969). Calculation of Gauss quadrature rules. *Mathematics of Computation*, 23(106), 221–230.
- González, J., Wiberg, M., & von Davier, A. A. (2016). A note on the Poisson's binomial distribution in item response theory. *Applied Psychological Measurement*, 40(4), 302–310.
- Goretzko, D., & Bühner, M. (2020). One model to rule them all? Using machine learning algorithms to determine the number of factors in exploratory factor analysis. *Psychological Methods*, 25(6), 776–786.
- Green, B. F., Bock, R. D., Humphreys, L. G., Linn, R. L., & Reckase, M. D. (1984). Technical guidelines for assessing computerized adaptive tests. *Journal of Educational Measurement*, 21(4), 347–360.
- Haebara, T. (1980). Equating logistic ability scales by a weighted least squares method. *Japanese Psychological Research*, 22(3), 144–149.
- Harman, H. H., & Jones, W. H. (1966). Factor analysis by minimizing residuals (minres). *Psychometrika*, 31(3), 351–368.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70.
- Hong, Y. (2013). On computing the distribution function for the Poisson binomial distribution. *Computational Statistics & Data Analysis*, 59, 41–51.
- Horn, J. L. (1965). A rationale and test for the number of factors in factor analysis. *Psychometrika*, 30(2), 179–185.
- Huber, P. J. (1967). The behavior of maximum likelihood estimates under nonstandard conditions. In *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability* (Vol. 1, pp. 221–233). University of California Press.
- Humphreys, L. G., & Ilgen, D. R. (1969). Note on a criterion for the number of common factors. *Educational and Psychological Measurement*, 29(3), 571–578.
- Jang, E. E., & Roussos, L. (2007). An investigation into the dimensionality of TOEFL using conditional covariance-based nonparametric approach. *Journal of Educational Measurement*, 44(1), 1–21.
- Jöreskog, K. G. (1967). Some contributions to maximum likelihood factor analysis. *Psychometrika*, 32(4), 443–482.
- Kim, S. (2006). A comparative study of IRT fixed parameter calibration methods. *Journal of Educational Measurement*, 43, 355–381.
- Kim, S., & Lee, W.-C. (2006). An extension of four IRT linking methods for mixed-format tests. *Journal of Educational Measurement*, 43(3), 193–213.
- Kolen, M. J., & Brennan, R. L. (2014). *Test Equating, Scaling, and Linking: Methods and Practices* (3rd ed.). Springer.
- Le Cam, L. (1960). An approximation theorem for the Poisson binomial distribution. *Pacific Journal of Mathematics*, 10(4), 1181–1197.
- Li, Y. H., & Lissitz, R. W. (2000). An evaluation of the accuracy of multidimensional IRT linking. *Applied Psychological Measurement*, 24(2), 115–138.
- Lord, F. M. (1980). *Applications of Item Response Theory to Practical Testing Problems*. Lawrence Erlbaum.
- Lord, F. M., & Wingersky, M. S. (1984). Comparison of IRT true-score and equipercentile observed-score "equatings." *Applied Psychological Measurement*, 8(4), 453–461.
- Lorenzo-Seva, U., Timmerman, M. E., & Kiers, H. A. L. (2011). The Hull method for selecting the number of common factors. *Multivariate Behavioral Research*, 46(2), 340–364.
- Louis, T. A. (1982). Finding the observed information matrix when using the EM algorithm. *JRSS B*, 44, 226–233.
- MacCallum, R. C., Browne, M. W., & Sugawara, H. M. (1996). Power analysis and determination of sample size for covariance structure modeling. *Psychological Methods*, 1(2), 130–149.
- Masters, G. N. (1982). A Rasch model for partial credit scoring. *Psychometrika*, 47, 149–174.
- Maydeu-Olivares, A., & Joe, H. (2005). Limited- and full-information estimation and goodness-of-fit testing in $2^n$ contingency tables: A unified framework. *Journal of the American Statistical Association*, 100(471), 1009–1020.
- Maydeu-Olivares, A., & Joe, H. (2006). Limited information goodness-of-fit testing in multidimensional contingency tables. *Psychometrika*, 71(4), 713–732.
- Maydeu-Olivares, A., & Joe, H. (2014). Assessing approximate fit in categorical data analysis. *Multivariate Behavioral Research*, 49(4), 305–328.
- McLachlan, G. J., & Peel, D. (2000). *Finite Mixture Models*. Wiley.
- Mislevy, R. J. (1984). Estimating latent distributions. *Psychometrika*, 49(3), 359–381.
- Mislevy, R. J. (1986). Bayes modal estimation in item response models. *Psychometrika*, 51, 177–195.
- Mislevy, R. J. (1991). Randomization-based inference about latent variables from complex samples. *Psychometrika*, 56(2), 177–196.
- Muraki, E. (1992). A generalized partial credit model: Application of an EM algorithm. *Applied Psychological Measurement*, 16, 159–176.
- Nandakumar, R., & Stout, W. (1993). Refinements of Stout's procedure for assessing latent trait unidimensionality. *Journal of Educational Statistics*, 18(1), 41–68.
- O'Connor, B. P. (2000). SPSS and SAS programs for determining the number of components using parallel analysis and Velicer's MAP test. *Behavior Research Methods, Instruments, & Computers*, 32(3), 396–402.
- Oakes, D. (1999). Direct calculation of the information matrix via the EM algorithm. *JRSS B*, 61, 479–482.
- Oehlert, G. W. (1992). A note on the delta method. *The American Statistician*, 46(1), 27–29.
- Ogasawara, H. (2001). Standard errors of item response theory equating/linking by response function methods. *Applied Psychological Measurement*, 25(1), 53–67.
- Olsson, U. (1979). Maximum likelihood estimation of the polychoric correlation coefficient. *Psychometrika*, 44(4), 443–460.
- Orlando, M., & Thissen, D. (2000). Likelihood-based item-fit indices for dichotomous item response theory models. *Applied Psychological Measurement*, 24(1), 50–64.
- Orlando, M., & Thissen, D. (2003). Further investigation of the performance of S-X2: An item fit index for use with dichotomous item response theory models. *Applied Psychological Measurement*, 27(4), 289–298.
- Oshima, T. C., Davey, T. C., & Lee, K. (2000). Multidimensional linking: Four practical approaches. *Journal of Educational Measurement*, 37(4), 357–373.
- Owen, D. B. (1956). Tables for computing bivariate normal probabilities. *The Annals of Mathematical Statistics*, 27(4), 1075–1090.
- Plackett, R. L. (1954). A reduction formula for normal multivariate integrals. *Biometrika*, 41(3–4), 351–360.
- Press, W. H., Teukolsky, S. A., Vetterling, W. T., & Flannery, B. P. (2007). *Numerical Recipes: The Art of Scientific Computing* (3rd ed.). Cambridge University Press.
- Ramsay, J. O. (1991). Kernel smoothing approaches to nonparametric item characteristic curve estimation. *Psychometrika*, 56(4), 611–630.
- Reckase, M. D. (2009). *Multidimensional Item Response Theory*. Springer.
- Revelle, W. (2024). *psych: Procedures for psychological, psychometric, and personality research* (R package). Northwestern University.
- Rodriguez, A., Reise, S. P., & Haviland, M. G. (2016). Evaluating bifactor models: Calculating and interpreting statistical indices. *Psychological Methods*, 21(2), 137–150.
- Rost, J. (1990). Rasch models in latent classes: An integration of two approaches to item analysis. *Applied Psychological Measurement*, 14(3), 271–282.
- Ruscio, J., & Kaczetow, W. (2008). Simulating multivariate nonnormal data using an iterative algorithm. *Multivariate Behavioral Research*, 43(3), 355–381.
- Ruscio, J., & Roche, B. (2012). Determining the number of factors to retain in an exploratory factor analysis using comparison data of known factorial structure. *Psychological Assessment*, 24(2), 282–292.
- Samejima, F. (1969). Estimation of latent ability using a response pattern of graded scores. *Psychometrika Monograph*, 17.
- Skinner, C. J. (1989). Domain means, regression and multivariate analysis. In C. J. Skinner, D. Holt & T. M. F. Smith (Eds.), *Analysis of Complex Surveys* (pp. 59–87). Wiley.
- Stocking, M. L., & Lord, F. M. (1983). Developing a common metric in item response theory. *Applied Psychological Measurement*, 7(2), 201–210.
- Stout, W. (1987). A nonparametric approach for assessing latent trait unidimensionality. *Psychometrika*, 52(4), 589–617.
- Stout, W. (1990). A new item response theory modeling approach with applications to unidimensionality assessment and ability estimation. *Psychometrika*, 55(2), 293–325.
- Stout, W., Froelich, A. G., & Gao, F. (2001). Using resampling methods to produce an improved DIMTEST procedure. In A. Boomsma, M. A. J. van Duijn, & T. A. B. Snijders (Eds.), *Essays on Item Response Theory* (pp. 357–375). Springer.
- Swaminathan, H., & Gifford, J. A. (1986). Bayesian estimation in the three-parameter logistic model. *Psychometrika*, 51, 589–601.
- Thissen, D., Cai, L., & Bock, R. D. (2010). The nominal categories item response model. In M. L. Nering & R. Ostini (Eds.), *Handbook of Polytomous Item Response Theory Models* (pp. 43–75). Routledge.
- Thissen, D., Pommerich, M., Billeaud, K., & Williams, V. S. L. (1995). Item response theory for scores on tests including polytomous items with ordered responses. *Applied Psychological Measurement*, 19(1), 39–49.
- Thissen, D., Steinberg, L., & Wainer, H. (1993). Detection of differential item functioning using the parameters of item response models. In P. W. Holland & H. Wainer (Eds.), *Differential Item Functioning* (pp. 67–113). Erlbaum.
- Thissen, D., & Wainer, H. (Eds.). (2001). *Test Scoring*. Lawrence Erlbaum Associates.
- Varadhan, R., & Roland, C. (2008). Simple and globally convergent methods for accelerating the convergence of any EM algorithm. *Scand. J. Statist.*, 35, 335–353.
- Velicer, W. F. (1976). Determining the number of components from the matrix of partial correlations. *Psychometrika*, 41(3), 321–327.
- Velicer, W. F., Eaton, C. A., & Fava, J. L. (2000). Construct explication through factor or component analysis: A review and evaluation of alternative procedures for determining the number of factors or components. In R. D. Goffin & E. Helmes (Eds.), *Problems and Solutions in Human Assessment* (pp. 41–71). Kluwer.
- Volkova, A. Y. (1996). A refinement of the central limit theorem for sums of independent random indicators. *Theory of Probability and Its Applications*, 40(4), 791–794.
- Wang, C. (2015). On latent trait estimation in multidimensional compensatory item response models. *Psychometrika*, 80(2), 428–449.
- Warm, T. A. (1989). Weighted likelihood estimation of ability in item response theory. *Psychometrika*, 54(3), 427–450.
- White, H. (1982). Maximum likelihood estimation of misspecified models. *Econometrica*, 50(1), 1–25.
- Woods, C. M. (2007). Empirical histograms in item response theory with ordinal data. *Educational and Psychological Measurement*, 67, 73–87.
- Woods, C. M., & Lin, N. (2009). Item response theory with estimation of the latent density using Davidian curves. *Applied Psychological Measurement*, 33, 102–117.
- Wothke, W. (1993). Nonpositive definite matrices in structural modeling. In K. A. Bollen & J. S. Long (Eds.), *Testing Structural Equation Models* (pp. 256–293). Sage.
- Yen, W. M. (1984). Effects of local item dependence on the fit and equating performance of the three-parameter logistic model. *Applied Psychological Measurement*, 8(2), 125–145.
- Zhang, D., & Davidian, M. (2001). Linear mixed models with flexible distributions of random effects for longitudinal data. *Biometrics*, 57, 795–802.
- Zhang, J. (2007). Conditional covariance theory and DETECT for polytomous items. *Psychometrika*, 72(1), 69–91.
- Zhang, J., & Stout, W. (1999). The theoretical DETECT index of dimensionality and its application to approximate simple structure. *Psychometrika*, 64(2), 213–249.
