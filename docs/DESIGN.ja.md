# flexpyirt 設計書（DESIGN）

[English](DESIGN.md) | 日本語

> 状態: ドラフト v0.2（2026-11-20、M1 実装を反映）。実装前の設計。決定事項は「決定」、未確定は §12「未決事項」に置く。

## 1. 目的と非目的

### 目的
- **Bock–Aitkin EM（BA-EM, Bock & Aitkin, 1981）による周辺最尤推定（MMLE）** を中核にした、Python の IRT 推定ライブラリ。
- R `mirt` が実務で担っている以下を、Python らしい API で提供する:
  - 多母集団（multiple-group）同時推定（群間の等値制約・部分アンカー）
  - 固定パラメタ推定（FIPC: fixed item parameter calibration）
  - 潜在分布の柔軟な指定（Normal / Empirical Histogram / 将来 Davidian・混合正規）
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
- 群ごとに反応行列（欠測は `-1`）をユニーク化し、度数 $c_p$ を得る。
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
- 将来: Davidian 曲線（Woods & Lin, 2009）、混合正規。いずれも `LatentDist` プロトコル（§6）に従えば Core に手を入れずに追加できる。

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
- **決定（M3, §15.14–§15.17）**: 公開の不変事前分布 `NormalPrior(mean, sd)`, `LogNormal(meanlog, sdlog)`,
  `Beta(a, b)` を提供する（各々 `log_pdf` とパラメタ検証を持つ。潜在分布 `Normal` との衝突を避けるため `NormalPrior`）。
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
  $-\nabla^2_\eta[\ell(\eta)+\log\pi(\eta)]$（Oakes の検証にも使う）、
  cross-product（$\sum_p c_p s_p s_p^\top$）、sandwich。
- MAP 時は `oakes`・`hessian` とも情報行列に $-\nabla^2_\eta\log\pi(\hat\eta)$ を加え、数学的恒等性を保つ。
  固定パラメタは $\eta$ に含まれないので自動的に除外される。

### 4.9 求積
- **決定**: 1 次元の既定は等間隔 61 点、$[-6,6]$（EH と同じ格子を共有でき、Normal でも重みを密度から作る）。
- Normal 専用に Gauss–Hermite も選べる（`Quadrature.gauss_hermite(n)`、未実装）。
- **多次元直積格子（M4, §15.30）**: 2・3 次元モデルは 1 次元等間隔格子の直積（`Quadrature.grid(n_factors)` → `(Q, D)`）を使う。次元あたりの既定節点数は `Quadrature.default(n_factors)` により mirt の `select_quadpts` に合わせ、1 次元で 61 点、2 次元で 31 点（$31^2 = 961$）、3 次元で 15 点（$15^3 = 3{,}375$）、区間 $[-6, 6]$ とする。4 次元以上は対象外（非目的）。

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
`families`（各カーネルの `n_factors`・`n_cat` を含む）と `n_groups` は静的（jit の static）、パラメタ値と配列は動的。

## 6. 拡張プロトコル

```python
class ItemKernel(Protocol):
    n_factors: int  # 1, 2, または 3
    param_names: tuple[str, ...]  # ("a", "d") または ("a1", ..., "aD", "d")
    n_cat: int
    slope_params: tuple[str, ...]
    intercept_params: tuple[str, ...]
    irt_names: tuple[str, ...]  # 表示用閾値（D > 1 では空）
    irt_dep_indices: tuple[tuple[int, ...], ...]

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
| Nominal | M5 以降（未定） | — |

表示用の IRT パラメタ化（2PL / Rasch / 3PL / 4PL は $b=-d/a$、GRM は $b_k=-d_k/a$、GPCM / PCM は
ステップ困難度 $b_k=-(d_k-d_{k-1})/a$、$d_0=0$）は Result 層（`FitResult.items`）で計算し、SE
（`b_se` / `bk_se`）は JAX の前向き自動微分によるヤコビアンを用いたデルタ法で求める（§15.18）。

## 8. スコアリング
`FitResult.score(data, *, method="EAP"|"MAP"|"ML"|"WLE"|"PV", group=None, item_names=None, key=None, n_draws=5)`:
- **EAP**（Bock & Mislevy, 1982）: 推定済みモデルと同じ求積格子上で、回答者の群の推定済み潜在分布を事前分布とする事後平均と事後 SD。
- **MAP**: 群の推定済み正規潜在分布を連続事前分布として $\log L(\theta\mid y) + \log\phi(\theta;\mu_g,\sigma_g^2)$ を最大化するベイズ最頻値推定。SE は最頻値における観測事後情報量 $(-\partial^2/\partial\theta^2)^{-1/2}$（§15.22）。
- **ML**: $\log L(\theta\mid y)$ を最大化する最尤推定。SE は観測情報量 $(-\partial^2\log L/\partial\theta^2)^{-1/2}$。有限の最大値を持たないパターンは $-\infty$ / $+\infty$（全欠測は NaN）、SE は NaN（§15.23）。
- **WLE**（Warm, 1989; Firth, 1993）: 観測項目にわたる Fisher テスト情報量 $I(\theta)$ と JAX 自動微分による $J(\theta) = \partial I/\partial\theta$ を用いて $\partial\log L/\partial\theta + J(\theta)/(2I(\theta)) = 0$（等価に $\log L(\theta\mid y) + \tfrac{1}{2}\log I(\theta)$ の最大化）を解く重み付き最尤推定。極端パターンでも有限。SE は mirt の既定に合わせて WLE 目的関数の観測曲率から計算（§15.24）。
- **PV**（Mislevy, 1991）: 細かい等間隔格子上の事後分布から区分線形逆 CDF 補間により回答者ごとに独立に `n_draws` 個を抽出し、`pv1..pvM` 列の DataFrame を返す。明示的な `key: jax.Array` が必須（未指定は `SpecificationError`）で、`key` に対して決定的（§15.25）。

すべての手法で欠測反応は尤度（および WLE の $I(\theta)$）から除く。すべての手法がユニークパターン圧縮と 2 のべき乗パディングを用いてデータ値に依存しない形状で `jit` 実行される（§15.21）。

## 9. 性能方針
- パターン圧縮 + バケットパディングで再コンパイルを抑える。
- item family ごとの `vmap`。異種項目混在でも family 数（静的）だけのループ。
- 既定は CPU で十分速いことを目標。GPU は大規模（N ≥ 10^5, J ≥ 100）で効けばよい。
- ベンチマーク: R mirt（`TOL=1e-4`、既定設定および検証設定）との wall time 比較を
  [benchmarks/](../benchmarks/README.ja.md) に置き、Apple M5 CPU での計測結果を
  [benchmarks/results/2026-11-21-apple-m5.md](../benchmarks/results/2026-11-21-apple-m5.md) に
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
6. **FIPC**: `Model.fix_from(bank, items=..., parameterization="slope-intercept" | "irt")` で項目パラメタを固定し、
   潜在分布は利用者が `with_latent(Normal.free())` で自由にする（既定は変えない。§4.5 の識別性検査で許可される）。
   IRT 表記（$b$）は $d=-ab$ で変換する（2PL のみ。GRM は $b_k$ → $d_k=-ab_k$）。
7. **GRM**（Samejima, 1969）: $P(y\ge k\mid\theta)=\sigma(a\theta+d_k)$、$d_1>\dots>d_{K-1}$、カテゴリは $0..K-1$。
   対数確率は端のカテゴリを $\log\sigma(-z_1)$、$\log\sigma(z_{K-1})$、中間を
   $\log\sigma(z_k)+\operatorname{log1mexp}(\log\sigma(z_{k+1})-\log\sigma(z_k))$ で安定に計算する
   （log1mexp は両分岐の入力を安全域に丸める二重 where で勾配の NaN を防ぐ）。順序制約は line search で
   有限値でない候補を棄却することで守る。family は `GRM(n_cat=K)` の $K$ ごと。データに現れないカテゴリが
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
   ParameterTable から自動的に現れる。`Model.fix_from()` は `"slope-intercept"`・`"irt"` のどちらでも
   `g`, `u` をそのまま渡し、3PL 項目では `u` 列を省略できる。
6. **HeywoodWarning**: 推定後、`fit()` はすべての自由な項目パラメタを検査し、有限の境界から `1e-4` 以内に
   到達したパラメタ（0 や 1 に近い `g`, `u`）または $|a|>20$ の自由な傾きがあれば、該当する項目名を列挙した
   `flexpyirt.errors.HeywoodWarning` を 1 回発する。
7. **識別性**: `g`, `u` は潜在変数の尺度も位置も担わない（`slope_params=("a",)`, `intercept_params=("d",)`）。
   したがって漸近線を固定しても潜在分布の平均や分散は識別されない。
8. **GPCM・PCM 項目モデル**: `GPCM(n_cat)`（Muraki, 1992）と `PCM(n_cat)`（Masters, 1982）は `n_cat` ごとに
   共通カーネル `GPCMKernel(n_cat)` を共有する。カテゴリは `0..K-1`（`n_cat >= 2` を検証）で、mirt の
   `itemtype="gpcm"` と同じ傾き・切片形式（採点係数 $\mathrm{ak}_k=k$ と $d_0=0$ を固定し、パラメタに含めない）:
   $$P(y=k\mid\theta)=\frac{\exp(k\,a\theta+d_k)}{\sum_{c=0}^{K-1}\exp(c\,a\theta+d_c)},\qquad d_0=0.$$
   内部パラメタは `(a, d1, ..., d{K-1})`。対数確率は `jax.nn.log_softmax` で計算する（数値安定で滑らかな微分を
   持ち、`d` に順序制約は不要）。`PCM(n_cat)` は既定で `a = 1` を固定し、単一群の既定潜在分布ルール（§4.5）を
   拡張して、全項目が `Rasch` または `PCM` のとき $N(0,\sigma^2)$（分散自由）とする（多値データに対する mirt の
   `itemtype="Rasch"` と同じ）。`n_cat = 2` では `GPCM(2)` は `TwoPL` に、`PCM(2)` は `Rasch` に厳密に一致する。
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
14. **事前分布と自然尺度 MAP 目的関数（M3, §4.7）**: `NormalPrior(mean, sd)`, `LogNormal(meanlog, sdlog)`,
    `Beta(a, b)` を `priors.py` の公開 frozen dataclass とし、パラメタ検証（`SpecificationError`）と `log_pdf(x)` を
    実装する。事前分布はヤコビアンなしの自然尺度で評価する（R mirt 1.46 の `LL.Priors` でも `expbeta` が
    `dbeta(plogis(logit_g), a, b, log = TRUE)` を logit ヤコビアンなしで評価することを確認済み。mirt 1.46 の
    `DerivativePriors` / `BL.LL` の不具合は VALIDATION.ja.md に記録）。`Beta.log_pdf` の $\log B(a, b)$ は
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
    （`LogNormal` は `(0, inf)`、`Beta` は `(0, 1)`）と交差させ、ParameterTable のモデル定義域を変えずに有界 Newton と
    SQUAREM が台を守るようにする。
17. **MAP の結果・履歴・SQUAREM 安全装置・事後 SE**: `FitResult.loglik` は周辺対数尤度のみとし、
    `FitResult.log_posterior` に $\ell(\hat\eta)+\log\pi(\hat\eta)$（事前分布なしでは `loglik` と同値）を返す。
    `history` は統一スキーマ `("iter", "loglik", "log_posterior", "max_change", "em_evaluations", "safeguard")` を持ち、
    EM・SQUAREM とも `log_posterior` が単調非減少で、SQUAREM 安全装置も `log_posterior` を比較する。自由パラメタに
    1 つ以上事前分布がある場合、`FitResult.aic` と `FitResult.bic` は `loglik` で計算しつつ `FlexpyirtWarning` を出す。
    `information_oakes` は $Q(\eta\mid\eta')$ に `log_prior(prob, e)` を足し、`information_hessian` は
    `log_posterior(prob, e)` を微分することで、両者とも厳密に $-\nabla^2_\eta\log\pi(\hat\eta)$ を加える。
18. **表示用 IRT 閾値のデルタ法 SE（M4, §7）**: 各 `ItemKernel` に `irt_names`（例: `("b",)` や
    `("b1", ..., "b{K-1}")`）、`irt_dep_indices`（各閾値に入る内部パラメタの添字）、純粋 JAX 関数
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
26. **`FitResult` の JSON 入出力 API とバージョン付き形式（M4, API §7）**: `FitResult.to_json(path=None, *, indent=2) -> str` は推定結果を厳格な JSON 文字列に直列化し、`path`（`str` または `os.PathLike`）が指定された場合は UTF-8 ファイルにも書き出す。`FitResult.from_json(source) -> FitResult` はファイルパスまたは JSON 文字列から再推定なしで復元する。素の JSON のみを使い、`pickle` や `eval` は一切使わない。最上位オブジェクトには `"format": "flexpyirt.FitResult"`、`"schema_version"`（§15.33 で多次元モデルを加えてから 2。version 1 のファイルも読める）、および書き出した版を記録する `"flexpyirt_version": flexpyirt.__version__` を持たせる。`FitResult` は `_model`・`_quadrature` とあわせて推定設定 `_method: BAEM` も保持し、メタデータ全体が往復保存できるようにする。`from_json` は入口で `check_x64()` を呼ぶ（スコアリングに float64 が必要なため、無効なら `PrecisionError`）。
27. **厳格な JSON と正確な float64 往復保存（M4）**: 書き出しは `json.dumps(..., allow_nan=False)` で厳格な RFC 8259 JSON のみを出力し、読み込みは `json.loads` に `parse_constant` を渡して生の `NaN`/`Infinity` リテラルを `SpecificationError` で拒否する。有限な IEEE-754 float64 値は通常の JSON 数値（Python の `repr` 厳密な最短 10 進表現）として書き出し、`-0.0` を含むすべての有限 float64 をビット単位で正確に往復保存する。浮動小数点型のフィールド（無制約の `lower`/`upper`、固定または未計算の `se`、非正定値の `vcov` など）に現れる非有限値（`NaN`, `+Inf`, `-Inf`）は明示的に文字列 `"NaN"`, `"Infinity"`, `"-Infinity"` として符号化する。これらを浮動小数点へ復号するのはスキーマ上 float 型のスロットのみであるため、項目名や群名などの文字列フィールドとは衝突しない。
28. **明示的レジストリと完全な状態保存（M4）**: 項目モデル（`"2PL"`, `"Rasch"`, `"3PL"`, `"4PL"`, `"GRM"`, `"GPCM"`, `"PCM"`、多値モデルでは `n_cat` を含む）、潜在分布（`"Normal"`）、事前分布（`"NormalPrior"`, `"LogNormal"`, `"Beta"`）、求積（`"equispaced"`）、推定手法（`"BAEM"`）は `_json.py` の明示的な名前レジストリで（逆）直列化する。未知の `format` タグ・スキーマ版・レジストリ名・不正な JSON・整合しない表や行列形状は明確なメッセージつきで `SpecificationError` を送出する。完全な `Model` 仕様（項目名と順序、項目モデル、`fixed`, `latent`, `groups`, `equal`, `priors`）と、`estimate`・`se` を含む検証済み `ParameterTable`（ユーザーによる `table=` 編集もそのまま保持）、`vcov`、`loglik`/`log_posterior`、収束情報と `history`、`quadrature`、`method`、`se_method`、`n_obs` をすべて保存する。`_eta` は `parameters` の検証済み自由行から直接復元し（`free_id` が `0..n_free-1` に連続であること、共有 `free_id` の `estimate` が一致すること、`vcov` 形状が整合することを検証）、推定値の重複を持たせない。復元後の `FitResult` はすべての公開属性・`items`・`latent`・`summary()`・`_repr_html_()`・`aic`/`bic`・全 5 手法の `score()` で完全に同一の結果を返す。
29. **確認的 2–3 次元 MIRT の仕様とカーネル（M4, §4.9, §6）**:
    `Model.with_factors(factors, *, correlated=True)` で 2 または 3 個の名前つき因子と負荷項目の確認的パターン（`Mapping[str, ItemSelector]`）を指定する。1 項目が複数因子に負荷すること（交差負荷）を許し、各因子には 1 項目以上が必要で、すべてのモデル項目がいずれか 1 因子以上に負荷しなければならない。負荷しない因子の傾きは 0 に固定する。`with_factors()` を呼ばない `Model.uniform(...)` / `Model.from_items(...)` は厳密に 1 次元（`n_factors=1`）のままビット単位で挙動を変えない。4 つのカーネル（`DichotomousKernel`, `FourPLKernel`, `GradedKernel`, `GPCMKernel`）はすべて静的な `n_factors: int = 1` と `with_factors(n_factors)` を持ち、線形予測子 $z = \mathbf a^\top\boldsymbol\theta + d$（$D=1$ は `params[0] * theta[:, 0]`、$D \in \{2, 3\}$ は `theta @ params[:D]`）を評価するため、既存の 7 項目モデル（`TwoPL`, `Rasch`, `ThreePL`, `FourPL`, `GRM`, `GPCM`, `PCM`）がすべて 2・3 次元でそのまま動作する。$D > 1$ では内部パラメタ名が mirt に合わせて `a1..aD`, `d`（GRM / GPCM / PCM は `d1..d{K-1}`）となり、1 次元用の表示閾値（`b` / `bk`）は省略する。$D > 1$ での多母集団、`with_latent()`、`fix_from(..., parameterization="irt")` は M4 では対象外とし `SpecificationError` を送出する。
30. **直積求積と多変量正規潜在分布の厳密 M-step（M4, §4.4, §4.9）**:
    `Quadrature.default(n_factors)` は mirt の `select_quadpts` に合わせた既定値（1 次元 `61`、2 次元 `31` → $961$ 点、3 次元 `15` → $3{,}375$ 点、区間 $[-6, 6]$）を返し、`Quadrature.grid(n_factors)` が `(Q, D)` の直積格子を返す（`nodes()` は 1 次元の `(n_points,)` のまま）。`CompiledProblem.nodes` の形状は `(Q, D)`。$D \in \{2, 3\}$ の多変量正規潜在分布は `ParameterTable` と `LatentBlock` 上で自然尺度の `(mean_1..D, var_1..D, cor_ij...)`（$L_2 = 5$, $L_3 = 9$、$1 \le i < j \le D$）としてパラメタ化する: $\Sigma_{ii} = v_i > 0$（`lower = 0`）、$\Sigma_{ij} = \rho_{ij}\sqrt{v_i v_j}$ かつ $\rho_{ij} \in (-1, 1)$（`lower = -1, upper = 1`）。分散が 1 固定（2PL / GRM / 3PL / 4PL / GPCM の既定）では `cor_ij` が $\Sigma_{ij}$ と一致するため mirt の `COV_ji` およびその SE と直接一致し、全項目が `Rasch` または `PCM` の場合は `cor_ij` に加えて `var_1..var_D` も既定で自由推定とする。`mvn_log_weights(nodes, params)` は $D = 1$ では従来の `normal_log_weights` にそのまま振り分け、$D \in \{2, 3\}$ では相関行列の正定値性を Sylvester の判定法（主座小行列式 $> 0$）と二重 `where` で保護しつつ、閉形式二次形式（$D = 2$）または Cholesky + `solve_triangular`（$D = 3$）と `log_softmax` で格子上の正規化対数重みを計算する。正定値領域内部では滑らかで NaN のない微分を与え、非正定値候補は NaN を返して境界つき Newton の直線探索および SQUAREM の安全装置で棄却される。1 次元（§4.4）と同様、潜在 M-step は離散化後の厳密な $Q_g^{\text{lat}} = \sum_q \bar n_{gq}\log w_{gq}$ を最大化する（EM の単調性と Oakes 恒等式の厳密性を保つ）。
31. **$D > 1$ の識別性・符号不定性・初期値（M4, §4.5, §4.10）**:
    `_check_identification_mirt` が推定前に必要条件を検査する: 各因子に有効な負荷（自由または非ゼロ固定）が 1 つ以上あること、自由な `var_m` には非ゼロ固定の `a{m}` が 1 つ以上あること、自由な `mean_m` にはその因子に負荷する項目のうち切片が固定されたものが 1 つ以上あること、確認的な回転識別のため $c_m$ 個の自由な相関に関わる因子 $m$ は少なくとも $c_m$ 個の固定負荷（通常は 0 固定）を持ち、非ゼロ固定負荷のない 2 因子が同一の自由負荷集合を共有しないこと（Bollen, 1989; Reckase, 2009）。**符号不定性**: $\theta_m$ とすべての $a_{jm}$・$\mu_m$・$\rho_{mm'}$（$m' \neq m$）の符号反転で周辺尤度は不変（1 次元と同様）。逆転項目や自然尺度 SE を不自然な境界で歪めないよう傾き自体は非有界（`(-inf, inf)`）のまま保ち、各因子の向きは因子別項目−残余相関から $1/\sqrt{L_j}$（$L_j$ は項目 $j$ が負荷する因子数。切片・漸近線には $\|\mathbf a_j^{(0)}\|$ を使用）で計算する厳密に正の既定初期傾き $a_{jm}^{(0)} \in [0.25, 3.0]$ と、常に正定値な相関初期値 `0.0`（$R^{(0)} = I_D$）によって正の方向に揃える。編集済み表である因子の初期傾きがすべて 0 の場合は `SpecificationError` で弾く。
32. **多次元の結果・Oakes SE・EAP スコアリング（M4, §4.8, §8）**:
    Core 層の E-step・M-step・SQUAREM・構造化 Oakes（`_oakes_own_hessian`, `_oakes_cross_hessian`）・チャンク分割 Hessian はすべて `(Q, D)` 節点と `(G, L_D)` 潜在ブロックに対して静的形状・自動微分のみでそのまま動作する。`FitResult.items` は `a1..aD`, `d`（`d1..`）とその SE を表示し、`FitResult.latent` は `mean_1..D`, `var_1..D`, `cor_ij` とその SE を表示する。`FitResult.score(data, method="EAP")` は宣言した因子順に `theta_<factor>`, `se_<factor>` 列を返す。`method="MAP"` も使える（§15.39）。$D > 1$ の `ML`・`WLE`・`PV` は `SpecificationError` を送出する。
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
    $D > 1$ の `ML`・`WLE`・`PV` は引き続き `SpecificationError`（ML/WLE は因子ごとに発散する方向の扱いが必要）。
40. **S-X2 項目適合度（M5、`FitResult.item_fit`）**: `Normal` 潜在分布の単一群 1 次元モデルと欠測のないデータに対する
    Orlando と Thissen（2000, 2003）の S-X2。期待割合
    $E_j[s,k]=\sum_q P_{jk}(\theta_q)L_{-j}(s-k\mid\theta_q)w_q/\sum_q L(s\mid\theta_q)w_q$ を、推定時の求積格子と潜在分布の重みの上で
    Lord–Wingersky 漸化式（Lord & Wingersky, 1984）から合計得点 $1,\dots,S-1$ について求め、各得点の観測人数を掛ける。
    度数の少ないセルは `mirt:::collapseCells`（mirt 1.46–1.48）を 1 行ずつ移植した処理でまとめる（1 件だけの行をまとめる前に
    計算した行のフラグを後で使う点も含め、統計量と自由度が一致するようにヒューリスティックをそのまま再現する）。
    $df_j$ = 残ったセル数 − 行数 − 項目 $j$ の自由パラメタ数 − 潜在分布の自由パラメタ数で、mirt と同じ（mirt はすべての項目から
    潜在分布のパラメタ数を引く）。RMSEA と $\chi^2$ の p 値も mirt に合わせる。表の計算は推定後の NumPy 処理（形がデータに依存
    するので `jit` の外）で、カテゴリ確率は JAX の項目カーネルから得る。多次元・多母集団・EH・欠測データは照合するまで
    `SpecificationError`（mirt は S-X2 で欠測を補完し、EH の重みを無視する）。

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
5. **微分可能な IRT 尺度リンキング（`link`、`LinkResult`、V20）**:
   別々に推定した 2 つの 1 次元尺度を共通項目を介して結ぶ特性曲線法（§1）。`source`（新フォーム）の尺度を
   `target`（基準フォーム）の尺度へ写す線形変換 $\theta^* = A\theta + B$（$A > 0$）の係数 $(A, B)$ を求める
   （Kolen & Brennan, 2014）。$\theta = (\theta^* - B)/A$ より、傾き–切片パラメタ化での項目パラメタは
   $a^* = a / A$ および以下で変換される。
   - `DichotomousKernel`（2PL/Rasch）・`FourPLKernel`（3PL/4PL）: $d^* = d - aB/A$（漸近線 $g, u$ は不変）。
   - `GradedKernel`（GRM）: $d_k^* = d_k - aB/A$（$k = 1, \dots, K-1$。順序 $d_1 > \dots > d_{K-1}$ を保存）。
   - `GPCMKernel`（GPCM/PCM）: $d_k^* = d_k - k\,aB/A$（$k = 1, \dots, K-1$、$d_0 = 0$）。
     $k a^* \theta^* + d_k^* = k a\theta + d_k$ とステップ困難度 $b_k^* = A b_k + B$ を保存する。
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
   全共通項目の表示用閾値をすべてまとめて平均するため、多値項目は項目単位の平均より重みが大きい。
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

## 参考文献
- Barton, M. A., & Lord, F. M. (1981). An upper asymptote for the three-parameter logistic item-response model. *ETS Research Report Series*, RR-81-20.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *JRSS B*, 57(1), 289–300.
- Bertsekas, D. P. (1982). Projected Newton methods for optimization problems with simple constraints. *SIAM Journal on Control and Optimization*, 20(2), 221–246.
- Birnbaum, A. (1968). Some latent trait models and their use in inferring an examinee's ability. In F. M. Lord & M. R. Novick, *Statistical Theories of Mental Test Scores* (pp. 397–479). Addison-Wesley.
- Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of item parameters: Application of an EM algorithm. *Psychometrika*, 46, 443–459.
- Bock, R. D., & Lieberman, M. (1970). Fitting a response model for n dichotomously scored items. *Psychometrika*, 35, 179–197.
- Bock, R. D., & Mislevy, R. J. (1982). Adaptive EAP estimation of ability in a microcomputer environment. *Applied Psychological Measurement*, 6(4), 431–444.
- Bollen, K. A. (1989). *Structural Equations with Latent Variables*. Wiley.
- Casella, G., & Berger, R. L. (2002). *Statistical Inference* (2nd ed.). Duxbury.
- Chalmers, R. P. (2012). mirt: A multidimensional item response theory package for the R environment. *Journal of Statistical Software*, 48(6).
- Firth, D. (1993). Bias reduction of maximum likelihood estimates. *Biometrika*, 80(1), 27–38.
- Haebara, T. (1980). Equating logistic ability scales by a weighted least squares method. *Japanese Psychological Research*, 22(3), 144–149.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70.
- Kim, S. (2006). A comparative study of IRT fixed parameter calibration methods. *Journal of Educational Measurement*, 43, 355–381.
- Kim, S., & Lee, W.-C. (2006). An extension of four IRT linking methods for mixed-format tests. *Journal of Educational Measurement*, 43(3), 193–213.
- Kolen, M. J., & Brennan, R. L. (2014). *Test Equating, Scaling, and Linking: Methods and Practices* (3rd ed.). Springer.
- Lord, F. M., & Wingersky, M. S. (1984). Comparison of IRT true-score and equipercentile observed-score "equatings." *Applied Psychological Measurement*, 8(4), 453–461.
- Louis, T. A. (1982). Finding the observed information matrix when using the EM algorithm. *JRSS B*, 44, 226–233.
- Masters, G. N. (1982). A Rasch model for partial credit scoring. *Psychometrika*, 47, 149–174.
- Mislevy, R. J. (1986). Bayes modal estimation in item response models. *Psychometrika*, 51, 177–195.
- Mislevy, R. J. (1991). Randomization-based inference about latent variables from complex samples. *Psychometrika*, 56(2), 177–196.
- Muraki, E. (1992). A generalized partial credit model: Application of an EM algorithm. *Applied Psychological Measurement*, 16, 159–176.
- Oakes, D. (1999). Direct calculation of the information matrix via the EM algorithm. *JRSS B*, 61, 479–482.
- Oehlert, G. W. (1992). A note on the delta method. *The American Statistician*, 46(1), 27–29.
- Ogasawara, H. (2001). Standard errors of item response theory equating/linking by response function methods. *Applied Psychological Measurement*, 25(1), 53–67.
- Orlando, M., & Thissen, D. (2000). Likelihood-based item-fit indices for dichotomous item response theory models. *Applied Psychological Measurement*, 24(1), 50–64.
- Orlando, M., & Thissen, D. (2003). Further investigation of the performance of S-X2: An item fit index for use with dichotomous item response theory models. *Applied Psychological Measurement*, 27(4), 289–298.
- Reckase, M. D. (2009). *Multidimensional Item Response Theory*. Springer.
- Samejima, F. (1969). Estimation of latent ability using a response pattern of graded scores. *Psychometrika Monograph*, 17.
- Stocking, M. L., & Lord, F. M. (1983). Developing a common metric in item response theory. *Applied Psychological Measurement*, 7(2), 201–210.
- Swaminathan, H., & Gifford, J. A. (1986). Bayesian estimation in the three-parameter logistic model. *Psychometrika*, 51, 589–601.
- Thissen, D., Steinberg, L., & Wainer, H. (1993). Detection of differential item functioning using the parameters of item response models. In P. W. Holland & H. Wainer (Eds.), *Differential Item Functioning* (pp. 67–113). Erlbaum.
- Varadhan, R., & Roland, C. (2008). Simple and globally convergent methods for accelerating the convergence of any EM algorithm. *Scand. J. Statist.*, 35, 335–353.
- Warm, T. A. (1989). Weighted likelihood estimation of ability in item response theory. *Psychometrika*, 54(3), 427–450.
- Woods, C. M. (2007). Empirical histograms in item response theory with ordinal data. *Educational and Psychological Measurement*, 67, 73–87.
- Woods, C. M., & Lin, N. (2009). Item response theory with estimation of the latent density using Davidian curves. *Applied Psychological Measurement*, 33, 102–117.
