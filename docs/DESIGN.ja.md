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
- **項目をまたぐ等値制約**（例: 全項目共通の $a$）は分解を壊す。v1 では非対応とし、M5 で「制約で連結された項目成分ごとに同時 Newton」を実装する。

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
- **Empirical Histogram (EH)**: $w_{gq}=\bar n_{gq}/N_g$。
  識別のため参照群 EH は各反復で平均 0・分散 1 に標準化する（Woods, 2007）。標準化の方法は §12 未決事項 1。
- 将来: Davidian 曲線（Woods & Lin, 2009）、混合正規。いずれも `LatentDist` プロトコル（§6）に従えば Core に手を入れずに追加できる。

### 4.5 識別性ルール（Compile 層で検査）
- 既定: 参照群は $N(0,1)$ 固定、焦点群の平均・分散は自由。焦点群を自由にするには、群間で全パラメタが等しいアンカー項目が必要
  （2PL なら最低 1 項目で理論上は識別、実務的には複数を推奨し、少ない場合は警告）。
- **FIPC**: 固定項目が尺度を決めるので、全群の平均・分散を自由にしてよい。固定項目が 0 なら参照群を固定。
  潜在分布を $N(0,1)$ に固定したまま固定項目以外を推定すると尺度がずれる（Kim, 2006）ので、
  FIPC では潜在分布（Normal の平均・分散、または EH の重み）を各 EM 反復で更新する方式（Kim の MWU-MEM に相当）を既定にする。
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
  パターン分割カテゴリ共起構造を活用する（§15.18）。
- 代替: 基底ベクトルをチャンク分割した前向き×後ろ向き自動微分
  （`jax.linearize(jax.grad(...))`）による観測（事後）情報行列
  $-\nabla^2_\eta[\ell(\eta)+\log\pi(\eta)]$（Oakes の検証にも使う）、
  cross-product（$\sum_p c_p s_p s_p^\top$）、sandwich。
- MAP 時は `oakes`・`hessian` とも情報行列に $-\nabla^2_\eta\log\pi(\hat\eta)$ を加え、数学的恒等性を保つ。
  固定パラメタは $\eta$ に含まれないので自動的に除外される。

### 4.9 求積
- **決定**: 1 次元の既定は等間隔 61 点、$[-6,6]$（EH と同じ格子を共有でき、Normal でも重みを密度から作る）。
- Normal 専用に Gauss–Hermite も選べる（`Quadrature.gauss_hermite(n)`）。
- 多次元（M4 以降）は直積格子、次元 3 まで。それ以上は対象外（非目的）。

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
    nodes: Array  # (Q, D)
    families: tuple[FamilyBlock, ...]  # 項目モデルごとの塊（静的）
    free_map: FreeMap
    latent: tuple[LatentBlock, ...]  # 群ごとの潜在分布（静的な種類 + 動的なパラメタ）
```
`families` と潜在分布の「種類」は静的（jit の static）、パラメタ値は動的。

## 6. 拡張プロトコル

```python
class ItemModel(Protocol):
    name: ClassVar[str]
    param_names: tuple[str, ...]  # 内部パラメタ名

    def log_prob(self, theta: Array, params: Array) -> Array:
        ...
        # theta (Q, D), params (n_params,) -> (Q, K) 各カテゴリの対数確率

    def init(self, y: Array) -> Array: ...  # 古典的初期値
    def to_irt(self, params: Array) -> dict[str, Array]: ...  # a,b 表示などへの変換（表示専用）
    def bounds(self) -> tuple[Array, Array]: ...


class LatentDist(Protocol):
    def log_weights(self, nodes: Array, params: Array) -> Array: ...  # (Q,)
    def m_step(self, nodes: Array, n_bar: Array, params: Array) -> Array: ...
    def n_params(self) -> int: ...
```
新しい項目モデルは `log_prob` を書くだけで、E-step・M-step（自動微分 Newton）・SE・スコアリングが全部動くことを設計目標にする。

## 7. 対応する項目モデル（段階的）
| モデル | 内部パラメタ | マイルストーン |
|---|---|---|
| Rasch / 2PL | $a, d$（$P=\sigma(a\theta+d)$） | M1 |
| 3PL / 4PL | $a, d, g, u$（自然尺度、$0<g<u\le 1$、§15.2） | M3 |
| GRM | $a, d_1>\dots>d_{K-1}$ | M2 |
| GPCM / PCM | $a, d_1..d_{K-1}$ | M3 |
| Nominal | M5 以降（未定） | — |

表示用の IRT パラメタ化（$b=-d/a$ など）は Result 層で変換し、SE はデルタ法で出す。

## 8. スコアリング
`FitResult.score(data, method="EAP"|"MAP"|"ML"|"WLE"|"PV", group=...)`。群ごとの推定済み潜在分布を事前分布として使う。EAP の事後 SD も返す。

## 9. 性能方針
- パターン圧縮 + バケットパディングで再コンパイルを抑える。
- item family ごとの `vmap`。異種項目混在でも family 数（静的）だけのループ。
- 既定は CPU で十分速いことを目標。GPU は大規模（N ≥ 10^5, J ≥ 100）で効けばよい。
- ベンチマーク: R mirt（`TOL=1e-4`, 既定設定）との wall time 比較を `benchmarks/` に置く（M3）。

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
1. **EH の識別の取り方**: (a) 毎反復で重みの平均・分散を 0/1 に合わせて節点をアフィン変換し項目パラメタを逆変換、(b) 参照群の EH を標準化した後で補間し直す、のどちらか。mirt の挙動を確認したうえで決める。
2. 多母集団で群の数が多いとき（例: 学校 × 学年）の扱い。群ごとの潜在分布を vmap できる形にしておく。
3. ~~欠測の設計~~ → MAR 前提の除外のみ（§14.8、2026-11-20）。
4. ~~パッケージ名~~ → `flexpyirt` に決定（2026-11-20）。
5. 項目をまたぐ等値制約の M-step（§4.3）の実装方式。

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
18. **構造化 Oakes とチャンク分割 Hessian による高速化・省メモリ化（M3, §4.8, §9）**:
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

## 参考文献
- Barton, M. A., & Lord, F. M. (1981). An upper asymptote for the three-parameter logistic item-response model. *ETS Research Report Series*, RR-81-20.
- Bertsekas, D. P. (1982). Projected Newton methods for optimization problems with simple constraints. *SIAM Journal on Control and Optimization*, 20(2), 221–246.
- Birnbaum, A. (1968). Some latent trait models and their use in inferring an examinee's ability. In F. M. Lord & M. R. Novick, *Statistical Theories of Mental Test Scores* (pp. 397–479). Addison-Wesley.
- Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of item parameters: Application of an EM algorithm. *Psychometrika*, 46, 443–459.
- Bock, R. D., & Lieberman, M. (1970). Fitting a response model for n dichotomously scored items. *Psychometrika*, 35, 179–197.
- Chalmers, R. P. (2012). mirt: A multidimensional item response theory package for the R environment. *Journal of Statistical Software*, 48(6).
- Louis, T. A. (1982). Finding the observed information matrix when using the EM algorithm. *JRSS B*, 44, 226–233.
- Masters, G. N. (1982). A Rasch model for partial credit scoring. *Psychometrika*, 47, 149–174.
- Mislevy, R. J. (1986). Bayes modal estimation in item response models. *Psychometrika*, 51, 177–195.
- Muraki, E. (1992). A generalized partial credit model: Application of an EM algorithm. *Applied Psychological Measurement*, 16, 159–176.
- Oakes, D. (1999). Direct calculation of the information matrix via the EM algorithm. *JRSS B*, 61, 479–482.
- Samejima, F. (1969). Estimation of latent ability using a response pattern of graded scores. *Psychometrika Monograph*, 17.
- Swaminathan, H., & Gifford, J. A. (1986). Bayesian estimation in the three-parameter logistic model. *Psychometrika*, 51, 589–601.
- Varadhan, R., & Roland, C. (2008). Simple and globally convergent methods for accelerating the convergence of any EM algorithm. *Scand. J. Statist.*, 35, 335–353.
- Woods, C. M. (2007). Empirical histograms in item response theory with ordinal data. *Educational and Psychological Measurement*, 67, 73–87.
- Woods, C. M., & Lin, N. (2009). Item response theory with estimation of the latent density using Davidian curves. *Applied Psychological Measurement*, 33, 102–117.
- Kim, S. (2006). A comparative study of IRT fixed parameter calibration methods. *Journal of Educational Measurement*, 43, 355–381.
