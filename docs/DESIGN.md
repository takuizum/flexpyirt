# flexpyirt 設計書（DESIGN）

> 状態: ドラフト v0.1（2026-11-20）。実装前の設計。決定事項は「決定」、未確定は §12「未決事項」に置く。

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
  $\mu_g=\frac{1}{N_g}\sum_q\bar n_{gq}\theta_q$、$\sigma_g^2=\frac{1}{N_g}\sum_q\bar n_{gq}(\theta_q-\mu_g)^2$（閉形式）。
- **Empirical Histogram (EH)**: $w_{gq}=\bar n_{gq}/N_g$。
  識別のため参照群 EH は各反復で平均 0・分散 1 に標準化する（Woods, 2007）。標準化の方法は §12 未決事項 1。
- 将来: Davidian 曲線（Woods & Lin, 2009）、混合正規。いずれも `LatentDist` プロトコル（§6）に従えば Core に手を入れずに追加できる。

### 4.5 識別性ルール（Compile 層で検査）
- 既定: 参照群は $N(0,1)$ 固定、焦点群の平均・分散は自由。焦点群を自由にするには、群間で全パラメタが等しいアンカー項目が必要
  （2PL なら最低 1 項目で理論上は識別、実務的には複数を推奨し、少ない場合は警告）。
- **FIPC**: 固定項目が尺度を決めるので、全群の平均・分散を自由にしてよい。固定項目が 0 なら参照群を固定。
  潜在分布を $N(0,1)$ に固定したまま固定項目以外を推定すると尺度がずれる（Kim, 2006）ので、
  FIPC では潜在分布（Normal の平均・分散、または EH の重み）を各 EM 反復で更新する方式（Kim の MWU-MEM に相当）を既定にする。
- 識別できない指定は `IdentificationError` を fit 前に投げる。

### 4.6 収束と加速
- 収束判定: 自由パラメタの最大絶対変化 $<$ `tol`（既定 1e-4、mirt と同じ思想）かつ対数尤度（MAP 時は対数事後）の変化が小さいこと。両方の履歴を保存。
- **SQUAREM**（Varadhan & Roland, 2008, スキーム S3）をオプションで。外挿後に目的関数が減ったら通常の EM ステップに戻す（単調性の安全装置）。
- EM ループは `lax.while_loop` でデバイス上に置く。進捗表示は `jax.debug.callback` を任意で。

### 4.7 MAP（Bayes modal）
- 事前分布は**自然なパラメタ尺度**（$a, b$ または $a, d$, $c$ など）で定義し、Jacobian 項は入れない（自然尺度での事後モード。mirt の Bayes modal と同じ）。
- 内部の最適化を変換後の尺度（例: $c$ を logit）で行っても、目的関数自体は自然尺度の事前分布で評価する。
- 結果には `loglik`（尤度のみ）と `log_posterior` を分けて返す。AIC/BIC は事前分布がある場合に警告付き。

### 4.8 標準誤差
- **決定**: 既定は Oakes (1999) の恒等式
  $$I(\hat\eta)=-\Big[\tfrac{\partial^2 Q(\eta\mid\eta')}{\partial\eta\,\partial\eta^\top}+\tfrac{\partial^2 Q(\eta\mid\eta')}{\partial\eta\,\partial\eta'^\top}\Big]_{\eta=\eta'=\hat\eta}$$
  を `jax.hessian` と `jax.jacfwd(jax.grad(...))` で計算。$Q$ は「E-step の重みを $\eta'$ で計算し、尤度を $\eta$ で評価する」2 引数関数として実装する。
- 代替: 直接 `jax.hessian` で周辺対数尤度の観測情報（小規模なら最も素直。Oakes の検証にも使う）、cross-product（$\sum_p c_p s_p s_p^\top$）、sandwich。
- MAP 時は事前分布の Hessian を加える。固定パラメタは $\eta$ に含まれないので自動的に除外。

### 4.9 求積
- **決定**: 1 次元の既定は等間隔 61 点、$[-6,6]$（EH と同じ格子を共有でき、Normal でも重みを密度から作る）。
- Normal 専用に Gauss–Hermite も選べる（`Quadrature.gauss_hermite(n)`）。
- 多次元（M4 以降）は直積格子、次元 3 まで。それ以上は対象外（非目的）。

### 4.10 初期値
- $d_j$: 正答率の logit、$a_j$: 項目−合計の点双列相関から変換（mirt と同様の古典的初期値）。GRM の閾値はカテゴリ累積比率の logit。
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
| `prior` | `Prior \| None` | 事前分布オブジェクト |
| `free_id` | int \| -1 | 自由パラメタ番号。**等値制約は同じ `free_id` を共有することで表す** |

- ユーザーは `spec.parameter_table(data)` で表を取り出し、pandas で編集して `fit(..., table=edited)` に渡せる（mirt の pars 編集ワークフローを残す）。
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
| 3PL / 4PL | $a,d,g,u$（$g,u$ は logit 尺度で最適化） | M3 |
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
- `HeywoodWarning`: 境界解（$g\to0$, $a$ 発散など）

## 11. テスト方針（詳細は VALIDATION.md）
- 単体: 各 `log_prob` の確率和 = 1、勾配の有限差分チェック、E-step の度数保存（$\sum_q\bar n_{gq}=N_g$）。
- 性質: hypothesis でランダムなパラメタ・データを生成し、EM の単調増加を検査。
- 回復: シミュレーションでのパラメタ回復（バイアス・RMSE）。
- 照合: R mirt の推定値・SE・対数尤度と許容誤差内で一致。

## 12. 未決事項
1. **EH の識別の取り方**: (a) 毎反復で重みの平均・分散を 0/1 に合わせて節点をアフィン変換し項目パラメタを逆変換、(b) 参照群の EH を標準化した後で補間し直す、のどちらか。mirt の挙動を確認したうえで決める。
2. 多母集団で群の数が多いとき（例: 学校 × 学年）の扱い。群ごとの潜在分布を vmap できる形にしておく。
3. 欠測の設計（MAR 前提の除外のみで十分か、not-reached の扱いを入れるか）。
4. ~~パッケージ名~~ → `flexpyirt` に決定（2026-11-20）。
5. 項目をまたぐ等値制約の M-step（§4.3）の実装方式。

## 参考文献
- Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of item parameters: Application of an EM algorithm. *Psychometrika*, 46, 443–459.
- Bock, R. D., & Lieberman, M. (1970). Fitting a response model for n dichotomously scored items. *Psychometrika*, 35, 179–197.
- Chalmers, R. P. (2012). mirt: A multidimensional item response theory package for the R environment. *Journal of Statistical Software*, 48(6).
- Louis, T. A. (1982). Finding the observed information matrix when using the EM algorithm. *JRSS B*, 44, 226–233.
- Oakes, D. (1999). Direct calculation of the information matrix via the EM algorithm. *JRSS B*, 61, 479–482.
- Varadhan, R., & Roland, C. (2008). Simple and globally convergent methods for accelerating the convergence of any EM algorithm. *Scand. J. Statist.*, 35, 335–353.
- Woods, C. M. (2007). Empirical histograms in item response theory with ordinal data. *Educational and Psychological Measurement*, 67, 73–87.
- Woods, C. M., & Lin, N. (2009). Item response theory with estimation of the latent density using Davidian curves. *Applied Psychological Measurement*, 33, 102–117.
- Kim, S. (2006). A comparative study of IRT fixed parameter calibration methods. *Journal of Educational Measurement*, 43, 355–381.
