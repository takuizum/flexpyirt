# flexpyirt 設計書（DESIGN）

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
- **Rasch（決定, M1）**: Rasch は $a_j=1$ 固定・$d_j$ 自由で、尺度は傾きで決まるので**潜在分布の分散を自由推定**し、
  平均は 0 に固定する（mirt の `itemtype="Rasch"` と同じ: `a1` 固定、`MEAN_1` 固定、`COV_11` 自由）。
  潜在分布を明示しない単一群モデルでは、**全項目が Rasch のとき既定を $N(0,\sigma^2)$（分散自由）**、
  それ以外は $N(0,1)$ 固定とする。
- 一般則（M1 で検査）: 分散を自由にするには傾きを固定した項目（値が 0 でない）が 1 つ以上、
  平均を自由にするには切片を固定した項目が 1 つ以上必要。
- 識別できない指定は `IdentificationError` を fit 前に投げる。

### 4.6 収束と加速
- 収束判定: 自由パラメタ（自然尺度の $\eta$）の最大絶対変化 $<$ `tol`（既定 1e-4、mirt の `TOL` と同じ定義）。
  対数尤度（MAP 時は対数事後）と最大変化の両方の履歴を保存し、単調性はテストで検査する。
  **決定（M1）**: 停止条件は最大絶対変化のみとする（対数尤度の変化量を条件に足すと mirt との対応が崩れるため）。
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
- **決定（M1, 2PL/Rasch）**: 正答率 $p_j$（$[0.01,0.99]$ に丸める）と項目−残余得点の点双列相関 $r_j$ から
  $a_j^{(0)}=\operatorname{clip}(1.7\,r_j/\sqrt{1-r_j^2},\,0.25,\,3)$（Rasch は 1）、
  $d_j^{(0)}=\operatorname{logit}(p_j)\sqrt{1+(a_j^{(0)}/1.7)^2}$（周辺確率を近似的に合わせるため）。
  分散の初期値は 1。初期値は収束点に影響しない前提で、照合テストもそれを確認する。
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
4. **依存**: pandas は必須依存（結果は DataFrame で返す）、polars は任意依存。
5. **SE**: $\eta$（自然尺度）に対する情報行列。Oakes は $Q(\eta\mid\eta')$ を
   `jax.jacfwd(jax.grad(Q, 0), argnums=(0, 1))` で微分。直接法は周辺対数尤度の `jax.hessian`。
   両者は恒等式として一致するので、内部整合テストは機械精度近く（相対 1e-6）で比較する。

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
