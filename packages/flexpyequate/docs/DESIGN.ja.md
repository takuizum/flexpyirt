# flexpyequate 設計（DESIGN）

[English](DESIGN.md) | 日本語

> 状態: ドラフト v0.2（2026-11-25）。マイルストーン E1〜E4 は実装し、R と照合済み（§7）。E5 以降は計画。
> 決めたことは「決定」と書く。

## 1. 目的と対象外

### 目的
- **得点分布**の等化。2 つのテスト版 X・Y の得点分布から、X 版の得点を Y 版の尺度に対応づける関数
  $e_Y(x)$ を求める。
- Kolen & Brennan (2014) の観測得点等化と、von Davier, Holland & Thayer (2004) のカーネル等化を扱う。
  どの手法も公刊された式から実装し、R（`equate`・`kequate`・`SNSequate`）と数値照合する。
- 軽い配布物にする。依存は NumPy・SciPy・pandas のみ。

### 対象外
- **項目パラメータが必要なもの**。IRT 真値等化・観測得点等化、尺度のリンキング、素点から尺度得点への
  換算表は flexpyirt に残す。flexpyequate は flexpyirt も JAX も import しない（issue #87 の決定）。
- 項目レベルのデータ。入力は常に得点分布（またはその元になる得点）。
- R パッケージの引数名やオブジェクト構造の再現。

### 原則（flexpyirt と共通）
1. 正しさ > 速さ > 機能の多さ。手法は R 参照値との照合テストが通って初めて「完成」とし、README には
   照合済みの手法だけを載せる。
2. 数式から実装する。コードは §4 の式から書き、R パッケージは参照値としてのみ使う。R パッケージが
   公刊された式と異なる場合は §7 に記録し、公刊された式を優先する。
3. 内部は配列に対する純粋関数、検証と DataFrame は公開層で扱う。

## 2. データ収集デザイン

| デザイン | データ | コンテナ |
|---|---|---|
| 等価群（EG） | 標本 1 から X、独立な標本 2 から Y | `ScoreDistribution` を 2 つ |
| 単一群（SG） | 同じ受検者から X と Y | `BivariateScoreDistribution` を 1 つ |
| カウンターバランス（CB） | 版の順序を入れ替えた 2 つの SG 標本 | `BivariateScoreDistribution` を 2 つ（E6） |
| アンカーテストつき非等価群（NEAT） | 母集団 P から (X, A)、母集団 Q から (Y, A) | `BivariateScoreDistribution` を 2 つ。アンカーは 2 番目の変数（線形手法は E4。E5〜E6 は計画） |

NEAT には 2 系統の手法があり、どちらも実装する計画である。**連鎖**等化（P 上で X → A、Q 上で A → Y）と、
**事後層化**（合成母集団 $T = wP + (1 - w)Q$。Tucker、Levine、Braun–Holland、頻度推定、カーネル等化の PSE）。

## 3. 記法

| 記号 | 意味 |
|---|---|
| $x_j,\ j = 1..J$; $y_k,\ k = 1..K$ | X・Y のとりうる得点（等間隔、刻み幅 $\delta$） |
| $n_j$, $N = \sum_j n_j$ | $x_j$ の度数、標本サイズ |
| $r_j = n_j / N$, $s_k$ | X・Y の得点確率 |
| $F$, $G$ | X・Y の離散累積分布 |
| $\mu_X, \sigma_X^2$ | X の平均と分散 |
| $e_Y(x)$ | 等化関数（X から Y の尺度へ） |

## 4. 手法

式の出典は著者・年・章で示す。式番号はまだ書いていない。書籍と突き合わせて確認してから追記する。

### 4.1 得点分布（E1）

`ScoreDistribution` は、等間隔のとりうる得点と、その度数（非負。整数でなくてもよい）を持つ。モーメントは
`ddof=1` を指定しない限り母集団形 $\sigma_X^2 = \sum_j r_j (x_j - \mu_X)^2$ を使う。

**パーセンタイル順位**（Kolen & Brennan, 2014, 2 章。ここでは割合で書く。公開メソッドは同書と同じ
0〜100 の尺度を使う）。離散得点は $[x_j - \delta/2,\, x_j + \delta/2)$ 上の一様分布とみなす。$x$ に最も
近いとりうる得点を $x^*$（中間は切り上げ）とすると、

$$P(x) = F(x^* - \delta) + \frac{x - (x^* - \delta/2)}{\delta}\,\bigl[F(x^*) - F(x^* - \delta)\bigr]$$

（$x_1 - \delta/2 \le x < x_J + \delta/2$）。それより下では $P = 0$、上では $P = 1$。

**パーセンタイル**（逆関数。Kolen & Brennan, 2014, 2 章）。割合 $p$ に対して、

$$P^{-1}_U(p) = x^*_U - \frac{\delta}{2} + \delta\,\frac{p - F(x^*_U - \delta)}{F(x^*_U) - F(x^*_U - \delta)},
\qquad x^*_U = \min\{x_j : F(x_j) > p\},$$

$$P^{-1}_L(p) = x^*_L + \frac{\delta}{2} + \delta\,\frac{p - F(x^*_L)}{F(x^*_L + \delta) - F(x^*_L)},
\qquad x^*_L = \max\{x_j : F(x_j) < p\},$$

ただし $P^{-1}_U(1) = x_J + \delta/2$、$P^{-1}_L(0) = x_1 - \delta/2$ とし、
$P^{-1}(p) = \tfrac12\,[P^{-1}_U(p) + P^{-1}_L(p)]$。両者が異なるのは、度数 0 が続いて $F$ が平らになる
ところだけで、そのとき平均は平らな区間の中点になる。

**決定**: 度数は整数でなくてもよい。事前平滑化した分布も同じコンテナで渡すため。

### 4.2 恒等・平均・線形等化（E1）

- 恒等: $e_Y(x) = x$。
- 平均（Kolen & Brennan, 2014, 2 章）: $e_Y(x) = x - \mu_X + \mu_Y$。
- 線形（2 章）: $e_Y(x) = \dfrac{\sigma_Y}{\sigma_X}(x - \mu_X) + \mu_Y$。

SG データでは 2 つの周辺分布に同じ式を使う（X と Y の従属性が効くのは標準誤差だけ。E7）。

**決定**: 既定は母集団分散（`ddof=0`）。Kolen & Brennan の式がそうであるため。`ddof=1` は $N - 1$ で
割る R `equate` を再現する（§7）。

### 4.3 等パーセンタイル等化（E1）

$$e_Y(x) = Q^{-1}\bigl(P(x)\bigr)$$

（Kolen & Brennan, 2014, 2 章）。$P$ は X のパーセンタイル順位関数、$Q^{-1}$ は Y のパーセンタイル関数
（§4.1）。等化後得点は $[y_1 - \delta_Y/2,\ y_K + \delta_Y/2]$ に入る。

### 4.4 対数線形モデルによる事前平滑化（E2）

得点度数に対する多項式対数線形モデル（Holland & Thayer, 2000）:

$$\log(N r_j) = \beta_0 + \sum_{i=1}^{C} \beta_i\, x_j^{\,i},$$

2 変量分布では

$$\log(N p_{jk}) = \beta_0 + \sum_{i=1}^{C_X} \beta_{Xi}\, x_j^{\,i} + \sum_{i=1}^{C_Y} \beta_{Yi}\, y_k^{\,i}
+ \sum_{(a,b)} \beta_{ab}\, x_j^{\,a} y_k^{\,b}.$$

最尤法（ポアソン尤度 = 多項尤度）をニュートン法で解く。あてはめた分布は、観測分布の 1〜$C$ 次の標本
モーメントを再現する（交差項を入れた場合は交差モーメント $E[X^aY^b]$ も）。これは単体テストにしている。

**決定**: あてはめの前に計画行列を正規直交化する（中心化・尺度調整したべき乗の QR 分解）。あてはめた
度数は基底によらない。一方、40〜100 点までの得点の素のべき乗を 6〜10 次まで使うと、ニュートン法には
条件が悪すぎる。そのため係数は素のべき乗の基底では報告しない。結果は、あてはめた分布、計画行列、
適合度統計量（ポアソン対数尤度、尤度比 $G^2$、Pearson $\chi^2$、Freeman–Tukey $\chi^2$、AIC、自由度）を
公開する。BIC は当面入れない。R の `BIC()` はポアソン `glm` のセル数を標本サイズとして使い、等化の文献は
受検者数を使うが、どちらを選ぶにしても照合できる参照値がないためである。

$\sqrt{N}(\hat r - r)$ の漸近共分散は、Holland & Thayer (1987) の「C 行列」を使って $C C^\top$ と書ける。
$B$ を計画行列（切片の列なし）とし、$M = (D_{\sqrt r} - \sqrt r\, r^\top) B = QR$ とすると、

$$C = D_{\sqrt r}\, Q, \qquad \operatorname{Cov}(\hat r) = \tfrac1N\, C C^\top .$$

事前平滑化をしない場合（飽和モデル）は $C = D_{\sqrt r} - r \sqrt r^\top$ で、多項分布の共分散
$D_r - r r^\top$ になる。

### 4.5 カーネル等化（EG は E3、他のデザインは E6）

5 つの段階からなる（von Davier, Holland & Thayer, 2004）。事前平滑化（§4.4）、デザイン関数による得点確率
$r, s$ の推定、連続化、等化、等化の標準誤差。

**連続化**（ガウスカーネル。4 章）。バンド幅 $h_X > 0$、$a_X^2 = \sigma_X^2 / (\sigma_X^2 + h_X^2)$ として、

$$F_{h}(x) = \sum_j r_j\, \Phi\bigl(R_{jX}(x)\bigr), \qquad
R_{jX}(x) = \frac{x - a_X x_j - (1 - a_X)\mu_X}{a_X h_X},$$

$$f_{h}(x) = \frac{1}{a_X h_X} \sum_j r_j\, \phi\bigl(R_{jX}(x)\bigr).$$

連続化した変数は、どの $h_X$ でも平均 $\mu_X$・分散 $\sigma_X^2$ を持つ。$h_X \to \infty$ でカーネル等化
関数は線形等化関数に近づく。

**バンド幅**（4 章）: 次を最小化する。

$$\mathrm{PEN}(h) = \sum_j \bigl(r_j - f_h(x_j)\bigr)^2 + K \sum_j A_j (1 - B_j),$$

ここで $f_h'(x_j - w) < 0$ なら $A_j = 1$、$f_h'(x_j + w) > 0$ なら $B_j = 0$（$w = 1/4$）。第 2 項は密度が
U 字になっている得点の数を数える。既定は $K = 1$。

**決定**: 最小点は、$h$ の格子探索（得点の刻み幅の 0.05〜10 倍を対数等間隔に 400 点）のあと、最良の格子点の
まわりで区間つきの 1 変数最小化を行って求める。ペナルティは一般に単峰でなく、$K > 0$ では不連続になる
ためである。

**決定**: ペナルティは、得点を刻み幅 $\delta$ を単位として測って評価する（同書は整数得点を前提にして
おり、そのとき密度と確率を比べられる）。バンド幅は得点の尺度で報告する。これにより、カーネル等化は
どちらの得点尺度の線形変換に対しても同変になる（単体テスト）。

**等化**（4 章）: $e_Y(x) = G_{h_Y}^{-1}\bigl(F_{h_X}(x)\bigr)$。区間で保護したニュートン反復で機械精度まで
解く。

**等化の標準誤差**（5 章）。$J_{e_Y} = (\partial e_Y/\partial r,\, \partial e_Y/\partial s)$ として、

$$\frac{\partial e_Y(x)}{\partial r_j} = \frac{1}{G'(e_Y(x))} \frac{\partial F_h(x)}{\partial r_j}, \qquad
\frac{\partial e_Y(x)}{\partial s_k} = -\frac{1}{G'(e_Y(x))} \frac{\partial G_h(y)}{\partial s_k}\Big|_{y = e_Y(x)},$$

$$\frac{\partial F_h(x)}{\partial r_j} = \Phi\bigl(R_{jX}(x)\bigr) - M_{jX}(x)\, f_h(x), \qquad
M_{jX}(x) = \tfrac12 (x - \mu_X)(1 - a_X^2)\Bigl(\frac{x_j - \mu_X}{\sigma_X}\Bigr)^2 + (1 - a_X)\, x_j,$$

EG で標本サイズ $N$・$M$ の独立な標本の場合、

$$\mathrm{SEE}_Y(x) = \sqrt{\tfrac1N \bigl\lVert \tfrac{\partial e_Y}{\partial r}^{\!\top} C_r \bigr\rVert^2
+ \tfrac1M \bigl\lVert \tfrac{\partial e_Y}{\partial s}^{\!\top} C_s \bigr\rVert^2 } .$$

微分の式は単体テストで有限差分と照合し、SEE はパラメトリック・ブートストラップと照合している。事前
平滑化をしない場合は多項分布の共分散を使う（$C = D_{\sqrt r} - r\sqrt r^\top$、§4.4）。

**相対誤差（PRE）**（4 章）: $\mathrm{PRE}(p) = 100\,[\mu_p(e_Y(X)) - \mu_p(Y)]/\mu_p(Y)$。$p$ 次の原点
モーメント（$p = 1..10$）で、$e_Y(X)$ は確率 $r_j$ で値 $e_Y(x_j)$ をとる。

**SEED**（E6）: 2 つの等化関数（たとえばカーネルと線形）の差の標準誤差
$\lVert (J_{e_1} - J_{e_2}) J_{DF} C \rVert$（5 章）。

### 4.6 NEAT の線形手法（E4）

母集団 1 が X とアンカー V を、母集団 2 が Y と V を受ける（Kolen & Brennan, 2014, 4 章）。各母集団を
1 つの `BivariateScoreDistribution` で表し、アンカーを 2 番目の変数にする。

**合成母集団** $T = w_1 P_1 + w_2 P_2$、$w_1 + w_2 = 1$。Tucker 法と Levine 観測得点法は $T$ の上で
線形等化を行う。$l_Y(x) = \dfrac{\sigma_s(Y)}{\sigma_s(X)}[x - \mu_s(X)] + \mu_s(Y)$ で、

$$\mu_s(X) = \mu_1(X) - w_2 \gamma_1 [\mu_1(V) - \mu_2(V)], \qquad
\mu_s(Y) = \mu_2(Y) + w_1 \gamma_2 [\mu_1(V) - \mu_2(V)],$$

$$\sigma_s^2(X) = \sigma_1^2(X) - w_2 \gamma_1^2 [\sigma_1^2(V) - \sigma_2^2(V)] + w_1 w_2 \gamma_1^2 [\mu_1(V) - \mu_2(V)]^2,$$

$$\sigma_s^2(Y) = \sigma_2^2(Y) + w_1 \gamma_2^2 [\sigma_1^2(V) - \sigma_2^2(V)] + w_1 w_2 \gamma_2^2 [\mu_1(V) - \mu_2(V)]^2 .$$

| 手法 | $\gamma_1$（$\gamma_2$ は母集団 2 の Y と V から同様に求める） |
|---|---|
| Tucker（X の V への回帰が線形で、両母集団で同じ） | $\sigma_1(X, V) / \sigma_1^2(V)$ |
| Levine、内部アンカー（古典的な同族モデル） | $\sigma_1^2(X) / \sigma_1(X, V)$ |
| Levine、外部アンカー | $[\sigma_1^2(X) + \sigma_1(X, V)] / [\sigma_1^2(V) + \sigma_1(X, V)]$ |

観測得点に適用する **Levine 真値**等化と、**連鎖線形**等化は合成母集団を使わない。

$$l_Y(x) = \frac{\gamma_2}{\gamma_1}[x - \mu_1(X)] + \mu_2(Y) + \gamma_2[\mu_1(V) - \mu_2(V)],$$

$$l_Y(x) = \mu_2(Y) + \frac{\sigma_2(Y)}{\sigma_2(V)}\Bigl[\mu_1(V) + \frac{\sigma_1(V)}{\sigma_1(X)}(x - \mu_1(X)) - \mu_2(V)\Bigr].$$

**Braun–Holland**: 頻度推定（事後層化）の分布の平均と標準偏差を使う線形等化（Kolen & Brennan, 2014,
5 章）。

$$f_s(x) = w_1 f_1(x) + w_2 \sum_v f_1(x \mid v)\, h_2(v), \qquad
g_s(y) = w_1 \sum_v g_2(y \mid v)\, h_1(v) + w_2\, g_2(y),$$

$h_1, h_2$ はアンカーの分布。同じ合成分布を E5 の頻度推定でも使う。

**決定**: 重みの既定は標本サイズに比例させる。$w_1 = N_1/(N_1 + N_2)$（`synthetic_weight=None`）。

**決定**: Levine の手法では `anchor="internal"` または `"external"` を明示させる。既定値は置かない。
選択を誤ると、警告なしに誤った答えになるためである。

**決定**: アンカー得点が片方の母集団にしか現れないとき、$f_1(x \mid v)$（または $g_2(y \mid v)$）は
推定できない。もう一方の母集団のその部分を除いて合成分布を正規化し直し、除いた確率を
`SparseDataWarning` で知らせる。同時分布を事前平滑化すればこの状況は起きない。

**決定**: `x` と `y` のとりうるアンカー得点は同じでなければならない。アンカーは両方の母集団が受ける
1 つのテストである。尺度の違うアンカーは、黙って比較せずにエラーにする。

**決定**: `ddof` は Tucker 法と Levine 観測得点法の分散・共分散に効く。$\gamma$、Levine 真値等化、連鎖線形
等化では打ち消し合う（1 つの母集団のモーメントの比であるため）。Braun–Holland 法は合成分布のモーメントを
使い、合成分布には固有の標本サイズがない。

### 4.7 NEAT の等パーセンタイル手法（E5、計画）

頻度推定（アンカーによる事後層化。Kolen & Brennan, 2014, 5 章）と連鎖等パーセンタイル等化。どちらも
2 変量分布の事前平滑化と組み合わせられるようにする。

## 5. 公開 API のスケッチ

```python
import flexpyequate as fe

x = fe.ScoreDistribution.from_scores(x_scores, score_range=(0, 40))
y = fe.ScoreDistribution.from_frequencies(scores=range(41), frequencies=y_counts)
xy = fe.BivariateScoreDistribution.from_scores(
    x_scores, y_scores, x_range=(0, 40), y_range=(0, 40)
)  # 単一群

result = fe.equate(x, y, method="equipercentile")  # EG
result = fe.equate(xy, method="linear")  # SG: 2 変量分布を 1 つ渡す
result.table  # DataFrame: score, equated
result.moments  # DataFrame: X・Y・等化後 X の mean, sd, skewness, kurtosis
result.transform(x_new)  # 任意の得点（整数でなくてもよい）を等化する

smooth_x = fe.presmooth(x, degree=6)  # LoglinearResult
smooth_xy = fe.presmooth(xy, degree=(4, 4), cross_terms=[(1, 1)])
smooth_x.distribution  # あてはめた度数を持つ ScoreDistribution
smooth_x.fit_statistics  # Series: likelihood_ratio, pearson, freeman_tukey, aic, ...
smooth_x.c_matrix  # Cov(r_hat) = C C' / N となる C

ke = fe.kernel_equate(smooth_x, smooth_y)  # ガウスカーネル、ペナルティによるバンド幅
ke = fe.kernel_equate(x, y, bandwidth_x=0.6, bandwidth_y=0.6)  # 観測分布のまま
ke.table  # DataFrame: score, equated, see
ke.continuized  # DataFrame: form, score, probability, cdf, density
ke.bandwidth_x, ke.bandwidth_y
ke.percent_relative_error
fe.penalty_bandwidth(smooth_x, penalty_weight=1.0)

# 非等価群: X 版とアンカー、Y 版とアンカー
xa = fe.BivariateScoreDistribution.from_scores(x_scores, x_anchor, x_range=(0, 36), y_range=(0, 12))
ya = fe.BivariateScoreDistribution.from_scores(y_scores, y_anchor, x_range=(0, 36), y_range=(0, 12))
tucker = fe.equate(xa, ya, method="tucker")  # w1 = N1 / (N1 + N2)
levine = fe.equate(xa, ya, method="levine_observed", anchor="internal", synthetic_weight=1.0)
tucker.gamma, tucker.synthetic_moments, tucker.synthetic_weight
```

- データより後ろの引数はすべて keyword-only。選択肢は小文字 snake_case の `Literal` 文字列:
  `method="identity" | "mean" | "linear" | "equipercentile"`（EG・SG）と
  `"tucker" | "levine_observed" | "levine_true" | "braun_holland" | "chained_linear"`（NEAT）。
- 結果は frozen dataclass（`EquatingResult`、`LoglinearResult`、`KernelEquatingResult`）。NumPy 配列
  （読み取り専用）を持ち、DataFrame はプロパティで返す。
- エラーは `flexpyequate.errors` のクラス: `ScoreDistributionError`（得点や度数が不正）、
  `SpecificationError`（引数が矛盾）、`ConvergenceWarning`、`SparseDataWarning`。
- 名前は Python の名前にする。役に立つところでは R の名前を docstring に書く（`ddof=1` ↔ R `equate` の
  分散、`penalty_weight` ↔ `kequate` の `KPEN`）。
- 計画（E5〜E6）: `equate(xa, ya, method="frequency_estimation" | "chained_equipercentile",
  synthetic_weight=...)`、`kernel_equate(..., design="single_group" | "counterbalanced" |
  "chained" | "post_stratification")`、`KernelEquatingResult.seed(other)`。

## 6. 構成

```
packages/flexpyequate/
  pyproject.toml        hatchling + hatch-vcs（バージョンはリポジトリのタグから。root = "../.."）
  README.md, README.ja.md
  docs/DESIGN.md, DESIGN.ja.md
  src/flexpyequate/
    distributions.py    ScoreDistribution, BivariateScoreDistribution
    equating.py         equate(), EquatingResult
    _neat.py            アンカーテストつき非等価群の式
    presmoothing.py     presmooth(), LoglinearResult
    kernel.py           kernel_equate(), KernelEquatingResult
    errors.py
  tests/                単体テスト、R 照合テスト、reference/*.json
validation/r/equate_*.R 参照値の生成スクリプト（リポジトリルート。CI では実行しない）
```

## 7. 検証計画

参照値は R で生成し、JSON として `tests/reference/` にコミットする。各ファイルに R とパッケージの
バージョンを記録する。CI では R を実行しない。再生成するにはリポジトリルートで次を実行する。

```bash
Rscript validation/r/equate_e1_reference.R
Rscript validation/r/equate_e2_reference.R
Rscript validation/r/equate_e3_reference.R
Rscript validation/r/equate_e4_reference.R
```

### ケース

| ID | 内容 | R の参照 | MS |
|---|---|---|---|
| Q1 | 平均・線形等化。EG（`ACTmath`、長さや刻み幅の異なるシミュレーションの版）と SG（2 変量シミュレーション） | `equate::equate(type = "mean", sx = 1, sy = 1)`、`equate(type = "linear")` | E1 |
| Q2 | パーセンタイル順位と等パーセンタイル等化。EG と SG。途中や両端に度数 0 がある分布、刻み幅が 1 でない得点を含む | `equate::px`、`equate::equate(type = "equipercentile")`、丸めをしない Kolen & Brennan 2 章の独立 R 実装 | E1 |
| Q3 | 対数線形モデルによる事前平滑化。1 変量（2〜8 次）、交差項あり・なしの 2 変量。平滑化後の分布の等パーセンタイル等化 | 厳しい収束判定の `stats::glm(family = poisson)`、`equate::presmoothing(smoothmethod = "loglinear")`、`equate(..., smoothmethod = "loglinear")` | E2 |
| Q4 | カーネル等化（EG）。バンド幅を与えたときの連続化 CDF、等化後得点、SEE、PRE。事前平滑化あり（3 データ）・なし（1 データ） | `kequate::kequate("EG", ...)`、`SNSequate::ker.eq`、厳しい求根精度と有限差分 SEE による独立 R 実装 | E3 |
| Q5 | ペナルティ関数の値と、ペナルティによるバンド幅（$K = 0$ と $K = 1$） | `kequate`（`KPEN = 0`）、`SNSequate::bandwidth`、独立 R 実装 | E3 |
| Q6 | Tucker、Levine 観測得点、Braun–Holland（それぞれ合成母集団の重み 4 通り）、Levine 真値、連鎖線形。内部アンカー（`KBneat`）と外部アンカー（シミュレーション。標本サイズと版の長さが異なる） | `equate::equate(type = "linear", method = "tucker" / "levine" / "braun/holland" / "chained", lts = )`、母集団モーメントで Kolen & Brennan の式を書き下した R 実装 | E4 |
| Q7 | 頻度推定、連鎖等パーセンタイル | `equate::equate(method = ...)` | E5 |
| Q8 | SG・CB・NEAT CE・NEAT PSE のカーネル等化、SEED | `kequate`、`SNSequate` | E6 |

### 許容誤差（断りがなければ最大絶対差）

| 量 | 参照 | 許容誤差 |
|---|---|---|
| モーメント、平均・線形等化の等化後得点、傾き、切片 | `equate`（`ddof=1`） | 1e-10 |
| パーセンタイル順位 | `equate::px` | 1e-12 |
| 等パーセンタイル等化の等化後得点 | 独立 R 実装 | 1e-10 |
| 等パーセンタイル等化の等化後得点（順位が 0 と 1 の間にあるもの） | `equate` | 1e-7 |
| NEAT の線形手法: 傾き、切片、$\gamma$、合成母集団の平均と標準偏差 | `equate`（`ddof=1`）、独立 R 実装（`ddof=0`） | 1e-10 |
| 頻度推定の合成確率 | 独立 R 実装 | 1e-13 |
| 事前平滑化後の度数（相対） | `glm`（`epsilon = 1e-13`） | 1e-8 |
| 事前平滑化後の度数（相対） | `equate::presmoothing`（`glm` の既定の収束判定） | 1e-5 |
| $G^2$、Pearson・Freeman–Tukey の $\chi^2$、対数尤度、AIC（相対） | `glm` | 1e-7 |
| 事前平滑化した分布の等パーセンタイル等化 | `equate` | 1e-5 |
| あてはめた確率の共分散 $CC^\top/N$（最大要素に対する相対） | `kequate` | 1e-9 |
| カーネルの CDF $F_h$・$G_h$、密度 | `kequate`、独立 R 実装 | 1e-12 |
| カーネル等化の等化後得点 | 独立 R 実装 | 1e-9 |
| カーネル等化の等化後得点 | 分散の係数を除いた `kequate` | 1e-6 |
| SEE（相対） | 分散の係数を除いた `kequate`、独立実装の有限差分 SEE | 1e-6 |
| PRE | 分散の係数を除いた `kequate` | 1e-5 |
| ペナルティの値（相対） | 独立 R 実装 | 1e-10 |
| バンド幅（$K = 0$） | 分散の係数を除いた `kequate` | 1e-6 |
| バンド幅（$K = 0$ と $K = 1$） | 独立 R 実装 | 1e-6 |
| バンド幅（$K = 0$ と $K = 1$） | `SNSequate` | 1e-3 |
| カーネル等化の等化後得点 / SEE（相対） | `SNSequate::ker.eq` | 1e-4 / 1e-4 |
| カーネル等化の等化後得点 / SEE（相対） / バンド幅 | 通常の `kequate` 呼び出し | 5e-3 / 1e-3 / 1e-4 |

実際の差はこれより十分小さい。`kequate` に対して SEE は 1e-8 程度、等化後得点は 2e-7 程度（後者は
`kequate` の求根精度による）。

### R パッケージとの既知の違い

- **`equate` は分散を $N - 1$ で割る**。Kolen & Brennan は $N$ を使う。`ddof=1` で `equate` を再現できる。
  既定は `ddof=0`。
- **`equate` の `type = "mean"`** は傾きが（Y の得点範囲）/（X の得点範囲）で、2 つの版の得点範囲が同じ
  ときだけ 1 になる。flexpyequate の平均等化は傾き 1（Kolen & Brennan の 2 章）。参照値は `sx = 1, sy = 1`
  を指定して生成している。
- **`equate` の Braun–Holland（および頻度推定）の合成分布**は、2 つの母集団を度数のまま混ぜる
  （$w\,n_1(x, v) + (1 - w)\,n_1(x, v)\,n_2(v)/n_1(v)$）。そのため母集団 1 の実質的な重みは $w$ でなく
  $wN_1/(wN_1 + (1 - w)N_2)$ になる（既定の $w$ では $N_1^2/(N_1^2 + N_2^2)$）。flexpyequate は Kolen &
  Brennan と同じく確率を混ぜる。実質的な重みを `synthetic_weight` に渡せば `equate` を再現できる
  （$N_1 = 1200$、$N_2 = 2000$ の例で切片の差は 0.02）。また `equate` の合成分布の標準偏差は総度数 − 1 で
  割るが、傾きでは打ち消し合う。
- **`equate` は丸める**。等パーセンタイルの処理の中で、パーセンタイル順位と累積割合を小数 10 桁に丸める。
  丸め誤差（最大 5e-11）が相対度数で割られるので、一致は 1e-8 程度までになる。そのため、厳しい照合には
  丸めをしない独立 R 実装を使う。
- **パーセンタイル順位がちょうど 0 または 1 のとき**、`equate` は Y の得点範囲の端
  （$y_1 - \delta/2$ または $y_K + \delta/2$）を返す。ここで採用している Kolen & Brennan の定義は上側と
  下側のパーセンタイルの平均で、Y のその端に観測がないときに値が異なる（ACT 数学の $x = 0$ では −0.5 で
  なく 0.0）。`equate` との比較ではこれらの得点を除き、独立実装との比較で確認している。
- **`kequate` は連続化の分散に $N/(N - 1)$ を掛ける**（$a_X^2 = \sigma_X^2/(\sigma_X^2 + h_X^2)$）。一方で
  微分の式は von Davier らのもの（$\sigma_X^2 = \sum_j r_j (x_j - \mu_X)^2$ を前提にする）をそのまま使って
  いる。flexpyequate は同書に従う（`SNSequate` も同じ）。厳しい照合では、`kequate` に確率ベクトルを渡し、
  標本サイズを $10^{8}$ 倍して係数の影響を無視できるようにし、SEE を $10^{4}$ 倍して戻す（SEE は
  $N^{-1/2}$ に正確に比例する）。もう 1 つの照合では通常の `kequate` 呼び出しを使い、この係数の影響を
  含む許容誤差にしている（$N \approx 600$、バンド幅 2 の例で等化後得点が最大 1.2e-3 動いた）。
- **`kequate` は $G_h(y) = F_h(x)$ を `uniroot(tol = 1e-6)` で解く**。そのため等化後得点の一致は 1e-6 程度
  までになる。
- **`kequate` のペナルティ第 2 項**は、密度が極大になる得点（左で $f' > 0$、右で $f' < 0$）も数える。
  同書の 4 章および `SNSequate` とは異なる。flexpyequate は同書に従う。$K = 0$（`kequate` の既定）では
  この項は入らない。flexpyequate の既定は同書と同じ $K = 1$。
- **`kequate` と `SNSequate` は得点が 0, 1, 2, … であることを前提にする**。それ以外の得点尺度は同変性の
  単体テストだけで確認している。
- **`SNSequate::ker.eq`** は等化後得点を $(-1, 101)$ の範囲で探し、−1 を下回る得点があると失敗する
  （試した 2 ケースのうち 1 つ）。バンド幅は 0.65 から始めた `optim()` で求めている。
- 独立実装の SEE では、得点確率が 5e-6 程度のところで、刻み 1e-6 の 2 点差分は精度が足りない（誤差
  1.4e-3）。生成スクリプトは刻み 1e-7 の 5 点差分を使う。

## 参考文献

- Holland, P. W., & Thayer, D. T. (1987). *Notes on the use of log-linear models for fitting
  discrete probability distributions* (ETS RR-87-31).
- Holland, P. W., & Thayer, D. T. (2000). Univariate and bivariate loglinear models for discrete
  test score distributions. *Journal of Educational and Behavioral Statistics, 25*, 133–183.
- Kolen, M. J., & Brennan, R. L. (2014). *Test equating, scaling, and linking* (3rd ed.). Springer.
- von Davier, A. A., Holland, P. W., & Thayer, D. T. (2004). *The kernel method of test equating*.
  Springer.
- Albano, A. D. (2016). equate: An R package for observed-score linking and equating. *Journal of
  Statistical Software, 74*(8).
- Andersson, B., Bränberg, K., & Wiberg, M. (2013). Performing the kernel method of test equating
  with the package kequate. *Journal of Statistical Software, 55*(6).
- González, J. (2014). SNSequate: Standard and nonstandard statistical models and methods for test
  equating. *Journal of Statistical Software, 59*(7).
