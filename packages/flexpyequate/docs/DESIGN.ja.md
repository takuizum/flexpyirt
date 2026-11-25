# flexpyequate 設計（DESIGN）

[English](DESIGN.md) | 日本語

> 状態: ドラフト v0.5（2026-11-25）。マイルストーン E1〜E7 は実装し、R と照合済み（§7）。
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
| カウンターバランス（CB） | 版の順序を入れ替えた 2 つの SG 標本 | (X, Y) の `BivariateScoreDistribution` を 2 つ。X を先に受けた標本、Y を先に受けた標本の順（カーネル等化、E6） |
| アンカーテストつき非等価群（NEAT） | 母集団 P から (X, A)、母集団 Q から (Y, A) | `BivariateScoreDistribution` を 2 つ。アンカーは 2 番目の変数（E4・E5。カーネル等化 E6） |

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

SG データでは 2 つの周辺分布に同じ式を使う（X と Y の従属性が効くのは標準誤差だけ。§4.8）。

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

### 4.5 カーネル等化（EG は E3、他のデザインと SEED は E6）

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

**デザイン関数**（E6。同書 2・3 章）。どのデザインでも、$r$ と $s$ は推定した同時確率の関数として与えられる。
SEE は $\lVert J_{e_Y} J_{DF}\, C \rVert$ で、$J_{DF}$ はデザイン関数のヤコビ行列、$C C^\top$ は独立な標本ごとの
推定確率の共分散（ブロックを横に並べる）。

| デザイン | $r$, $s$ |
|---|---|
| EG | 2 組の得点確率そのもの |
| SG、(X, Y) の同時分布 $P$ | $r_j = \sum_k p_{jk}$、$s_k = \sum_j p_{jk}$ |
| CB、同時分布 $P_{12}$（X が先）と $P_{21}$（Y が先）、周辺を $r_1, s_1$、$r_2, s_2$ とする | $r = w_X r_1 + (1 - w_X) r_2$、$s = w_Y s_1 + (1 - w_Y) s_2$ |
| NEAT PSE、(X, A) の $P$ と (Y, A) の $Q$、アンカーの周辺 $t_P$・$t_Q$、$T = wP + (1 - w)Q$ | $r_j = \sum_l \bigl[w + (1 - w)\tfrac{t_{Ql}}{t_{Pl}}\bigr] p_{jl}$、$s_k = \sum_l \bigl[(1 - w) + w\tfrac{t_{Pl}}{t_{Ql}}\bigr] q_{kl}$ |

PSE では

$$dr_j = \sum_l \Bigl[w + (1 - w)\frac{t_{Ql}}{t_{Pl}}\Bigr] dp_{jl}
- (1 - w)\sum_l \frac{p_{jl}\, t_{Ql}}{t_{Pl}^2}\, dt_{Pl} + (1 - w)\sum_l \frac{p_{jl}}{t_{Pl}}\, dt_{Ql},$$

$s$ も対称な形になる。

**連鎖等化（NEAT CE）**は 4 つの分布を連続化する。P での X と A（$F_P$・$H_P$）、Q での A と Y
（$H_Q$・$G_Q$）で、それぞれ別のバンド幅を持つ。

$$e_Y(x) = G_Q^{-1}\Bigl(H_Q\bigl(H_P^{-1}(F_P(x))\bigr)\Bigr).$$

$a = H_P^{-1}(F_P(x))$ とすると、連鎖律から独立な 2 つの部分が得られる。

$$\mathrm{SEE}_Y(x)^2 =
\Bigl\lVert \frac{H_Q'(a)}{G_Q'(e_Y(x))}\,\frac{1}{H_P'(a)}
\Bigl(\frac{\partial F_P(x)}{\partial r_P} J_r - \frac{\partial H_P(a)}{\partial t_P} J_t\Bigr) C_P \Bigr\rVert^2
+ \Bigl\lVert \frac{1}{G_Q'(e_Y(x))}
\Bigl(\frac{\partial H_Q(a)}{\partial t_Q} J_t - \frac{\partial G_Q(e_Y(x))}{\partial s_Q} J_s\Bigr) C_Q \Bigr\rVert^2 ,$$

$J_r$・$J_t$ はセル確率をアンカー方向・合計得点方向に足し上げる行列。相対誤差（PRE）は 2 つのリンク
（P での X → A、Q での A → Y）ごとに報告する。

**SEED**（E6）: 同じデータから推定した 2 つの等化関数の差の標準誤差は $\lVert v_1(x) - v_2(x) \rVert$。
$v(x) = J_{e}(x) J_{DF} C$ は、ノルムが SEE になるベクトル（5 章）。`KernelEquatingResult` はこのベクトルを
保持し（`see_vectors`）、`seed(other)` がその差をとる。カーネルと線形（`linear=True`。`kequate` と同じく
標準偏差の 1000 倍のバンド幅）、2 通りのバンド幅、連鎖と事後層化の比較などに使う。

**決定**: 同時分布を 2 つ渡す場合はデザインを指定させる（`design="counterbalanced"`・`"chained"`・
`"post_stratification"`）。この 3 つはデータからは区別できない。等価群と単一群は入力から判断する。

**決定**: PSE の重みの既定は §4.6 の手法と同じ $w = N_P/(N_P + N_Q)$（`kequate` の既定は 0.5）。CB の
重みの既定は $(1/2, 1/2)$。

**決定**: カーネル PSE では、アンカー得点の確率が片方の母集団でだけ 0 のときはエラーにする（§4.6 のように
除外はしない）。そこではデザイン関数のヤコビ行列が定義できず、カーネル法は事前平滑化を前提にしている
ためである。

**決定**: `seed()` が確認するのは確認できることだけ（デザインの系統、X の得点、`see_vectors` の列数が
同じこと）。列は各 C 行列の QR 分解に依存するので、2 つの結果は同じ事前平滑化の結果から作る必要がある。
これは文書に書くだけで、強制はしない。

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

### 4.7 NEAT の等パーセンタイル手法（E5）

Kolen & Brennan (2014, 5 章)。どちらの手法も §4.1 のパーセンタイル順位関数とパーセンタイル関数を使う。

**頻度推定**（アンカーによる事後層化）: §4.6 の合成分布 $f_s$・$g_s$ の等パーセンタイル等化。

$$e_Y(x) = Q_s^{-1}\bigl(P_s(x)\bigr).$$

**連鎖等パーセンタイル**等化: 母集団 1 で X をアンカーに、母集団 2 でアンカーを Y に等化する。

$$e_Y(x) = Q_2^{-1}\Bigl(H_2\bigl(H_1^{-1}(P_1(x))\bigr)\Bigr),$$

$P_1$・$H_1$ は母集団 1 での X・V のパーセンタイル順位関数、$H_2$・$Q_2$ は母集団 2 での V・Y のもの。
$H_2$ は、整数でないアンカー上の等化値に対して、§4.1 の連続なパーセンタイル順位関数で評価する。

事前平滑化は `equate()` のオプションにしない。事前平滑化した同時分布
（`presmooth(xa, degree=(…, …), cross_terms=…).distribution`）を渡す。

**決定**: 頻度推定では `EquatingResult.x`・`.y` を合成分布（総度数 $N_1 + N_2$）にする。等化が定義されるのが
その母集団だからで、`moments` も合成母集団を表す。連鎖等化では母集団 1 の X と母集団 2 の Y とし、2 つの
アンカー分布を `anchor_distributions` に保持して、`transform()` で任意の得点を等化できるようにする。

### 4.8 従来手法の標準誤差（E7）

`equate(..., standard_error=...)` は X の取りうる得点での等化の標準誤差を返す
（`EquatingResult.see`、`table` の `see` 列）。

**デルタ法**（`"analytic"`。Kolen, 1985; Kolen & Brennan, 2014, 7 章; Zu & Yuan, 2012）。
Braun–Holland 以外の線形手法は、標本の平均・分散・共分散 $\theta$ の関数
$e_Y(x) = c(\theta) + b(\theta)x$ なので、

$$\mathrm{SE}[\hat e_Y(x)]^2 = g(x)^\top \Omega\, g(x), \qquad g(x) = \nabla c + x\,\nabla b,
\qquad \Omega = \operatorname{Cov}(\hat\theta),$$

$\Omega$ は独立な標本ごとのブロックからなる。大きさ $N$ の標本の得点 $A, B, \dots$ と中心モーメント
$\mu$ について、

$$N\operatorname{Cov}(\bar A, \bar B) = \sigma_{AB}, \qquad
N\operatorname{Cov}(\bar A, s_{BC}) = \mu_{ABC}, \qquad
N\operatorname{Cov}(s_{AB}, s_{CD}) = \mu_{ABCD} - \sigma_{AB}\sigma_{CD},$$

これはセル比率の多項分布の共分散をモーメントに移したもので、分布の仮定を置かない。EG の線形等化では
Braun & Holland (1982) の式になる。

$$\mathrm{SE}^2 = \sigma_Y^2\Bigl\{\frac{1}{N_X} + \frac{1}{N_Y}
+ \Bigl[\frac{\mathrm{sk}(X)}{N_X} + \frac{\mathrm{sk}(Y)}{N_Y}\Bigr] z
+ \Bigl[\frac{\mathrm{ku}(X) - 1}{4N_X} + \frac{\mathrm{ku}(Y) - 1}{4N_Y}\Bigr] z^2\Bigr\},
\qquad z = \frac{x - \mu_X}{\sigma_X}.$$

勾配: 平均・線形等化では $\partial b/\partial\sigma_X^2 = -b/(2\sigma_X^2)$、
$\partial b/\partial\sigma_Y^2 = b/(2\sigma_Y^2)$、
$\nabla c = \nabla\mu_Y - b\nabla\mu_X - \mu_X\nabla b$（平均等化は $b = 1$）。Tucker 法と Levine
観測得点法では、$\Delta_\mu = \mu_1(V) - \mu_2(V)$、$\Delta_\sigma = \sigma_1^2(V) - \sigma_2^2(V)$
として（§4.6）、

$$d\mu_s(X) = d\mu_1(X) - w_2\Delta_\mu\, d\gamma_1 - w_2\gamma_1\, d\Delta_\mu, \qquad
d\mu_s(Y) = d\mu_2(Y) + w_1\Delta_\mu\, d\gamma_2 + w_1\gamma_2\, d\Delta_\mu,$$

$$d\sigma_s^2(X) = d\sigma_1^2(X) + 2\gamma_1(-w_2\Delta_\sigma + w_1w_2\Delta_\mu^2)\, d\gamma_1
- w_2\gamma_1^2\, d\Delta_\sigma + 2w_1w_2\gamma_1^2\Delta_\mu\, d\Delta_\mu,$$

$$d\sigma_s^2(Y) = d\sigma_2^2(Y) + 2\gamma_2(w_1\Delta_\sigma + w_1w_2\Delta_\mu^2)\, d\gamma_2
+ w_1\gamma_2^2\, d\Delta_\sigma + 2w_1w_2\gamma_2^2\Delta_\mu\, d\Delta_\mu,$$

$$db = \frac b2\Bigl[\frac{d\sigma_s^2(Y)}{\sigma_s^2(Y)} - \frac{d\sigma_s^2(X)}{\sigma_s^2(X)}\Bigr],
\qquad dc = d\mu_s(Y) - b\, d\mu_s(X) - \mu_s(X)\, db,$$

$d\gamma$ は $\gamma$ の定義から求める（Tucker: $\partial\gamma/\partial\sigma_{XV} =
1/\sigma_V^2$、$\partial\gamma/\partial\sigma_V^2 = -\sigma_{XV}/\sigma_V^4$。Levine 内部アンカー:
$\sigma_X^2$ について $1/\sigma_{XV}$、$\sigma_{XV}$ について $-\sigma_X^2/\sigma_{XV}^2$。Levine
外部アンカー: $\sigma_X^2$、$\sigma_{XV}$、$\sigma_V^2$ についてそれぞれ
$1/(\sigma_V^2 + \sigma_{XV})$、$(\sigma_V^2 - \sigma_X^2)/(\sigma_V^2 + \sigma_{XV})^2$、
$-(\sigma_X^2 + \sigma_{XV})/(\sigma_V^2 + \sigma_{XV})^2$）。Levine 真値等化と連鎖線形等化も §4.6 の
式から同じように微分する。合成母集団の重みは定数として扱う。

**Braun–Holland 法**は同時分布全体に依存するので、セル確率 $p_{jl}$（母集団 1 の X と V）と
$q_{kl}$（母集団 2 の Y と V）について微分する。$e = \mu_s(Y) + b\,(x - \mu_s(X))$、
$A_j = -b\,x_j - b\,(x - \mu_s(X))(x_j - \mu_s(X))^2 / (2\sigma_s^2(X))$、
$B_k = y_k + b\,(x - \mu_s(X))(y_k - \mu_s(Y))^2 / (2\sigma_s^2(Y))$、
$\bar A_l = \sum_j A_j p_{jl}$、$\bar B_l = \sum_k B_k q_{kl}$ として、

$$\frac{\partial e}{\partial p_{jl}} = A_j\Bigl[w_1 + w_2\frac{h_{2l}}{h_{1l}}\Bigr]
- w_2\frac{h_{2l}}{h_{1l}^2}\bar A_l + w_1\frac{\bar B_l}{h_{2l}}, \qquad
\frac{\partial e}{\partial q_{kl}} = B_k\Bigl[w_2 + w_1\frac{h_{1l}}{h_{2l}}\Bigr]
- w_1\frac{h_{1l}}{h_{2l}^2}\bar B_l + w_2\frac{\bar A_l}{h_{1l}},$$

分散は $\sum_{\text{標本}}[\sum p\,(\partial e)^2 - (\sum p\,\partial e)^2]/N$。

**正規理論**（`"analytic_normal"`）: 勾配は同じで、$\Omega$ を（2 変量）正規分布のもとでの値にする。
つまり 3 次モーメントは 0、$\mu_{ABCD} - \sigma_{AB}\sigma_{CD} =
\sigma_{AC}\sigma_{BD} + \sigma_{AD}\sigma_{BC}$。線形等化では EG で
$\sigma_Y^2 (1/N_X + 1/N_Y)(1 + z^2/2)$、SG で $\sigma_Y^2 (1 - \rho)[z^2(1 + \rho) + 2]/N$ になる
（Lord, 1950; Angoff, 1971。単体テストで確認）。

**等パーセンタイル等化（EG）**（Lord, 1982; Kolen & Brennan, 2014, 7 章）。$p = P(x)$（割合）、
$y^*_U = \min\{y_k : G(y_k) > p\}$、$G_0 = G(y^*_U - \delta_Y)$、$g = G(y^*_U) - G_0$ として、

$$\mathrm{SE}^2 = \frac{\delta_Y^2}{g^2}\Bigl\{\frac{p(1 - p)}{N_X}
+ \frac{1}{N_Y}\Bigl[G_0 - p^2 + \frac{(p - G_0)^2}{g}\Bigr]\Bigr\}.$$

第 2 項は $Q(y)$ のデルタ法による分散そのもの。第 1 項はパーセンタイル順位を二項比率として扱っており、
離散得点ではその分散を $f(x)/(4N_X)$ だけ大きく見積もる。したがってこの式は近似である。参照用の 3 つの
データでは、パーセンタイル順位が 2〜98 の得点で、ブートストラップ標準誤差（2000 回）と Lord の式の比は
0.87〜1.13、平均は 0.98〜1.01 だった。

**ブートストラップ**（`"bootstrap"`。Kolen & Brennan, 2014, 7 章。`equate::bootstrap` と同じ）。
`n_replications` 回の反復それぞれで、各標本から $N$ 人を復元抽出する。度数データでは、観測された
相対度数を確率とする多項分布からの抽出になる（同時分布はセルを再抽出するので、2 つの得点の従属性は
保たれる）。X の標本を先に、Y の標本を後に抽出する。各ブートストラップ標本を同じ手法で等化し、等化後
得点の標準偏差（分母 $B - 1$）を標準誤差とする。`bootstrap_mean` はその平均。

| 手法（デザイン） | `analytic` | `analytic_normal` | `bootstrap` |
|---|---|---|---|
| 恒等 | 0 | 0 | 0 |
| 平均・線形（EG） | あり（Braun & Holland, 1982） | あり | あり |
| 平均・線形（SG） | あり | あり | あり |
| 等パーセンタイル（EG） | あり（Lord, 1982） | なし | あり |
| 等パーセンタイル（SG） | なし | なし | あり |
| Tucker、Levine 観測得点、連鎖線形 | あり | あり | あり |
| Levine 真値 | あり | あり | あり |
| Braun–Holland | あり（セル確率） | なし | あり |
| 頻度推定、連鎖等パーセンタイル | なし | なし | あり |

**決定**: 標準誤差は別の関数にせず `equate()` のオプションにする（既定は `standard_error=None` で、
これまでの呼び出しは変わらない）。入力も検証済みのオプションも同じだから。

**決定**: `"analytic"` は分布を仮定しないデルタ法を指す。R `equate` はアンカーテストの手法で両方を、
EG では分布を仮定しないものだけを返す。正規理論版は別の選択肢 `"analytic_normal"` にする。

**決定**: 解析的な式に代入する分散・共分散は、等化関数と同じく `ddof` に従う。3 次・4 次モーメントは
常に $N$ で割る。`ddof=1` なら EG の線形等化で `equate` の標準誤差と完全に一致する。

**決定**: ブートストラップは `rng`（`numpy.random.Generator` か非負の整数シード）を必須にし、なければ
エラーにする。どのブートストラップ結果も再現できるようにするため。引数名は flexpyctt と同じ `rng`、
反復回数は flexpyirt と同じ `n_replications`。

**決定**: 合成母集団の重みの既定値 $N_1/(N_1 + N_2)$ はデータから一度だけ計算し、ブートストラップ標本
の間で固定する（標本サイズは変わらない）。

**決定**: 等化できないブートストラップ標本（分散のない得点、Levine 法で共分散が正でない、結果が有限で
ない）は除き、`SparseDataWarning` で数を知らせる。結果の `n_replications` は実際に使った数。使える
標本が 2 つ未満ならエラー。

**決定**: $x$ のパーセンタイル順位が 0 か 1 のところでは、Lord の標準誤差を 0 とする（`equate` と
同じ）。式が値を持たないため。

**決定**: 3 種類とも度数を観測値として扱う。事前平滑化した分布を入力にすると、解析的な式はモデルを
当てはめたことを無視し、ブートストラップは平滑化した分布から再当てはめなしに抽出する（モデルを固定
したパラメトリック・ブートストラップ）。標本サイズは度数の合計（ブートストラップでは四捨五入）。
事前平滑化を考慮した標準誤差は扱わない（下記）。カーネル等化にはある。

**決定**: Braun–Holland 法の解析的標準誤差は、一方の母集団にしかないアンカー得点があるとエラーにする
（微分が存在しない）。ブートストラップは使える。

**扱わないもの**: SG の等パーセンタイル等化、頻度推定、連鎖等パーセンタイル等化の解析的標準誤差
（文献に式はある。例: Jarjoura & Kolen, 1985; Liou & Cheng, 1995。R `equate` に照合先がないので
ブートストラップのみ）。`equate()` で事前平滑化を考慮した標準誤差（`equate::bootstrap` のように
ブートストラップ標本ごとに対数線形モデルを当てはめ直す）。`kernel_equate()` のブートストラップ標準
誤差。ブートストラップによるバイアス、RMSE、パーセンタイル区間。

### 4.9 他のカーネルとバンド幅の選び方（E7）

**カーネル**（Lee & von Davier, 2011）。分布関数 $K$、密度 $k$、平均 0、分散 $\sigma_V^2$ のカーネル
変数 $V$ を使うと、§4.5 の連続化は次のようになる。

$$F_h(x) = \sum_j r_j\, K\bigl(R_{jX}(x)\bigr), \qquad
R_{jX}(x) = \frac{x - a_X x_j - (1 - a_X)\mu_X}{a_X h_X}, \qquad
a_X^2 = \frac{\sigma_X^2}{\sigma_X^2 + \sigma_V^2 h_X^2},$$

$$f_h(x) = \frac{1}{a_X h_X}\sum_j r_j\, k\bigl(R_{jX}(x)\bigr), \qquad
f_h'(x) = \frac{1}{(a_X h_X)^2}\sum_j r_j\, k'\bigl(R_{jX}(x)\bigr),$$

SEE に使う微分は形が変わらず、
$\partial F_h(x)/\partial r_j = K(R_{jX}(x)) - M_{jX}(x) f_h(x)$（$M_{jX}$ は同じ、$a_X$ はその
カーネルのもの）。平均と分散はどのカーネルでも保たれる（単体テスト）。

| `kernel` | $K(z)$ | $k(z)$ | $k'(z)$ | $\sigma_V^2$ |
|---|---|---|---|---|
| `"gaussian"` | $\Phi(z)$ | $\phi(z)$ | $-z\phi(z)$ | 1 |
| `"logistic"` | $1/(1 + e^{-z})$ | $K(1 - K)$ | $k\,(1 - 2K)$ | $\pi^2/3$ |
| `"uniform"` | $\min\{\max(z + \tfrac12, 0), 1\}$ | $[-\tfrac12, \tfrac12)$ で 1 | 0 | $1/12$ |

§4.5 のすべてのデザイン、SEE、SEED、PRE は、カーネルを差し替えた同じコードで計算する。§4.5 の
ペナルティはそのカーネルの密度で計算する。一様カーネルでは第 2 項が 0 になる。

**決定**: ロジスティックカーネルは標準ロジスティック、一様カーネルは $(-1/2, 1/2)$ 上の一様分布とする。
Lee & von Davier の「標準」版であり、`kequate`（`slog = 1`、`bunif = 0.5`）と `SNSequate` の既定でも
ある。尺度を変えた版はバンド幅を変えた同じ連続化（$\sigma_{V1} h_1 = \sigma_{V2} h_2$）なので、尺度
パラメータは用意しない。

**決定**: 一様カーネルの密度は半開区間 $[-1/2, 1/2)$ で 1 とする。隣り合うカーネルが接するところで
二重に数えないため（R の `dunif` は両端を含む）。違いが出るのは測度 0 の集合だけ。

**決定**: 一様カーネルでは、連続化した分布の台の外と、$h$ が得点の刻み幅より小さいときの得点の間で
$F_h$ が平らになる。分位点は台の中に収め、等化後得点で Y の密度が 0 のところは SEE を `nan` にする
（デルタ法が使えない）。バンド幅を刻み幅以上にすれば隙間はできない。

**決定**: 一様カーネルでは `linear=True` を受け付けない。台が有界なので、バンド幅を大きくしても線形
関数にならない。ロジスティックカーネルでは線形になる。`kequate` は線形関数と SEED を常にガウス
カーネルで計算する。こちらでは `result.seed(kernel_equate(..., linear=True))` がそれにあたる。

**決定**: 一様カーネルのペナルティは、カーネルの端が取りうる得点を通過するところで不連続に変わる。
他のカーネルと同じ探索（格子、続いて区間を限った局所探索）を使い、見つかった最良の点を返す。

**バンド幅の選び方**（`bandwidth_method`、ガウスカーネル）。ペナルティのほかに:

- `"rule_of_thumb"`（Andersson & von Davier, 2014）:
  $h = 9\sigma/\sqrt{100N^{2/5} - 81}$。実際に使う密度に対する Silverman の経験則
  $a h = 0.9\,\sigma N^{-1/5}$ を $h$ について解いたもの。
- `"double_smoothing"`（Häggström & Wiberg, 2014）。大きなバンド幅 $g$ のパイロット連続化 $f_g$ を
  未知の密度の代わりに使う。取りうる得点とその中点からなる格子 $x^*_l$（$l = 1..2J - 1$）の上で、

  $$\mathrm{DS}(h) = \sum_l \Bigl[\sum_j f_g(x_j)\,\frac{1}{a h}\phi\bigl(R_j(x^*_l)\bigr)
  - r^*_l\Bigr]^2,$$

  $r^*_l$ は取りうる得点では $r_j$、中点では $f_g(x^*_l)$。
- `"cross_validation"`: `kequate` の 1 個抜き基準。

  $$\mathrm{CV}(h) = \sum_j \bigl(r_j - \hat r_{(j)}\bigr)^2, \qquad
  \hat r_{(j)} = \frac{\sum_{i \ne j} r_i\,\phi(R_i(x_j))}{\sum_{i \ne j}\phi(R_i(x_j))}.$$

ペナルティと同じく、後の 2 つは得点を刻み幅の単位で測って計算し、§4.5 の格子探索と局所探索で最小化
する。

**決定**: 3 つの規則は `kernel_equate()` のオプション（`bandwidth_method`）にする。使ったバンド幅は
結果に入る。公開関数は増やさない。

**決定**: 経験則の $\sigma$ は連続化する分布の標準偏差（連続化の他の部分と同じ母集団形。`kequate` は
分散に $N/(N - 1)$ を掛ける）。$N$ はその版を受けた標本の大きさ: EG では $N_X$・$N_Y$、SG では $N$、
アンカーテストのデザインでは X（とそのアンカー）に母集団 P、Y（とそのアンカー）に母集団 Q の大きさ
（`kequate` と同じ）。各版が 2 つの標本で観測される CB デザインでは受け付けない。

**決定**: 二重平滑化のパイロットバンド幅は得点の刻み幅の 4 倍とし、候補はそれ未満で探す。出典は
パイロットを「大きく、主観的に選ぶ」としており、4 は `kequate` の値（探索区間の上端）。上限を置かない
と、短いテスト（参照データの 9 点のアンカー）では基準が最大のバンド幅で最小になる。

**決定**: `"cross_validation"` は `kequate` の基準（`CV = TRUE`）で、名前もそれに合わせる。標本を
無作為に 2 つに分けて繰り返す Liang & von Davier (2014) の尤度交差検証ではない。今回は原論文を入手
できず、上の式は Wallin, Häggström & Wiberg (2021) と `kequate` のソースによる。検証の限界として
ここに記す。

**決定**: SEE は、どの規則でも選んだバンド幅を固定値として扱う（同書がペナルティについてそうして
いるのと同じ）。Andersson & von Davier (2014) は、$h$ が $\hat\sigma$ に比例することを考慮した経験則
用の SEE を与えている。`kequate` のその実装（`altopt = TRUE`）はこちらで導いた微分と一致しない
（§7）ので、使える参照がなく、扱わない。

**扱わないもの**: `SNSequate` の Epanechnikov カーネルと適応カーネル。`kequate` の
`"stdgaussian"` カーネル。ロジスティック・一様カーネルの尺度パラメータ。ガウス以外のカーネルでの
バンド幅規則。Liang & von Davier (2014) の尤度交差検証。データに依存するバンド幅についての
Andersson & von Davier (2014) の SEE。CB デザインでの経験則。

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

sg = fe.kernel_equate(smooth_xy)  # 単一群: 同時分布を 1 つ
cb = fe.kernel_equate(smooth_12, smooth_21, design="counterbalanced")
pse = fe.kernel_equate(smooth_xa, smooth_ya, design="post_stratification", synthetic_weight=0.5)
ce = fe.kernel_equate(smooth_xa, smooth_ya, design="chained")
ce.anchor_bandwidths, ce.chain_percent_relative_error
pse.seed(fe.kernel_equate(smooth_xa, smooth_ya, design="post_stratification", linear=True))
ce.seed(pse)  # DataFrame: score, difference, seed

# 非等価群: X 版とアンカー、Y 版とアンカー
xa = fe.BivariateScoreDistribution.from_scores(x_scores, x_anchor, x_range=(0, 36), y_range=(0, 12))
ya = fe.BivariateScoreDistribution.from_scores(y_scores, y_anchor, x_range=(0, 36), y_range=(0, 12))
tucker = fe.equate(xa, ya, method="tucker")  # w1 = N1 / (N1 + N2)
levine = fe.equate(xa, ya, method="levine_observed", anchor="internal", synthetic_weight=1.0)
tucker.gamma, tucker.synthetic_moments, tucker.synthetic_weight
fe.equate(xa, ya, method="frequency_estimation")
fe.equate(xa, ya, method="chained_equipercentile")

# 等化の標準誤差（E7）。
fe.equate(x, y, method="linear", standard_error="analytic").table  # score, equated, see
fe.equate(xa, ya, method="tucker", standard_error="analytic_normal").see
boot = fe.equate(
    xa,
    ya,
    method="chained_equipercentile",
    standard_error="bootstrap",
    n_replications=1000,
    rng=20261125,
)
boot.see, boot.bootstrap_mean, boot.n_replications

# 他のカーネルとバンド幅の選び方（E7）。
fe.kernel_equate(smooth_x, smooth_y, kernel="logistic")
fe.kernel_equate(smooth_x, smooth_y, kernel="uniform", bandwidth_x=1.0, bandwidth_y=1.0)
fe.penalty_bandwidth(smooth_x, kernel="logistic")
fe.kernel_equate(smooth_x, smooth_y, bandwidth_method="double_smoothing")  # "rule_of_thumb"、
ke.kernel, ke.bandwidth_method  # "cross_validation" も
```

- データより後ろの引数はすべて keyword-only。選択肢は小文字 snake_case の `Literal` 文字列:
  `method="identity" | "mean" | "linear" | "equipercentile"`（EG・SG）と
  `"tucker" | "levine_observed" | "levine_true" | "braun_holland" | "chained_linear" |
  "frequency_estimation" | "chained_equipercentile"`（NEAT）。
- 結果は frozen dataclass（`EquatingResult`、`LoglinearResult`、`KernelEquatingResult`）。NumPy 配列
  （読み取り専用）を持ち、DataFrame はプロパティで返す。
- エラーは `flexpyequate.errors` のクラス: `ScoreDistributionError`（得点や度数が不正）、
  `SpecificationError`（引数が矛盾）、`ConvergenceWarning`、`SparseDataWarning`。
- 名前は Python の名前にする。役に立つところでは R の名前を docstring に書く（`ddof=1` ↔ R `equate` の
  分散、`penalty_weight` ↔ `kequate` の `KPEN`）。
- `equate()` の `standard_error="analytic" | "analytic_normal" | "bootstrap"`（`n_replications` と
  `rng` を伴う）。`kernel_equate()` の `kernel="gaussian" | "logistic" | "uniform"` と
  `bandwidth_method="penalty" | "rule_of_thumb" | "double_smoothing" | "cross_validation"`
  （§4.8、§4.9）。

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
    _standard_errors.py 従来手法のデルタ法による標準誤差
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
Rscript validation/r/equate_e5_reference.R
Rscript validation/r/equate_e6_reference.R
Rscript validation/r/equate_e7_standard_errors_reference.R  # 約 10 分（equate::bootstrap）
Rscript validation/r/equate_e7_kernel_reference.R           # E3・E6 のファイルを読む
```

E7 の 2 つの生成スクリプトは、コミット済みの E1・E3・E4・E6 の参照ファイルからデータを読む。それらを
先に生成しておくこと。

### ケース

| ID | 内容 | R の参照 | MS |
|---|---|---|---|
| Q1 | 平均・線形等化。EG（`ACTmath`、長さや刻み幅の異なるシミュレーションの版）と SG（2 変量シミュレーション） | `equate::equate(type = "mean", sx = 1, sy = 1)`、`equate(type = "linear")` | E1 |
| Q2 | パーセンタイル順位と等パーセンタイル等化。EG と SG。途中や両端に度数 0 がある分布、刻み幅が 1 でない得点を含む | `equate::px`、`equate::equate(type = "equipercentile")`、丸めをしない Kolen & Brennan 2 章の独立 R 実装 | E1 |
| Q3 | 対数線形モデルによる事前平滑化。1 変量（2〜8 次）、交差項あり・なしの 2 変量。平滑化後の分布の等パーセンタイル等化 | 厳しい収束判定の `stats::glm(family = poisson)`、`equate::presmoothing(smoothmethod = "loglinear")`、`equate(..., smoothmethod = "loglinear")` | E2 |
| Q4 | カーネル等化（EG）。バンド幅を与えたときの連続化 CDF、等化後得点、SEE、PRE。事前平滑化あり（3 データ）・なし（1 データ） | `kequate::kequate("EG", ...)`、`SNSequate::ker.eq`、厳しい求根精度と有限差分 SEE による独立 R 実装 | E3 |
| Q5 | ペナルティ関数の値と、ペナルティによるバンド幅（$K = 0$ と $K = 1$） | `kequate`（`KPEN = 0`）、`SNSequate::bandwidth`、独立 R 実装 | E3 |
| Q6 | Tucker、Levine 観測得点、Braun–Holland（それぞれ合成母集団の重み 4 通り）、Levine 真値、連鎖線形。内部アンカー（`KBneat`）と外部アンカー（シミュレーション。標本サイズと版の長さが異なる） | `equate::equate(type = "linear", method = "tucker" / "levine" / "braun/holland" / "chained", lts = )`、母集団モーメントで Kolen & Brennan の式を書き下した R 実装 | E4 |
| Q7 | 頻度推定（合成母集団の重み 4 通り）と連鎖等パーセンタイル等化。`KBneat` と外部アンカーのシミュレーションデータ。2 変量対数線形モデルで事前平滑化した `KBneat` での両手法 | `equate::equate(type = "equipercentile", method = "frequency estimation" / "chained", smoothmethod = )`、丸めをしない Kolen & Brennan の R 実装 | E5 |
| Q8 | SG（事前平滑化あり・なし）、CB（重み 3 通り）、NEAT PSE（重み 3 通り）、NEAT CE のカーネル等化。等化後得点、SEE、線形版とカーネル対線形の SEED、PRE、ペナルティによるバンド幅 | `kequate::kequate("SG" / "CB" / "NEAT_PSE" / "NEAT_CE", ...)`、デザイン関数を書き下し全セル確率についての有限差分で SEE を求める R 実装 | E6 |
| Q9 | 解析的標準誤差。平均・線形等化（EG の 3 データと SG）、等パーセンタイル等化の Lord の標準誤差（EG）、Tucker と Levine 観測得点（合成母集団の重み 3 通り）、Levine 真値、連鎖線形、Braun–Holland（内部アンカーの `KBneat` と外部アンカーのシミュレーションデータ）。分布を仮定しない版と正規理論版 | `equate::equate(..., verbose = TRUE)$concordance$se`（EG の線形と等パーセンタイル。Tucker、Levine 観測得点、連鎖は `se.g` と `se.n`）、全度数を 1e6 倍した同じ呼び出し、全セル確率（正規理論版はモーメント）についての有限差分ヤコビアンでデルタ法を書き下した R 実装 | E7 |
| Q10 | ブートストラップ標準誤差（2000 回）。EG と SG の線形・等パーセンタイル、Tucker、連鎖線形、頻度推定、連鎖等パーセンタイル | `equate::bootstrap(reps = 2000)`。2 つのモンテカルロ推定値の**統計的な**比較（R の乱数は再現できない）。解析的標準誤差との単体テスト、再抽出の手順の単体テスト | E7 |
| Q11 | すべてのデザイン（EG、SG、CB は重み 2 通り、PSE は重み 2 通り、CE）でのロジスティック・一様カーネル。連続化 CDF と密度、等化後得点、SEE、ガウスの線形関数に対する SEED、PRE、ペナルティの値とペナルティによるバンド幅 | `kequate::kequate(..., kernel = "logistic" / "uniform")`、`kequate:::PEN`、`SNSequate::bandwidth(kert = "logis" / "unif")`、一般のカーネルの連続化を書き下し有限差分で SEE を求める R 実装 | E7 |
| Q12 | 経験則、二重平滑化、1 個抜き交差検証によるバンド幅（取りうる得点が 9〜41 個の 8 分布）。格子上の基準の値。EG と CE での一連の計算 | `kequate:::bandwidth.select("altopt" / "DS" / "CV")`、`kequate:::DS`、`kequate:::CV`、`kequate(..., DS = TRUE)`、`kequate(..., CV = TRUE)`、基準を書き下した R 実装 | E7 |

### 許容誤差（断りがなければ最大絶対差）

| 量 | 参照 | 許容誤差 |
|---|---|---|
| モーメント、平均・線形等化の等化後得点、傾き、切片 | `equate`（`ddof=1`） | 1e-10 |
| パーセンタイル順位 | `equate::px` | 1e-12 |
| 等パーセンタイル等化の等化後得点 | 独立 R 実装 | 1e-10 |
| 等パーセンタイル等化の等化後得点（順位が 0 と 1 の間にあるもの） | `equate` | 1e-7 |
| NEAT の線形手法: 傾き、切片、$\gamma$、合成母集団の平均と標準偏差 | `equate`（`ddof=1`）、独立 R 実装（`ddof=0`） | 1e-10 |
| 頻度推定の合成確率 | 独立 R 実装 | 1e-13 |
| 頻度推定・連鎖等パーセンタイルの等化後得点 | 独立 R 実装 | 1e-10 |
| 頻度推定・連鎖等パーセンタイルの等化後得点（順位が 0 と 1 の間にあるもの） | `equate` | 5e-7 |
| 同上、事前平滑化した同時分布（`equate` が平滑化した表、および平滑化から通した場合） | `equate` | 5e-6 |
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
| SG・CB・PSE・CE: カーネル等化の等化後得点、SEE（相対） | 独立 R 実装 | 1e-9、1e-6 |
| SG・CB・PSE・CE: カーネル等化の等化後得点、SEE（相対）、PRE、バンド幅（$K = 0$） | 分散の係数を除いた `kequate` | 1e-6、1e-6、1e-5、1e-6 |
| SG・CB・PSE・CE: 線形版の等化後得点、線形版の SEE とカーネル対線形の SEED（相対） | 分散の係数を除いた `kequate` | 1e-6、1e-5 |
| カーネル等化の等化後得点 / SEE（相対） / バンド幅 | 通常の `kequate` 呼び出し | 5e-3 / 1e-3 / 1e-4 |
| EG の線形等化の標準誤差（`ddof=1`、相対） | `equate` | 1e-10 |
| 連鎖線形の標準誤差（分布を仮定しない版・正規理論版、相対） | `equate` | 1e-10 |
| Tucker、Levine 観測得点の標準誤差（両方の版、相対） | 度数を 1e6 倍した `equate` | 1e-8 |
| 同上（`ddof=0` と `ddof=1`、相対） | 通常の `equate` 呼び出し | 2e-3 |
| 平均・線形（EG・SG）、Tucker、Levine 観測得点・真値、連鎖線形、Braun–Holland の標準誤差と正規理論版（相対） | 独立 R 実装（有限差分） | 1e-8 |
| EG の等パーセンタイル等化の Lord の標準誤差（絶対・相対） | `equate` | 1e-6 |
| ブートストラップ標準誤差（2000 回、相対）: 各得点 / 得点についての平均 | `equate::bootstrap`（2000 回） | 0.12 / 0.06（統計的） |
| ロジスティック・一様カーネル: CDF、密度 | 独立 R 実装 | 1e-12 |
| ロジスティック・一様カーネル、全デザイン: 等化後得点、SEE（相対） | 独立 R 実装 | 1e-9、1e-6 |
| ロジスティック・一様カーネル、全デザイン: 等化後得点、SEE（相対）、ガウスの線形関数に対する SEED（相対）、PRE | 分散の係数を除いた `kequate` | 1e-6、1e-6、1e-5、5e-5 |
| ロジスティック・一様カーネルのペナルティの値（相対） | `kequate:::PEN`（$K = 0$）、独立 R 実装（$K = 0, 1$） | 1e-10 |
| ロジスティック・一様カーネルのペナルティによるバンド幅 | `kequate`（$K = 0$）、独立 R 実装 | 1e-6 |
| ロジスティック・一様カーネルのペナルティによるバンド幅（$K = 1$） | `SNSequate::bandwidth` | 1e-5 |
| 二重平滑化・交差検証の基準の値（相対） | `kequate:::DS`、`kequate:::CV`、独立 R 実装 | 1e-10 |
| 二重平滑化のバンド幅 | 分散の係数を除いた `kequate`、独立 R 実装 | 1e-6 |
| 交差検証のバンド幅 | 分散の係数を除いた `kequate`、独立 R 実装 | 1e-5 |
| 経験則のバンド幅（相対） | 係数 $\sqrt{N/(N-1)}$ を掛けた上での `kequate`、独立 R 実装 | 1e-10 |
| 二重平滑化・交差検証の一連の計算（EG、CE）: 等化後得点、SEE（相対） | 分散の係数を除いた `kequate(..., DS = TRUE / CV = TRUE)` | 1e-5、1e-5 |

実際の差はこれより十分小さい。`kequate` に対して SEE は 1e-8 程度、等化後得点は 2e-7 程度（後者は
`kequate` の求根精度による）。

E7 での実測値（全ケースでの最大）:

| 量 | 参照 | 実測 |
|---|---|---|
| EG の線形等化の標準誤差、`ddof=1`（相対） | `equate` | 1.9e-14 |
| 連鎖線形の標準誤差（相対） | `equate` | 3.6e-15 |
| Tucker、Levine 観測得点の標準誤差（相対） | 度数を 1e6 倍した `equate` | 8.1e-10 |
| Tucker、Levine 観測得点の標準誤差（相対） | 通常の `equate` | 7.6e-4（`ddof=0`）、3.4e-4（`ddof=1`） |
| デルタ法の標準誤差すべて、両方の版（相対） | 独立 R 実装 | 4.3e-10 |
| 正規性のもとでの SG の線形等化の標準誤差（相対） | 閉形式 $\sigma_Y^2(1-\rho)[z^2(1+\rho)+2]/N$ | 4.6e-15 |
| Lord の標準誤差（絶対、相対） | `equate` | 1.4e-7、1.7e-7 |
| 線形手法のブートストラップ標準誤差（相対、最悪の得点） | `equate::bootstrap` | 0.054 |
| 等パーセンタイル手法のブートストラップ標準誤差、パーセンタイル順位 2〜98（相対、最悪の得点） | `equate::bootstrap` | 0.074 |
| ブートストラップ標準誤差の得点についての平均（相対、最悪のケース） | `equate::bootstrap` | 0.038 |
| ロジスティック・一様カーネル: 等化後得点、SEE（相対） | 独立 R 実装 | 4.4e-11、1.3e-9 |
| ロジスティック・一様カーネル: 等化後得点、SEE（相対）、SEED（相対）、PRE | 分散の係数を除いた `kequate` | 5.5e-7、1.5e-7、5.3e-8、9.3e-6 |
| ペナルティの値（相対） | `kequate:::PEN`、独立実装 | 2.4e-13 |
| ロジスティック・一様カーネルのペナルティによるバンド幅 | `kequate`、独立実装、`SNSequate` | 1.3e-8、7.4e-9、6.6e-9 |
| 二重平滑化・交差検証の基準の値（相対） | `kequate`、独立実装 | 9.8e-15 |
| 二重平滑化のバンド幅 | `kequate`、独立実装 | 1.8e-8、9.4e-9 |
| 交差検証のバンド幅 | `kequate`、独立実装 | 2.6e-7、1.3e-6 |
| 経験則のバンド幅（相対） | `kequate`、独立実装 | 2.5e-13、4.9e-15 |
| 一連の計算（EG、CE）: 等化後得点、SEE（相対） | `kequate(DS = TRUE / CV = TRUE)` | 7.7e-7、7.2e-7 |

**ブートストラップの R との一致は統計的なもので、数値的なものではない。** R と NumPy の乱数生成器は
異なるので、2 つのブートストラップ標準誤差は同じ量の 2 つのモンテカルロ推定値である。それぞれ
$B = 2000$ 回なら、比の標準偏差は得点ごとに約 $1/\sqrt{B} = 2.2\%$。ブートストラップは 3 通りに検証
している。(a) 手法とデザインの 8 通りの組み合わせで解析的標準誤差と比べる（1500 回、許容誤差は
モンテカルロ標準誤差の 5 倍に 3% を加えたもの）。(b) 文書に書いた再抽出の手順（多項分布の度数、X の
標本を先に Y の標本を後に、分母 $B - 1$ の標準偏差）をテスト内で書き下し、同じシードから `see` と
`bootstrap_mean` が丸め誤差の範囲で再現されることを EG・SG・アンカーテストのデザインで確かめる
（決定的なテスト）。(c) `equate::bootstrap` と上の許容誤差で比べる。

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
- **`kequate` のカウンターバランスデザイン**は重みが `wcb` の 1 つで、$r = w r_1 + (1 - w) r_2$、
  $s = w s_1 + (1 - w) s_2$ とする。ところが SEE では Y の周辺の重みが逆になっている（第 1 標本に
  $1 - w$）。そのため SEE が等化関数と対応するのは $w = 1/2$ のときだけである。参照データで `wcb = 0.3` と
  すると、有限差分による SEE と最大 29% 違う（flexpyequate の SEE は 1e-6 で一致する）。flexpyequate は
  重みを 2 つ $(w_X, w_Y)$ 持ち、`kequate` とは $(1/2, 1/2)$ で照合している。
- **`kequate` は $G_h(y) = F_h(x)$ を `uniroot(tol = 1e-6)` で解く**。そのため等化後得点の一致は 1e-6 程度
  までになる。
- **`kequate` のペナルティ第 2 項**は、密度が極大になる得点（左で $f' > 0$、右で $f' < 0$）も数える。
  同書の 4 章および `SNSequate` とは異なる。flexpyequate は同書に従う。$K = 0$（`kequate` の既定）では
  この項は入らない。flexpyequate の既定は同書と同じ $K = 1$。
- **`kequate` と `SNSequate` は得点が 0, 1, 2, … であることを前提にする**。それ以外の得点尺度は同変性の
  単体テストだけで確認している。
- **`SNSequate::ker.eq`** は等化後得点を $(-1, 101)$ の範囲で探し、−1 を下回る得点があると失敗する
  （試した 2 ケースのうち 1 つ）。バンド幅は 0.65 から始めた `optim()` で求めている。
- **`equate` の Tucker・Levine 観測得点等化の標準誤差**は、母集団形（$N$ で割る）の分散・共分散から
  計算するが、$\gamma$ と合成母集団の平均・標準偏差には $N - 1$ を使う等化関数のものを使う。この混在で
  標準誤差は相対的に $1/N$ 程度変わる（参照データで最大 7.6e-4）。flexpyequate の `ddof` をどちらに
  しても同じ。全度数を $10^6$ 倍して（結果を $10^3$ 倍して）比べると 2 つの形は一致し、差は 1e-9 に
  なる。連鎖線形等化は影響を受けない（1 つの母集団の中での比だから）。
- **`equate` には解析的標準誤差がない**手法がある: 平均等化、SG デザイン、Levine 真値等化、
  Braun–Holland 法、アンカーテストの等パーセンタイル手法（最初の 2 つの関数は `NULL` を返す）。
  flexpyequate のものは有限差分による独立実装と照合し、最後のものはブートストラップのみとする。
- **`equate` の等パーセンタイル等化の標準誤差**には得点の刻み幅が入っていないので、Y の刻み幅が
  $\delta_Y \ne 1$ のときは刻み幅の単位になる。flexpyequate は $\delta_Y$ を掛ける。パーセンタイル
  順位は小数 10 桁に丸められる（一致は 2e-7）。
- **`equate::bootstrap` と得点範囲の両端**: パーセンタイル順位が 0 か 1 のところで `equate` と
  flexpyequate は等パーセンタイル等化の値の定義が異なり（上記）、最低点・最高点付近のブートストラップ
  標準誤差に差が出る（参照データで比が 0.48 まで下がる）。比較にはパーセンタイル順位が 2〜98 の得点を
  使う。頻度推定では `equate` の実効的な合成母集団の重み（上記）を渡す。
- **`kequate` のロジスティックカーネルのペナルティ**は、密度の微分に余分な係数 $1/(a h)$ を掛けている。
  正の数なので、符号もペナルティも変わらない。
- **`kequate` はバンド幅の探索範囲**を $(0, 4)$（一様カーネルでは $(0, 2.618)$）に限り、局所探索の
  `optimize()` を使う。flexpyequate はまず刻み幅の $(0.05, 10)$ 倍の格子を調べる（二重平滑化は
  パイロットバンド幅 4 まで）。参照ケースではすべて同じバンド幅が選ばれた。
- **`kequate` の `altopt`**（経験則）は、バージョン 1.6.4 では `CV = TRUE` と併用したときだけバンド幅
  に反映され、$h \propto \hat\sigma$ を考慮した別の微分 $\partial F/\partial r$ に切り替わる。
  $h = c\sigma$ として $F$ を微分すると $\tfrac12 a\,z_j^2 \sum_k x_k r_k \phi(R_k)/(a h)$ という項が
  出るが、`kequate` のものには係数 $a$ がない。また EG のコードは Y の版に X の平均を渡している。その
  ため、この SEE は参照に使わず、バンド幅だけを比べる。
- **`kequate` の `CV`** は得点についての 1 個抜き基準で、標本を分割する Liang & von Davier (2014) の
  尤度交差検証ではない。`kequate` の `DS` は差の 2 乗の項だけを（平均として）返す。最小点は和の場合と
  同じ。
- 独立実装の SEE では、得点確率が 5e-6 程度のところで、刻み 1e-6 の 2 点差分は精度が足りない（誤差
  1.4e-3）。生成スクリプトは刻み 1e-7 の 5 点差分を使う。

## 参考文献

- Andersson, B., & von Davier, A. A. (2014). Improving the bandwidth selection in kernel
  equating. *Journal of Educational Measurement, 51*, 223–238.
- Angoff, W. H. (1971). Scales, norms, and equivalent scores. In R. L. Thorndike (Ed.),
  *Educational measurement* (2nd ed., pp. 508–600). American Council on Education.
- Braun, H. I., & Holland, P. W. (1982). Observed-score test equating: A mathematical analysis
  of some ETS equating procedures. In P. W. Holland & D. B. Rubin (Eds.), *Test equating*
  (pp. 9–49). Academic Press.
- Häggström, J., & Wiberg, M. (2014). Optimal bandwidth selection in observed-score kernel
  equating. *Journal of Educational Measurement, 51*, 201–211.
- Jarjoura, D., & Kolen, M. J. (1985). Standard errors of equipercentile equating for the
  common item nonequivalent populations design. *Journal of Educational Statistics, 10*,
  143–160.
- Kolen, M. J. (1985). Standard errors of Tucker equating. *Applied Psychological Measurement,
  9*, 209–223.
- Lee, Y.-H., & von Davier, A. A. (2011). Equating through alternative kernels. In A. A. von
  Davier (Ed.), *Statistical models for test equating, scaling, and linking* (pp. 159–173).
  Springer.（ETS RR-08-12 として参照）
- Liang, T., & von Davier, A. A. (2014). Cross-validation: An alternative bandwidth selection
  method in kernel equating. *Applied Psychological Measurement, 38*, 281–295.
- Liou, M., & Cheng, P. E. (1995). Asymptotic standard error of equipercentile equating.
  *Journal of Educational and Behavioral Statistics, 20*, 259–286.
- Lord, F. M. (1950). *Notes on comparable scales for test scores* (ETS RB-50-48).
- Lord, F. M. (1982). The standard error of equipercentile equating. *Journal of Educational
  Statistics, 7*, 165–174.
- Wallin, G., Häggström, J., & Wiberg, M. (2021). How important is the choice of bandwidth in
  kernel equating? *Applied Psychological Measurement, 45*, 518–535.
- Zu, J., & Yuan, K.-H. (2012). Standard error of linear observed-score equating for the NEAT
  design with nonnormally distributed data. *Journal of Educational Measurement, 49*, 190–213.
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
