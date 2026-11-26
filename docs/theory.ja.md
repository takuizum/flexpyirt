# 理論

[English](theory.md) | 日本語

flexpyirt の背後にある考え方の要点: どんなモデルか、何を最大化するか、EM アルゴリズムがそれをどう行うか、
標準誤差・スコア・適合度統計量・リンキング係数がどこから来るか。各節の最後に、コード例のある
[ユーザーガイド](guide.ja.md) の節を示す。詳しくは末尾の参考文献（Baker & Kim, 2004; Chalmers, 2012 など）を参照。

1. [項目反応モデル](#1-項目反応モデル)
2. [周辺尤度](#2-周辺尤度)
3. [識別性](#3-識別性)
4. [Bock–Aitkin EM アルゴリズム](#4-bockaitkin-em-アルゴリズム)
5. [潜在分布](#5-潜在分布)
6. [多母集団と FIPC](#6-多母集団と-fipc)
7. [事前分布と MAP 推定](#7-事前分布と-map-推定)
8. [欠測反応](#8-欠測反応)
9. [標準誤差](#9-標準誤差)
10. [受検者のスコアリング](#10-受検者のスコアリング)
11. [モデル適合と DIF](#11-モデル適合と-dif)
12. [尺度リンキング](#12-尺度リンキング)

**記法.** 受検者 $i=1,\dots,N$、項目 $j=1,\dots,J$、反応 $y_{ij}\in\{0,\dots,K_j-1\}$、潜在特性 $\theta$
（多次元モデルではベクトル $\boldsymbol\theta$）、群 $g=1,\dots,G$。$\sigma(z)=1/(1+e^{-z})$ はロジスティック関数。
$\xi_j$ は項目 $j$ のパラメタ、$\eta$ は自由パラメタ全体のベクトル。

## 1. 項目反応モデル

項目反応モデルは、各反応カテゴリの確率を $\theta$ の関数として与える。flexpyirt は（mirt と同じく）
**傾き–切片（slope–intercept）型**を使う: 線形予測子は傾き $a$ と切片 $d$ による $z=a\theta+d$。

| モデル | カテゴリ確率 | 備考 |
|---|---|---|
| 2PL（Birnbaum, 1968） | $P(y=1\mid\theta)=\sigma(a\theta+d)$ | |
| Rasch（Rasch, 1960） | $P(y=1\mid\theta)=\sigma(\theta+d)$ | $a=1$。尺度の単位は潜在分散が担う |
| 3PL / 4PL | $P(y=1\mid\theta)=g+(u-g)\,\sigma(a\theta+d)$ | 下側漸近線 $g$（当て推量）、上側漸近線 $u$（3PL では $u=1$） |
| GRM（Samejima, 1969） | $P(y\ge k\mid\theta)=\sigma(a\theta+d_k)$、$P(y=k)=P(y\ge k)-P(y\ge k+1)$ | 切片は順序つき $d_1>\dots>d_{K-1}$ |
| GPCM（Muraki, 1992） | $P(y=k\mid\theta)\propto\exp(k\,a\theta+d_k)$、$d_0=0$ | 隣接カテゴリのロジット $\log\frac{P(y=k)}{P(y=k-1)}=a\theta+d_k-d_{k-1}$ |
| PCM（Masters, 1982） | $a=1$ の GPCM | |
| Nominal（Bock, 1972） | $P(y=k\mid\theta)\propto\exp(a\,\mathrm{ak}_k\,\theta+d_k)$ | 順序のないカテゴリ。得点係数 $\mathrm{ak}_0=0$、$\mathrm{ak}_{K-1}=K-1$、$d_0=0$（Thissen, Cai & Bock, 2010） |

**IRT（困難度）パラメタ表記.** $a\theta+d=a(\theta-b)$ と書くと、おなじみの困難度 $b=-d/a$ になる:
2PL で $P(y=1)=1/2$ となる $\theta$ である。同様に GRM のカテゴリ境界は $b_k=-d_k/a$、GPCM / PCM の
ステップ困難度（隣り合う 2 カテゴリが等確率になる点）は $b_k=-(d_k-d_{k-1})/a$。flexpyirt は数値的に
扱いやすい $(a,d)$（$a=0$ でも定義できる）を推定し、$b$ はデルタ法の標準誤差つきで表示用に報告する
（[9 節](#9-標準誤差)）。

**定数 $D=1.7$.** ロジスティック曲線を正規累積（正規オージブ）に近づけるため $\sigma(1.7\,a(\theta-b))$ と書く
ソフトウェアもある。flexpyirt は（mirt と同じく）この定数を使わない。「$D=1.7$」のソフトウェアの傾きは
flexpyirt の傾きの $1/1.7$ になる。

**多次元モデル.** 因子が $D$ 個のとき、$z=\mathbf a^\top\boldsymbol\theta+d$ で因子ごとに傾き
（$a_1,\dots,a_D$）を持ち、$\boldsymbol\theta$ は多変量正規分布に従う。*確認的*モデルでは各項目は指定した因子にだけ
負荷し、ほかの傾きは 0 に固定される。

**情報量.** 項目の Fisher 情報量は、その項目が各 $\theta$ でどれだけ精密に測れるかを表す。2PL では
$I_j(\theta)=a_j^2P_j(\theta)\{1-P_j(\theta)\}$ で、$\theta=b_j$ で最大になる。テスト情報量
$I(\theta)=\sum_jI_j(\theta)$ は WLE に現れる（[10 節](#10-受検者のスコアリング)）。因子が複数あると情報量は
行列 $\mathbf I_j(\boldsymbol\theta)=\sum_kP_{jk}\,\nabla\log P_{jk}\,\nabla\log P_{jk}^\top$ になる。多次元 2PL では
$P_j(1-P_j)\,\mathbf a_j\mathbf a_j^\top$ で、項目は傾きベクトル $\mathbf a_j$ の方向についてだけ情報をもつ。方向
$\mathbf u$（$\|\mathbf u\|=1$）の情報量は $\mathbf u^\top\mathbf I(\boldsymbol\theta)\mathbf u$（Reckase & McKinley, 1991;
Reckase, 2009）。

ガイド: [3 節](guide.ja.md#3-モデルを指定する)。

## 2. 周辺尤度

**局所独立.** $\theta$ を与えたとき異なる項目への反応は独立なので、受検者 $i$ の反応パターン
$\mathbf y_i$ の尤度は
$$L(\mathbf y_i\mid\theta)=\prod_j P_j(y_{ij}\mid\theta;\xi_j).$$

**周辺最尤推定（MML）.** 受検者パラメタ $\theta_i$ を項目と同時に推定することはしない（$N$ を大きくしても
項目パラメタの推定値が一致性を持たない）。代わりに $\theta$ を母集団分布 $f(\theta)$ からの無作為抽出とみなし、
積分で消去する（Bock & Lieberman, 1970; Bock & Aitkin, 1981）:
$$\ell(\eta)=\sum_{i=1}^N\log\int L(\mathbf y_i\mid\theta)\,f(\theta)\,d\theta .$$
項目パラメタと $f$ のパラメタ（平均と分散、または形）を、$\ell$ を最大化して推定する。

**求積.** 積分を、節点 $\theta_1,\dots,\theta_Q$ と重み $w_q$ による和で置き換える:
$$\ell(\eta)\approx\sum_i\log\sum_{q=1}^Q L(\mathbf y_i\mid\theta_q)\,w_q .$$
既定の格子は $[-6,6]$ の等間隔 61 点で、$w_q$ は密度 $f(\theta_q)$ に比例し、格子上で和が 1 になるよう正規化する。
**flexpyirt が厳密に最大化し `loglik` として報告するのは、この離散化した尤度である。** 同じ反応パターンの
受検者は同じ項を寄与するので、パターンを度数つきで 1 回だけ数える。正規の $f$ には Gauss–Hermite 求積も
使える。多次元モデルは 1 次元格子の直積を使う。

重みは有限の格子上で正規化されるので、平均が 0 から遠い、または分散が大きい正規分布は格子の端で切れる。
報告される `var` は切れる前の正規密度のパラメタで、格子上の分布の分散はそれよりわずかに小さい。格子を広げれば
切れる影響は無視できる（ガイド [13 節](guide.ja.md#13-求積格子の選び方)）。

## 3. 識別性

$\theta$ の原点と単位は任意である。任意の $m$ と $s>0$ について、$\theta$ を $\theta^*=s\theta+m$ に、
項目パラメタを $a^*=a/s$、$d^*=d-am/s$ に置き換えても、$a^*\theta^*+d^*=a\theta+d$ なのですべての反応確率は
変わらない。何かが尺度を固定する必要がある:

- **既定**: 潜在分布を $N(0,1)$（平均・分散とも固定）とする。
- **Rasch / PCM**: 傾きが 1 に固定され単位が決まるので、潜在**分散を推定**し、平均だけを 0 に固定する。
- **固定項目（FIPC）**: 既知の値に固定した項目が項目バンクの尺度を持ち込むので、潜在分布の平均・分散を推定できる。
- **多母集団**: 参照群を固定し、焦点群の平均・分散は、パラメタを群間で等しくしたアンカー項目を通じて識別される。

一般則: 潜在分散を自由にするには（同じ連結した群の集まりの中に）0 でない値に固定した傾きが 1 つ以上、
潜在平均を自由にするには固定した切片が 1 つ以上必要。flexpyirt はこれを推定前に検査し、満たさなければ
`IdentificationError` を出す。確認的多次元モデルでは、0 に固定した負荷が $\boldsymbol\theta$ の回転の不定性も取り除く。

ガイド: [8 節](guide.ja.md#8-潜在分布と識別性)。

## 4. Bock–Aitkin EM アルゴリズム

$\ell(\eta)$ は和の対数を含むので直接の最大化は扱いにくい。Bock & Aitkin (1981) が IRT に適用した EM アルゴリズム
（Dempster, Laird & Rubin, 1977）は、各受検者がどの節点に「いる」かを欠測データとみなし、2 つのステップを交互に行う。

**E ステップ.** 現在のパラメタ $\eta'$ で、各パターンの節点上の事後重み
$$h_{iq}=\frac{L(\mathbf y_i\mid\theta_q)\,w_q}{\sum_{q'}L(\mathbf y_i\mid\theta_{q'})\,w_{q'}}$$
を求め、そこから**期待度数**（「人工データ」）を作る: 節点 $q$ にいる受検者の期待人数と、そのうち項目 $j$ で
カテゴリ $k$ を選ぶ期待人数
$$\bar n_q=\sum_i h_{iq},\qquad \bar r_{jkq}=\sum_i h_{iq}\,\mathbf 1[y_{ij}=k].$$

**M ステップ.** 期待完全データ対数尤度
$$Q(\eta\mid\eta')=\sum_j\sum_q\sum_k\bar r_{jkq}\log P_{jk}(\theta_q;\xi_j)+\sum_q\bar n_q\log w_q$$
を最大化する。これは項目ごとの小さな問題（第 1 項）と潜在分布の問題（第 2 項）に分かれる。flexpyirt は各項目の問題を、
自動微分による勾配・Hessian を使った数回の Newton 法と、$Q$ を決して減らさない直線探索で解く。

**単調性.** $\ell(\eta)=Q(\eta\mid\eta')-H(\eta\mid\eta')$ で、$H$ は $\eta=\eta'$ で最大になるので、$Q$ を増やす
ステップは $\ell$ も増やす（*一般化* EM）。したがって対数尤度は反復ごとに決して減らない。`history["loglik"]`
で確かめられる。

**潜在分布の M ステップ.** 正規の $f$ では、重みは格子上で正規化した $w_q(\mu,\sigma^2)\propto\phi(\theta_q;\mu,\sigma^2)$
である。よく使われる近道は $\mu$ と $\sigma^2$ を期待度数 $\bar n_q$ の平均・分散にすることだが、これは*連続*の
正規尤度の最大化であって離散化した尤度の最大化ではなく、厳密な EM ステップではない。flexpyirt は
$\sum_q\bar n_q\log w_q(\mu,\sigma^2)$ を厳密に最大化する。格子が分布を十分に覆っていれば両者は一致し、格子が
分布を切ると少しずれる（mirt との小さな差はこれによる。[精度と検証](accuracy.ja.md) を参照）。

**収束.** 2 回の反復の間の自由パラメタの最大絶対変化が `tol`（既定 $10^{-4}$、mirt の `TOL` と同じ規則）を
下回ったら EM を止める。

**SQUAREM.** EM は線形収束で、潜在分布が弱くしか決まらない場合（分散の推定、多群、ヒストグラム）には遅い。
SQUAREM（Varadhan & Roland, 2008）は EM の経路に沿って外挿する: EM 写像を $F$、$\mathbf r=F(\eta)-\eta$、
$\mathbf v=F(F(\eta))-2F(\eta)+\eta$ として $\eta-2\alpha\mathbf r+\alpha^2\mathbf v$
（$\alpha=-\lVert\mathbf r\rVert/\lVert\mathbf v\rVert$）へ進み、安定化のために EM を 1 回適用する。外挿点が実行不可能か
対数尤度を下げる場合、flexpyirt は通常の EM の点に戻るので、単調性が保たれ、不動点も同じになる。

ガイド: [4 節](guide.ja.md#4-推定する)。

## 5. 潜在分布

潜在分布 $f(\theta)$ は積分のための道具であるだけでなく、母集団のモデルでもある。真の分布が歪んでいたり
二峰だったりすると、正規性の仮定が項目パラメタに偏りをもたらすことがある（Woods, 2007）。flexpyirt は 4 つの形を持つ。

- **正規分布** $N(\mu,\sigma^2)$: パラメタ 2 つ。
- **経験ヒストグラム（EH）**（Bock & Aitkin, 1981; Woods, 2007）: 重み $w_1,\dots,w_Q$ そのものを自由パラメタにする。
  M ステップは確率単体上で $\sum_q\bar n_q\log w_q$ を最大化し、$w_q=\bar n_q/N$（ごく小さい下限つき）となる。
  ヒストグラムは位置と尺度が自由なので、固定項目が尺度を決めない限り、flexpyirt は参照群のヒストグラムを
  M ステップの*中で*平均 0・分散 1 に制約する（厳密な制約付き最大化なので EM の単調性は保たれる）。柔軟だが
  パラメタが $Q-1$ 個あるので大きな標本が必要。
- **Davidian 曲線**（Zhang & Davidian, 2001; Woods & Lin, 2009）: 滑らかな半ノンパラメトリック密度
  $h_k(z)=P_k(z)^2\varphi(z)$。$k$ 次多項式の 2 乗と標準正規密度の積である。多項式の係数は（$h_k$ の積分が 1 に
  なるよう）単位球面上に制約され、$k$ 個の角度でパラメタ化される。角度がすべて 0 なら正規分布。flexpyirt は
  曲線を解析的に標準化して、どんな形でも平均・分散がちょうど $\mu$・$\sigma^2$ になるようにし、形を位置・尺度から
  切り離す。次数 $k$（1–10）はふつう AIC や BIC で選ぶ。
- **混合正規**（Mislevy, 1984）: $f(\theta)=\sum_c\pi_c\,N(\mu_c,\sigma_c^2)$。成分数 $k$、項目パラメタは全体で共通で、
  下位母集団からなる母集団に向く。混合分布は同じように平均 $\mu$・分散 $\sigma^2$ に標準化される。成分は交換可能
  （番号は任意）で、尤度には複数の局所最大があり得るので、いくつかの初期値から推定し直すのがよい。

正規分布の近くでは Davidian 曲線や混合正規の形のパラメタはほとんど決まらず、正規モデルに対する尤度比検定は
通常の $\chi^2$ 分布に従わない。モデルの比較には AIC / BIC を使う。

ガイド: [8 節](guide.ja.md#8-潜在分布と識別性)。

## 6. 多母集団と FIPC

**多母集団の同時推定**（Bock & Zimowski, 1997）は、それぞれ固有の潜在分布 $f_g$ を持つ複数の母集団を同時に推定する。
周辺対数尤度は群についての和 $\ell=\sum_g\sum_{i\in g}\log\sum_q L(\mathbf y_i\mid\theta_q)w_{gq}$ である。項目パラメタは
群ごとに違ってよいが、群間で等しく制約した**アンカー項目**は例外で、アンカーがすべての群を参照群の尺度に載せるので、
焦点群の平均・分散が推定可能になる。等値制約は複数のパラメタが 1 つの自由パラメタを共有することで実装しており、
項目ごとの M ステップは保たれる。

**特異項目機能（DIF）.** $\theta$ が同じでも群によって反応確率が違う項目は DIF を示す。アンカーで尺度を結んだうえで、
その項目のパラメタを群間で自由にしたモデルと等しくしたモデルを比べて検定する（[11 節](#11-モデル適合と-dif)）。

**固定項目パラメタ推定（FIPC）**（Kim, 2006）は新しい項目を既存の項目バンクの尺度に載せる: バンクの項目を既知の値に
固定し、新しい項目と潜在分布の平均・分散を 1 回の EM で推定する。新しい標本がバンクの推定標本と違うとき、潜在分布を
$N(0,1)$ のままにすると固定項目と矛盾するので、潜在分布を反復ごとに更新する（Kim の「重みを毎回更新し、EM を複数回
回す」方式 MWU-MEM に相当）。

ガイド: [9 節](guide.ja.md#9-多母集団)、[10 節](guide.ja.md#10-固定項目パラメタ推定fipc)。

## 7. 事前分布と MAP 推定

小さな標本や、反応がパラメタをほとんど制約しない項目（3PL の下側漸近線など）では、推定値が不安定になったり
境界に張り付いたりする。ベイズモーダル推定（Mislevy, 1986）は対数事前密度を目的関数に加え、対数事後
$$\ell(\eta)+\log\pi(\eta)$$
を最大化する。EM では事前分布は M ステップに入るだけで、各項目は $Q_j(\xi_j)+\log\pi(\xi_j)$ を最大化し、EM は
対数事後について単調になる。事前分布は（mirt と同じく）自然なパラメタ尺度で、ヤコビアン項なしで定義する
（例: $g$ そのものへのベータ事前分布、$a$ への対数正規事前分布）。結果は事後モードなので、その点の対数尤度は
最大値ではない。このとき AIC・BIC は本来の意味を失うので、flexpyirt は求められると警告する。

ガイド: [11 節](guide.ja.md#11-事前分布と-map-推定)。

## 8. 欠測反応

欠測した反応は尤度から単に抜ける: 積 $L(\mathbf y_i\mid\theta)=\prod_jP_j(y_{ij}\mid\theta)$ は観測された項目だけを
走る。これが正しいのは反応が**ランダムな欠測（MAR）**のときである（Rubin, 1976）: 欠測するかどうかは観測された量
（群や実施したテストフォームを含む）に依存してよいが、それを与えたもとで、観測されなかった反応そのものには依存しない。
不完全計画（共通項目を持つ別フォーム）による計画的な欠測はこれを満たす。受検者の能力に依存する無答や未到達は満たさない
ことがあり、明示的に採点するか欠測のモデルで扱う方がよい。

## 9. 標準誤差

漸近的な標準誤差は、推定値における**観測情報行列** $I(\hat\eta)=-\partial^2\ell/\partial\eta\,\partial\eta^\top$
（MAP では $-\partial^2\log\pi/\partial\eta\,\partial\eta^\top$ を加える）の逆行列から得る。

- **Hessian**（`se="hessian"`）: 周辺対数尤度の 2 階微分を自動微分で直接計算する。
- **Oakes の恒等式**（Oakes, 1999; `se="oakes"`、既定）: 同じ行列を EM の目的関数から計算する:
  $$I(\hat\eta)=-\left[\frac{\partial^2Q(\eta\mid\eta')}{\partial\eta\,\partial\eta^\top}+\frac{\partial^2Q(\eta\mid\eta')}{\partial\eta\,\partial\eta'^\top}\right]_{\eta=\eta'=\hat\eta}.$$
  第 1 項は完全データの情報量、第 2 項は $\theta$ が観測されないことで失われる情報の補正である（Louis, 1982 に
  同値な形がある）。自動微分ではどちらの項も厳密なので、2 つの方法は丸め誤差の範囲で一致する。

どちらも、収束した内点解でのみ意味がある。行列が正定値でない場合（推定値が境界上にある、識別が弱い）、flexpyirt は
NaN を報告する。

**クロス積とサンドイッチ。** 推定値における受検者 $i$ のスコアを $s_i=\partial\log L_i/\partial\eta$、その
クロス積を $B=\sum_i s_is_i^\top$ とする。モデルが正しければ $B$ と $I$ は同じ行列を推定する（情報等式）ので、
$B^{-1}$ も共分散の推定になる（`se="crossprod"`）。モデルが正しくないと両者は一致せず、推定量は真の分布を
最もよく近似するパラメタに収束し、その共分散は**サンドイッチ**（Huber, 1967; White, 1982; `se="sandwich"`）
$$\operatorname{Var}(\hat\eta)\approx I(\hat\eta)^{-1}\,B\,I(\hat\eta)^{-1}$$
になる。情報等式のもとでは $I^{-1}$ に一致する。MAP では、事前分布の曲率
$-\partial^2\log\pi/\partial\eta\,\partial\eta^\top$ を $I$ と同じように $B$ にも加えるので、モデルが正しければ
3 つの推定はやはり一致する。スコアは厳密である（各応答パターンの周辺尤度の自動微分）。受検者は互いに独立と
仮定する。

**デルタ法.** $b=-d/a$ のような関数 $b=h(\eta)$ について
$\operatorname{Var}(\hat b)\approx\nabla h^\top\,\operatorname{Var}(\hat\eta)\,\nabla h$。flexpyirt は表示用の IRT 閾値と、
標準化した混合成分にこれを使う。

ガイド: [6 節](guide.ja.md#6-標準誤差)。

## 10. 受検者のスコアリング

項目を推定したあと、項目パラメタを既知として各受検者の $\theta$ を推定する。

| 手法 | 推定値 | 標準誤差 |
|---|---|---|
| EAP（Bock & Mislevy, 1982） | 求積格子上の事後平均 $\hat\theta=\sum_q\theta_q h_q$ | 事後 SD |
| MAP | 事後モード: $\log L(\mathbf y\mid\theta)+\log\phi(\theta;\mu,\sigma^2)$ を最大化 | モードでの $\{-\partial^2/\partial\theta^2\}^{-1/2}$ |
| ML | $\log L(\mathbf y\mid\theta)$ を最大化 | 観測情報量による $I(\hat\theta)^{-1/2}$ |
| WLE（Warm, 1989） | $\partial\log L/\partial\theta+I'(\theta)/\{2I(\theta)\}=0$ を解く | 重み付き尤度の曲率から |
| PV（Mislevy, 1991） | 各受検者の事後分布からの無作為抽出 | — |

- **ベイズ的なスコア**（EAP、MAP）は推定した母集団分布を事前分布に使う。個人の平均二乗誤差は最も小さいが、
  群の平均へ縮小され（短いテストほど強い）、その分散は母集団の分散を過小評価する。
- **ML** は事前分布を使わないので縮小されないが、短いテストでは $1/J$ のオーダーの偏りがあり、全最低・全最高の
  パターンでは無限大になる。**WLE**（$\log L+\tfrac12\log I(\theta)$ の最大化と同値; Firth, 1993）は ML の主要な
  $O(1/J)$ の偏りを取り除き、有限のままである。
- **Plausible values** は母集団レベルの分析（平均、回帰）のためのもの: 抽出したセットごとに分析し、結果を Rubin の
  規則で統合すると、点推定値では伝わらない測定誤差を正しく反映できる。

ガイド: [7 節](guide.ja.md#7-スコアリング)。

## 11. モデル適合と DIF

**情報量規準.** 自由パラメタ数を $k$ として $\mathrm{AIC}=-2\ell+2k$、$\mathrm{BIC}=-2\ell+k\log N$。同じデータに
当てはめたモデルを比べる（小さいほどよい）。BIC の方が複雑さへの罰則が強い。

**尤度比検定.** 入れ子のモデルでは $G^2=2(\ell_{\text{大}}-\ell_{\text{小}})$ が、自由パラメタ数の差を自由度とする
$\chi^2$ に近似的に従う。ただし小さいモデルが大きいモデルの境界上にないことが条件である（たとえば潜在分布の形を
正規分布に対して検定するときは成り立たない）。DIF 検定はこれを 1 項目ずつ使う。

**項目適合: S-X2**（Orlando & Thissen, 2000）. 受検者を合計得点 $s$ でまとめる。各得点・カテゴリについて、
項目 $j$ でカテゴリ $k$ を選んだ観測割合を、モデルが示す割合
$$E_{jk}(s)=\frac{\sum_qP_{jk}(\theta_q)\,L_{-j}(s-k\mid\theta_q)\,w_q}{\sum_qL(s\mid\theta_q)\,w_q}$$
と比べる。$L(s\mid\theta)$ は合計得点 $s$ の確率（Lord–Wingersky の漸化式で求める）、$L_{-j}$ は項目 $j$ を除いた
もの。推定した $\theta$ ではなく観測された合計得点でまとめるので、疎なセルをまとめたあとの統計量が近似的に
$\chi^2$ に従う。

**限定情報の適合度: $M_2$**（Maydeu-Olivares & Joe, 2005, 2006）. すべての反応パターン（$\prod_jK_j$ 通り）を使う
完全情報の適合度検定は、ほとんどのパターンが観測されないため破綻する。$M_2$ は 1 次・2 次の周辺（項目平均と
ペアごとの交差積）だけを使う。それらの残差を $\mathbf e_2$、漸近共分散を $\Xi_2$、パラメタについての微分を $\Delta_2$
として
$$M_2=N\,\mathbf e_2^\top C_2\,\mathbf e_2,\qquad C_2=\Xi_2^{-1}-\Xi_2^{-1}\Delta_2(\Delta_2^\top\Xi_2^{-1}\Delta_2)^{-1}\Delta_2^\top\Xi_2^{-1}$$
で、（周辺の数 − パラメタ数）を自由度とする $\chi^2$ に近似的に従う。そこから導く RMSEA は、標本サイズに
よらない尺度で近似的な適合を表す。

ガイド: [5 節](guide.ja.md#5-結果を読む)、[9 節](guide.ja.md#9-多母集団)。

## 12. 尺度リンキング

2 つのテストフォームを別々に（それぞれの $N(0,1)$ の尺度で）推定すると、パラメタは線形変換
$\theta^*=A\theta+B$ だけ異なる尺度に載る（3 節）。共通項目から $A$ と $B$ を推定できる。この変換のもとで、
元（source）フォームの傾き–切片パラメタは $a^*=a/A$、$d^*=d-aB/A$ になる（GPCM では $d_k^*=d_k-k\,aB/A$。
名義反応モデルでは $d_k^*=d_k-\mathrm{ak}_k\,aB/A$ で、スコアリング係数 $\mathrm{ak}_k$ は変わらない）。
3PL・4PL の漸近線は変わらない。

**特性曲線法**は、変換した source の項目が基準（target）母集団上で target の項目の反応曲線を再現するように $(A,B)$ を
選ぶ:

- **Haebara**（1980）: カテゴリ反応曲線の差の 2 乗
  $\sum_qw_q\sum_j\sum_k\{P^{\text{tgt}}_{jk}(\theta_q)-P^{\text{src}}_{jk}((\theta_q-B)/A)\}^2$
  （すべてのカテゴリについて和をとる。Kim & Lee, 2006 による多値への拡張）。
- **Stocking–Lord**（1983）: テスト特性曲線（共通項目の期待合計得点）の差の 2 乗
  $\sum_qw_q\{T^{\text{tgt}}(\theta_q)-T^{\text{src}}((\theta_q-B)/A)\}^2$。
- **対称版**（Kim & Lee, 2006）: 逆向きの同じ規準を加える。B を A に結んだ結果が、A を B に結んだ変換のちょうど
  逆になる。

特性曲線法は一般にモーメント法（mean/mean、mean/sigma）より安定で、flexpyirt はモーメント法を初期値にだけ使う。
$(A,B)$ の標準誤差は、規準の勾配が 0 という条件にデルタ法を適用して得る（Ogasawara, 2001）。両方の推定の
標本誤差が反映される。リンキングの質は共通項目で決まるので、まず 2 つのフォーム間で共通項目の DIF を確かめる。

**多次元モデル。** 因子が $D$ 個あるとき、尺度の違いは
$\boldsymbol\theta^*=\mathbf A\boldsymbol\theta+\boldsymbol\beta$ で、$\mathbf a^*=\mathbf A^{-\top}\mathbf a$、
$d^*=d-\mathbf a^{*\top}\boldsymbol\beta$ になる（Oshima, Davey & Lee, 2000）。同じ 2 つの規準を $D$ 次元の格子の上で
和をとり、source の曲線は $\mathbf A^{-1}(\boldsymbol\theta_q-\boldsymbol\beta)$ で評価する。$\mathbf A$ がどんな行列で
ありうるかは、2 つの推定をどう識別したかで決まる。0 に固定した傾きの構造が同じで因子の分散を 1 にした確認的モデルどうし
なら、違いは因子ごとの単位と原点だけなので $\mathbf A$ は対角行列になる。因子軸そのものが違いうるなら、$\mathbf A$ は
回転と拡大の積（Li & Lissitz, 2000）か一般の正則行列（Oshima, Davey & Lee, 2000）になり、変換後の項目は source で 0 だった
傾きが 0 でなくなる。

ガイド: [15 節](guide.ja.md#15-尺度リンキング)。

## 参考文献

- Baker, F. B., & Kim, S.-H. (2004). *Item Response Theory: Parameter Estimation Techniques* (2nd ed.). Marcel Dekker.
- Birnbaum, A. (1968). Some latent trait models and their use in inferring an examinee's ability. In F. M. Lord & M. R. Novick, *Statistical Theories of Mental Test Scores*. Addison-Wesley.
- Bock, R. D. (1972). Estimating item parameters and latent ability when responses are scored in two or more nominal categories. *Psychometrika*, 37, 29–51.
- Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of item parameters: Application of an EM algorithm. *Psychometrika*, 46, 443–459.
- Bock, R. D., & Lieberman, M. (1970). Fitting a response model for n dichotomously scored items. *Psychometrika*, 35, 179–197.
- Bock, R. D., & Mislevy, R. J. (1982). Adaptive EAP estimation of ability in a microcomputer environment. *Applied Psychological Measurement*, 6, 431–444.
- Bock, R. D., & Zimowski, M. F. (1997). Multiple group IRT. In W. J. van der Linden & R. K. Hambleton (Eds.), *Handbook of Modern Item Response Theory*. Springer.
- Chalmers, R. P. (2012). mirt: A multidimensional item response theory package for the R environment. *Journal of Statistical Software*, 48(6).
- Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *JRSS B*, 39, 1–38.
- Firth, D. (1993). Bias reduction of maximum likelihood estimates. *Biometrika*, 80, 27–38.
- Haebara, T. (1980). Equating logistic ability scales by a weighted least squares method. *Japanese Psychological Research*, 22, 144–149.
- Huber, P. J. (1967). The behavior of maximum likelihood estimates under nonstandard conditions. In *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability* (Vol. 1, pp. 221–233). University of California Press.
- Kim, S. (2006). A comparative study of IRT fixed parameter calibration methods. *Journal of Educational Measurement*, 43, 355–381.
- Kim, S., & Lee, W.-C. (2006). An extension of four IRT linking methods for mixed-format tests. *Journal of Educational Measurement*, 43, 193–213.
- Kolen, M. J., & Brennan, R. L. (2014). *Test Equating, Scaling, and Linking* (3rd ed.). Springer.
- Li, Y. H., & Lissitz, R. W. (2000). An evaluation of the accuracy of multidimensional IRT linking. *Applied Psychological Measurement*, 24, 115–138.
- Lord, F. M., & Wingersky, M. S. (1984). Comparison of IRT true-score and equipercentile observed-score "equatings". *Applied Psychological Measurement*, 8, 453–461.
- Louis, T. A. (1982). Finding the observed information matrix when using the EM algorithm. *JRSS B*, 44, 226–233.
- Masters, G. N. (1982). A Rasch model for partial credit scoring. *Psychometrika*, 47, 149–174.
- Maydeu-Olivares, A., & Joe, H. (2005). Limited- and full-information estimation and goodness-of-fit testing in $2^n$ contingency tables. *Journal of the American Statistical Association*, 100, 1009–1020.
- Maydeu-Olivares, A., & Joe, H. (2006). Limited information goodness-of-fit testing in multidimensional contingency tables. *Psychometrika*, 71, 713–732.
- Mislevy, R. J. (1984). Estimating latent distributions. *Psychometrika*, 49, 359–381.
- Mislevy, R. J. (1986). Bayes modal estimation in item response models. *Psychometrika*, 51, 177–195.
- Mislevy, R. J. (1991). Randomization-based inference about latent variables from complex samples. *Psychometrika*, 56, 177–196.
- Muraki, E. (1992). A generalized partial credit model: Application of an EM algorithm. *Applied Psychological Measurement*, 16, 159–176.
- Oakes, D. (1999). Direct calculation of the information matrix via the EM algorithm. *JRSS B*, 61, 479–482.
- Ogasawara, H. (2001). Standard errors of item response theory equating/linking by response function methods. *Applied Psychological Measurement*, 25, 53–67.
- Oshima, T. C., Davey, T. C., & Lee, K. (2000). Multidimensional linking: Four practical approaches. *Journal of Educational Measurement*, 37, 357–373.
- Orlando, M., & Thissen, D. (2000). Likelihood-based item-fit indices for dichotomous item response theory models. *Applied Psychological Measurement*, 24, 50–64.
- Rasch, G. (1960). *Probabilistic Models for Some Intelligence and Attainment Tests*. Danish Institute for Educational Research.
- Reckase, M. D. (2009). *Multidimensional Item Response Theory*. Springer.
- Reckase, M. D., & McKinley, R. L. (1991). The discriminating power of items that measure more than one dimension. *Applied Psychological Measurement*, 15(4), 361–373.
- Rubin, D. B. (1976). Inference and missing data. *Biometrika*, 63, 581–592.
- Samejima, F. (1969). Estimation of latent ability using a response pattern of graded scores. *Psychometrika Monograph*, 17.
- Stocking, M. L., & Lord, F. M. (1983). Developing a common metric in item response theory. *Applied Psychological Measurement*, 7, 201–210.
- Thissen, D., Cai, L., & Bock, R. D. (2010). The nominal categories item response model. In M. L. Nering & R. Ostini (Eds.), *Handbook of Polytomous Item Response Theory Models* (pp. 43–75). Routledge.
- Varadhan, R., & Roland, C. (2008). Simple and globally convergent methods for accelerating the convergence of any EM algorithm. *Scandinavian Journal of Statistics*, 35, 335–353.
- Warm, T. A. (1989). Weighted likelihood estimation of ability in item response theory. *Psychometrika*, 54, 427–450.
- White, H. (1982). Maximum likelihood estimation of misspecified models. *Econometrica*, 50(1), 1–25.
- Woods, C. M. (2007). Empirical histograms in item response theory with ordinal data. *Educational and Psychological Measurement*, 67, 73–87.
- Woods, C. M., & Lin, N. (2009). Item response theory with estimation of the latent density using Davidian curves. *Applied Psychological Measurement*, 33, 102–117.
- Zhang, D., & Davidian, M. (2001). Linear mixed models with flexible distributions of random effects for longitudinal data. *Biometrics*, 57, 795–802.
