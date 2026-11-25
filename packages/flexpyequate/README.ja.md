# flexpyequate

[English](https://github.com/takuizum/flexpyirt/blob/main/packages/flexpyequate/README.md) | 日本語

テスト得点分布の等化を行う Python ライブラリです。2 つのテスト版の得点分布から、X 版の得点を Y 版の
尺度に対応づける関数を求めます。

- **R と数値照合済み**: すべての手法を R `equate`、および公刊された式の独立実装と数値で比較しています。
- **軽量**: 依存は NumPy・SciPy・pandas のみです。
- **範囲**: 得点分布にもとづく観測得点等化。項目パラメータを使う等化（IRT 真値等化・観測得点等化、
  尺度のリンキング）は [flexpyirt](https://github.com/takuizum/flexpyirt) にあります。

> 開発中（0.x）。1.0.0 までは公開 API が変わることがあります。

## 機能

| 機能 | 内容 | 照合先 |
|---|---|---|
| 得点分布 | 素点または度数表から作る `ScoreDistribution` と `BivariateScoreDistribution`。等間隔なら刻み幅は任意。モーメント、パーセンタイル順位とパーセンタイル（Kolen & Brennan, 2014） | R `equate` |
| デザイン | 等価群（2 つの分布）と単一群（1 つの 2 変量分布） | R `equate` |
| 平均等化・線形等化 | `equate(x, y, method="mean")`、`method="linear"`（標準偏差は母集団形または標本形） | R `equate` |
| 等パーセンタイル等化 | `equate(x, y, method="equipercentile")`。Kolen & Brennan (2014) のパーセンタイル順位の定義に従い、度数 0 の得点を含む分布にも対応 | R `equate`、教科書の式の独立 R 実装 |

## インストール

Python 3.11 以降。

```bash
pip install flexpyequate  # または: uv add flexpyequate
```

## 例

```python
import flexpyequate as fe

# 等価な 2 群が受けた 2 つの版の、得点 0..4 の度数
x = fe.ScoreDistribution.from_frequencies(scores=range(5), frequencies=[4, 6, 6, 3, 1])
y = fe.ScoreDistribution.from_frequencies(scores=range(5), frequencies=[2, 4, 6, 4, 4])

result = fe.equate(x, y, method="equipercentile")
print(result.table)
#    score   equated
# 0    0.0  0.500000
# 1    1.0  1.666667
# 2    2.0  2.750000
# 3    3.0  3.875000
# 4    4.0  4.375000

result.transform([1.5, 2.5])   # とりうる得点の間の値も等化できる
result.moments                 # X・Y・等化後得点の平均、標準偏差、歪度、尖度

linear = fe.equate(x, y, method="linear")
linear.slope, linear.intercept
```

受検者ごとの得点から作る場合と、同じ受検者が両方の版を受けた単一群の場合:

```python
x = fe.ScoreDistribution.from_scores(x_scores, score_range=(0, 40))
xy = fe.BivariateScoreDistribution.from_scores(
    x_scores, y_scores, x_range=(0, 40), y_range=(0, 40)
)
fe.equate(xy, method="equipercentile")   # 2 つの周辺分布を等化する
```

## R との一致についての注意

- R `equate` は分散を N − 1 で割ります。flexpyequate の既定は Kolen & Brennan の母集団形です。
  `ddof=1` を指定すると `equate` の線形等化と一致します。
- R `equate` の `type = "mean"` は得点に 2 つの得点範囲の比を掛けます。flexpyequate の平均等化は常に
  傾き 1 です（得点範囲が同じ版どうしでは両者は一致します）。
- X 版の得点のパーセンタイル順位がちょうど 0 または 100 のとき、flexpyequate は Kolen & Brennan の
  定義どおり Y 版の上側・下側パーセンタイルの平均を返します。`equate` は Y 版の得点範囲の端を返します。
  両者が異なるのは、Y 版の最低点（または最高点）付近を取った受検者がいないときだけです。

## 参考文献

- Kolen, M. J., & Brennan, R. L. (2014). *Test equating, scaling, and linking* (3rd ed.). Springer.
- Albano, A. D. (2016). equate: An R package for observed-score linking and equating. *Journal of
  Statistical Software, 74*(8).

## ライセンス

MIT
