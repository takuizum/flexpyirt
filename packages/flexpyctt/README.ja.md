# flexpyctt

[English](https://github.com/takuizum/flexpyirt/blob/main/packages/flexpyctt/README.md) | 日本語

推定済みの IRT モデルを使わない項目・テスト分析のライブラリです。ポリコリック相関、テストの次元数を調べる道具、
観測得点による DIF の検出を提供します。依存は NumPy、SciPy、pandas だけです（JAX は不要）。

flexpyctt は、IRT モデルを推定する [flexpyirt](https://pypi.org/project/flexpyirt/) と対になるパッケージです。
flexpyirt をインストールすると flexpyctt も入ります。モデルを推定しない場合は flexpyctt だけをインストールできます。

- **R と数値照合済み**: 結果を lavaan、psych、EFAtools、EFA.dimensions、sirt、difR と数値で比べています
  （[精度と検証](https://takuizum.github.io/flexpyirt/ja/accuracy/)）。
- **軽い**: コンパイルも 64 ビットの設定も要りません。

> 開発中（0.x）。1.0.0 までは公開 API が変わることがあります。

**ドキュメント: <https://takuizum.github.io/flexpyirt/ja/>**

## 機能

| 機能 | 内容 | 照合先 |
|---|---|---|
| ポリコリック相関 | `polychoric()`: 2 段階最尤法によるポリコリック・テトラコリック相関行列。欠測は利用可能なケースで扱う。lavaan と同じ 2 x 2 表のゼロセル補正。正定値でない行列の平滑化 | lavaan `lavCor`、psych |
| 平行分析 | `parallel_analysis()`: ポリコリックまたは Pearson 相関、PCA または SMC の固有値、並べ替えまたは正規乱数の参照データ、分位点または平均の規則 | EFAtools |
| MAP と EKC | `velicer_map()`（原法と改訂版）、`empirical_kaiser()` | psych、EFA.dimensions、EFAtools |
| 因子分析と Hull 法 | `efa()`（回転なしの ULS / ML）、`hull()` | psych、EFAtools |
| 比較データ法 | `comparison_data()`（Ruscio & Roche, 2012） | EFAtools |
| DETECT | `detect()`、`exploratory_detect()`（条件付き共分散、DETECT、ASSI、RATIO） | sirt |
| DIMTEST | `dimtest()`（ブートストラップによるバイアス補正） | シミュレーション |
| 観測得点による DIF | `mantel_haenszel()`（χ²、共通オッズ比、ETS のデルタと A / B / C 分類）、`mantel()`（順序のある項目、Liu–Agresti のオッズ比）、`generalized_mantel_haenszel()`（多群、名義カテゴリ）、`logistic_dif()`（均一・不均一、Nagelkerke の ΔR²）。マッチング得点の純化つき | difR、`mantelhaen.test` |

## インストール

Python 3.12 以上。

```bash
pip install flexpyctt  # または: uv add flexpyctt
```

## クイックスタート

1 列が 1 項目、1 行が 1 人の DataFrame（または配列）を渡します。反応は順序のあるカテゴリのコードで、欠測は NaN です。

```python
import numpy as np
import pandas as pd

from flexpyctt import empirical_kaiser, parallel_analysis, polychoric, velicer_map

# 相関のある 2 因子を測る 2 値項目 10 個をシミュレーションする。
rng = np.random.default_rng(1)
factors = rng.normal(size=(1000, 2))
factors[:, 1] = 0.3 * factors[:, 0] + np.sqrt(1 - 0.3**2) * factors[:, 1]
loadings = np.zeros((10, 2))
loadings[:5, 0] = loadings[5:, 1] = 0.75
latent = factors @ loadings.T + rng.normal(size=(1000, 10)) * np.sqrt(1 - 0.75**2)
df = pd.DataFrame(
    (latent > np.linspace(-0.8, 0.8, 10)).astype(int),
    columns=[f"q{j:02d}" for j in range(1, 11)],
)

pc = polychoric(df)  # すべて 2 値項目なのでテトラコリック相関
print(pc.correlation.round(2))

pa = parallel_analysis(df, rng=0)  # rng: シードまたは numpy.random.Generator
print(pa.n_factors, velicer_map(pc).n_factors, empirical_kaiser(pc).n_factors)
```

乱数を使う関数は `rng`（整数のシードまたは `numpy.random.Generator`）を受け取るので、結果を再現できます。

各関数の詳細は[ユーザーガイド](https://takuizum.github.io/flexpyirt/ja/guide/)と
[API リファレンス](https://takuizum.github.io/flexpyirt/ja/reference/)を参照してください。

## ライセンス

MIT
