# flexpyirt

[English](https://github.com/takuizum/flexpyirt/blob/main/README.md) | 日本語

JAX による項目反応理論（IRT）の推定ライブラリ。Bock–Aitkin EM による周辺最尤推定（MMLE）を中核に、
多母集団の同時推定や固定項目パラメタ推定（FIPC）を Python から扱えるようにする。

- **正しさを検証済み**: 対応済みの機能はすべて R `mirt` と数値照合している（パラメタ・対数尤度・標準誤差が
  おおむね 1e-8 で一致。[精度と検証](https://takuizum.github.io/flexpyirt/ja/accuracy/)）。
- **厳密な微分**: 勾配・Hessian・標準誤差（Oakes の恒等式）は数値近似ではなく JAX の自動微分で計算する。
  推定は JAX でコンパイルされ、GPU でも動く。
- **ParameterTable**: mirt の `pars = "values"` に相当する表で、固定・等値・初期値を編集できる。

> 開発中（0.x）。公開 API は 1.0.0 まで変わることがある。

**ドキュメント: <https://takuizum.github.io/flexpyirt/ja/>**

## できること

| 機能 | 内容 | mirt と照合 |
|---|---|---|
| 項目モデル | 2PL、Rasch（潜在分散を自由推定）、3PL、4PL、GRM（段階反応モデル）、GPCM、PCM（潜在分散を自由推定）、Nominal（名義反応モデル。1 因子・2 因子で mirt と照合済み） | ✓ |
| 事前分布と MAP | 項目パラメタの事前分布 `NormalPrior`・`LogNormalPrior`・`BetaPrior`（`Model.prior()`）、自然尺度、`FitResult.log_posterior` | ✓ |
| 多母集団 | 群ごとの項目パラメタ、アンカー項目の群間等値、焦点群の平均・分散の推定 | ✓ |
| 項目間の等値制約 | 項目・項目モデルをまたぐ共通パラメタ（`Model.equal_across_items()`、例: 全項目共通の傾き）。群間等値と組み合わせ可 | ✓ |
| FIPC | 項目バンクの値で一部の項目を固定し、残りの項目と潜在分布を推定 | ✓ |
| 欠測 | 欠測（MAR）を含むデータ | ✓ |
| 標準誤差 | Oakes の恒等式（既定）／周辺対数尤度（MAP 時は対数事後）の Hessian、表示用 IRT 閾値のデルタ法 SE（`b_se` / `bk_se`） | ✓ |
| スコアリング | EAP・MAP・ML・WLE（1–3 因子、標準誤差つき）および plausible values（`PV`、1–3 因子、明示的な JAX PRNG `key`）。ベイズ手法は群ごとの潜在分布を事前分布に使う | ✓ |
| 多次元モデル | 確認的 2–3 因子 MIRT（`Model.with_factors()`、多母集団も可）。交差負荷、相関／直交の多変量正規因子、直積求積格子、Oakes/Hessian SE、EAP・MAP・ML・WLE スコアリング。次元縮約による特殊因子数に制限のない双因子モデル（`Model.with_bifactor()`、単一群、EAP スコアリング）。相関してよい主因子 1〜2 個の two-tier モデル（`Model.with_two_tier()`） | ✓ |
| 経験ヒストグラム | 歪んだ・多峰の潜在分布のための `EmpiricalHistogram()`（Woods, 2007）。参照群は厳密な制約付き M-step で平均 0・分散 1 に標準化、FIPC では自由。重みと 4 次までのモーメントを報告 | ✓ |
| Davidian 曲線 | 単一群 1 次元モデル向けの半ノンパラメトリック潜在分布 `DavidianCurve(degree)`（Zhang & Davidian, 2001; Woods & Lin, 2009）。厳密な連続 `mean`/`var` へ解析的に標準化（Rasch・PCM・FIPC で `free_mean`/`free_var` 可）、項目・潜在パラメタの同時 Oakes/Hessian SE と EAP/PV 採点に対応 | ✓ |
| 混合正規 | 単一群 1 次元モデル向けの、項目を共通とする有限混合正規潜在密度 `NormalMixture(n_components)`（Mislevy, 1984）。mirt の `dentype = "mixture-k"` のクラスのパラメタ化を厳密な連続 `mean`/`var` へ解析的に標準化、標準化した成分（`pi_c`・`mean_c`・`var_c`）とデルタ法 SE、項目・潜在パラメタの同時 Oakes/Hessian SE、EAP/PV 採点、退化解の警告 | ✓ |
| 結果の保存 | `FitResult.to_json()` / `FitResult.from_json()`（厳密なスキーマ、pickle 不使用） | — |
| EM の加速 | 単調性の安全装置つき SQUAREM（スキーム S3）、`BAEM(squarem=True)`。有無どちらでも mirt と一致 | ✓ |
| 項目適合度 | Orlando–Thissen の S-X2（`FitResult.item_fit()`、多母集団は群ごと）。mirt と同じセルのまとめ方・自由度・RMSEA・p 値 | ✓ |
| 限定情報の適合度 | Maydeu-Olivares & Joe の M2 / M2*（`FitResult.m2()`）。自由度・p 値・RMSEA と信頼区間・SRMSR・2 値の TLI / CFI | ✓ |
| DIF | 多母集団の尤度比検定による DIF（`FitResult.dif()`。mirt の `DIF()` の `add` / `drop` 方式、AIC・SABIC・HQ・BIC の差、Holm または BH で補正した p 値） | ✓ |
| 尺度リンキング | 微分可能な特性曲線法リンキング（`link()`, `LinkResult`: Haebara・Stocking–Lord の非対称・対称版、Ogasawara デルタ法 SE、項目・$\theta$ の変換。2PL・3PL・GRM・GPCM と 2PL + GPCM 混在の共通項目で R の独立な実装と照合済み） | ✓ |
| ポリコリック相関 | ポリコリック・テトラコリック相関行列（`polychoric()`: two-step の最尤法、欠測は利用可能なケース、lavaan と同じ 2 x 2 表の空セル補正、正定値でない行列の平滑化。lavaan の `lavCor` と照合済み） | ✓ |
| 次元数 | 平行分析（`parallel_analysis()`: ポリコリックまたは Pearson 相関、PCA または SMC の固有値、並べ替えまたは正規の参照データ、分位点または平均の基準）、Velicer の MAP（`velicer_map()`、元の版と改訂版）、経験的カイザー基準（`empirical_kaiser()`）、回転しない ULS / ML の因子分析（`efa()`）、Hull 法（`hull()`）、比較データ法（`comparison_data()`）。EFAtools・psych・EFA.dimensions と照合済み） | ✓ |
| 局所依存と一次元性 | Q3、Q3\*、Chen & Thissen の LD $X^2$ / $G^2$（`FitResult.local_dependence()`、mirt の `residuals()` と同じ）、ジャックナイフ傾き指標（`FitResult.jsi()`、mirt と同じ。速い近似もある）、モデルに基づく平行分析（`model_parallel_analysis()`、ltm の `unidimTest` と同じ）、確認的・探索的 DETECT（`detect()`、`exploratory_detect()`、sirt と同じ）、ブートストラップでバイアス補正する DIMTEST（`dimtest()`、シミュレーションで確認）、複数の手法を実行して合意を別々に示す `dimensionality()` のレポート | ✓ |
| 識別性の検査 | 識別できない指定は推定前に `IdentificationError` | — |

## インストール

Python 3.11 以上。

```bash
pip install flexpyirt  # または uv add flexpyirt
```

polars の DataFrame を使う場合は `flexpyirt[polars]`。GPU を使う場合は、先に CUDA 版の `jax` を入れる
（[JAX のインストール手順](https://docs.jax.dev/en/latest/installation.html)）。

flexpyirt は **64 ビット浮動小数点**で計算する。JAX の設定はユーザーのものなのでライブラリからは変えない。
推定の前に有効にしておく（無効なら `fit()` が `PrecisionError` を出す）。

```python
import jax

jax.config.update("jax_enable_x64", True)
```

## クイックスタート

例として 2PL のデータをシミュレーションして推定する。実データでは、列が項目・行が回答者の
DataFrame（値は 0/1、欠測は NaN）を渡せばよい。

```python
import numpy as np
import pandas as pd

from flexpyirt import Model, TwoPL, fit

rng = np.random.default_rng(1)
n_persons, n_items = 1000, 10
a = rng.lognormal(0.0, 0.3, n_items)  # 識別力
d = rng.normal(0.0, 1.0, n_items)  # 切片
theta = rng.normal(0.0, 1.0, n_persons)
p = 1.0 / (1.0 + np.exp(-(np.outer(theta, a) + d)))
df = pd.DataFrame(
    (rng.random(p.shape) < p).astype(int),
    columns=[f"q{j:02d}" for j in range(1, n_items + 1)],
)

spec = Model.uniform(TwoPL(), items=df.columns)
res = fit(spec, df)

print(res)  # 対数尤度・パラメタ数・収束状況
print(res.items.head())  # item, model, a, d, a_se, d_se, b（= -d/a）, b_se
print(res.loglik, res.aic, res.bic)
scores = res.score(df)  # 回答者ごとの EAP（theta）と事後標準偏差（se）
```

パラメタは mirt と同じ傾き・切片形式 $P(y=1\mid\theta)=\sigma(a\theta+d)$ で推定し、
`items` には表示用に $b=-d/a$ も載せる。

## 使い方の例

詳しくは [ユーザーガイド](https://takuizum.github.io/flexpyirt/ja/guide/)、関数ごとの説明は [API リファレンス](https://takuizum.github.io/flexpyirt/ja/reference/)。

**Rasch モデル**（傾きは 1 に固定、潜在分布の分散を推定）

```python
from flexpyirt import Rasch

rasch = fit(Model.uniform(Rasch(), items=df.columns), df)
print(rasch.latent)  # group, mean（0 に固定）, var（推定値）, mean_se, var_se
```

**多母集団（アンカー項目で 2 群をつなぐ）**

```python
from flexpyirt import Normal

group = np.repeat(["ref", "focal"], n_persons // 2)
spec2 = (
    Model.uniform(TwoPL(), items=df.columns)
    .with_groups(ref=Normal.standard(), focal=Normal.free())  # 最初の群が参照群
    .equal_across_groups(items=["q01", "q02", "q03", "q04"])  # アンカー項目
)
res2 = fit(spec2, df, group=group)
print(res2.latent)  # focal の平均・分散が推定される
```

**FIPC（項目バンクの値で固定して新しい項目を推定）**

```python
bank = pd.DataFrame({"item": ["q01", "q02", "q03"], "a": a[:3], "d": d[:3]})
spec3 = Model.uniform(TwoPL(), items=df.columns).fix_from(bank).with_latent(Normal.free())
res3 = fit(spec3, df)
```

## 性能

R `mirt`（既定設定、`TOL = 1e-4`、CPU・61 点求積）との実行時間の比較
（[Apple M5 での計測結果](https://github.com/takuizum/flexpyirt/blob/main/benchmarks/results/2026-11-21-apple-m5.md)）では、初回の JAX/XLA コンパイル（cold の EM 推定で約 0.8–1.1 秒）以降は、テストした全ケースで
warm の CPU 推定が `mirt` より速く（`mirt` 比 `0.12×–0.60×`）、標準誤差なしの 1 回限りの cold 推定では
`mirt` の方が速い。

## ドキュメント

- [ユーザーガイド](https://takuizum.github.io/flexpyirt/ja/guide/) — モデルの指定、多母集団、FIPC、ParameterTable の編集、推定オプション
- [理論](https://takuizum.github.io/flexpyirt/ja/theory/) — モデル、推定アルゴリズム、標準誤差、スコアリング、適合度、リンキングの要点
- [API リファレンス](https://takuizum.github.io/flexpyirt/ja/reference/) — 公開クラス・関数の一覧と引数
- [精度と検証](https://takuizum.github.io/flexpyirt/ja/accuracy/) — R mirt との照合方法、許容誤差、mirt との既知の違い

すべての文書に英語版（`*.md`）がある。

## フィードバック

不具合の報告や質問は [GitHub Issues](https://github.com/takuizum/flexpyirt/issues) へ。

## ライセンス

MIT
