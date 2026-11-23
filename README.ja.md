# flexpyirt

[English](https://github.com/takuizum/flexpyirt/blob/main/README.md) | 日本語

JAX による項目反応理論（IRT）の推定ライブラリ。Bock–Aitkin EM による周辺最尤推定（MMLE）を中核に、
多母集団の同時推定や固定項目パラメタ推定（FIPC）を Python から扱えるようにする。

- **正しさを検証済み**: 対応済みの機能はすべて R `mirt` と数値照合している（パラメタ・対数尤度・標準誤差が
  おおむね 1e-8 で一致。[検証の詳細](https://takuizum.github.io/flexpyirt/ja/VALIDATION/)）。
- **自動微分**: M-step の勾配・Hessian も標準誤差（Oakes の恒等式）も JAX の自動微分で計算する。
  新しい項目モデルは対数確率を書くだけで推定・SE・スコアリングが動く設計。
- **ParameterTable**: mirt の `pars = "values"` に相当する表で、固定・等値・初期値を編集できる。

> 開発中（0.x）。公開 API は 1.0.0 まで変わることがある。

**ドキュメント: <https://takuizum.github.io/flexpyirt/ja/>**

## できること

| 機能 | 内容 | mirt 照合 |
|---|---|---|
| 項目モデル | 2PL、Rasch（潜在分散を自由推定）、3PL、4PL、GRM（段階反応モデル）、GPCM、PCM（潜在分散を自由推定）、Nominal（名義反応モデル。1 因子・2 因子で照合済み、3 因子は単体テストのみ） | V1, V2, V3, V8, V11, V23 |
| 事前分布と MAP | 項目パラメタの事前分布 `NormalPrior`・`LogNormalPrior`・`BetaPrior`（`Model.prior()`）、自然尺度、`FitResult.log_posterior` | V7 |
| 多母集団 | 群ごとの項目パラメタ、アンカー項目の群間等値、焦点群の平均・分散の推定 | V4 |
| 項目間の等値制約 | 項目・項目モデルをまたぐ共通パラメタ（`Model.equal_across_items()`、例: 全項目共通の傾き）。群間等値と組み合わせ可 | V16 |
| FIPC | 項目バンクの値で一部の項目を固定し、残りの項目と潜在分布を推定 | V5 |
| 欠測 | 欠測（MAR）を含むデータ | V9 |
| 標準誤差 | Oakes の恒等式（既定）／周辺対数尤度（MAP 時は対数事後）の Hessian、表示用 IRT 閾値のデルタ法 SE（`b_se` / `bk_se`） | V7, V10, V12, V14 |
| スコアリング | EAP・MAP・ML・WLE（1–3 因子、標準誤差つき）および plausible values（`PV`、1–3 因子、明示的な JAX PRNG `key`）。ベイズ手法は群ごとの潜在分布を事前分布に使う | V1–V5, V7–V9, V11, V13, V14 |
| 多次元モデル | 確認的 2–3 因子 MIRT（`Model.with_factors()`、多母集団も可）。交差負荷、相関／直交の多変量正規因子、直積求積格子、Oakes/Hessian SE、EAP・MAP・ML・WLE スコアリング | V14, V21 |
| 経験ヒストグラム | 歪んだ・多峰の潜在分布のための `EmpiricalHistogram()`（Woods, 2007）。参照群は厳密な制約付き M-step で平均 0・分散 1 に標準化、FIPC では自由。重みと 4 次までのモーメントを報告 | V6 |
| Davidian 曲線 | 単一群 1 次元モデル向けの半ノンパラメトリック潜在分布 `DavidianCurve(degree)`（Zhang & Davidian, 2001; Woods & Lin, 2009）。厳密な連続 `mean`/`var` へ解析的に標準化（Rasch・PCM・FIPC で `free_mean`/`free_var` 可）、項目・潜在パラメタの同時 Oakes/Hessian SE と EAP/PV 採点に対応 | V19 |
| 混合正規 | 単一群 1 次元モデル向けの、項目を共通とする有限混合正規潜在密度 `NormalMixture(n_components)`（Mislevy, 1984）。mirt の `dentype = "mixture-k"` のクラスのパラメタ化を厳密な連続 `mean`/`var` へ解析的に標準化、標準化した成分（`pi_c`・`mean_c`・`var_c`）とデルタ法 SE、項目・潜在パラメタの同時 Oakes/Hessian SE、EAP/PV 採点、退化解の警告 | V22 |
| 結果の保存 | `FitResult.to_json()` / `FitResult.from_json()`（厳密なスキーマ、pickle 不使用） | — |
| EM の加速 | 単調性の安全装置つき SQUAREM（スキーム S3）、`BAEM(squarem=True)`。有無どちらでも mirt と一致 | V1–V5, V7–V9, V11, V14 |
| 項目適合度 | Orlando–Thissen の S-X2（`FitResult.item_fit()`、多母集団は群ごと）。mirt と同じセルのまとめ方・自由度・RMSEA・p 値 | V15 |
| 限定情報の適合度 | Maydeu-Olivares & Joe の M2 / M2*（`FitResult.m2()`）。自由度・p 値・RMSEA と信頼区間・SRMSR・2 値の TLI / CFI | V18 |
| DIF | 多母集団の尤度比検定による DIF（`FitResult.dif()`。mirt の `DIF()` の `add` / `drop` 方式、AIC・SABIC・HQ・BIC の差、Holm または BH で補正した p 値） | V17 |
| 尺度リンキング | 微分可能な特性曲線法リンキング（`link()`, `LinkResult`: Haebara・Stocking–Lord の非対称・対称版、Ogasawara デルタ法 SE、項目・$\theta$ の変換。2PL・3PL・GRM・GPCM と 2PL + GPCM 混在の共通項目で照合済み） | V20 |
| 識別性の検査 | 識別できない指定は推定前に `IdentificationError` | — |

今後の予定は[ロードマップ](https://takuizum.github.io/flexpyirt/ja/ROADMAP/)を参照。

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

R `mirt`（既定設定、`TOL = 1e-4`、CPU・61 点求積）との wall time ベンチマークを
[benchmarks/](https://github.com/takuizum/flexpyirt/blob/main/benchmarks/README.ja.md) に、Apple M5 での計測結果を
[benchmarks/results/2026-11-21-apple-m5.md](https://github.com/takuizum/flexpyirt/blob/main/benchmarks/results/2026-11-21-apple-m5.md) に
置いている。初回の JAX/XLA コンパイル（cold の EM 推定で約 0.8–1.1 秒）以降は、テストした全ケースで
warm の CPU 推定が `mirt` より速く（`mirt` 比 `0.12×–0.60×`）、標準誤差なしの 1 回限りの cold 推定では
`mirt` の方が速い。

## ドキュメント

- [ユーザーガイド](https://takuizum.github.io/flexpyirt/ja/guide/) — モデルの指定、多母集団、FIPC、ParameterTable の編集、推定オプション
- [API リファレンス](https://takuizum.github.io/flexpyirt/ja/reference/) — 公開クラス・関数の一覧と引数
- [検証](https://takuizum.github.io/flexpyirt/ja/VALIDATION/) — R mirt との照合方法、許容誤差、判明した違い
- [設計書](https://takuizum.github.io/flexpyirt/ja/DESIGN/) — 数式とアルゴリズム、実装上の決定
- [ロードマップ](https://takuizum.github.io/flexpyirt/ja/ROADMAP/)
- [開発フロー](https://github.com/takuizum/flexpyirt/blob/main/CONTRIBUTING.ja.md) — ブランチ運用、レビュー、リリース

すべての文書に英語版（`*.md`）がある。

## 開発

```bash
uv sync --all-extras --dev
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
```

ブランチ運用とリリースの手順は [CONTRIBUTING.md](https://github.com/takuizum/flexpyirt/blob/main/CONTRIBUTING.ja.md) を参照。

## ライセンス

MIT
