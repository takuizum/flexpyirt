# flexpyirt 公開 API スケッチ

[English](API.md) | 日本語

> **実装済みの API は [API リファレンス](reference.ja.md) と [ユーザーガイド](guide.ja.md) を参照。**
> このファイルは設計段階のスケッチで、未実装の機能（EH、Gauss–Hermite、`TruncNormal`、M4 のスコアリング・MIRT など）も含む。

> 目標: 「仕様（Spec）・推定（fit）・結果（FitResult）」を分ける。仕様は不変オブジェクトで、メソッドチェーンは新しい仕様を返す。
> ここに書いたシグネチャは設計案。実装時に変える場合はこのファイルを先に更新する。

## 1. 最小例（単一群 2PL）

```python
import jax

jax.config.update("jax_enable_x64", True)

import pandas as pd
from flexpyirt import Model, TwoPL, fit

df = pd.read_csv("responses.csv")  # 列 = 項目, 0/1, 欠測 NaN
spec = Model.uniform(TwoPL(), items=df.columns)
res = fit(spec, df)

res  # Jupyter では表（_repr_html_）
res.items  # DataFrame: item, a, b(=-d/a), a_se, b_se, ...
res.loglik, res.aic, res.bic
res.score(df, method="EAP")  # DataFrame: theta, se
```

## 2. 混合形式 + 事前分布（MAP）

```python
from flexpyirt import Model, ThreePL, GRM, LogNormal, Beta, fit

spec = (
    Model.from_items(
        {
            **{f"mc{i:02d}": ThreePL() for i in range(1, 31)},
            **{f"cr{i}": GRM(n_cat=4) for i in range(1, 6)},
        }
    )
    .prior(param="a", dist=LogNormal(0.0, 0.5))
    .prior(param="g", dist=Beta(5, 17), items=lambda name: name.startswith("mc"))
)
res = fit(spec, df)
res.log_posterior, res.loglik
```

## 3. 多母集団（部分アンカー）

```python
from flexpyirt import Model, TwoPL, Normal, EmpiricalHistogram, fit

anchors = [f"q{i:02d}" for i in range(1, 11)]

spec = (
    Model.uniform(TwoPL(), items=df.columns.drop("form"))
    .with_groups(
        ref=Normal.standard(),  # 参照群 N(0,1) 固定
        focal=Normal.free(),  # 平均・分散を推定
    )
    .equal_across_groups(items=anchors)  # アンカー項目は全パラメタ群間等値
)
res = fit(spec, df.drop(columns="form"), group=df["form"])
res.latent  # DataFrame: group, mean, sd, se
res.items  # group 列つき（等値項目は同値）
```

EH にしたい場合は `focal=EmpiricalHistogram()`。参照群も EH にする場合は `ref=EmpiricalHistogram(standardize=True)`。

## 4. 固定パラメタ推定（FIPC）

```python
bank = pd.read_csv("item_bank.csv")  # item, a, d（または a, b）
spec = (
    Model.uniform(TwoPL(), items=df.columns)
    .fix_from(bank, items=anchors, parameterization="irt")  # a, b で与えてもよい
    .with_latent(Normal.free())  # FIPC では潜在分布を推定（DESIGN §4.5）
)
res = fit(spec, df)
```

個別に固定: `.fix(item="q01", a=1.2, d=0.3)`。

## 5. パラメタ表を直接編集（上級者向け、mirt の pars='values' 相当）

```python
table = spec.parameter_table(df)  # 長形式 DataFrame（DESIGN §5.1）
table.loc[(table.item == "q05") & (table.param == "a"), ["value", "free"]] = [1.0, False]
res = fit(spec, df, table=table)  # 表が spec より優先
```

## 6. 推定オプション

```python
from flexpyirt import BAEM, Quadrature

res = fit(
    spec,
    df,
    method=BAEM(tol=1e-4, max_iter=2000, squarem=True, m_step_newton_steps=5),
    quadrature=Quadrature.equispaced(61, lim=(-6, 6)),  # または Quadrature.gauss_hermite(41)
    se="oakes",  # "oakes" | "hessian" | "crossprod" | "sandwich" | None
    verbose=False,
)
res.converged, res.n_iter, res.history  # history: DataFrame(iter, loglik, max_change)
```

## 6.1 M1 時点で実装済みの範囲
- 項目モデル: `TwoPL`, `Rasch`。潜在分布: `Normal`（`Normal.standard()`, `Normal.free()`, `Normal(free_var=True)`）。
  全項目 Rasch の単一群では既定で分散自由（DESIGN §4.5）。
- `fit(spec, data, *, group=None, table=None, item_names=None, method=None, quadrature=None, se="oakes")`。
  `se` は `"oakes" | "hessian" | None`。`group=` は M2 まで `NotImplementedError`。
- `BAEM(tol=1e-4, max_iter=2000, m_step_newton_steps=3)`（`squarem` は M3）。
  `Quadrature.equispaced(n_points=61, lim=(-6, 6))`。Gauss–Hermite は `Quadrature.gauss_hermite(n)`（DESIGN §15.44）。
- `Model.uniform`, `Model.from_items`, `.fix(item=..., a=..., d=...)`, `.with_latent(...)`, `.parameter_table(data)`。
- `FitResult`: `items`, `latent`, `parameters`（表 + `estimate`, `se`）, `vcov`, `loglik`, `aic`, `bic`,
  `converged`, `n_iter`, `history`, `score(data, method="EAP")`。
- 入力は 0/1 の pandas / polars DataFrame または numpy 配列（`item_names=`）。欠測は M2 まで `SpecificationError`。

## 6.2 M2 で追加された範囲
- 項目モデル `GRM(n_cat=K)`（カテゴリ $0..K-1$）。欠測（NaN, MAR）を受け付ける。
- 多母集団: `Model.with_groups(ref=Normal.standard(), focal=Normal.free())`（最初が参照群）、
  `.equal_across_groups(items=..., params=None)`、`fit(..., group=labels)`。`FitResult.items` は `group` 列つき、
  `score(data, group=labels)`。
- FIPC: `Model.fix_from(bank, items=None, parameterization="slope-intercept" | "irt")` + `.with_latent(Normal.free())`。
- 識別不能な仕様（アンカーなしで焦点群自由、固定項目なしで平均自由など）は fit 前に `IdentificationError`。

## 6.3 M3 で追加された範囲（進行中）
- 項目モデル: `ThreePL`, `FourPL`, `GPCM(n_cat=K)`, `PCM(n_cat=K)`（境界解で `HeywoodWarning`）。
- SQUAREM 加速: `BAEM(squarem=True)`（スキーム S3 と単調性・実行可能性の安全装置。
  `FitResult.n_em_evaluations`, `FitResult.n_safeguards`）。
- 事前分布と MAP 推定: `log_pdf` を持つ `NormalPrior(mean, sd)`, `LogNormal(meanlog, sdlog)`,
  `Beta(a, b)` を `Model.prior(*, param, dist, items=None)` または ParameterTable の `prior` 列で指定。
  `FitResult.log_posterior`（事前分布なしでは `loglik` と同値）、`history["log_posterior"]`、
  事前分布ありの `aic`/`bic` での `FlexpyirtWarning`、`se="oakes"`・`se="hessian"` 両方への事後情報行列。
  潜在分布パラメタへの事前分布は `SpecificationError`。

## 7. 主なクラス・関数の一覧

| 名前 | 種類 | 役割 |
|---|---|---|
| `Model` | frozen dataclass | 仕様全体。`uniform`, `from_items` で生成、`.fix`, `.fix_from`, `.prior`, `.equal_across_groups`, `.equal_across_items`, `.with_groups`, `.with_latent`, `.parameter_table` |
| `Rasch`, `TwoPL`, `ThreePL`, `FourPL`, `GRM`, `GPCM` | frozen dataclass（`ItemModel` を満たす） | 項目モデル |
| `Normal`, `EmpiricalHistogram` | frozen dataclass（`LatentDist` を満たす） | 潜在分布。`Normal.standard()`, `Normal.free()` |
| `LogNormal`, `NormalPrior`, `Beta`, `TruncNormal` | frozen dataclass | 事前分布（`log_pdf` を持つ）。潜在分布の `Normal` と名前がぶつからないよう事前分布側は `NormalPrior` |
| `BAEM`, `Quadrature` | frozen dataclass | 推定設定 |
| `fit` | 関数 | 推定。`FitResult` を返す |
| `FitResult` | クラス | 結果（`items`, `latent`, `se`, `vcov`, `loglik`, `score`, `summary`, `item_fit`, `dif`, `m2`, `to_json` / `from_json`） |

## 8. 設計上の約束
- 公開関数は keyword-only を基本にする（`fit(spec, data, *, group=None, ...)`）。
- 文字列で選ぶ選択肢は `Literal` で型付け（`se: Literal["oakes", ...]`）。
- 入力は pandas / polars / numpy を受け付ける。列名を項目名に、NaN を欠測に。numpy の場合は `item_names=` で指定。
- 仕様の誤りは fit の**前**に、どの項目・どのパラメタが悪いかを含むメッセージで `SpecificationError` を投げる。
- 乱数を使う処理（plausible values など）は `key: jax.Array` を明示的に受け取る（グローバル乱数状態を持たない）。
