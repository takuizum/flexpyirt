# flexpyirt 公開 API スケッチ

[English](API.md) | 日本語

> **実装済みの API は [API リファレンス](../reference.ja.md) と [ユーザーガイド](../guide.ja.md) を参照。**
> このファイルは設計段階のスケッチで、未実装の機能（`TruncNormal` など）も含む。

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
res.person_fit(df)  # DataFrame: theta, lz, infit, z_infit, outfit, z_outfit, n_items
res.item_fit(df, statistics=("infit", "outfit"))  # item, infit, z_infit, outfit, z_outfit
```

## 2. 混合形式 + 事前分布（MAP）

```python
from flexpyirt import Model, ThreePL, GRM, LogNormalPrior, BetaPrior, fit

spec = (
    Model.from_items(
        {
            **{f"mc{i:02d}": ThreePL() for i in range(1, 31)},
            **{f"cr{i}": GRM(n_categories=4) for i in range(1, 6)},
        }
    )
    .prior(param="a", dist=LogNormalPrior(0.0, 0.5))
    .prior(param="g", dist=BetaPrior(5, 17), items=lambda name: name.startswith("mc"))
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
res.latent  # DataFrame: group, mean, var, mean_se, var_se
res.items  # group 列つき（等値項目は同値）
```

EH にしたい場合は全群を `EmpiricalHistogram()` にする（EH と `Normal` は群の間で混在できない）。参照群のヒストグラムは標準化される。

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
    quadrature=Quadrature.equispaced(61, bounds=(-6, 6)),  # または Quadrature.gauss_hermite(41)
    se="oakes",  # "oakes" | "hessian" | "crossprod" | "sandwich" | None
)
# 標本ウェイトとクラスタ抽出（学校の中の生徒）: 標本設計に基づくサンドイッチ SE。
res_cl = fit(spec, df, weights=w, cluster=school, se="sandwich")
(
    res.converged,
    res.n_iter,
    res.history,
)  # history: DataFrame(iter, loglik, log_posterior, max_change, ...)
```

## 6.1 M1 時点で実装済みの範囲
- 項目モデル: `TwoPL`, `Rasch`。潜在分布: `Normal`（`Normal.standard()`, `Normal.free()`, `Normal(free_var=True)`）。
  全項目 Rasch の単一群では既定で分散自由（DESIGN §4.5）。
- `fit(spec, data, *, group=None, table=None, item_names=None, method=None, quadrature=None, se="oakes")`。
  `se` は `"oakes" | "hessian" | None`。`group=` は M2 まで `NotImplementedError`。
- `BAEM(tol=1e-4, max_iter=2000, m_step_newton_steps=3)`（`squarem` は M3）。
  `Quadrature.equispaced(n_points=61, bounds=(-6, 6))`。Gauss–Hermite は `Quadrature.gauss_hermite(n)`（DESIGN §15.44）。準モンテカルロは 2〜5 因子用の `Quadrature.qmc(n_points=5000)`（DESIGN §28。4〜5 因子では必須。mirt との照合はまだ）。
- `Model.uniform`, `Model.from_items`, `.fix(item=..., a=..., d=...)`, `.with_latent(...)`, `.parameter_table(data)`。
- `FitResult`: `items`, `latent`, `parameters`（表 + `estimate`, `se`）, `vcov`, `loglik`, `aic`, `bic`,
  `converged`, `n_iter`, `history`, `score(data, method="EAP")`。
- 入力は 0/1 の pandas / polars DataFrame または numpy 配列（`item_names=`）。欠測は M2 まで `SpecificationError`。

## 6.2 M2 で追加された範囲
- 項目モデル `GRM(n_categories=K)`（カテゴリ $0..K-1$）。欠測（NaN, MAR）を受け付ける。
- 多母集団: `Model.with_groups(ref=Normal.standard(), focal=Normal.free())`（最初が参照群。#79 以降は
  位置引数のマッピング `with_groups({"ref": ..., "s01": ...})` も受け付け、多数の群や識別子でないラベルに使う。#75 以降は
  全群が同じ次数の `DavidianCurve` または同じ成分数の `NormalMixture` を使ってもよく、群ごとに形を持つ）、
  `.equal_across_groups(items=..., params=None)`、`fit(..., group=labels)`。`FitResult.items` は `group` 列つき、
  `score(data, group=labels)`。
- FIPC: `Model.fix_from(bank, items=None, parameterization="slope_intercept" | "irt")` + `.with_latent(Normal.free())`。
- 識別不能な仕様（アンカーなしで焦点群自由、固定項目なしで平均自由など）は fit 前に `IdentificationError`。

## 6.3 M3 で追加された範囲（進行中）
- 項目モデル: `ThreePL`, `FourPL`, `GPCM(n_categories=K)`, `PCM(n_categories=K)`（境界解で `HeywoodWarning`）。
- SQUAREM 加速: `BAEM(squarem=True)`（スキーム S3 と単調性・実行可能性の安全装置。
  `FitResult.n_em_evaluations`, `FitResult.n_safeguards`）。
- 事前分布と MAP 推定: `log_pdf` を持つ `NormalPrior(mean, sd)`, `LogNormalPrior(mean_log, sd_log)`,
  `BetaPrior(a, b)` を `Model.prior(*, param, dist, items=None)` または ParameterTable の `prior` 列で指定。
  `FitResult.log_posterior`（事前分布なしでは `loglik` と同値）、`history["log_posterior"]`、
  事前分布ありの `aic`/`bic` での `FlexpyirtWarning`、`se="oakes"`・`se="hessian"` 両方への事後情報行列。
  潜在分布パラメタへの事前分布は `SpecificationError`。

## 7. 主なクラス・関数の一覧

推定済みのモデルを使わない道具（下の表の `polychoric` から `logistic_dif` まで。`model_parallel_analysis` と
`dimensionality` を除く）は flexpyctt で定義し、flexpyirt が再公開する。これらは JAX の `key` の代わりに `rng`
（シードまたは `numpy.random.Generator`）を取る。`flexpyirt.polychoric` は JAX 版、`flexpyctt.polychoric` は
NumPy / SciPy 版（DESIGN.md §16.18）。

| 名前 | 種類 | 役割 |
|---|---|---|
| `Model` | frozen dataclass | 仕様全体。`uniform`, `from_items` で生成、`.fix`, `.fix_from`, `.prior`, `.equal_across_groups`, `.equal_across_items`, `.with_groups`, `.with_latent`, `.parameter_table` |
| `Rasch`, `TwoPL`, `ThreePL`, `FourPL`, `GRM`, `GPCM`, `PCM`, `Nominal` | frozen dataclass（`ItemModel` を満たす） | 項目モデル |
| `items.ItemModel`, `items.ItemKernel`, `items.register_item_model` | Protocol、関数 | ユーザー定義の項目モデルとその JSON 登録（DESIGN §20） |
| `Normal`, `EmpiricalHistogram`, `DavidianCurve`, `NormalMixture` | frozen dataclass（`LatentDist` を満たす） | 潜在分布。`Normal.standard()`, `Normal.free()`, `DavidianCurve(degree)`, `NormalMixture(n_components)` |
| `LogNormalPrior`, `NormalPrior`, `BetaPrior`, `LogitNormalPrior`, `TruncNormal`（未実装） | frozen dataclass | 事前分布（`log_pdf` を持つ）。潜在分布の `Normal` と名前がぶつからないよう事前分布側は `NormalPrior` |
| `BAEM`, `Quadrature` | frozen dataclass | 推定設定 |
| `fit` | 関数 | 推定。`FitResult` を返す |
| `FitResult` | クラス | 結果（`items`, `latent`, `parameters`, `vcov`, `loglik`, `score`, `summary`, `item_fit`, `person_fit`, `local_dependence`, `jsi`, `dif`, `m2`, `to_json` / `from_json`） |
| `dif_screen`, `dif_effect_sizes` | 関数 | 1 次元の多母集団モデルの DIF の分析手順: アンカーの純化つきの全項目のスクリーニング（`strategy="all_others"` / `"sequential_free"` / `"sequential_constrain"`、尤度比検定または Wald 検定。`table`・`history`・`flagged_items`・`anchor_items`・最終モデル `fit` を持つ `DIFScreenResult` を返す）と効果量（期待得点の符号つき・符号なしの差、ESSD、RMSD、テストの行）。`FitResult.dif(..., test="wald")` で Wald 検定（DESIGN §16.22） |
| `link` | 関数 | 微分可能な IRT 尺度リンキング（Haebara・Stocking–Lord の非対称・対称版。多次元モデルでは `transformation=` で対角・直交・斜交の $\mathbf A$ を選ぶ）。`LinkResult` を返す |
| `LinkResult` | クラス | 尺度リンキング結果（`A`, `B`, `A_se`, `B_se`, `vcov`, `criterion`, `transformation`, `items`, `transform_theta`, `summary`） |
| `screen_common_items` | 関数 | リンキングの共通項目の点検と項目パラメータドリフト（頑健 z・変位・期待得点の RMSD。`link()` による反復的な純化）。`CommonItemScreeningResult` を返す |
| `CommonItemScreeningResult` | クラス | 点検の結果（`table`, `link`, `initial_link`, `kept_items`, `removed_items`, `history`, `flagged_items`, `summary`） |
| `stack_screening_tables` | 関数 | 複数の実施回の点検の表を `administration` 列を付けて積む |
| `irt_equate` | 関数 | 同じ尺度上の（または `LinkResult` を介した）新フォームの合計得点を旧フォームへ IRT 真値等化・観測得点等化する。`IRTEquatingResult`（`table`, `new_distribution`, `old_distribution`, `summary`）を返す |
| `score_distribution` | 関数 | フォーム（項目・群）の推定母集団におけるモデル上の合計得点分布 |
| `scale_table` | 関数 | 素点-尺度得点変換表: 合計得点ごとの $\theta$（真値または EAPsum）、その SE、切って丸めた線形の尺度得点 |
| `polychoric` | 関数 | ポリコリック・テトラコリック相関行列（2 段階の最尤法、平滑化）。`PolychoricResult` を返す |
| `parallel_analysis`, `velicer_map`, `empirical_kaiser` | 関数 | 次元数の診断。`ParallelAnalysisResult`, `MAPResult`, `EmpiricalKaiserResult`（`n_factors` を持つ frozen dataclass）を返す |
| `comparison_data` | 関数 | 比較データ法（Ruscio & Roche, 2012）。`ComparisonDataResult` を返す |
| `efa`, `hull` | 関数 | 相関行列の回転なしの ULS / ML 因子分析（EFAtools と同じ適合度指標）と Hull 法。`EFAResult`, `HullResult` を返す |
| `dimtest` | 関数 | ブートストラップでバイアス補正する DIMTEST。`DimtestResult` を返す |
| `mantel_haenszel`, `mantel`, `generalized_mantel_haenszel`, `logistic_dif` | 関数 | 観測得点による DIF（flexpyctt）: ETS 分類つきの Mantel–Haenszel、Mantel 検定、一般化 MH、ロジスティック回帰。マッチング得点の純化つき。`DIFResult`（`table`, `anchor_items`, `n_iter`, `converged`, `flag_history`）を返す |
| `detect`, `exploratory_detect` | 関数 | 確認的・探索的 DETECT。`DetectResult`, `ExploratoryDetectResult` を返す |
| `model_parallel_analysis`, `dimensionality` | 関数 | `FitResult` のモデルに基づく平行分析と、まとめのレポート。`ModelParallelAnalysisResult`, `DimensionalityResult` を返す |
| `item_information`, `test_information`, `expected_score` | 関数 | `FitResult` の、与えた `theta` での項目・テスト情報量（条件付き SE つき）と期待得点。`theta` をインデックスにした DataFrame を返す（DESIGN §18.2） |
| `information_matrix` | 関数 | 与えた `theta` でのテスト（または 1 項目）の Fisher 情報行列。形 `(T, D, D)` の配列。多次元モデルでは `item_information` / `test_information` が `direction=`（方向別の情報量と合成値の SE）と形 `(T, D)` の `theta` をとり、因子名の `MultiIndex` をインデックスにする（DESIGN §18.3） |
| `item_probabilities`, `empirical_item_curve`, `wright_map_table` | 関数 | 図の元になる数値: トレースライン（縦長の DataFrame）、EAP の $\theta$ の等人数区間ごとの観測カテゴリ比率とモデルの確率、Wright マップの受検者の推定値と項目のサーストン閾値（`WrightMapTable(persons, items)`）（DESIGN §21） |
| `flexpyirt.plotting` | モジュール | `plot_item_curves`（観測比率つき。群ごとの曲線が DIF の図）、`plot_information`、`plot_expected_score`、`plot_wright_map`。matplotlib は任意依存 `flexpyirt[plot]` から遅延 import（なければ `MissingDependencyError`）（DESIGN §21） |
| `reliability`, `summed_score_table` | 関数 | 周辺信頼性・経験的信頼性と、合計得点 EAP の表（`FitResult.score(method="EAPsum")` も）。DataFrame を返す |
| `summed_score_distribution` | 関数 | カテゴリ確率に対する Lord–Wingersky / Poisson 二項分布のカーネル（DESIGN §18.1） |
| `compare_fits` | 関数 | 同じデータへの当てはめの比較表: `n_params`、`loglik`、AIC、SABIC、HQ、BIC と直前の行に対する尤度比検定（mirt の `anova()`。DESIGN §24.1） |
| `simulate_responses`, `SimulatedData` | 関数、frozen dataclass | `FitResult` または `Model` + 項目バンクから JAX の `key` で回答データを生成する。`responses`、`theta`、`group` を返す（DESIGN §24.2） |
| `classification_accuracy` | 関数 | カット得点による分類の正確さと一貫性（合計得点上の Lee, 2010、$\theta$ 上の Rudner, 2001, 2005。D 法と P 法）。1 次元の `FitResult` に対し `ClassificationResult`（`summary`、`by_category`、`by_cut`、`conditional`、`accuracy_table`、`consistency_table`。DESIGN §18.4）を返す |
| `LocalDependenceResult` | frozen dataclass | `FitResult.local_dependence()` の戻り値（`pairs`, `q3`, `q3_mean`, `max_q3_star`） |
| `JSIResult` | frozen dataclass | `FitResult.jsi()` の戻り値（`matrix`, `pairs`, `method`, `converged`） |
| `PolychoricResult` | frozen dataclass | `correlation`, `unsmoothed_correlation`, `thresholds`, `categories`, `n_obs`、ペアごとの `converged` / `at_boundary` / `zero_cell_corrected`、`smoothed`, `min_eigenvalue` |

## 8. 設計上の約束
- 公開関数は keyword-only を基本にする（`fit(spec, data, *, group=None, ...)`）。
- 文字列で選ぶ選択肢は `Literal` で型付け（`se: Literal["oakes", ...]`）。
- 入力は pandas / polars / numpy を受け付ける。列名を項目名に、NaN を欠測に。numpy の場合は `item_names=` で指定。
- 仕様の誤りは fit の**前**に、どの項目・どのパラメタが悪いかを含むメッセージで `SpecificationError` を投げる。
- 乱数を使う処理（plausible values など）は `key: jax.Array` を明示的に受け取る（グローバル乱数状態を持たない）。
