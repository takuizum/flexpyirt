# flexpyequate design (DESIGN)

English | [日本語](DESIGN.ja.md)

> Status: draft v0.5 (2026-11-25). Milestones E1–E7 are implemented and validated against R
> (§7). Decisions are marked "Decision".

## 1. Purpose and non-goals

### Purpose
- Equating of **score distributions**: given the score distributions of two test forms X and Y,
  find the function $e_Y(x)$ that puts form-X scores on the form-Y scale.
- Cover the observed-score methods of Kolen & Brennan (2014) and the kernel method of von Davier,
  Holland & Thayer (2004), each implemented from the published formulas and checked numerically
  against R (`equate`, `kequate`, `SNSequate`).
- A light distribution: NumPy, SciPy and pandas only.

### Non-goals
- **Anything that needs item parameters**: IRT true-score and observed-score equating, scale
  linking and raw-to-scale tables stay in flexpyirt. flexpyequate never imports flexpyirt or JAX
  (decision of issue #87).
- Item-level data. The input is always a score distribution (or the scores it is tabulated from).
- Reproducing the argument names or object structure of the R packages.

### Principles (shared with flexpyirt)
1. Correctness over speed over number of features. A method is "done" when its comparison test
   against an R reference passes; only validated methods appear in the README.
2. Formulas first: the code is written from the formulas in §4, and the R packages are reference
   values only. Where an R package deviates from the published formula, the deviation is recorded
   in §7 and the published formula wins.
3. Pure functions on arrays inside, validation and DataFrames in the public layer.

## 2. Data collection designs

| Design | Data | Container |
|---|---|---|
| Equivalent groups (EG) | X from sample 1, Y from an independent sample 2 | two `ScoreDistribution` |
| Single group (SG) | X and Y from the same examinees | one `BivariateScoreDistribution` |
| Counterbalanced (CB) | two SG samples taking the forms in opposite orders | two `BivariateScoreDistribution` of (X, Y): the sample that took X first, then the one that took Y first (kernel equating, E6) |
| Non-equivalent groups with anchor test (NEAT) | (X, A) from population P and (Y, A) from population Q | two `BivariateScoreDistribution`, anchor as the second variable (E4, E5; kernel equating E6) |

For NEAT, two families of methods exist and both are planned: **chained** equating (X → A on P,
then A → Y on Q) and **post-stratification** (a synthetic population
$T = wP + (1 - w)Q$; Tucker, Levine, Braun–Holland, frequency estimation, kernel PSE).

## 3. Notation

| Symbol | Meaning |
|---|---|
| $x_j,\ j = 1..J$; $y_k,\ k = 1..K$ | possible scores of X and Y (equally spaced, increment $\delta$) |
| $n_j$, $N = \sum_j n_j$ | frequency of $x_j$, sample size |
| $r_j = n_j / N$, $s_k$ | score probabilities of X and Y |
| $F$, $G$ | discrete cumulative distributions of X and Y |
| $\mu_X, \sigma_X^2$ | mean and variance of X |
| $e_Y(x)$ | equating function, X to the Y scale |

## 4. Methods

Formulas are cited by author, year and chapter. Equation numbers are deliberately not given yet:
they are to be added after checking them against the printed books.

### 4.1 Score distributions (E1)

`ScoreDistribution` holds the equally spaced possible scores and their (non-negative, possibly
non-integer) frequencies. Moments use the population form
$\sigma_X^2 = \sum_j r_j (x_j - \mu_X)^2$ unless `ddof=1` is asked for.

**Percentile rank** (Kolen & Brennan, 2014, ch. 2; written here as a proportion, the public
methods use the 0–100 scale of the book). A discrete score is
treated as uniform on $[x_j - \delta/2,\, x_j + \delta/2)$. With $x^*$ the possible score nearest
to $x$ (halves rounded up),

$$P(x) = F(x^* - \delta) + \frac{x - (x^* - \delta/2)}{\delta}\,\bigl[F(x^*) - F(x^* - \delta)\bigr]$$

for $x_1 - \delta/2 \le x < x_J + \delta/2$, with $P = 0$ below and $P = 1$ above.

**Percentile** (inverse; Kolen & Brennan, 2014, ch. 2). For a proportion $p$,

$$P^{-1}_U(p) = x^*_U - \frac{\delta}{2} + \delta\,\frac{p - F(x^*_U - \delta)}{F(x^*_U) - F(x^*_U - \delta)},
\qquad x^*_U = \min\{x_j : F(x_j) > p\},$$

$$P^{-1}_L(p) = x^*_L + \frac{\delta}{2} + \delta\,\frac{p - F(x^*_L)}{F(x^*_L + \delta) - F(x^*_L)},
\qquad x^*_L = \max\{x_j : F(x_j) < p\},$$

with $P^{-1}_U(1) = x_J + \delta/2$ and $P^{-1}_L(0) = x_1 - \delta/2$, and
$P^{-1}(p) = \tfrac12\,[P^{-1}_U(p) + P^{-1}_L(p)]$. The two differ only where a run of zero
frequencies makes $F$ flat; the average is then the middle of the flat stretch.

**Decision**: frequencies may be non-integer, because presmoothed distributions are passed through
the same container.

### 4.2 Identity, mean and linear equating (E1)

- Identity: $e_Y(x) = x$.
- Mean (Kolen & Brennan, 2014, ch. 2): $e_Y(x) = x - \mu_X + \mu_Y$.
- Linear (ch. 2): $e_Y(x) = \dfrac{\sigma_Y}{\sigma_X}(x - \mu_X) + \mu_Y$.

For SG data the same formulas apply to the two margins (the dependence between X and Y matters
only for standard errors, §4.8).

**Decision**: the population variance (`ddof=0`) is the default because that is the formula in
Kolen & Brennan; `ddof=1` reproduces R `equate`, which divides by $N - 1$ (§7).

### 4.3 Equipercentile equating (E1)

$$e_Y(x) = Q^{-1}\bigl(P(x)\bigr)$$

(Kolen & Brennan, 2014, ch. 2), with $P$ the percentile-rank function of X and $Q^{-1}$ the
percentile function of Y from §4.1. Equated scores lie in
$[y_1 - \delta_Y/2,\ y_K + \delta_Y/2]$.

### 4.4 Log-linear presmoothing (E2)

Polynomial log-linear model for the score frequencies (Holland & Thayer, 2000):

$$\log(N r_j) = \beta_0 + \sum_{i=1}^{C} \beta_i\, x_j^{\,i},$$

and for a bivariate distribution

$$\log(N p_{jk}) = \beta_0 + \sum_{i=1}^{C_X} \beta_{Xi}\, x_j^{\,i} + \sum_{i=1}^{C_Y} \beta_{Yi}\, y_k^{\,i}
+ \sum_{(a,b)} \beta_{ab}\, x_j^{\,a} y_k^{\,b}.$$

Fitted by maximum likelihood (Poisson = multinomial likelihood) with Newton's method. The fitted
distribution reproduces the first $C$ sample moments of the observed one (and the cross moments
$E[X^aY^b]$ for the included cross terms); this is a unit test.

**Decision**: the design matrix is orthonormalized (QR of the centred, scaled powers) before
fitting. Fitted frequencies do not depend on the basis, and raw powers of scores up to 40–100 with
degree 6–10 are too ill-conditioned for Newton's method. Coefficients are therefore not reported in
the raw-power basis; the result exposes the fitted distribution, the design matrix and the fit
statistics (Poisson log-likelihood, likelihood-ratio $G^2$, Pearson $\chi^2$, Freeman–Tukey
$\chi^2$, AIC and degrees of freedom). BIC is left out for now: R's `BIC()` on the Poisson `glm`
uses the number of cells as the sample size, the equating literature uses the number of
examinees, and there is no reference to validate a choice against.

The asymptotic covariance of $\sqrt{N}(\hat r - r)$ is $C C^\top$ with the "C-matrix" of Holland &
Thayer (1987): with $B$ the design matrix (no intercept column) and
$M = (D_{\sqrt r} - \sqrt r\, r^\top) B = QR$,

$$C = D_{\sqrt r}\, Q, \qquad \operatorname{Cov}(\hat r) = \tfrac1N\, C C^\top .$$

Without presmoothing (saturated model) $C = D_{\sqrt r} - r \sqrt r^\top$, which gives the
multinomial covariance $D_r - r r^\top$.

### 4.5 Kernel equating (E3 for EG; E6 for the other designs and SEED)

Five steps (von Davier, Holland & Thayer, 2004): presmoothing (§4.4), estimation of the score
probabilities $r, s$ through the design function, continuization, equating, and the standard error
of equating.

**Continuization** with a Gaussian kernel (ch. 4). With bandwidth $h_X > 0$ and
$a_X^2 = \sigma_X^2 / (\sigma_X^2 + h_X^2)$,

$$F_{h}(x) = \sum_j r_j\, \Phi\bigl(R_{jX}(x)\bigr), \qquad
R_{jX}(x) = \frac{x - a_X x_j - (1 - a_X)\mu_X}{a_X h_X},$$

$$f_{h}(x) = \frac{1}{a_X h_X} \sum_j r_j\, \phi\bigl(R_{jX}(x)\bigr).$$

The continuized variable has mean $\mu_X$ and variance $\sigma_X^2$ for every $h_X$. As
$h_X \to \infty$ the kernel equating function tends to the linear one.

**Bandwidth** (ch. 4): minimize

$$\mathrm{PEN}(h) = \sum_j \bigl(r_j - f_h(x_j)\bigr)^2 + K \sum_j A_j (1 - B_j),$$

where $A_j = 1$ if $f_h'(x_j - w) < 0$ and $B_j = 0$ if $f_h'(x_j + w) > 0$ ($w = 1/4$): the second
term counts the scores at which the density is U-shaped. $K = 1$ by default.

**Decision**: the minimizer is found by a grid search on $h$ (400 points, log-spaced between
0.05 and 10 score increments) followed by a bounded scalar minimization around the best grid
point, because the penalty is not unimodal in general and is discontinuous when $K > 0$.

**Decision**: the penalty is evaluated with scores measured in units of their increment
$\delta$ (the book assumes integer scores, where a density and a probability are comparable), and
the bandwidth is reported on the score scale. Kernel equating is then equivariant under linear
transformations of either score scale (a unit test).

**Equating** (ch. 4): $e_Y(x) = G_{h_Y}^{-1}\bigl(F_{h_X}(x)\bigr)$, solved by a safeguarded
Newton iteration to machine precision.

**Standard error of equating** (ch. 5). With $J_{e_Y} = (\partial e_Y/\partial r,\,
\partial e_Y/\partial s)$,

$$\frac{\partial e_Y(x)}{\partial r_j} = \frac{1}{G'(e_Y(x))} \frac{\partial F_h(x)}{\partial r_j}, \qquad
\frac{\partial e_Y(x)}{\partial s_k} = -\frac{1}{G'(e_Y(x))} \frac{\partial G_h(y)}{\partial s_k}\Big|_{y = e_Y(x)},$$

$$\frac{\partial F_h(x)}{\partial r_j} = \Phi\bigl(R_{jX}(x)\bigr) - M_{jX}(x)\, f_h(x), \qquad
M_{jX}(x) = \tfrac12 (x - \mu_X)(1 - a_X^2)\Bigl(\frac{x_j - \mu_X}{\sigma_X}\Bigr)^2 + (1 - a_X)\, x_j,$$

and, for EG with independent samples of sizes $N$ and $M$,

$$\mathrm{SEE}_Y(x) = \sqrt{\tfrac1N \bigl\lVert \tfrac{\partial e_Y}{\partial r}^{\!\top} C_r \bigr\rVert^2
+ \tfrac1M \bigl\lVert \tfrac{\partial e_Y}{\partial s}^{\!\top} C_s \bigr\rVert^2 } .$$

The derivative formulas are checked against finite differences in the unit tests, and the SEE
against a parametric bootstrap. Without presmoothing the multinomial covariance is used
($C = D_{\sqrt r} - r\sqrt r^\top$, §4.4).

**Percent relative error** (ch. 4): $\mathrm{PRE}(p) = 100\,[\mu_p(e_Y(X)) - \mu_p(Y)]/\mu_p(Y)$
for the $p$-th raw moments, $p = 1..10$, with $e_Y(X)$ taking the value $e_Y(x_j)$ with
probability $r_j$.

**Design functions** (E6; ch. 2 and 3 of the book). Every design gives $r$ and $s$ as functions
of the estimated joint probabilities, and the SEE is $\lVert J_{e_Y} J_{DF}\, C \rVert$ with
$J_{DF}$ the Jacobian of the design function and $C C^\top$ the covariance of the estimated
probabilities of each independent sample (blocks side by side).

| Design | $r$, $s$ |
|---|---|
| EG | the two sets of score probabilities |
| SG, joint $P$ of (X, Y) | $r_j = \sum_k p_{jk}$, $s_k = \sum_j p_{jk}$ |
| CB, joints $P_{12}$ (X first) and $P_{21}$ (Y first) with margins $r_1, s_1$, $r_2, s_2$ | $r = w_X r_1 + (1 - w_X) r_2$, $s = w_Y s_1 + (1 - w_Y) s_2$ |
| NEAT PSE, $P$ of (X, A) and $Q$ of (Y, A), anchor margins $t_P$, $t_Q$, $T = wP + (1 - w)Q$ | $r_j = \sum_l \bigl[w + (1 - w)\tfrac{t_{Ql}}{t_{Pl}}\bigr] p_{jl}$, $s_k = \sum_l \bigl[(1 - w) + w\tfrac{t_{Pl}}{t_{Ql}}\bigr] q_{kl}$ |

For PSE,

$$dr_j = \sum_l \Bigl[w + (1 - w)\frac{t_{Ql}}{t_{Pl}}\Bigr] dp_{jl}
- (1 - w)\sum_l \frac{p_{jl}\, t_{Ql}}{t_{Pl}^2}\, dt_{Pl} + (1 - w)\sum_l \frac{p_{jl}}{t_{Pl}}\, dt_{Ql},$$

and symmetrically for $s$.

**Chained equating (NEAT CE)** continuizes four distributions, X and A in P ($F_P$, $H_P$) and A
and Y in Q ($H_Q$, $G_Q$), each with its own bandwidth:

$$e_Y(x) = G_Q^{-1}\Bigl(H_Q\bigl(H_P^{-1}(F_P(x))\bigr)\Bigr).$$

With $a = H_P^{-1}(F_P(x))$, the chain rule gives two independent parts,

$$\mathrm{SEE}_Y(x)^2 =
\Bigl\lVert \frac{H_Q'(a)}{G_Q'(e_Y(x))}\,\frac{1}{H_P'(a)}
\Bigl(\frac{\partial F_P(x)}{\partial r_P} J_r - \frac{\partial H_P(a)}{\partial t_P} J_t\Bigr) C_P \Bigr\rVert^2
+ \Bigl\lVert \frac{1}{G_Q'(e_Y(x))}
\Bigl(\frac{\partial H_Q(a)}{\partial t_Q} J_t - \frac{\partial G_Q(e_Y(x))}{\partial s_Q} J_s\Bigr) C_Q \Bigr\rVert^2 ,$$

where $J_r$ and $J_t$ sum the cell probabilities over the anchor and over the total score. The
percent relative error is reported for each link (X to A in P, A to Y in Q).

**SEED** (E6): the standard error of the difference between two equating functions estimated
from the same data is $\lVert v_1(x) - v_2(x) \rVert$, where $v(x) = J_{e}(x) J_{DF} C$ is the
vector whose norm is the SEE (ch. 5). `KernelEquatingResult` keeps these vectors
(`see_vectors`) and `seed(other)` takes the difference: kernel against linear (`linear=True`,
bandwidths of 1000 standard deviations as in `kequate`), two sets of bandwidths, or chained
against post-stratification equating.

**Decision**: with two joint distributions the design must be named (`design="counterbalanced"`,
`"chained"` or `"post_stratification"`); the three cannot be told apart from the data.
Equivalent groups and single group are inferred.

**Decision**: the default PSE weight is $w = N_P/(N_P + N_Q)$, as for the methods of §4.6
(`kequate` defaults to 0.5). The CB weights default to $(1/2, 1/2)$.

**Decision**: kernel PSE raises an error when an anchor score has probability zero in one
population only, instead of dropping it as §4.6 does: the Jacobian of the design function is not
defined there, and the kernel method presupposes presmoothing.

**Decision**: `seed()` checks only what it can (same design family, X scores and number of
columns of `see_vectors`). The columns depend on the QR factorization behind each C-matrix, so
both results must come from the same presmoothing results; this is documented, not enforced.

### 4.6 NEAT, linear methods (E4)

Population 1 takes X and the anchor V, population 2 takes Y and V (Kolen & Brennan, 2014, ch. 4).
Each population is one `BivariateScoreDistribution` with the anchor as the second variable.

**Synthetic population** $T = w_1 P_1 + w_2 P_2$, $w_1 + w_2 = 1$. The Tucker and the Levine
observed-score methods equate linearly in $T$,
$l_Y(x) = \dfrac{\sigma_s(Y)}{\sigma_s(X)}[x - \mu_s(X)] + \mu_s(Y)$, with

$$\mu_s(X) = \mu_1(X) - w_2 \gamma_1 [\mu_1(V) - \mu_2(V)], \qquad
\mu_s(Y) = \mu_2(Y) + w_1 \gamma_2 [\mu_1(V) - \mu_2(V)],$$

$$\sigma_s^2(X) = \sigma_1^2(X) - w_2 \gamma_1^2 [\sigma_1^2(V) - \sigma_2^2(V)] + w_1 w_2 \gamma_1^2 [\mu_1(V) - \mu_2(V)]^2,$$

$$\sigma_s^2(Y) = \sigma_2^2(Y) + w_1 \gamma_2^2 [\sigma_1^2(V) - \sigma_2^2(V)] + w_1 w_2 \gamma_2^2 [\mu_1(V) - \mu_2(V)]^2 .$$

| Method | $\gamma_1$ ($\gamma_2$ likewise from Y and V in population 2) |
|---|---|
| Tucker (regression of X on V is linear and the same in both populations) | $\sigma_1(X, V) / \sigma_1^2(V)$ |
| Levine, internal anchor (classical congeneric model) | $\sigma_1^2(X) / \sigma_1(X, V)$ |
| Levine, external anchor | $[\sigma_1^2(X) + \sigma_1(X, V)] / [\sigma_1^2(V) + \sigma_1(X, V)]$ |

**Levine true-score** equating, applied to observed scores, and **chained linear** equating do
not use a synthetic population:

$$l_Y(x) = \frac{\gamma_2}{\gamma_1}[x - \mu_1(X)] + \mu_2(Y) + \gamma_2[\mu_1(V) - \mu_2(V)],$$

$$l_Y(x) = \mu_2(Y) + \frac{\sigma_2(Y)}{\sigma_2(V)}\Bigl[\mu_1(V) + \frac{\sigma_1(V)}{\sigma_1(X)}(x - \mu_1(X)) - \mu_2(V)\Bigr].$$

**Braun–Holland**: linear equating with the means and standard deviations of the
frequency-estimation (post-stratification) distributions (Kolen & Brennan, 2014, ch. 5),

$$f_s(x) = w_1 f_1(x) + w_2 \sum_v f_1(x \mid v)\, h_2(v), \qquad
g_s(y) = w_1 \sum_v g_2(y \mid v)\, h_1(v) + w_2\, g_2(y),$$

with $h_1, h_2$ the anchor distributions. The same synthetic distributions serve frequency
estimation in E5.

**Decision**: the default weight is proportional to the sample sizes, $w_1 = N_1/(N_1 + N_2)$
(`synthetic_weight=None`).

**Decision**: the Levine methods require `anchor="internal"` or `"external"` explicitly. There
is no default, because the wrong choice gives a wrong answer without any warning.

**Decision**: where an anchor score occurs in one population only, $f_1(x \mid v)$ (or
$g_2(y \mid v)$) is not estimable. That part of the other population is left out, the synthetic
distribution is renormalized, and a `SparseDataWarning` reports the probability left out.
Presmoothing the joint distributions avoids the situation.

**Decision**: `x` and `y` must have the same possible anchor scores. The anchor is one test
taken by both populations; different anchor scales are rejected rather than silently compared.

**Decision**: `ddof` applies to the variances and covariances of the Tucker and Levine
observed-score methods. It cancels in $\gamma$, in Levine true-score and in chained linear
equating (ratios of moments of one population), and the Braun–Holland method uses the moments of
the synthetic distributions, which have no sample size of their own.

### 4.7 NEAT, equipercentile methods (E5)

Kolen & Brennan (2014, ch. 5). Both methods reuse the percentile-rank and percentile functions of
§4.1.

**Frequency estimation** (post-stratification on the anchor): equipercentile equating of the
synthetic distributions $f_s$ and $g_s$ of §4.6,

$$e_Y(x) = Q_s^{-1}\bigl(P_s(x)\bigr).$$

**Chained equipercentile** equating: X is equated to the anchor in population 1 and the anchor to
Y in population 2,

$$e_Y(x) = Q_2^{-1}\Bigl(H_2\bigl(H_1^{-1}(P_1(x))\bigr)\Bigr),$$

with $P_1$, $H_1$ the percentile-rank functions of X and V in population 1 and $H_2$, $Q_2$ those
of V and Y in population 2. $H_2$ is evaluated at the non-integer anchor equivalents with the
continuous percentile-rank function of §4.1.

Presmoothing is not an option of `equate()`: pass the presmoothed joint distributions
(`presmooth(xa, degree=(…, …), cross_terms=…).distribution`).

**Decision**: for frequency estimation, `EquatingResult.x` and `.y` are the synthetic
distributions (total frequency $N_1 + N_2$), because that is the population in which the
equating is defined; `moments` then describes the synthetic population. For chained equating
they are X in population 1 and Y in population 2, and the two anchor distributions are kept in
`anchor_distributions` so that `transform()` can equate arbitrary scores.

### 4.8 Standard errors of the traditional methods (E7)

`equate(..., standard_error=...)` returns the standard error of equating at the possible scores
of X (`EquatingResult.see`, a column `see` of `table`).

**Delta method** (`"analytic"`; Kolen, 1985; Kolen & Brennan, 2014, ch. 7; Zu & Yuan, 2012).
Every linear method except Braun–Holland is a function $e_Y(x) = c(\theta) + b(\theta)x$ of the
means, variances and covariances $\theta$ of the samples, so

$$\mathrm{SE}[\hat e_Y(x)]^2 = g(x)^\top \Omega\, g(x), \qquad g(x) = \nabla c + x\,\nabla b,
\qquad \Omega = \operatorname{Cov}(\hat\theta),$$

with one block of $\Omega$ per independent sample. For a sample of size $N$ with scores $A, B,
\dots$ and central moments $\mu$,

$$N\operatorname{Cov}(\bar A, \bar B) = \sigma_{AB}, \qquad
N\operatorname{Cov}(\bar A, s_{BC}) = \mu_{ABC}, \qquad
N\operatorname{Cov}(s_{AB}, s_{CD}) = \mu_{ABCD} - \sigma_{AB}\sigma_{CD},$$

which is the multinomial covariance of the cell proportions carried to the moments (no
distributional assumption). For linear equating of equivalent groups this gives the formula of
Braun & Holland (1982),

$$\mathrm{SE}^2 = \sigma_Y^2\Bigl\{\frac{1}{N_X} + \frac{1}{N_Y}
+ \Bigl[\frac{\mathrm{sk}(X)}{N_X} + \frac{\mathrm{sk}(Y)}{N_Y}\Bigr] z
+ \Bigl[\frac{\mathrm{ku}(X) - 1}{4N_X} + \frac{\mathrm{ku}(Y) - 1}{4N_Y}\Bigr] z^2\Bigr\},
\qquad z = \frac{x - \mu_X}{\sigma_X}.$$

The gradients: for mean and linear equating, $\partial b/\partial\sigma_X^2 = -b/(2\sigma_X^2)$,
$\partial b/\partial\sigma_Y^2 = b/(2\sigma_Y^2)$ and
$\nabla c = \nabla\mu_Y - b\nabla\mu_X - \mu_X\nabla b$ ($b = 1$ for mean equating). For the
Tucker and Levine observed-score methods, with $\Delta_\mu = \mu_1(V) - \mu_2(V)$ and
$\Delta_\sigma = \sigma_1^2(V) - \sigma_2^2(V)$ (§4.6),

$$d\mu_s(X) = d\mu_1(X) - w_2\Delta_\mu\, d\gamma_1 - w_2\gamma_1\, d\Delta_\mu, \qquad
d\mu_s(Y) = d\mu_2(Y) + w_1\Delta_\mu\, d\gamma_2 + w_1\gamma_2\, d\Delta_\mu,$$

$$d\sigma_s^2(X) = d\sigma_1^2(X) + 2\gamma_1(-w_2\Delta_\sigma + w_1w_2\Delta_\mu^2)\, d\gamma_1
- w_2\gamma_1^2\, d\Delta_\sigma + 2w_1w_2\gamma_1^2\Delta_\mu\, d\Delta_\mu,$$

$$d\sigma_s^2(Y) = d\sigma_2^2(Y) + 2\gamma_2(w_1\Delta_\sigma + w_1w_2\Delta_\mu^2)\, d\gamma_2
+ w_1\gamma_2^2\, d\Delta_\sigma + 2w_1w_2\gamma_2^2\Delta_\mu\, d\Delta_\mu,$$

$$db = \frac b2\Bigl[\frac{d\sigma_s^2(Y)}{\sigma_s^2(Y)} - \frac{d\sigma_s^2(X)}{\sigma_s^2(X)}\Bigr],
\qquad dc = d\mu_s(Y) - b\, d\mu_s(X) - \mu_s(X)\, db,$$

and $d\gamma$ from the definition of $\gamma$ (Tucker: $\partial\gamma/\partial\sigma_{XV} =
1/\sigma_V^2$, $\partial\gamma/\partial\sigma_V^2 = -\sigma_{XV}/\sigma_V^4$; Levine, internal:
$1/\sigma_{XV}$ for $\sigma_X^2$ and $-\sigma_X^2/\sigma_{XV}^2$ for $\sigma_{XV}$; Levine,
external: $1/(\sigma_V^2 + \sigma_{XV})$, $(\sigma_V^2 - \sigma_X^2)/(\sigma_V^2 +
\sigma_{XV})^2$ and $-(\sigma_X^2 + \sigma_{XV})/(\sigma_V^2 + \sigma_{XV})^2$ for
$\sigma_X^2$, $\sigma_{XV}$, $\sigma_V^2$). Levine true-score and chained linear equating are
differentiated in the same way from their formulas in §4.6. The synthetic weight is a constant.

**Braun–Holland** depends on the whole joint distributions, so it is differentiated with respect
to the cell probabilities $p_{jl}$ (X, V in population 1) and $q_{kl}$ (Y, V in population 2).
With $e = \mu_s(Y) + b\,(x - \mu_s(X))$,
$A_j = -b\,x_j - b\,(x - \mu_s(X))(x_j - \mu_s(X))^2 / (2\sigma_s^2(X))$,
$B_k = y_k + b\,(x - \mu_s(X))(y_k - \mu_s(Y))^2 / (2\sigma_s^2(Y))$,
$\bar A_l = \sum_j A_j p_{jl}$ and $\bar B_l = \sum_k B_k q_{kl}$,

$$\frac{\partial e}{\partial p_{jl}} = A_j\Bigl[w_1 + w_2\frac{h_{2l}}{h_{1l}}\Bigr]
- w_2\frac{h_{2l}}{h_{1l}^2}\bar A_l + w_1\frac{\bar B_l}{h_{2l}}, \qquad
\frac{\partial e}{\partial q_{kl}} = B_k\Bigl[w_2 + w_1\frac{h_{1l}}{h_{2l}}\Bigr]
- w_1\frac{h_{1l}}{h_{2l}^2}\bar B_l + w_2\frac{\bar A_l}{h_{1l}},$$

and the variance is $\sum_{\text{samples}}[\sum p\,(\partial e)^2 - (\sum p\,\partial e)^2]/N$.

**Normal theory** (`"analytic_normal"`): the same gradients with $\Omega$ under (bivariate)
normality, i.e. third moments zero and $\mu_{ABCD} - \sigma_{AB}\sigma_{CD} =
\sigma_{AC}\sigma_{BD} + \sigma_{AD}\sigma_{BC}$. For linear equating this gives
$\sigma_Y^2 (1/N_X + 1/N_Y)(1 + z^2/2)$ for equivalent groups and
$\sigma_Y^2 (1 - \rho)[z^2(1 + \rho) + 2]/N$ for a single group (Lord, 1950; Angoff, 1971; unit
tests).

**Equipercentile equating, equivalent groups** (Lord, 1982; Kolen & Brennan, 2014, ch. 7). With
$p = P(x)$ as a proportion, $y^*_U = \min\{y_k : G(y_k) > p\}$, $G_0 = G(y^*_U - \delta_Y)$ and
$g = G(y^*_U) - G_0$,

$$\mathrm{SE}^2 = \frac{\delta_Y^2}{g^2}\Bigl\{\frac{p(1 - p)}{N_X}
+ \frac{1}{N_Y}\Bigl[G_0 - p^2 + \frac{(p - G_0)^2}{g}\Bigr]\Bigr\}.$$

The second term is the exact delta-method variance of $Q(y)$; the first treats the percentile
rank as a binomial proportion, which overstates its variance by $f(x)/(4N_X)$ for a discrete
score. The formula is therefore an approximation: over the scores with percentile rank between
2 and 98 the ratio of the bootstrap standard error (2000 replications) to Lord's lies between
0.87 and 1.13, with a mean of 0.98 to 1.01, in the three reference data sets.

**Bootstrap** (`"bootstrap"`; Kolen & Brennan, 2014, ch. 7; as `equate::bootstrap`). Each of
`n_replications` replications draws $N$ examinees with replacement from every sample, which for
frequency data is a multinomial draw with the observed relative frequencies (the cells of a
joint distribution are resampled, so the dependence between the two scores is kept); the
sample of X is drawn first, then that of Y. Each bootstrap sample is equated with the same
method, and the standard error is the standard deviation (divisor $B - 1$) of the equated
scores. `bootstrap_mean` is their mean.

| Method (design) | `analytic` | `analytic_normal` | `bootstrap` |
|---|---|---|---|
| identity | 0 | 0 | 0 |
| mean, linear (EG) | yes (Braun & Holland, 1982) | yes | yes |
| mean, linear (SG) | yes | yes | yes |
| equipercentile (EG) | yes (Lord, 1982) | no | yes |
| equipercentile (SG) | no | no | yes |
| Tucker, Levine observed score, chained linear | yes | yes | yes |
| Levine true score | yes | yes | yes |
| Braun–Holland | yes (cell probabilities) | no | yes |
| frequency estimation, chained equipercentile | no | no | yes |

**Decision**: the standard error is an option of `equate()` (`standard_error=None` by default,
so existing calls do not change), not a separate function: it needs the same inputs and the
same validated options.

**Decision**: `"analytic"` means the distribution-free delta method. R `equate` reports both
versions for the anchor-test methods and only the distribution-free one for equivalent groups;
the normal-theory version is the separate option `"analytic_normal"`.

**Decision**: the variances and covariances that are plugged into the analytic formulas follow
`ddof`, like the equating function itself; third and fourth moments always divide by $N$. With
`ddof=1` this reproduces `equate`'s standard error of linear equating for equivalent groups
exactly.

**Decision**: the bootstrap requires `rng` (a `numpy.random.Generator` or a non-negative
integer seed) and raises without it, so that every bootstrap result can be reproduced. The
argument is called `rng` as in flexpyctt, and the number of replications `n_replications` as in
flexpyirt.

**Decision**: the default synthetic weight $N_1/(N_1 + N_2)$ is computed once from the data and
kept fixed over the bootstrap samples (their sizes are the same).

**Decision**: a bootstrap sample that cannot be equated (a score without variance, a
non-positive covariance for Levine's methods, a non-finite result) is left out, and a
`SparseDataWarning` reports how many; `n_replications` of the result is the number used. Fewer
than two usable samples is an error.

**Decision**: where the percentile rank of $x$ is 0 or 1, Lord's standard error is 0 (as in
`equate`): the formula has no value there.

**Decision**: all three kinds treat the frequencies as observed. With a presmoothed
distribution as input the analytic formulas ignore that a model was fitted and the bootstrap
draws from the smoothed distribution without refitting it (a parametric bootstrap with a fixed
model); the total frequency, rounded for the bootstrap, is the sample size. Standard errors
that account for presmoothing are not covered (see below); kernel equating has them.

**Decision**: the analytic standard error of Braun–Holland raises an error when an anchor score
was observed in one population only (the derivative does not exist); the bootstrap works.

**Not covered**: analytic standard errors of equipercentile equating for a single group, of
frequency estimation and of chained equipercentile equating (formulas exist in the literature,
e.g. Jarjoura & Kolen, 1985; Liou & Cheng, 1995; R `equate` has none to compare with, so they
are bootstrap only); standard errors that account for presmoothing in `equate()` (refitting the
log-linear model in every bootstrap sample, as `equate::bootstrap` can do); bootstrap standard
errors for `kernel_equate()`; bootstrap bias, RMSE and percentile intervals.

### 4.9 Further kernels and bandwidth rules (E7)

**Kernels** (Lee & von Davier, 2011). With a kernel variable $V$ with distribution function
$K$, density $k$, mean 0 and variance $\sigma_V^2$, the continuization of §4.5 becomes

$$F_h(x) = \sum_j r_j\, K\bigl(R_{jX}(x)\bigr), \qquad
R_{jX}(x) = \frac{x - a_X x_j - (1 - a_X)\mu_X}{a_X h_X}, \qquad
a_X^2 = \frac{\sigma_X^2}{\sigma_X^2 + \sigma_V^2 h_X^2},$$

$$f_h(x) = \frac{1}{a_X h_X}\sum_j r_j\, k\bigl(R_{jX}(x)\bigr), \qquad
f_h'(x) = \frac{1}{(a_X h_X)^2}\sum_j r_j\, k'\bigl(R_{jX}(x)\bigr),$$

and the derivative used for the SEE keeps its form,
$\partial F_h(x)/\partial r_j = K(R_{jX}(x)) - M_{jX}(x) f_h(x)$ with the same $M_{jX}$ and the
kernel's own $a_X$. Mean and variance are preserved for every kernel (unit test).

| `kernel` | $K(z)$ | $k(z)$ | $k'(z)$ | $\sigma_V^2$ |
|---|---|---|---|---|
| `"gaussian"` | $\Phi(z)$ | $\phi(z)$ | $-z\phi(z)$ | 1 |
| `"logistic"` | $1/(1 + e^{-z})$ | $K(1 - K)$ | $k\,(1 - 2K)$ | $\pi^2/3$ |
| `"uniform"` | $\min\{\max(z + \tfrac12, 0), 1\}$ | 1 on $[-\tfrac12, \tfrac12)$ | 0 | $1/12$ |

Every design of §4.5, the SEE, the SEED and the percent relative error use the same code with
the kernel exchanged. The penalty of §4.5 is used with the kernel's density; for the uniform
kernel its second term is zero.

**Decision**: the logistic kernel is the standard logistic and the uniform kernel is uniform on
$(-1/2, 1/2)$: the "standard" versions of Lee & von Davier and the defaults of `kequate`
(`slog = 1`, `bunif = 0.5`) and `SNSequate`. Their rescaled versions are the same continuization
with another bandwidth ($\sigma_{V1} h_1 = \sigma_{V2} h_2$), so no scale parameter is offered.

**Decision**: the density of the uniform kernel is taken on the half-open interval
$[-1/2, 1/2)$, so that kernels that touch do not count twice (R's `dunif` includes both ends);
this matters on a set of measure zero.

**Decision**: with the uniform kernel $F_h$ is flat outside the support of the continuized
distribution and, when $h$ is below the score increment, between possible scores. The quantile
is then kept inside the support, and where the density of Y is zero at an equated score the SEE
is `nan` (the delta method does not apply). A bandwidth of at least one increment avoids gaps.

**Decision**: `linear=True` is refused with the uniform kernel: its support is bounded, so a
large bandwidth does not give the linear function. With the logistic kernel it does. `kequate`
always computes its linear function and SEED with the Gaussian kernel, which here is
`result.seed(kernel_equate(..., linear=True))`.

**Decision**: the penalty of the uniform kernel jumps where the edge of a kernel passes a
possible score. The same search as for the other kernels is used (grid, then a bounded local
search); it returns the best point found.

**Bandwidth rules** (`bandwidth_method`, Gaussian kernel) besides the penalty:

- `"rule_of_thumb"` (Andersson & von Davier, 2014):
  $h = 9\sigma/\sqrt{100N^{2/5} - 81}$, which is Silverman's rule
  $a h = 0.9\,\sigma N^{-1/5}$ for the density actually used, solved for $h$.
- `"double_smoothing"` (Häggström & Wiberg, 2014). A pilot continuization $f_g$ with a large
  bandwidth $g$ stands in for the unknown density. On the grid $x^*_l$, $l = 1..2J - 1$, of the
  possible scores and the points halfway between them,

  $$\mathrm{DS}(h) = \sum_l \Bigl[\sum_j f_g(x_j)\,\frac{1}{a h}\phi\bigl(R_j(x^*_l)\bigr)
  - r^*_l\Bigr]^2,$$

  with $r^*_l = r_j$ at a possible score and $f_g(x^*_l)$ halfway between two.
- `"cross_validation"`: the leave-one-out criterion of `kequate`,

  $$\mathrm{CV}(h) = \sum_j \bigl(r_j - \hat r_{(j)}\bigr)^2, \qquad
  \hat r_{(j)} = \frac{\sum_{i \ne j} r_i\,\phi(R_i(x_j))}{\sum_{i \ne j}\phi(R_i(x_j))}.$$

Like the penalty, the last two are evaluated with scores in units of their increment and
minimized by the grid and local search of §4.5.

**Decision**: the three rules are options of `kernel_equate()` (`bandwidth_method`), which
reports the bandwidths it used; no new public function is added.

**Decision**: $\sigma$ of the rule of thumb is the standard deviation of the distribution that
is continuized (population form, like everything else in the continuization; `kequate` uses
$N/(N - 1)$ times the variance), and $N$ is the size of the sample that took the form: $N_X$,
$N_Y$ for EG, $N$ for SG, the size of population P for X (and its anchor) and of Q for Y (and
its anchor) in the anchor-test designs, as in `kequate`. The counterbalanced design, where each
form is observed in two samples, is refused.

**Decision**: the pilot bandwidth of double smoothing is 4 score increments and candidates are
searched below it. The source describes the pilot as "large, subjectively chosen"; 4 is
`kequate`'s value (the upper end of its search interval). Without the bound the criterion has
its global minimum at the largest bandwidth for short tests (the 9-point anchor of the
reference data).

**Decision**: `"cross_validation"` is `kequate`'s criterion (`CV = TRUE`), named after it. It is
not the likelihood cross-validation of Liang & von Davier (2014), which splits the sample at
random into two halves and repeats; the papers were not available for this work and the
formulas above are taken from Wallin, Häggström & Wiberg (2021) and the `kequate` source, which
is recorded here as a limit of the verification.

**Decision**: the SEE treats the selected bandwidth as fixed for every rule, as the book does
for the penalty. Andersson & von Davier (2014) give a SEE for the rule of thumb that accounts
for $h$ being proportional to $\hat\sigma$; `kequate`'s version of it (`altopt = TRUE`) differs
from the derivative we obtain (§7), so there is no usable reference and it is not covered.

**Not covered**: the Epanechnikov and adaptive kernels of `SNSequate`; `kequate`'s
`"stdgaussian"` kernel; scale parameters of the logistic and uniform kernels; the bandwidth
rules for kernels other than the Gaussian; the likelihood cross-validation of Liang & von
Davier (2014); the SEE of Andersson & von Davier (2014) for a data-dependent bandwidth; the
rule of thumb for the counterbalanced design.

## 5. Public API sketch

```python
import flexpyequate as fe

x = fe.ScoreDistribution.from_scores(x_scores, score_range=(0, 40))
y = fe.ScoreDistribution.from_frequencies(scores=range(41), frequencies=y_counts)
xy = fe.BivariateScoreDistribution.from_scores(
    x_scores, y_scores, x_range=(0, 40), y_range=(0, 40)
)  # single group

result = fe.equate(x, y, method="equipercentile")  # EG
result = fe.equate(xy, method="linear")  # SG: one bivariate distribution
result.table  # DataFrame: score, equated
result.moments  # DataFrame: mean, sd, skewness, kurtosis of X, Y and equated X
result.transform(x_new)  # equate arbitrary (also non-integer) scores

smooth_x = fe.presmooth(x, degree=6)  # LoglinearResult
smooth_xy = fe.presmooth(xy, degree=(4, 4), cross_terms=[(1, 1)])
smooth_x.distribution  # ScoreDistribution with the fitted frequencies
smooth_x.fit_statistics  # Series: likelihood_ratio, pearson, freeman_tukey, aic, ...
smooth_x.c_matrix  # C with Cov(r_hat) = C C' / N

ke = fe.kernel_equate(smooth_x, smooth_y)  # Gaussian kernel, penalty bandwidths
ke = fe.kernel_equate(x, y, bandwidth_x=0.6, bandwidth_y=0.6)  # observed distributions
ke.table  # DataFrame: score, equated, see
ke.continuized  # DataFrame: form, score, probability, cdf, density
ke.bandwidth_x, ke.bandwidth_y
ke.percent_relative_error
fe.penalty_bandwidth(smooth_x, penalty_weight=1.0)

sg = fe.kernel_equate(smooth_xy)  # single group: one joint distribution
cb = fe.kernel_equate(smooth_12, smooth_21, design="counterbalanced")
pse = fe.kernel_equate(smooth_xa, smooth_ya, design="post_stratification", synthetic_weight=0.5)
ce = fe.kernel_equate(smooth_xa, smooth_ya, design="chained")
ce.anchor_bandwidths, ce.chain_percent_relative_error
pse.seed(fe.kernel_equate(smooth_xa, smooth_ya, design="post_stratification", linear=True))
ce.seed(pse)  # DataFrame: score, difference, seed

# Non-equivalent groups: form X with the anchor, form Y with the anchor.
xa = fe.BivariateScoreDistribution.from_scores(x_scores, x_anchor, x_range=(0, 36), y_range=(0, 12))
ya = fe.BivariateScoreDistribution.from_scores(y_scores, y_anchor, x_range=(0, 36), y_range=(0, 12))
tucker = fe.equate(xa, ya, method="tucker")  # w1 = N1 / (N1 + N2)
levine = fe.equate(xa, ya, method="levine_observed", anchor="internal", synthetic_weight=1.0)
tucker.gamma, tucker.synthetic_moments, tucker.synthetic_weight
fe.equate(xa, ya, method="frequency_estimation")
fe.equate(xa, ya, method="chained_equipercentile")

# Standard errors of equating (E7).
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

# Other kernels and bandwidth rules (E7).
fe.kernel_equate(smooth_x, smooth_y, kernel="logistic")
fe.kernel_equate(smooth_x, smooth_y, kernel="uniform", bandwidth_x=1.0, bandwidth_y=1.0)
fe.penalty_bandwidth(smooth_x, kernel="logistic")
fe.kernel_equate(smooth_x, smooth_y, bandwidth_method="double_smoothing")  # or "rule_of_thumb",
ke.kernel, ke.bandwidth_method  # "cross_validation"
```

- All arguments after the data are keyword-only. Options are `Literal` strings in lower
  snake_case: `method="identity" | "mean" | "linear" | "equipercentile"` (EG, SG) and
  `"tucker" | "levine_observed" | "levine_true" | "braun_holland" | "chained_linear" |
  "frequency_estimation" | "chained_equipercentile"` (NEAT).
- Results are frozen dataclasses (`EquatingResult`, `LoglinearResult`, `KernelEquatingResult`)
  holding NumPy arrays (read-only) and exposing DataFrames as properties.
- Errors are `flexpyequate.errors` classes: `ScoreDistributionError` (invalid scores or
  frequencies), `SpecificationError` (inconsistent arguments), `ConvergenceWarning`,
  `SparseDataWarning`.
- Names are Python names; the R name is given in the docstring where it helps (`ddof=1` ↔ R
  `equate`'s variance, `penalty_weight` ↔ `kequate`'s `KPEN`).
- `standard_error="analytic" | "analytic_normal" | "bootstrap"` (with `n_replications` and
  `rng`) for `equate()`; `kernel="gaussian" | "logistic" | "uniform"` and
  `bandwidth_method="penalty" | "rule_of_thumb" | "double_smoothing" | "cross_validation"` for
  `kernel_equate()` (§4.8, §4.9).

## 6. Layout

```
packages/flexpyequate/
  pyproject.toml        hatchling + hatch-vcs (version from the repository tag, root = "../..")
  README.md, README.ja.md
  docs/DESIGN.md, DESIGN.ja.md
  src/flexpyequate/
    distributions.py    ScoreDistribution, BivariateScoreDistribution
    equating.py         equate(), EquatingResult
    _neat.py            formulas for non-equivalent groups with an anchor test
    _standard_errors.py delta-method standard errors of the traditional methods
    presmoothing.py     presmooth(), LoglinearResult
    kernel.py           kernel_equate(), KernelEquatingResult
    errors.py
  tests/                unit tests, R comparison tests, reference/*.json
validation/r/equate_*.R reference generators (repository root; not run in CI)
```

## 7. Validation plan

Reference values come from R and are committed as JSON under `tests/reference/`; each file records
the R and package versions. CI does not run R. To regenerate, run from the repository root:

```bash
Rscript validation/r/equate_e1_reference.R
Rscript validation/r/equate_e2_reference.R
Rscript validation/r/equate_e3_reference.R
Rscript validation/r/equate_e4_reference.R
Rscript validation/r/equate_e5_reference.R
Rscript validation/r/equate_e6_reference.R
Rscript validation/r/equate_e7_standard_errors_reference.R  # ~10 min (equate::bootstrap)
Rscript validation/r/equate_e7_kernel_reference.R           # reads the E3 and E6 files
```

The two E7 generators read their data from the committed E1, E3, E4 and E6 reference files, so
those must be generated first.

### Cases

| ID | Contents | R reference | MS |
|---|---|---|---|
| Q1 | Mean and linear equating, EG (`ACTmath`, simulated forms with different lengths and increments) and SG (bivariate simulated) | `equate::equate(type = "mean", sx = 1, sy = 1)`, `equate(type = "linear")` | E1 |
| Q2 | Percentile ranks and equipercentile equating, EG and SG, including distributions with zero frequencies inside and at the ends, and non-unit score increments | `equate::px`, `equate::equate(type = "equipercentile")`; an independent R implementation of Kolen & Brennan ch. 2 without rounding | E1 |
| Q3 | Log-linear presmoothing: univariate (degrees 2–8), bivariate with and without cross terms; equipercentile equating of the smoothed distributions | `stats::glm(family = poisson)` with a strict tolerance; `equate::presmoothing(smoothmethod = "loglinear")`; `equate(..., smoothmethod = "loglinear")` | E2 |
| Q4 | Kernel equating, EG: continuized CDFs, equated scores, SEE, PRE with given bandwidths; with presmoothing (three data sets) and without (one) | `kequate::kequate("EG", ...)`; `SNSequate::ker.eq`; an independent R implementation with a strict root tolerance and a finite-difference SEE | E3 |
| Q5 | Penalty function values and penalty bandwidths ($K = 0$ and $K = 1$) | `kequate` (`KPEN = 0`), `SNSequate::bandwidth`, independent R implementation | E3 |
| Q6 | Tucker, Levine observed score, Braun–Holland (four synthetic weights each), Levine true score, chained linear; internal anchor (`KBneat`) and external anchor (simulated, unequal sample sizes and form lengths) | `equate::equate(type = "linear", method = "tucker" / "levine" / "braun/holland" / "chained", lts = )`; the Kolen & Brennan formulas written out in R with population moments | E4 |
| Q7 | Frequency estimation (four synthetic weights) and chained equipercentile equating on `KBneat` and the simulated external-anchor data; both methods on `KBneat` presmoothed with a bivariate log-linear model | `equate::equate(type = "equipercentile", method = "frequency estimation" / "chained", smoothmethod = )`; Kolen & Brennan written out in R without rounding | E5 |
| Q8 | Kernel equating for SG (presmoothed and observed), CB (three pairs of weights), NEAT PSE (three weights) and NEAT CE: equated scores, SEE, linear version and SEED between kernel and linear, PRE, penalty bandwidths | `kequate::kequate("SG" / "CB" / "NEAT_PSE" / "NEAT_CE", ...)`; the design functions written out in R with a finite-difference SEE over all cell probabilities | E6 |
| Q9 | Analytic standard errors: mean and linear equating, EG (three data sets) and SG; Lord's standard error of equipercentile equating, EG; Tucker and Levine observed score (three synthetic weights), Levine true score, chained linear and Braun–Holland on `KBneat` (internal anchor) and the simulated external-anchor data; distribution-free and normal-theory versions | `equate::equate(..., verbose = TRUE)$concordance$se` (linear and equipercentile EG; Tucker, Levine observed, chained: `se.g` and `se.n`), also with all frequencies times 1e6; the delta method written out in R with finite-difference Jacobians with respect to all cell probabilities (and to the moments, for the normal-theory version) | E7 |
| Q10 | Bootstrap standard errors (2000 replications): linear and equipercentile, EG and SG; Tucker, chained linear, frequency estimation and chained equipercentile | `equate::bootstrap(reps = 2000)`: **statistical** comparison of two Monte Carlo estimates (R's random numbers cannot be reproduced); unit tests against the analytic standard errors and of the resampling mechanics | E7 |
| Q11 | Logistic and uniform kernels in every design (EG, SG, CB with two pairs of weights, PSE with two weights, CE): continuized CDF and density, equated scores, SEE, SEED against the Gaussian linear function, PRE; penalty values and penalty bandwidths | `kequate::kequate(..., kernel = "logistic" / "uniform")`, `kequate:::PEN`; `SNSequate::bandwidth(kert = "logis" / "unif")`; the general-kernel continuization written out in R with a finite-difference SEE | E7 |
| Q12 | Rule-of-thumb, double-smoothing and leave-one-out cross-validation bandwidths (eight distributions with 9 to 41 possible scores); criterion values on a grid; end to end for EG and CE | `kequate:::bandwidth.select("altopt" / "DS" / "CV")`, `kequate:::DS`, `kequate:::CV`, `kequate(..., DS = TRUE)`, `kequate(..., CV = TRUE)`; the criteria written out in R | E7 |

### Tolerances (maximum absolute difference unless stated)

| Quantity | Reference | Tolerance |
|---|---|---|
| Moments, mean / linear equated scores, slope, intercept | `equate` (`ddof=1`) | 1e-10 |
| Percentile ranks | `equate::px` | 1e-12 |
| Equipercentile equated scores | independent R implementation | 1e-10 |
| Equipercentile equated scores (ranks strictly between 0 and 1) | `equate` | 1e-7 |
| NEAT linear methods: slope, intercept, $\gamma$, synthetic means and SDs | `equate` (`ddof=1`); independent R implementation (`ddof=0`) | 1e-10 |
| Frequency-estimation synthetic probabilities | independent R implementation | 1e-13 |
| Frequency estimation and chained equipercentile equated scores | independent R implementation | 1e-10 |
| Frequency estimation and chained equipercentile equated scores (ranks strictly between 0 and 1) | `equate` | 5e-7 |
| The same on presmoothed joint distributions (tables smoothed by `equate`, and end to end) | `equate` | 5e-6 |
| Presmoothed frequencies, relative | `glm` (`epsilon = 1e-13`) | 1e-8 |
| Presmoothed frequencies, relative | `equate::presmoothing` (default `glm` tolerance) | 1e-5 |
| $G^2$, Pearson and Freeman–Tukey $\chi^2$, log-likelihood, AIC (relative) | `glm` | 1e-7 |
| Equipercentile equating of presmoothed distributions | `equate` | 1e-5 |
| Covariance of the fitted probabilities $CC^\top/N$ (relative to its largest entry) | `kequate` | 1e-9 |
| Kernel CDFs $F_h$, $G_h$; density | `kequate`; independent R implementation | 1e-12 |
| Kernel equated scores | independent R implementation | 1e-9 |
| Kernel equated scores | `kequate` without its variance factor | 1e-6 |
| SEE (relative) | `kequate` without its variance factor; independent finite-difference SEE | 1e-6 |
| PRE | `kequate` without its variance factor | 1e-5 |
| Penalty values (relative) | independent R implementation | 1e-10 |
| Bandwidth, $K = 0$ | `kequate` without its variance factor | 1e-6 |
| Bandwidth, $K = 0$ and $K = 1$ | independent R implementation | 1e-6 |
| Bandwidth, $K = 0$ and $K = 1$ | `SNSequate` | 1e-3 |
| Kernel equated scores / SEE (relative) | `SNSequate::ker.eq` | 1e-4 / 1e-4 |
| SG, CB, PSE, CE: kernel equated scores; SEE (relative) | independent R implementation | 1e-9; 1e-6 |
| SG, CB, PSE, CE: kernel equated scores; SEE (relative); PRE; bandwidths ($K = 0$) | `kequate` without its variance factor | 1e-6; 1e-6; 1e-5; 1e-6 |
| SG, CB, PSE, CE: linear equated scores; linear SEE and SEED kernel vs linear (relative) | `kequate` without its variance factor | 1e-6; 1e-5 |
| Kernel equated scores / SEE (relative) / bandwidth | ordinary `kequate` call | 5e-3 / 1e-3 / 1e-4 |
| Linear SE, EG (`ddof=1`), relative | `equate` | 1e-10 |
| Chained linear SE, distribution-free and normal-theory, relative | `equate` | 1e-10 |
| Tucker, Levine observed-score SE, both versions, relative | `equate` with frequencies times 1e6 | 1e-8 |
| The same (`ddof=0` and `ddof=1`), relative | ordinary `equate` call | 2e-3 |
| Mean, linear (EG, SG), Tucker, Levine observed and true score, chained linear, Braun–Holland SE; normal-theory versions; relative | independent R implementation (finite differences) | 1e-8 |
| Lord's SE of equipercentile equating, EG (absolute and relative) | `equate` | 1e-6 |
| Bootstrap SE (2000 replications), relative, each score / mean over scores | `equate::bootstrap` (2000 replications) | 0.12 / 0.06 (statistical) |
| Logistic, uniform kernel: CDF, density | independent R implementation | 1e-12 |
| Logistic, uniform kernel, all designs: equated scores; SEE (relative) | independent R implementation | 1e-9; 1e-6 |
| Logistic, uniform kernel, all designs: equated scores; SEE (relative); SEED against the Gaussian linear function (relative); PRE | `kequate` without its variance factor | 1e-6; 1e-6; 1e-5; 5e-5 |
| Penalty values, logistic and uniform kernel (relative) | `kequate:::PEN` ($K = 0$); independent R implementation ($K = 0, 1$) | 1e-10 |
| Penalty bandwidth, logistic and uniform kernel | `kequate` ($K = 0$); independent R implementation | 1e-6 |
| Penalty bandwidth, logistic and uniform kernel ($K = 1$) | `SNSequate::bandwidth` | 1e-5 |
| Double-smoothing and cross-validation criterion values (relative) | `kequate:::DS`, `kequate:::CV`; independent R implementation | 1e-10 |
| Double-smoothing bandwidth | `kequate` without its variance factor; independent R implementation | 1e-6 |
| Cross-validation bandwidth | `kequate` without its variance factor; independent R implementation | 1e-5 |
| Rule-of-thumb bandwidth (relative) | `kequate` after its factor $\sqrt{N/(N-1)}$; independent R implementation | 1e-10 |
| Double smoothing, cross-validation end to end (EG, CE): equated scores; SEE (relative) | `kequate(..., DS = TRUE / CV = TRUE)` without its variance factor | 1e-5; 1e-5 |

Observed differences are well inside these: about 1e-8 for the SEE and 2e-7 for the equated
scores against `kequate` (the latter is `kequate`'s root tolerance).

Measured in E7 (largest over all cases):

| Quantity | Reference | Measured |
|---|---|---|
| Linear SE, EG, `ddof=1` (relative) | `equate` | 1.9e-14 |
| Chained linear SE (relative) | `equate` | 3.6e-15 |
| Tucker, Levine observed-score SE (relative) | `equate`, frequencies times 1e6 | 8.1e-10 |
| Tucker, Levine observed-score SE (relative) | ordinary `equate` | 7.6e-4 (`ddof=0`), 3.4e-4 (`ddof=1`) |
| All delta-method SEs, both versions (relative) | independent R implementation | 4.3e-10 |
| SG linear SE under normality (relative) | closed form $\sigma_Y^2(1-\rho)[z^2(1+\rho)+2]/N$ | 4.6e-15 |
| Lord's SE (absolute; relative) | `equate` | 1.4e-7; 1.7e-7 |
| Bootstrap SE, linear methods (relative, worst score) | `equate::bootstrap` | 0.054 |
| Bootstrap SE, equipercentile methods, percentile rank 2–98 (relative, worst score) | `equate::bootstrap` | 0.074 |
| Bootstrap SE, mean over scores (relative, worst case) | `equate::bootstrap` | 0.038 |
| Logistic / uniform kernel: equated scores; SEE (relative) | independent R implementation | 4.4e-11; 1.3e-9 |
| Logistic / uniform kernel: equated scores; SEE (relative); SEED (relative); PRE | `kequate` without its variance factor | 5.5e-7; 1.5e-7; 5.3e-8; 9.3e-6 |
| Penalty values (relative) | `kequate:::PEN`; independent | 2.4e-13 |
| Penalty bandwidths, logistic / uniform | `kequate`; independent; `SNSequate` | 1.3e-8; 7.4e-9; 6.6e-9 |
| Double-smoothing, cross-validation criterion values (relative) | `kequate`; independent | 9.8e-15 |
| Double-smoothing bandwidth | `kequate`; independent | 1.8e-8; 9.4e-9 |
| Cross-validation bandwidth | `kequate`; independent | 2.6e-7; 1.3e-6 |
| Rule-of-thumb bandwidth (relative) | `kequate`; independent | 2.5e-13; 4.9e-15 |
| End to end (EG, CE): equated scores; SEE (relative) | `kequate(DS = TRUE / CV = TRUE)` | 7.7e-7; 7.2e-7 |

**Bootstrap agreement with R is statistical, not numerical.** R's and NumPy's random number
generators differ, so the two bootstrap standard errors are two Monte Carlo estimates of the
same quantity; with $B = 2000$ replications each their ratio has a standard deviation of about
$1/\sqrt{B} = 2.2\%$ per score. The bootstrap is validated in three ways: (a) against the
analytic standard errors for eight method and design combinations (1500 replications,
tolerance five Monte Carlo standard errors plus 3%), (b) by a deterministic test that writes
the documented resampling out (multinomial counts, X's sample first, then Y's, standard
deviation with divisor $B - 1$) and reproduces `see` and `bootstrap_mean` to rounding error
from the same seed, for EG, SG and the anchor-test design, and (c) against
`equate::bootstrap` within the tolerances above.

### Known differences from the R packages

- **`equate` divides variances by $N - 1$**; Kolen & Brennan use $N$. `ddof=1` reproduces
  `equate`; the default is `ddof=0`.
- **`equate`'s `type = "mean"`** has slope (range of Y) / (range of X), which is 1 only when the
  two forms have the same score range. flexpyequate's mean equating has slope 1 (ch. 2 of
  Kolen & Brennan); the reference values are generated with `sx = 1, sy = 1`.
- **`equate`'s Braun–Holland (and frequency-estimation) synthetic distributions** mix the two
  populations as counts, $w\,n_1(x, v) + (1 - w)\,n_1(x, v)\,n_2(v)/n_1(v)$, so the weight of
  population 1 is in effect $wN_1/(wN_1 + (1 - w)N_2)$ instead of $w$ (with the default $w$:
  $N_1^2/(N_1^2 + N_2^2)$). flexpyequate mixes probabilities, as in Kolen & Brennan; `equate` is
  reproduced by passing the effective weight as `synthetic_weight` (intercept difference 0.02 for
  $N_1 = 1200$, $N_2 = 2000$). Its synthetic standard deviations also divide by the total count
  minus 1, which cancels in the slope.
- **`equate` rounds** percentile ranks and cumulative proportions to 10 decimals inside its
  equipercentile routine. The rounding error (up to 5e-11) is divided by a relative frequency, so
  agreement is limited to about 1e-8; hence the unrounded independent R implementation for the
  tight check.
- **Percentile rank exactly 0 or 1**: `equate` returns the end of the Y range
  ($y_1 - \delta/2$ or $y_K + \delta/2$). Kolen & Brennan's definition, followed here, is the mean
  of the upper and lower percentiles, which differs when Y has no observations at that end (ACT
  mathematics, $x = 0$: 0.0 instead of −0.5). The comparison with `equate` leaves these scores
  out; the independent implementation covers them.
- **`kequate` multiplies the variance by $N/(N - 1)$** in the continuization
  ($a_X^2 = \sigma_X^2/(\sigma_X^2 + h_X^2)$) while keeping the derivative formulas of von Davier
  et al., which assume $\sigma_X^2 = \sum_j r_j (x_j - \mu_X)^2$. flexpyequate follows the book
  (as `SNSequate` does). For the tight comparison `kequate` is called with probability vectors and
  sample sizes multiplied by $10^{8}$, which makes the factor negligible, and its SEE is rescaled
  by $10^{4}$ (the SEE is exactly proportional to $N^{-1/2}$). A second comparison uses the
  ordinary `kequate` call with tolerances that cover the factor (it moved the equated scores by
  up to 1.2e-3 with $N \approx 600$ and a bandwidth of 2).
- **`kequate`'s counterbalanced design** has one weight `wcb` and forms
  $r = w r_1 + (1 - w) r_2$, $s = w s_1 + (1 - w) s_2$, but its SEE weights the Y margins the
  other way round ($1 - w$ for the first sample). The SEE therefore belongs to the equating
  function only for $w = 1/2$: with `wcb = 0.3` it differs from the finite-difference SEE by up
  to 29% in the reference data, while flexpyequate's agrees to 1e-6. flexpyequate has two
  weights, $(w_X, w_Y)$; `kequate` is compared at $(1/2, 1/2)$.
- **`kequate` solves $G_h(y) = F_h(x)$ with `uniroot(tol = 1e-6)`**, which limits agreement of the
  equated scores to about 1e-6.
- **`kequate`'s second penalty term** also counts scores at which the density has a local maximum
  ($f' > 0$ on the left and $< 0$ on the right), unlike ch. 4 of the book and `SNSequate`.
  flexpyequate follows the book; with $K = 0$ (`kequate`'s default) the term does not enter.
  flexpyequate's default is $K = 1$, as in the book.
- **`kequate` and `SNSequate` assume scores 0, 1, 2, …**; other score scales are covered by the
  equivariance unit test only.
- **`SNSequate::ker.eq`** searches the equated score in $(-1, 101)$ and fails when a score
  equates below −1 (one of the two cases tried); its bandwidths come from `optim()` started at
  0.65.
- **`equate`'s standard errors for Tucker and Levine observed-score equating** are computed from
  population (divisor $N$) variances and covariances, but with $\gamma$, the synthetic means and
  the synthetic standard deviations of its equating function, which uses $N - 1$. The mixture
  changes the standard error by about $1/N$ in relative terms (7.6e-4 at most in the reference
  data), whichever `ddof` flexpyequate uses. With all frequencies multiplied by $10^6$ (and the
  result by $10^3$) the two forms coincide and the agreement is 1e-9. Chained linear equating
  is not affected (ratios within one population).
- **`equate` has no analytic standard errors** for mean equating, the single-group design,
  Levine true-score equating, the Braun–Holland method and the equipercentile anchor-test
  methods (its functions for the first two return `NULL`). flexpyequate's are checked against
  the independent finite-difference implementation, and the last group is bootstrap only.
- **`equate`'s standard error of equipercentile equating** has no score increment in it, so for
  a form Y with increment $\delta_Y \ne 1$ it is in units of the increment; flexpyequate
  multiplies by $\delta_Y$. Percentile ranks are rounded to 10 decimals (agreement 2e-7).
- **`equate::bootstrap` and the ends of the score range**: `equate` and flexpyequate define the
  equipercentile equivalent differently where the percentile rank is 0 or 1 (above), which
  shows in the bootstrap standard errors of the lowest and highest scores (ratios down to 0.48
  in the reference data). The comparison uses the scores with percentile rank between 2 and
  98. For frequency estimation `equate`'s effective synthetic weight (above) is passed.
- **`kequate`'s penalty for the logistic kernel** multiplies the derivative of the density by an
  extra factor $1/(a h)$; it is positive, so the sign and the penalty are unaffected.
- **`kequate` restricts the bandwidth search** to $(0, 4)$ (to $(0, 2.618)$ for the uniform
  kernel) with `optimize()`, a local search; flexpyequate searches a grid on $(0.05, 10)$
  increments first (double smoothing: up to the pilot bandwidth 4). The selected bandwidths
  agreed in all reference cases.
- **`kequate`'s `altopt`** (rule of thumb) takes effect on the bandwidth only together with
  `CV = TRUE` in version 1.6.4, and it switches to a different derivative $\partial F/\partial
  r$ that accounts for $h \propto \hat\sigma$. Differentiating $F$ with $h = c\sigma$ gives the
  term $\tfrac12 a\,z_j^2 \sum_k x_k r_k \phi(R_k)/(a h)$ where `kequate` has the same without
  the factor $a$, and its equivalent-groups code passes the mean of X for form Y. That SEE is
  therefore not used as a reference; only the bandwidth is compared.
- **`kequate`'s `CV`** is a leave-one-out criterion over score points, not the split-sample
  likelihood cross-validation of Liang & von Davier (2014). `kequate`'s `DS` returns the
  squared-difference term only (as a mean; the minimizer is that of the sum).
- A two-point finite difference with step 1e-6 is not accurate enough for the independent SEE
  where a score probability is about 5e-6 (error 1.4e-3); the generator uses a five-point stencil
  with step 1e-7.

## References

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
  Springer. (Read as ETS RR-08-12.)
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
