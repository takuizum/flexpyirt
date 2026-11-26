"""Limited-information goodness of fit: the M2 / M2* statistic (Result layer, DESIGN.md §16.3).

Model-implied item-score moments and their Jacobian come from the fitted Core problem on its
quadrature grid (the Jacobian by JAX forward-mode autodiff); the asymptotic covariance, SVD
orthogonal complement, and fit indices run in NumPy / SciPy outside ``jit``. Nothing here depends
on the number of factors: for a confirmatory multidimensional model the ``Q`` nodes are the
product grid and the weights are multivariate normal (DESIGN.md §16.19). Multiple groups, missing
responses (pairwise-available margins with their exact covariance under MCAR) and MAP fits follow
DESIGN.md §16.20.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from scipy import optimize, stats

from flexpyirt import _core
from flexpyirt._core import CompiledProblem
from flexpyirt.errors import FlexpyirtWarning, SpecificationError

__all__ = [
    "M2Result",
    "SampleMargins",
    "ci_column_names",
    "conditional_moments",
    "m2_statistic",
    "model_margins",
    "null_m2",
    "rmsea",
    "rmsea_ci",
    "sample_margins",
]


@dataclass(frozen=True)
class M2Result:
    """Values reported by :meth:`FitResult.m2` before DataFrame formatting."""

    m2: float
    df: int
    p: float
    rmsea: float
    rmsea_lower: float
    rmsea_upper: float
    srmsr_by_group: tuple[float, ...]
    tli: float
    cfi: float

    @property
    def srmsr(self) -> float:
        """SRMSR of a single-group model (``NaN`` with several groups; see ``srmsr_by_group``)."""
        return self.srmsr_by_group[0] if len(self.srmsr_by_group) == 1 else float("nan")


def ci_column_names(ci: float) -> tuple[str, str]:
    """Column names for the RMSEA confidence interval (e.g. ``rmsea_5``, ``rmsea_95`` for 0.90)."""
    lo_pct = round(50.0 * (1.0 - ci), 10)
    hi_pct = round(50.0 * (1.0 + ci), 10)
    return f"rmsea_{lo_pct:g}", f"rmsea_{hi_pct:g}"


def _pairs(n_items: int) -> tuple[np.ndarray, np.ndarray]:
    """Return pairs ``(i, j)`` with ``i < j`` in mirt's column-major ``lower.tri`` order."""
    ii, jj = [], []
    for i in range(n_items):
        for j in range(i + 1, n_items):
            ii.append(i)
            jj.append(j)
    return np.asarray(ii, dtype=int), np.asarray(jj, dtype=int)


def conditional_moments(prob: CompiledProblem, eta: Array, group: int = 0) -> tuple[Array, Array]:
    r"""Return ``(J, Q)`` conditional first and second moments of item scores in ``group``.

    .. math::
        m_{1,jq} = \sum_{k=0}^{K_j-1} k\,P_{jk}(\theta_q),\qquad
        m_{2,jq} = \sum_{k=0}^{K_j-1} k^2 P_{jk}(\theta_q).
    """
    n_items = int(prob.y.shape[1])
    n_nodes = int(prob.nodes.shape[0])
    m1 = jnp.zeros((n_items, n_nodes), dtype=prob.nodes.dtype)
    m2 = jnp.zeros_like(m1)
    for fam, logp in zip(prob.families, _core._item_log_probs(prob, eta), strict=True):
        p = jnp.exp(logp[group])  # (J_f, Q, K_f)
        k = jnp.arange(fam.kernel.n_categories, dtype=p.dtype)
        m1 = m1.at[fam.columns, :].set(jnp.sum(p * k, axis=-1))
        m2 = m2.at[fam.columns, :].set(jnp.sum(p * k**2, axis=-1))
    return m1, m2


def model_margins(prob: CompiledProblem, eta: Array, group: int = 0) -> Array:
    r"""Return ``(s,)`` model-implied means and cross-products of item scores in ``group``.

    Maydeu-Olivares & Joe (2005, 2006); mirt's ``M2*``: :math:`E[y_j]` for every item, then
    :math:`E[y_iy_j]` for :math:`i < j` in column-major order.
    """
    m1, _ = conditional_moments(prob, eta, group)
    w = jnp.exp(_core._latent_log_weights(prob, eta)[group])  # (Q,)
    ii, jj = _pairs(int(m1.shape[0]))
    e1 = m1 @ w
    e2 = (m1[ii, :] * m1[jj, :]) @ w
    return jnp.concatenate([e1, e2])


def _xi(m1: np.ndarray, m2: np.ndarray, w: np.ndarray, e: np.ndarray) -> np.ndarray:
    r"""Asymptotic covariance :math:`\Xi_2` of sample margins under local independence.

    For margins :math:`a, b` (single items or pairs of distinct items; Maydeu-Olivares & Joe,
    2005, 2006), :math:`\Xi_{ab} = E[y_a y_b] - E[y_a] E[y_b]`, where the conditional moment of
    the product uses :math:`m_2` for any item appearing in both :math:`a` and :math:`b`.
    """
    n_items = m1.shape[0]
    ii, jj = _pairs(n_items)
    n_pairs = ii.size
    p_idx = np.arange(n_pairs)
    e1, e2 = e[:n_items], e[n_items:]

    # Block (1, 1): E[y_a y_b] - E[y_a] E[y_b].
    xi11 = (m1 * w) @ m1.T
    xi11[np.diag_indices(n_items)] = m2 @ w
    xi11 -= np.outer(e1, e1)

    # Block (1, 2): E[y_a y_i y_j] - E[y_a] E[y_i y_j].
    p1 = m1[ii] * m1[jj]  # (P, Q)
    xi12 = (m1 * w) @ p1.T
    xi12[ii, p_idx] = (m2[ii] * m1[jj]) @ w
    xi12[jj, p_idx] = (m1[ii] * m2[jj]) @ w
    xi12 -= np.outer(e1, e2)

    # Block (2, 2): E[y_i y_j y_k y_l] - E[y_i y_j] E[y_k y_l].
    xi22 = (p1 * w) @ p1.T
    # Pairs sharing exactly one item x: overwrite the (J-1) x (J-1) submatrix of pairs containing x.
    for x in range(n_items):
        mask_i = ii == x
        mask_j = jj == x
        p_x = np.flatnonzero(mask_i | mask_j)
        other_x = np.where(mask_i[p_x], jj[p_x], ii[p_x])
        m1_other = m1[other_x]
        xi22[np.ix_(p_x, p_x)] = ((m2[x] * w) * m1_other) @ m1_other.T
    # Diagonal (pairs sharing both items): E[y_i^2 y_j^2].
    xi22[p_idx, p_idx] = (m2[ii] * m2[jj]) @ w
    xi22 -= np.outer(e2, e2)

    xi = np.block([[xi11, xi12], [xi12.T, xi22]])
    return np.asarray(0.5 * (xi + xi.T), dtype=float)


def _svd_complement(delta: np.ndarray) -> tuple[np.ndarray | None, int]:
    r"""Return an orthonormal basis for :math:`\operatorname{col}(\Delta)^\perp` and SVD rank.

    Returns ``(None, rank)`` if :math:`\Delta` is rank-deficient (``rank < n_cols``) or has no
    orthogonal complement (``rank >= n_margins``).
    """
    n_margins, n_cols = delta.shape
    if n_cols == 0:
        return np.eye(n_margins, dtype=float), 0
    u, sv, _ = np.linalg.svd(delta, full_matrices=True)
    tol = float(sv.max(initial=0.0)) * max(n_margins, n_cols) * float(np.finfo(float).eps)
    rank = int(np.sum(sv > tol))
    if rank < n_cols or rank >= n_margins:
        return None, rank
    return u[:, rank:], rank


def _solve_quadratic_form(resid: np.ndarray, comp: np.ndarray, xi: np.ndarray) -> float:
    r"""Evaluate :math:`e^\top \Delta_c (\Delta_c^\top \Xi_2 \Delta_c)^{-1} \Delta_c^\top e`."""
    proj = comp.T @ resid
    mid = comp.T @ xi @ comp
    mid = 0.5 * (mid + mid.T)
    try:
        sol = np.linalg.solve(mid, proj)
    except np.linalg.LinAlgError as exc:
        raise SpecificationError(
            "Could not invert the orthogonal complement covariance matrix for M2."
        ) from exc
    return float(abs(proj @ sol))


def _quadratic_form(resid: np.ndarray, delta: np.ndarray, xi: np.ndarray) -> float:
    r"""Evaluate :math:`e^\top \Delta_c (\Delta_c^\top \Xi_2 \Delta_c)^{-1} \Delta_c^\top e`.

    Uses an orthonormal basis :math:`\Delta_c` for the orthogonal complement of
    :math:`\operatorname{col}(\Delta)` (numerical rank by SVD). Returns ``NaN`` when
    :math:`\Delta` is rank-deficient or :math:`\Delta_c` is empty.
    """
    comp, _ = _svd_complement(delta)
    if comp is None:
        return float("nan")
    return _solve_quadratic_form(resid, comp, xi)


def rmsea(x2: float, df: int, n: int) -> float:
    r"""Compute :math:`\sqrt{\max(0,\,X^2/df - 1)/(N - 1)}` (mirt's ``rmsea``).

    Returns ``NaN`` when :math:`df \le 0`, :math:`N \le 1`, or ``x2`` is non-finite.
    """
    if df <= 0 or n <= 1 or not np.isfinite(x2):
        return float("nan")
    return float(np.sqrt(max(0.0, x2 / df - 1.0) / (n - 1)))


def rmsea_ci(x2: float, df: int, n: int, level: float = 0.90) -> tuple[float, float]:
    r"""Compute RMSEA confidence interval by inverting the noncentral :math:`\chi^2` CDF.

    MacCallum, Browne & Sugawara (1996): :math:`\sqrt{\lambda / (N\,df)}` matching mirt's
    ``RMSEA.CI``. Returns ``0.0`` for a bound when :math:`P(\chi^2_{df} \le X^2)` is already at or
    below the target probability, and doubles the upper search bracket from
    :math:`\max(N, 5 X^2)` if needed so a valid upper bound never collapses to 0.
    """
    if df <= 0 or n <= 1 or not np.isfinite(x2):
        return float("nan"), float("nan")
    alpha = (1.0 - level) / 2.0

    def _root(target: float, initial_upper: float) -> float:
        def f(lam: float) -> float:
            cdf = float(stats.ncx2.cdf(x2, df, lam) if lam > 0.0 else stats.chi2.cdf(x2, df))
            return cdf - target

        if f(0.0) <= 0.0:
            return 0.0
        upper = max(initial_upper, 1.0)
        while f(upper) > 0.0 and upper < 1e12:
            upper *= 2.0
        if f(upper) > 0.0:
            return float("nan")
        return float(optimize.brentq(f, 0.0, upper, xtol=1e-12, rtol=1e-14))

    lam_l = _root(1.0 - alpha, x2)
    lam_u = _root(alpha, max(float(n), 5.0 * x2))
    lo = float(np.sqrt(lam_l / (n * df))) if np.isfinite(lam_l) else float("nan")
    hi = float(np.sqrt(lam_u / (n * df))) if np.isfinite(lam_u) else float("nan")
    return lo, hi


def null_m2(y: np.ndarray) -> tuple[float, int]:
    r""":math:`M_2` of the dichotomous independence (null) model, for TLI and CFI.

    With every slope fixed at 0 and :math:`J` free item intercepts (whose MLE reproduces the
    sample proportions :math:`\bar y_j`), the orthogonal complement of :math:`\Delta_{\text{null}}`
    maps residuals to sample covariances and diagonalizes :math:`\Xi_2`, simplifying exactly to

    .. math::
        M_{2,\text{null}} = N \sum_{i < j} r_{ij}^2, \qquad
        df_{\text{null}} = \binom{J}{2},

    where :math:`r_{ij}` is the sample Pearson correlation between items :math:`i` and :math:`j`.
    """
    n_obs, n_items = y.shape
    df_null = n_items * (n_items - 1) // 2
    if df_null <= 0:
        return float("nan"), df_null
    yf = y.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.corrcoef(yf, rowvar=False)
    ii, jj = _pairs(n_items)
    r_pairs = corr[ii, jj]
    if not np.all(np.isfinite(r_pairs)):
        return float("nan"), df_null
    return float(n_obs * np.sum(r_pairs**2)), df_null


@dataclass(frozen=True)
class SampleMargins:
    r"""Pairwise-available sample margins of one group (DESIGN.md §16.20).

    ``obs`` holds :math:`\hat p_a=N_a^{-1}\sum_n o_{na}y_{na}` for the :math:`J` items and
    :math:`\binom J2` pairs, ``count`` the :math:`N_a`, ``joint`` the :math:`N_{ab}` (``None``
    for complete data, where every entry is ``n_rows``), and ``cov`` the pairwise-available
    item-score covariance used by SRMSR.
    """

    obs: np.ndarray
    count: np.ndarray
    joint: np.ndarray | None
    cov: np.ndarray
    n_rows: int


def sample_margins(y: np.ndarray, item_names: Sequence[str] | None = None) -> SampleMargins:
    r"""Return the pairwise-available margins of ``y`` (``(N, J)``, missing coded negative).

    .. math::
        N_a=\sum_n o_{na},\qquad \hat p_a=N_a^{-1}\sum_n o_{na}\,y_{na},\qquad
        N_{ab}=\sum_n o_{na}\,o_{nb},

    where :math:`o_{na}=1` when respondent :math:`n` answered every item of margin :math:`a`
    and :math:`y_{na}` is the item score or the product of the two scores (DESIGN.md §16.20).

    Raises
    ------
    SpecificationError
        If an item or a pair of items has no respondent who answered it (``N_a = 0``).
    """
    n_rows, n_items = y.shape
    names = list(item_names) if item_names is not None else [str(j) for j in range(n_items)]
    ii, jj = _pairs(n_items)
    seen = y >= 0
    yz = np.where(seen, y, 0).astype(float)
    ind = np.concatenate([seen, seen[:, ii] & seen[:, jj]], axis=1).astype(float)  # (N, s)
    vals = np.concatenate([yz, yz[:, ii] * yz[:, jj]], axis=1)
    count = ind.sum(axis=0)
    empty = np.flatnonzero(count == 0)
    if empty.size:
        a = int(empty[0])
        what = (
            f"item {names[a]!r}"
            if a < n_items
            else f"items {names[int(ii[a - n_items])]!r} and {names[int(jj[a - n_items])]!r}"
        )
        raise SpecificationError(
            f"M2 needs every item and every pair of items answered by at least one respondent "
            f"(in each group); nobody answered {what}. Limited-information fit on second-order "
            "margins is not defined for designs where items are never given together."
        )
    obs = vals.sum(axis=0) / count
    complete = bool(seen.all())
    joint = None if complete else ind.T @ ind
    mean = obs[:n_items]
    cov = np.empty((n_items, n_items), dtype=float)
    cov[ii, jj] = obs[n_items:] - mean[ii] * mean[jj]
    cov[jj, ii] = cov[ii, jj]
    cov[np.diag_indices(n_items)] = (yz**2).sum(axis=0) / count[:n_items] - mean**2
    return SampleMargins(obs=obs, count=count, joint=joint, cov=cov, n_rows=int(n_rows))


def _residual_and_gamma(
    sample: SampleMargins, margins: np.ndarray, xi: np.ndarray, mirt_convention: bool
) -> tuple[np.ndarray, np.ndarray]:
    r"""Return the scaled residual :math:`r_g` and its covariance :math:`\Gamma_g` (§16.20).

    Default: :math:`r_g=\sqrt{N_g}(\hat p_g-\pi_g)` and
    :math:`\Gamma_{g,ab}=N_g N_{ab}\,\Xi_{ab}/(N_aN_b)`, the exact covariance of
    :math:`r_g` under MCAR (:math:`\Xi_2` itself for complete data). ``mirt_convention``
    reproduces mirt 1.46.1: :math:`r_{g,a}=N_a(\hat p_a-\pi_a)/\sqrt{N_g}` with :math:`\Xi_2`.
    """
    n_g = float(sample.n_rows)
    diff = sample.obs - margins
    if mirt_convention:
        return sample.count * diff / np.sqrt(n_g), xi
    resid = np.sqrt(n_g) * diff
    if sample.joint is None:
        return resid, xi
    gamma = xi * (n_g * sample.joint / np.outer(sample.count, sample.count))
    return resid, 0.5 * (gamma + gamma.T)


def _null_m2(
    samples: Sequence[SampleMargins], n_items: int, mirt_convention: bool
) -> tuple[float, int]:
    r""":math:`M_2` of the dichotomous independence model with intercepts in every group.

    The model has :math:`\pi_{g,j}=\hat p_{g,j}` (the available-case proportion, its MLE) and
    :math:`\pi_{g,ij}=\hat p_{g,i}\hat p_{g,j}`; :math:`\Delta` has one block of :math:`J`
    logit-intercept columns per group with
    :math:`\partial\pi_j/\partial\tau_j=v_j=\hat p_j(1-\hat p_j)` and
    :math:`\partial\pi_{ij}/\partial\tau_i=v_i\hat p_j`. With complete data this equals
    :func:`null_m2` summed over groups. Returns ``(NaN, df)`` if :math:`\Delta` is
    rank-deficient (an item without variance in some group).
    """
    ii, jj = _pairs(n_items)
    s = samples[0].obs.size
    n_groups = len(samples)
    n_total = float(sum(smp.n_rows for smp in samples))
    df_null = n_groups * (s - n_items)
    if df_null <= 0:
        return float("nan"), df_null
    resids, gammas = [], []
    delta = np.zeros((n_groups * s, n_groups * n_items), dtype=float)
    for g, smp in enumerate(samples):
        p = smp.obs[:n_items]
        v = p * (1.0 - p)
        e = np.concatenate([p, p[ii] * p[jj]])
        xi = _xi(p[:, None], p[:, None], np.ones(1), e)
        r, gam = _residual_and_gamma(smp, e, xi, mirt_convention)
        resids.append(r)
        gammas.append(gam)
        blk = np.zeros((s, n_items), dtype=float)
        blk[np.arange(n_items), np.arange(n_items)] = v
        rows = n_items + np.arange(ii.size)
        blk[rows, ii] = v[ii] * p[jj]
        blk[rows, jj] = v[jj] * p[ii]
        if not mirt_convention:
            blk *= np.sqrt(smp.n_rows / n_total)
        delta[g * s : (g + 1) * s, g * n_items : (g + 1) * n_items] = blk
    comp, _ = _svd_complement(delta)
    if comp is None:
        return float("nan"), df_null
    return _solve_quadratic_form(np.concatenate(resids), comp, _block_diag(gammas)), df_null


def _block_diag(blocks: Sequence[np.ndarray]) -> np.ndarray:
    """Block-diagonal matrix of square ``blocks``."""
    size = sum(b.shape[0] for b in blocks)
    out = np.zeros((size, size), dtype=float)
    pos = 0
    for b in blocks:
        k = b.shape[0]
        out[pos : pos + k, pos : pos + k] = b
        pos += k
    return out


def _srmsr(sample: SampleMargins, margins: np.ndarray, m2: np.ndarray, w: np.ndarray) -> float:
    """Root mean squared residual of item-score correlations (pairwise-available, as mirt)."""
    n_items = sample.cov.shape[0]
    ii, jj = _pairs(n_items)
    mean_e = margins[:n_items]
    cov_e = np.zeros((n_items, n_items), dtype=float)
    cov_e[ii, jj] = margins[n_items:] - mean_e[ii] * mean_e[jj]
    cov_e = cov_e + cov_e.T
    cov_e[np.diag_indices(n_items)] = m2 @ w - mean_e**2
    cov_o = sample.cov
    with np.errstate(divide="ignore", invalid="ignore"):
        r_o = cov_o / np.sqrt(np.outer(np.diag(cov_o), np.diag(cov_o)))
        r_e = cov_e / np.sqrt(np.outer(np.diag(cov_e), np.diag(cov_e)))
    return float(np.sqrt(np.mean((r_o[ii, jj] - r_e[ii, jj]) ** 2)))


def m2_statistic(
    prob: CompiledProblem,
    eta: Array,
    y: np.ndarray,
    *,
    n_params: int,
    n_categories: Sequence[int],
    groups: np.ndarray | None = None,
    item_names: Sequence[str] | None = None,
    level: float = 0.90,
    _latent_in_delta: bool = True,
    _mirt_convention: bool = False,
) -> M2Result:
    r"""Compute the :math:`M_2` / :math:`M_2^*` statistic (Maydeu-Olivares & Joe, 2005, 2006).

    .. math::
        M_2 = r^\top \Delta_c (\Delta_c^\top \Gamma \Delta_c)^{-1} \Delta_c^\top r,\qquad
        df = G\,s - q,

    where :math:`r=(\sqrt{N_g}(\hat p_g - \pi_g))_g` stacks the residuals of the
    :math:`G` groups, :math:`\Gamma=\operatorname{blockdiag}(\Gamma_g)` is their asymptotic
    covariance (:math:`\Xi_2` of group :math:`g` under local independence; with missing
    responses rescaled by :math:`N_gN_{ab}/(N_aN_b)`, DESIGN.md §16.20), :math:`\Delta_c` is an
    orthonormal basis for the orthogonal complement of
    :math:`\Delta^\ast=(\sqrt{N_g/N}\,\partial\pi_g/\partial\eta^\top)_g` (shared parameters
    once), :math:`s = J + \binom{J}{2}`, and :math:`q = \text{n\_params}`. For one group with
    complete data this is :math:`N\,e^\top\Delta_c(\Delta_c^\top\Xi_2\Delta_c)^{-1}\Delta_c^\top e`.

    ``groups`` holds each row's group code (default: all rows in group 0). By default
    ``_latent_in_delta=True`` includes free latent parameters in :math:`\Delta`
    (Maydeu-Olivares & Joe, 2005, 2006); ``_latent_in_delta=False`` drops them while keeping
    them in ``df``, and ``_mirt_convention=True`` leaves the group rows of :math:`\Delta`
    unscaled and uses mirt's missing-data weighting; together they reproduce R mirt 1.46.1
    (validation only).
    """
    n_obs, n_items = y.shape
    codes = np.zeros(n_obs, dtype=int) if groups is None else np.asarray(groups, dtype=int)
    n_groups = int(np.asarray(_core._latent_log_weights(prob, eta)).shape[0])
    samples = [sample_margins(y[codes == g], item_names) for g in range(n_groups)]

    def all_margins(e: Array) -> Array:
        return jnp.concatenate([model_margins(prob, e, g) for g in range(n_groups)])

    margins_all = np.asarray(all_margins(eta), dtype=float)
    jac = np.array(jax.jacfwd(all_margins)(eta), dtype=float)
    if not _latent_in_delta:
        lat_fids = {int(i) for i in np.asarray(prob.latent.free_index).ravel() if int(i) >= 0}
        if lat_fids:
            keep = [c for c in range(jac.shape[1]) if c not in lat_fids]
            jac = jac[:, keep]
    s = margins_all.size // n_groups
    log_w = np.asarray(_core._latent_log_weights(prob, eta), dtype=float)

    resids, gammas, srmsr = [], [], []
    for g, smp in enumerate(samples):
        margins = margins_all[g * s : (g + 1) * s]
        m1, m2 = (np.asarray(a, dtype=float) for a in conditional_moments(prob, eta, g))
        w = np.exp(log_w[g])
        xi = _xi(m1, m2, w, margins)
        r, gam = _residual_and_gamma(smp, margins, xi, _mirt_convention)
        resids.append(r)
        gammas.append(gam)
        if not _mirt_convention:
            jac[g * s : (g + 1) * s] *= np.sqrt(smp.n_rows / n_obs)
        srmsr.append(_srmsr(smp, margins, m2, w))

    df = int(n_groups * s - n_params)
    if df <= 0:
        warnings.warn(
            f"M2 has non-positive degrees of freedom (df={df}); the model is not locally "
            "identified with residual degrees of freedom from first- and second-order moments, "
            "so M2 and derived fit indices are NaN.",
            FlexpyirtWarning,
            stacklevel=3,
        )
        stat = float("nan")
    else:
        comp, rank = _svd_complement(jac)
        if comp is None:
            warnings.warn(
                f"M2 Jacobian is rank-deficient (rank {rank} < {jac.shape[1]} free parameters); "
                "the model is not locally identified from first- and second-order moments, "
                "so M2 and derived fit indices are NaN.",
                FlexpyirtWarning,
                stacklevel=3,
            )
            stat = float("nan")
        else:
            stat = _solve_quadratic_form(np.concatenate(resids), comp, _block_diag(gammas))

    if df > 0 and np.isfinite(stat):
        pval = float(stats.chi2.sf(stat, df))
        rmsea_val = rmsea(stat, df, n_obs)
        lo, hi = rmsea_ci(stat, df, n_obs, level)
    else:
        pval = float("nan")
        rmsea_val = float("nan")
        lo, hi = float("nan"), float("nan")

    tli_val = float("nan")
    cfi_val = float("nan")
    if df > 0 and np.isfinite(stat) and all(k == 2 for k in n_categories):
        x2_null, df_null = _null_m2(samples, n_items, _mirt_convention)
        if np.isfinite(x2_null) and df_null > 0:
            if x2_null > stat:
                denom_tli = x2_null / df_null - 1.0
                if denom_tli != 0.0:
                    tli_val = float((x2_null / df_null - stat / df) / denom_tli)
                denom_cfi = x2_null - df_null
                if denom_cfi != 0.0:
                    cfi_val = float(np.clip(1.0 - (stat - df) / denom_cfi, 0.0, 1.0))
            else:
                warnings.warn(
                    "Null model M2 is smaller than or equal to the fitted model M2; "
                    "TLI and CFI are NaN.",
                    FlexpyirtWarning,
                    stacklevel=3,
                )

    return M2Result(stat, df, pval, rmsea_val, lo, hi, tuple(srmsr), tli_val, cfi_val)
