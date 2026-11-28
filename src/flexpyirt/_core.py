"""Core layer: pure JAX functions on pytrees (DESIGN.md §2 ③, §4).

Nothing here knows about names, strings or DataFrames. All shapes are static; variable-length
data is padded and masked (zero-count pattern rows, ``-1`` free indices).

Notation (DESIGN.md §3): ``P`` patterns, ``J`` items, ``Q`` nodes, ``G`` groups, ``K`` categories,
``eta`` the free-parameter vector on the natural scale.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from flexpyirt._newton import newton_ascent
from flexpyirt.items import ItemKernel
from flexpyirt.latent import (
    davidian_log_weights,
    eh_constrained_weights,
    eh_project_floor,
    mixture_log_weights,
    mvn_log_weights,
)
from flexpyirt.priors import PRIOR_NONE, eval_log_prior

__all__ = [
    "BifactorBlock",
    "CompiledProblem",
    "CoupledBlock",
    "CoupledMember",
    "EMResult",
    "FamilyBlock",
    "GroupBlocks",
    "LatentBlock",
    "PriorBlock",
    "eap",
    "em",
    "information_hessian",
    "information_oakes",
    "log_posterior",
    "log_prior",
    "marginal_loglik",
    "squarem",
]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FamilyBlock:
    """Items sharing one kernel, for all groups (DESIGN.md §5.2, §14.1–14.2).

    Attributes
    ----------
    columns : (J_f,) int
        Column of each item in the response matrix.
    fixed_values : (G, J_f, P_f) float
        Values of fixed parameters (and starting values of free ones).
    free_index : (G, J_f, P_f) int
        Index into ``eta``, or ``-1`` if fixed.
    local_index : (G, J_f, P_f) int
        Slot in the item's local vector ``u_j`` (length ``S``), or ``-1`` if fixed. Parameters
        equal across groups share a slot.
    slot_free_index : (J_f, S) int
        Index into ``eta`` of each local slot, or ``-1`` for unused (padding) slots.
    slot_lower, slot_upper : (J_f, S) float
        Bounds of each local slot.
    kernel : ItemKernel
        Static log-probability kernel.
    complete : bool
        Static: ``True`` only if no response pattern has a missing response on any item of the
        family. Lets :func:`information_oakes` drop the category-0 indicator columns
        (DESIGN.md §15.20). Code that replaces ``y`` must reset it (:func:`with_responses`).
    """

    columns: Array
    fixed_values: Array
    free_index: Array
    local_index: Array
    slot_free_index: Array
    slot_lower: Array
    slot_upper: Array
    kernel: ItemKernel = field(metadata={"static": True})
    complete: bool = field(default=False, metadata={"static": True})


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LatentBlock:
    """Latent distribution of each group: multivariate normal or ``EmpiricalHistogram``.

    ``Normal`` parameter order matches :func:`~flexpyirt.latent.latent_param_names`:
    ``(mean, var)`` for ``D = 1`` (length 2) and ``(mean_1..D, var_1..D, cor_ij...)`` for
    ``D in {2, 3}`` (length 5 or 9). EH is unidimensional only (DESIGN.md §4.4, §5.2).

    Attributes
    ----------
    fixed_values, free_index, lower, upper : (G, L_D)
        ``Normal`` parameters (all ``free_index == -1`` for EH).
    eh_weights : (G, Q) float
        Empirical histogram weights per group (dummy ``(G, 1)`` for ``Normal``).
    eh_min_weight : (G,) float
        Strict positive floor per group for EH (dummy ``(G,)`` for ``Normal``).
    is_eh : bool
        Static flag distinguishing ``EmpiricalHistogram`` from ``Normal``.
    standardize_mean, standardize_var : bool
        Static flags selecting the reference-group moment constraints (DESIGN §4.4, §15.34).
    davidian_degree : int
        Static polynomial degree for :class:`~flexpyirt.DavidianCurve` (``0`` for Normal / EH).
    mixture_components : int
        Static number of components for :class:`~flexpyirt.NormalMixture` (``0`` otherwise);
        the parameters then follow :func:`~flexpyirt.latent.mixture_param_names`.
    """

    fixed_values: Array
    free_index: Array
    lower: Array
    upper: Array
    eh_weights: Array
    eh_min_weight: Array
    is_eh: bool = field(default=False, metadata={"static": True})
    standardize_mean: bool = field(default=False, metadata={"static": True})
    standardize_var: bool = field(default=False, metadata={"static": True})
    davidian_degree: int = field(default=0, metadata={"static": True})
    mixture_components: int = field(default=0, metadata={"static": True})


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PriorBlock:
    """Branch-free prior specification per free parameter ``eta`` (DESIGN.md §4.7).

    Attributes
    ----------
    kind : (n_free,) int32
        ``0`` none, ``1`` NormalPrior, ``2`` LogNormalPrior, ``3`` BetaPrior,
        ``4`` LogitNormalPrior, ``5`` LogitNormalPrior with ``mode_scale="logit"``.
    param1, param2 : (n_free,) float64
        Distribution parameters (unused entries hold ``0.0`` and ``1.0``).
    """

    kind: Array
    param1: Array
    param2: Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CoupledMember:
    """Items of one family inside a :class:`CoupledBlock` (DESIGN.md §16.1).

    Attributes
    ----------
    positions : (n,) int
        Positions of the member items within the family (indices into ``J_f``).
    local_index : (G, n, P_f) int
        Slot of each parameter in the component's vector ``u_c``, or ``-1`` if fixed.
    family : int
        Static index of the family in ``CompiledProblem.families``.
    """

    positions: Array
    local_index: Array
    family: int = field(metadata={"static": True})


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CoupledBlock:
    """Items connected by equality constraints across items, solved jointly (DESIGN.md §16.1).

    Attributes
    ----------
    slot_free_index : (S,) int
        Index into ``eta`` of each slot of the component's vector ``u_c`` (all free).
    slot_lower, slot_upper : (S,) float
        Bounds of each slot (intersected with the prior's support).
    members : tuple[CoupledMember, ...]
        One entry per family with items in the component.
    """

    slot_free_index: Array
    slot_lower: Array
    slot_upper: Array
    members: tuple[CoupledMember, ...]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class BifactorBlock:
    r"""Dimension reduction of a bifactor or two-tier model (DESIGN.md §17.1, §17.2).

    The first ``P`` factors are primary (``P = 1``, the general factor, for a bifactor model;
    Gibbons & Hedeker, 1992; Cai, 2010) and are integrated on their product grid of
    :math:`Q_p = n^P` nodes. Block :math:`b = 0..S-1` holds the items of specific factor
    :math:`b+1`; block :math:`S` holds the items without a specific factor, integrated with
    uniform "specific" weights (their log-probabilities do not depend on the specific node, so
    the inner sum is exact).

    Attributes
    ----------
    block_nodes : (B, Q_p n, D) float
        Nodes of each block (``B = S + 1``): columns ``0..P-1`` the primary node, column
        ``P + b`` the specific node, other columns 0; the primary index is major.
    primary_nodes : (Q_p, P) float
        Product grid of the primary factors.
    primary_base : (Q_p,) float
        Fixed log base weights of the primary nodes (§15.44).
    nodes_1d, base_1d : (n,) float
        One-dimensional nodes and their fixed log base weights (§15.44).
    family_block : tuple of (J_f,) int
        Block of each item, per family.
    n_specific : int
        Static number of specific factors ``S``.
    n_primary : int
        Static number of primary factors ``P``.
    """

    block_nodes: Array
    primary_nodes: Array
    primary_base: Array
    nodes_1d: Array
    base_1d: Array
    family_block: tuple[Array, ...]
    n_specific: int = field(metadata={"static": True})
    n_primary: int = field(default=1, metadata={"static": True})


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class GroupBlocks:
    """Group-blocked view of the pattern rows for many groups (DESIGN.md §26).

    Each group's pattern rows are split into blocks of a common size ``B``, ordered group-major.
    The flat pattern arrays of :class:`CompiledProblem` are unchanged; this is an index into
    them, built by ``compile_problem`` only.

    Attributes
    ----------
    rows : (N_B, B) int
        Row of each block slot in the flat pattern arrays; padding slots hold the sentinel ``P``
        (a zero-count, all-missing row appended at run time).
    group : (N_B,) int
        Group of each block (0 for empty padding blocks).
    last : (N_B,) bool
        Whether the block is the last one of its group (``False`` for padding blocks).
    """

    rows: Array
    group: Array
    last: Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CompiledProblem:
    r"""Everything the Core needs (DESIGN.md §5.2).

    Attributes
    ----------
    y : (P, J) int
        Unique response patterns, missing as ``-1``; padding rows have count 0.
    counts : (P,) float
        Pattern frequencies.
    group : (P,) int
        Group of each pattern.
    nodes : (Q, D) float
        Product quadrature nodes (``D in {1, 2, 3}``).
    node_log_base : (Q,) float
        Fixed log base weight of each node, added inside the normalization of the latent weights:
        0 for equispaced grids, :math:`\\log w^{GH}_q + \tfrac12\\|\theta_q\\|^2` for
        Gauss–Hermite (DESIGN.md §15.44).
    families : tuple[FamilyBlock, ...]
    latent : LatentBlock
    priors : PriorBlock
        Prior specification per free parameter (counted once per ``free_id``).
    n_groups : int
        Static number of groups.
    coupled : tuple[CoupledBlock, ...]
        Components of items connected by equality constraints across items (DESIGN.md §16.1);
        empty when every ``free_id`` stays within one item.
    reduction : BifactorBlock or None
        Bifactor / two-tier dimension reduction (DESIGN.md §17.1, §17.2). When set, ``nodes``
        is the primary grid embedded in ``D`` columns, item expected counts live on each item's
        block grid ``(Q_p n)``, and ``ExpectedCounts.n`` has shape ``(G, Q_p + B n)`` (primary
        marginal, then each block's specific marginal, flattened).
    blocks : GroupBlocks or None
        Group-blocked pattern index (DESIGN.md §26); set for non-reduced models with two or
        more groups, ``None`` otherwise (the flat path).
    """

    y: Array
    counts: Array
    group: Array
    nodes: Array
    families: tuple[FamilyBlock, ...]
    latent: LatentBlock
    priors: PriorBlock
    node_log_base: Array
    n_groups: int = field(metadata={"static": True})
    coupled: tuple[CoupledBlock, ...] = ()
    reduction: BifactorBlock | None = None
    blocks: GroupBlocks | None = None


def _expand(eta: Array, fixed: Array, free_index: Array) -> Array:
    """Return ``eta[free_index]`` where free and ``fixed`` elsewhere (DESIGN.md §13.2)."""
    if eta.shape[0] == 0:  # static: nothing is free
        return fixed
    return jnp.where(free_index >= 0, eta[jnp.maximum(free_index, 0)], fixed)


def _scatter(eta: Array, values: Array, free_index: Array) -> Array:
    """Write ``values`` of free slots back into ``eta`` (fixed slots are dropped)."""
    idx = jnp.where(free_index >= 0, free_index, eta.shape[0])
    return eta.at[idx.ravel()].set(values.ravel(), mode="drop")


def _family_nodes(prob: CompiledProblem, f: int) -> tuple[Array, int | None]:
    """Nodes of family ``f``'s items and their ``vmap`` axis over items.

    The shared ``(Q, D)`` grid (axis ``None``), or each item's bifactor block grid
    ``(J_f, n^2, D)`` (axis 0; DESIGN.md §17.1).
    """
    if prob.reduction is None:
        return prob.nodes, None
    return prob.reduction.block_nodes[prob.reduction.family_block[f]], 0


def _item_log_probs(prob: CompiledProblem, eta: Array) -> list[Array]:
    """Per family, ``(G, J_f, Q, K_f)`` log-probabilities on each item's nodes."""
    out = []
    for f, fam in enumerate(prob.families):
        xi = _expand(eta, fam.fixed_values, fam.free_index)  # (G, J_f, P_f)
        nodes, axis = _family_nodes(prob, f)

        def per_item(p: Array, x: Array, k: ItemKernel = fam.kernel) -> Array:
            return k.log_prob(x, p)

        per_group = jax.vmap(per_item, in_axes=(0, axis))
        out.append(jax.vmap(per_group, in_axes=(0, None))(xi, nodes))
    return out


def _one_hots(prob: CompiledProblem) -> list[Array]:
    """Per family, ``(P, J_f, K_f)`` one-hot responses; a missing response is an all-zero row."""
    return [
        jax.nn.one_hot(prob.y[:, fam.columns], fam.kernel.n_categories, dtype=prob.nodes.dtype)
        for fam in prob.families
    ]


LatentKind = tuple[int, int]
"""Static ``(davidian_degree, mixture_components)`` of a parametric latent block."""


def _latent_kind(lat: LatentBlock) -> LatentKind:
    """Return the static :data:`LatentKind` of ``lat``."""
    return (lat.davidian_degree, lat.mixture_components)


def _latent_log_weights_group(
    nodes: Array, base: Array, params: Array, kind: LatentKind = (0, 0)
) -> Array:
    """``(Q,)`` log weights for one parametric group (Normal / MVN / Davidian / NormalMixture).

    ``base`` is the fixed node log base weight (§15.44). Davidian curves are restricted to
    equispaced grids (``base`` is 0 there; ``compile_problem`` rejects Gauss–Hermite), so their
    weights are the density normalized on the grid as validated in V19. Normal mixtures
    normalize each component with ``base`` like ``Normal`` (§16.7).
    """
    davidian_degree, mixture_components = kind
    if davidian_degree > 0:
        return davidian_log_weights(nodes, params)
    if mixture_components > 0:
        return mixture_log_weights(nodes, params, base)
    return _normal_logw(nodes, base, params)


def _latent_log_weights(prob: CompiledProblem, eta: Array) -> Array:
    """``(G, Q)`` log quadrature weights."""
    if prob.latent.is_eh:
        return jnp.log(prob.latent.eh_weights)
    lat = _expand(eta, prob.latent.fixed_values, prob.latent.free_index)
    deg = _latent_kind(prob.latent)
    return jax.vmap(lambda p: _latent_log_weights_group(prob.nodes, prob.node_log_base, p, deg))(
        lat
    )


def _reduced_log_weights_group(red: BifactorBlock, params: Array) -> tuple[Array, Array]:
    r"""Primary ``(Q_p,)`` and block ``(B, n)`` log weights of one group (§17.1, §17.2).

    ``params`` is ``(mean_1..D, var_1..D, cor_ij...)`` with correlations among the primary
    factors only. The primary factors have the normalized multivariate-normal weights
    :math:`\log w^{(0)}=\log\mathrm{softmax}(b+\log\phi_P(\cdot;\boldsymbol\mu,\boldsymbol\Sigma))`
    on their product grid; each specific factor :math:`m` has the one-dimensional weights
    :math:`\log w^{(m)}=\log\mathrm{softmax}(b+\log\phi(\cdot;\mu_m,\sigma^2_m))`; the block of
    items without a specific factor uses uniform weights.
    """
    n_primary = red.n_primary
    n_factors = n_primary + red.n_specific
    means = params[:n_factors]
    variances = params[n_factors : 2 * n_factors]
    primary_params = jnp.concatenate(
        [means[:n_primary], variances[:n_primary], params[2 * n_factors :]]
    )
    log_wp = _normal_logw(red.primary_nodes, red.primary_base, primary_params)
    pairs = jnp.stack([means[n_primary:], variances[n_primary:]], axis=-1)  # (S, 2)
    log_ws = jax.vmap(lambda pair: _normal_logw(red.nodes_1d, red.base_1d, pair))(pairs)
    n_pts = red.nodes_1d.shape[0]
    uniform = jnp.full((1, n_pts), -jnp.log(n_pts), dtype=log_ws.dtype)
    return log_wp, jnp.concatenate([log_ws, uniform], axis=0)


def _bifactor_log_weights(prob: CompiledProblem, eta: Array) -> tuple[Array, Array]:
    """Primary ``(G, Q_p)`` and block ``(G, B, n)`` log weights of a reduced model (§17.1)."""
    red = prob.reduction
    assert red is not None
    lat = _expand(eta, prob.latent.fixed_values, prob.latent.free_index)  # (G, L)
    log_wp, log_wb = jax.vmap(lambda params: _reduced_log_weights_group(red, params))(lat)
    return log_wp, log_wb


def _reduced_latent_q(red: BifactorBlock, n: Array, params: Array) -> Array:
    r"""Latent part of Q of a reduced model for one group (DESIGN.md §17.2).

    .. math::
        Q^{\mathrm{lat}}=\sum_g\bar n^{(0)}_g\log w^{(0)}_g+\sum_{b,t}\bar n^{(b)}_t\log w^{(b)}_t

    with ``n`` the flat ``(Q_p + B n,)`` expected counts.
    """
    log_wp, log_wb = _reduced_log_weights_group(red, params)
    return jnp.sum(n * jnp.concatenate([log_wp, log_wb.ravel()]))


def _bifactor_parts(prob: CompiledProblem, eta: Array) -> tuple[Array, Array, Array]:
    r"""Factorized log joint of a bifactor model (Gibbons & Hedeker, 1992; DESIGN.md §17.1).

    With :math:`\ell^{(b)}_{pgt}=\sum_{j\in b}\log P_j(y_{pj}\mid\theta_g,\theta_t)`,

    .. math::
        u_{pbgt}=\ell^{(b)}_{pgt}+\log w^{(b)}_t,\qquad
        L^{(b)}_{pg}=\operatorname{logsumexp}_t u_{pbgt},\qquad
        \lambda_{pg}=\log w^{(0)}_g+\sum_b L^{(b)}_{pg}.

    For a two-tier model :math:`g` runs over the :math:`Q_p` nodes of the primary product grid
    (Cai, 2010; §17.2). Returns ``(lambda (P, Q_p), u (P, B, Q_p, n), L (P, B, Q_p))``.
    """
    red = prob.reduction
    assert red is not None
    n_blocks = red.n_specific + 1
    n_pts = red.nodes_1d.shape[0]
    n_prim = red.primary_nodes.shape[0]
    n_pat = prob.y.shape[0]
    ell = jnp.zeros((n_pat, prob.n_groups, n_blocks, n_prim * n_pts), dtype=prob.nodes.dtype)
    for f, (oh, logp) in enumerate(zip(_one_hots(prob), _item_log_probs(prob, eta), strict=True)):
        blk = jax.nn.one_hot(red.family_block[f], n_blocks, dtype=oh.dtype)  # (J_f, B)
        # Mask the small (G, B, J_f, K, Q) table first so the pattern contraction is one matmul
        # per (group, block) without a (P, J_f, B, Q) intermediate.
        masked = jnp.einsum("gjqk,jb->gbjkq", logp, blk)
        ell = ell + jnp.einsum("pjk,gbjkq->pgbq", oh, masked)
    own = jnp.take_along_axis(ell, prob.group[:, None, None, None], axis=1)[:, 0]
    log_wg, log_wb = _bifactor_log_weights(prob, eta)
    inner = own.reshape(n_pat, n_blocks, n_prim, n_pts) + log_wb[prob.group][:, :, None, :]
    block_ll = jax.scipy.special.logsumexp(inner, axis=3)  # (P, B, Q_p)
    lam = log_wg[prob.group] + jnp.sum(block_ll, axis=1)
    return lam, inner, block_ll


def _log_joint(prob: CompiledProblem, eta: Array) -> Array:
    r"""``(P, Q)`` :math:`\ell_{pq}+\log w_{g(p)q}` (DESIGN.md §4.2, §14.3).

    The item part is computed for every group, ``(P, G, Q)``, and each pattern's own group is
    then selected, avoiding a ``(P, J, Q, K)`` intermediate. For a bifactor model this is the
    general-factor log joint :math:`\lambda_{pg}` of :func:`_bifactor_parts` (§17.1).
    """
    if prob.reduction is not None:
        return _bifactor_parts(prob, eta)[0]
    if prob.blocks is not None:
        return _blocked_log_joint(prob, eta)
    n_pat, n_nodes = prob.y.shape[0], prob.nodes.shape[0]
    ell = jnp.zeros((n_pat, prob.n_groups, n_nodes), dtype=prob.nodes.dtype)
    for oh, logp in zip(_one_hots(prob), _item_log_probs(prob, eta), strict=True):
        ell = ell + jnp.einsum("pjk,gjqk->pgq", oh, logp)
    own = jnp.take_along_axis(ell, prob.group[:, None, None], axis=1)[:, 0, :]
    return own + _latent_log_weights(prob, eta)[prob.group]


_BLOCK_MAP_ROWS = 512
"""Rows evaluated together per ``lax.map`` step of the blocked log joint (DESIGN.md §26)."""


def _padded_rows(prob: CompiledProblem) -> tuple[Array, Array]:
    """Return ``y`` and ``counts`` with the sentinel row ``P`` appended (all missing, count 0)."""
    y = jnp.concatenate([prob.y, jnp.full((1, prob.y.shape[1]), -1, dtype=prob.y.dtype)])
    counts = jnp.concatenate([prob.counts, jnp.zeros((1,), dtype=prob.counts.dtype)])
    return y, counts


def _block_log_joint(
    prob: CompiledProblem,
    log_probs: list[Array],
    log_w: Array,
    y_block: Array,
    g: Array,
) -> tuple[Array, list[Array]]:
    r"""``(B, Q)`` log joint of one single-group block and its per-family one-hots (§26).

    .. math:: \lambda_{pq}=\log w_{gq}+\sum_f\sum_{j,k}\mathbf 1[y_{pj}=k]\log P^{(g)}_{jkq}
    """
    out = jnp.broadcast_to(log_w[g], (y_block.shape[0], log_w.shape[1]))
    one_hots = []
    for fam, lp in zip(prob.families, log_probs, strict=True):
        oh = jax.nn.one_hot(y_block[:, fam.columns], fam.kernel.n_categories, dtype=log_w.dtype)
        out = out + jnp.einsum("pjk,jqk->pq", oh, lp[g])
        one_hots.append(oh)
    return out, one_hots


def _unblock(prob: CompiledProblem, per_block: Array) -> Array:
    """Scatter ``(N_B, B, Q)`` block rows back to the flat ``(P, Q)`` rows (sentinels dropped).

    Flat padding rows, which are in no block, are 0.
    """
    blk = prob.blocks
    assert blk is not None
    n_nodes = per_block.shape[-1]
    out = jnp.zeros((prob.y.shape[0], n_nodes), dtype=per_block.dtype)
    return out.at[blk.rows.ravel()].set(per_block.reshape(-1, n_nodes), mode="drop")


def _blocked_log_joint(prob: CompiledProblem, eta: Array) -> Array:
    r"""``(P, Q)`` log joint evaluated block by block, each block under its own group (§26).

    Costs :math:`O(N_B B\,JKQ)` instead of the flat path's :math:`O(PGJKQ)`; flat padding rows
    (count 0, in no block) get 0.
    """
    blk = prob.blocks
    assert blk is not None
    log_probs = _item_log_probs(prob, eta)
    log_w = _latent_log_weights(prob, eta)
    y_pad, _ = _padded_rows(prob)

    def one(xs: tuple[Array, Array]) -> Array:
        rows, g = xs
        return _block_log_joint(prob, log_probs, log_w, y_pad[rows], g)[0]

    batch = max(1, _BLOCK_MAP_ROWS // blk.rows.shape[1])
    per_block = jax.lax.map(one, (blk.rows, blk.group), batch_size=batch)
    return _unblock(prob, per_block)


def marginal_loglik(prob: CompiledProblem, eta: Array) -> Array:
    r"""Marginal log-likelihood (Bock & Aitkin, 1981).

    .. math::
        \ell(\eta)=\sum_p c_p\log\sum_q w_{g(p)q}\prod_j P_j(y_{pj}\mid\theta_q)
    """
    return jnp.sum(prob.counts * jax.scipy.special.logsumexp(_log_joint(prob, eta), axis=1))


def log_prior(prob: CompiledProblem, eta: Array) -> Array:
    r"""Sum of log priors over all free parameters on the natural scale (DESIGN.md §4.7).

    Returns ``0.0`` when no priors are present or ``eta`` is empty. Each free parameter
    (``free_id``) is counted once; fixed parameters are not in ``eta`` and contribute nothing.
    """
    if eta.shape[0] == 0:
        return jnp.zeros((), dtype=prob.nodes.dtype)
    return jnp.sum(eval_log_prior(eta, prob.priors.kind, prob.priors.param1, prob.priors.param2))


def log_posterior(prob: CompiledProblem, eta: Array) -> Array:
    r"""Log posterior :math:`\ell(\eta) + \log\pi(\eta)` (DESIGN.md §4.7).

    Equals :func:`marginal_loglik` when no priors are attached.
    """
    return marginal_loglik(prob, eta) + log_prior(prob, eta)


class ExpectedCounts(NamedTuple):
    r"""E-step output.

    ``r``: :math:`\bar r_{gjkq}` per family, shape ``(G, J_f, K_f, Q)``; ``n``: :math:`\bar n_{gq}`.
    """

    r: tuple[Array, ...]
    n: Array


def _bifactor_posterior(prob: CompiledProblem, eta: Array) -> tuple[Array, Array, Array]:
    r"""Posterior of a bifactor model (DESIGN.md §17.1).

    .. math::
        h_{pg}=\mathrm{softmax}_g\lambda_{pg},\qquad
        h^{(b)}_{pgt}=h_{pg}\exp(u_{pbgt}-L^{(b)}_{pg})

    Returns ``(h (P, Q_p), h_block (P, B, Q_p, n), log L_p (P,))``.
    """
    lam, inner, block_ll = _bifactor_parts(prob, eta)
    lse = jax.scipy.special.logsumexp(lam, axis=1)
    h = jnp.exp(lam - lse[:, None])
    h_block = h[:, None, :, None] * jnp.exp(inner - block_ll[..., None])
    return h, h_block, lse


def _bifactor_e_step(prob: CompiledProblem, eta: Array) -> tuple[ExpectedCounts, Array, Array]:
    r"""E-step of a bifactor model on the block grids (DESIGN.md §17.1).

    .. math::
        \bar r_{gjkq}=\sum_{p\in g}c_p h^{(b(j))}_{pq}\mathbf 1[y_{pj}=k],\qquad
        q=(g', t)

    ``n`` is the primary marginal followed by each block's specific marginal, flattened to
    ``(G, Q_p + B n)`` (§17.2).
    """
    red = prob.reduction
    assert red is not None
    n_blocks = red.n_specific + 1
    h, h_block, lse = _bifactor_posterior(prob, eta)
    n_pat = h.shape[0]
    group_one_hot = jax.nn.one_hot(prob.group, prob.n_groups, dtype=h.dtype)
    cg = prob.counts[:, None] * group_one_hot  # (P, G)
    weighted = h_block.reshape(n_pat, n_blocks, -1)
    r_list = []
    for f, oh in enumerate(_one_hots(prob)):
        blk = jax.nn.one_hot(red.family_block[f], n_blocks, dtype=h.dtype)  # (J_f, B)
        coh = jnp.einsum("pa,pjk->pajk", cg, oh)
        # One (A J_f K) x P x Q matmul per block, then keep each item's own block.
        per_block = jnp.einsum("pajk,pbq->bajkq", coh, weighted)
        r_list.append(jnp.einsum("bajkq,jb->ajkq", per_block, blk))
    r_bar = tuple(r_list)
    n_general = jnp.einsum("pa,pg->ag", cg, h)
    n_specific = jnp.einsum("pa,pbgt->abt", cg, h_block)
    n_bar = jnp.concatenate([n_general, n_specific.reshape(prob.n_groups, -1)], axis=1)
    loglik = jnp.sum(prob.counts * lse)
    return ExpectedCounts(r_bar, n_bar), loglik, h


def _e_step_details(prob: CompiledProblem, eta: Array) -> tuple[ExpectedCounts, Array, Array]:
    """Run one E-step and return ``(counts, loglik, h)`` with ``h`` of shape ``(P, Q)``."""
    if prob.reduction is not None:
        return _bifactor_e_step(prob, eta)
    if prob.blocks is not None:
        return _blocked_e_step(prob, eta)
    log_joint = _log_joint(prob, eta)
    lse = jax.scipy.special.logsumexp(log_joint, axis=1, keepdims=True)
    h = jnp.exp(log_joint - lse)  # softmax_q(log_joint), (P, Q)
    weighted = prob.counts[:, None] * h  # c_p h_pq, (P, Q)
    group_one_hot = jax.nn.one_hot(prob.group, prob.n_groups, dtype=weighted.dtype)
    n_bar = jnp.einsum("pg,pq->gq", group_one_hot, weighted)
    r_bar = tuple(
        jnp.einsum("pg,pjk,pq->gjkq", group_one_hot, oh, weighted) for oh in _one_hots(prob)
    )
    loglik = jnp.sum(prob.counts * lse[:, 0])
    return ExpectedCounts(r_bar, n_bar), loglik, h


def _blocked_e_step(prob: CompiledProblem, eta: Array) -> tuple[ExpectedCounts, Array, Array]:
    r"""E-step folded over single-group blocks with ``lax.scan`` (DESIGN.md §26).

    .. math::
        \bar n_{gq}=\sum_{b:g(b)=g}\sum_{p\in b}c_ph_{pq},\qquad
        \bar r_{gjkq}=\sum_{b:g(b)=g}\sum_{p\in b}c_ph_{pq}\mathbf 1[y_{pj}=k]

    The counts are accumulated in place at each block's group, so peak memory does not grow
    with the number of blocks. Returns ``(counts, loglik, h)`` with ``h`` on the flat rows.
    """
    blk = prob.blocks
    assert blk is not None
    log_probs = _item_log_probs(prob, eta)
    log_w = _latent_log_weights(prob, eta)
    y_pad, c_pad = _padded_rows(prob)
    dtype = log_w.dtype
    n_groups, n_nodes = log_w.shape
    init = (
        jnp.zeros((n_groups, n_nodes), dtype=dtype),
        tuple(
            jnp.zeros((n_groups, fam.columns.shape[0], fam.kernel.n_categories, n_nodes), dtype)
            for fam in prob.families
        ),
        jnp.zeros((), dtype=dtype),
    )

    def body(
        carry: tuple[Array, tuple[Array, ...], Array], xs: tuple[Array, Array]
    ) -> tuple[tuple[Array, tuple[Array, ...], Array], Array]:
        n_acc, r_acc, ll = carry
        rows, g = xs
        lam, one_hots = _block_log_joint(prob, log_probs, log_w, y_pad[rows], g)
        c_b = c_pad[rows]
        lse = jax.scipy.special.logsumexp(lam, axis=1)
        h = jnp.exp(lam - lse[:, None])
        weighted = c_b[:, None] * h
        n_acc = n_acc.at[g].add(jnp.sum(weighted, axis=0))
        r_acc = tuple(
            r.at[g].add(jnp.einsum("pjk,pq->jkq", oh, weighted))
            for r, oh in zip(r_acc, one_hots, strict=True)
        )
        return (n_acc, r_acc, ll + jnp.sum(c_b * lse)), h

    (n_bar, r_bar, loglik), h_blocks = jax.lax.scan(body, init, (blk.rows, blk.group))
    return ExpectedCounts(r_bar, n_bar), loglik, _unblock(prob, h_blocks)


def e_step(prob: CompiledProblem, eta: Array) -> tuple[ExpectedCounts, Array]:
    r"""E-step (DESIGN.md §4.2).

    .. math::
        h_{pq}=\mathrm{softmax}_q(\ell_{pq}+\log w_{g(p)q}),\quad
        \bar n_{gq}=\sum_{p\in g}c_ph_{pq},\quad
        \bar r_{gjkq}=\sum_{p\in g}c_ph_{pq}\mathbf 1[y_{pj}=k]

    Returns the expected counts and the marginal log-likelihood at ``eta``.
    """
    counts, loglik, _ = _e_step_details(prob, eta)
    return counts, loglik


def _item_q(kernel: ItemKernel, nodes: Array, r: Array, params: Array) -> Array:
    r""":math:`Q_{jg}(\xi)=\sum_q\sum_k\bar r_{gjkq}\log P_{jk}(\theta_q;\xi)` for one group (§4.3).

    Missing responses contribute nothing because their one-hot rows are zero.
    """
    return jnp.sum(r.T * kernel.log_prob(nodes, params))


def _normal_logw(nodes: Array, base: Array, params: Array) -> Array:
    r"""Return normalized log latent weights :math:`\log\mathrm{softmax}_q(b_q+\log\phi_q)`.

    ``base`` is the fixed node log base weight (0 on equispaced grids; Gauss–Hermite, §15.44).
    """
    out: Array = jax.nn.log_softmax(base + mvn_log_weights(nodes, params))
    return out


def _latent_q(
    nodes: Array, base: Array, n: Array, params: Array, kind: LatentKind = (0, 0)
) -> Array:
    r"""Latent part of Q (DESIGN.md §4.4, §16.4, §16.7).

    .. math:: Q_g^{\mathrm{lat}}=\sum_q\bar n_{gq}\log w_{gq}(\text{params}_g)
    """
    logw = _latent_log_weights_group(nodes, base, params, kind)
    if kind[0] > 0:
        # Guard 0 * (-inf) when a polynomial root coincides with a zero-count node.
        return jnp.sum(jnp.where(n > 0.0, n * logw, 0.0))
    return jnp.sum(n * logw)


def _local_to_xi(u: Array, local: Array, fixed: Array) -> Array:
    """Item parameters of every group, ``(G, P_f)``, from the item's local vector ``u``."""
    return jnp.where(local >= 0, u[jnp.maximum(local, 0)], fixed)


def expected_complete_loglik(prob: CompiledProblem, eta: Array, counts: ExpectedCounts) -> Array:
    r""":math:`Q(\eta\mid\eta')` with the expected counts computed at :math:`\eta'`."""
    total = jnp.zeros((), dtype=prob.nodes.dtype)
    for f, (fam, r) in enumerate(zip(prob.families, counts.r, strict=True)):
        xi = _expand(eta, fam.fixed_values, fam.free_index)  # (G, J_f, P_f)
        nodes, axis = _family_nodes(prob, f)

        def per_item(rj: Array, p: Array, x: Array, k: ItemKernel = fam.kernel) -> Array:
            return _item_q(k, x, rj, p)

        per_group = jax.vmap(per_item, in_axes=(0, 0, axis))
        total = total + jnp.sum(jax.vmap(per_group, in_axes=(0, 0, None))(r, xi, nodes))
    if prob.reduction is not None:
        # Primary factors jointly, each specific factor on its own (§17.1, §17.2).
        log_wp, log_wb = _bifactor_log_weights(prob, eta)
        log_w = jnp.concatenate([log_wp, log_wb.reshape(prob.n_groups, -1)], axis=1)
        return total + jnp.sum(counts.n * log_w)
    if prob.latent.is_eh:
        return total + jnp.sum(counts.n * jnp.log(prob.latent.eh_weights))
    lat = _expand(eta, prob.latent.fixed_values, prob.latent.free_index)
    deg = _latent_kind(prob.latent)
    q_lat = jax.vmap(lambda n, p: _latent_q(prob.nodes, prob.node_log_base, n, p, deg))(
        counts.n, lat
    )
    return total + jnp.sum(q_lat)


def _coupled_masks(prob: CompiledProblem) -> list[Array]:
    """Per family, ``(J_f,)`` mask of the items that belong to a :class:`CoupledBlock`."""
    masks = [jnp.zeros(fam.columns.shape, dtype=jnp.bool_) for fam in prob.families]
    for block in prob.coupled:
        for mem in block.members:
            masks[mem.family] = masks[mem.family].at[mem.positions].set(True)
    return masks


def _m_step_coupled(
    prob: CompiledProblem,
    eta: Array,
    new_eta: Array,
    counts: ExpectedCounts,
    n_steps: int,
    unconstrained: bool = False,
) -> Array:
    r"""Joint masked Newton per component of items linked across items (DESIGN.md §16.1).

    For component :math:`c` with free vector :math:`u_c`, maximizes

    .. math::
        Q_c(u_c)=\sum_{j\in c}\sum_g\sum_q\sum_k\bar r_{gjkq}
        \log P_{jk}(\theta_q;\xi_j^{(g)}(u_c))+\sum_s\log\pi(u_{cs}),

    the part of :math:`Q(\eta\mid\eta')` (Bock & Aitkin, 1981; Dempster, Laird & Rubin, 1977)
    that depends on :math:`u_c`, with every prior counted once. Reads ``u_c`` from ``eta`` (the
    previous iterate) and writes it into ``new_eta``.
    """
    for block in prob.coupled:
        fids = block.slot_free_index
        u0 = eta[fids]
        kind = prob.priors.kind[fids]
        p1 = prob.priors.param1[fids]
        p2 = prob.priors.param2[fids]

        def coupled_q(
            v: Array, blk: CoupledBlock = block, sk: Array = kind, sp1: Array = p1, sp2: Array = p2
        ) -> Array:
            total = jnp.sum(eval_log_prior(v, sk, sp1, sp2))
            for mem in blk.members:
                fam = prob.families[mem.family]
                r_m = counts.r[mem.family][:, mem.positions]  # (G, n, K, Q)
                fix_m = fam.fixed_values[:, mem.positions]  # (G, n, P_f)
                xi = _local_to_xi(v, mem.local_index, fix_m)  # (G, n, P_f)
                nodes, axis = _family_nodes(prob, mem.family)
                if axis is not None:
                    nodes = nodes[mem.positions]

                def per_item(rj: Array, pj: Array, x: Array, k: ItemKernel = fam.kernel) -> Array:
                    return _item_q(k, x, rj, pj)

                per_group = jax.vmap(per_item, in_axes=(0, 0, axis))
                total = total + jnp.sum(jax.vmap(per_group, in_axes=(0, 0, None))(r_m, xi, nodes))
            return total

        mask = jnp.ones(fids.shape, dtype=jnp.bool_)
        u_new = newton_ascent(
            coupled_q,
            u0,
            mask,
            block.slot_lower,
            block.slot_upper,
            n_steps,
            unconstrained=unconstrained,
        )
        new_eta = new_eta.at[fids].set(u_new)
    return new_eta


def _m_step_items(
    prob: CompiledProblem,
    eta: Array,
    counts: ExpectedCounts,
    n_steps: int,
    unconstrained: bool = False,
) -> Array:
    """Item-wise masked Newton M-step, then the coupled components (DESIGN.md §4.3, §14.2, §16.1).

    Items in a :class:`CoupledBlock` are held in the item-wise ``vmap`` and solved jointly by
    :func:`_m_step_coupled`.
    """
    new_eta = eta
    coupled_masks = _coupled_masks(prob)
    for f, (fam, r, cmask) in enumerate(zip(prob.families, counts.r, coupled_masks, strict=True)):
        nodes, axis = _family_nodes(prob, f)
        u0 = _expand(eta, jnp.zeros(fam.slot_free_index.shape, eta.dtype), fam.slot_free_index)
        slot_free = fam.slot_free_index >= 0
        if eta.shape[0] == 0:
            slot_kind = jnp.full(fam.slot_free_index.shape, PRIOR_NONE, dtype=jnp.int32)
            slot_p1 = jnp.zeros(fam.slot_free_index.shape, dtype=eta.dtype)
            slot_p2 = jnp.ones(fam.slot_free_index.shape, dtype=eta.dtype)
        else:
            safe_fid = jnp.maximum(fam.slot_free_index, 0)
            slot_kind = jnp.where(slot_free, prob.priors.kind[safe_fid], PRIOR_NONE)
            slot_p1 = jnp.where(slot_free, prob.priors.param1[safe_fid], 0.0)
            slot_p2 = jnp.where(slot_free, prob.priors.param2[safe_fid], 1.0)
        r_items = jnp.swapaxes(r, 0, 1)  # (J_f, G, K, Q)
        local = jnp.swapaxes(fam.local_index, 0, 1)  # (J_f, G, P_f)
        fixed = jnp.swapaxes(fam.fixed_values, 0, 1)

        def solve_item(
            rj: Array,
            loc: Array,
            fix: Array,
            u: Array,
            free: Array,
            lo: Array,
            hi: Array,
            sk: Array,
            sp1: Array,
            sp2: Array,
            x: Array,
            k: ItemKernel = fam.kernel,
        ) -> Array:
            def q_item(v: Array) -> Array:
                xi = _local_to_xi(v, loc, fix)  # (G, P_f)
                q_lik = jnp.sum(jax.vmap(lambda rg, pg: _item_q(k, x, rg, pg))(rj, xi))
                return q_lik + jnp.sum(eval_log_prior(v, sk, sp1, sp2))

            return newton_ascent(q_item, u, free, lo, hi, n_steps, unconstrained=unconstrained)

        u_new = jax.vmap(solve_item, in_axes=(0,) * 10 + (axis,))(
            r_items,
            local,
            fixed,
            u0,
            slot_free & ~cmask[:, None],
            fam.slot_lower,
            fam.slot_upper,
            slot_kind,
            slot_p1,
            slot_p2,
            nodes,
        )
        new_eta = _scatter(new_eta, u_new, fam.slot_free_index)
    if not prob.coupled:
        return new_eta
    return _m_step_coupled(prob, eta, new_eta, counts, n_steps, unconstrained)


def _nodes_1d(prob: CompiledProblem) -> Array:
    """``(Q,)`` nodes of a unidimensional problem (EH is unidimensional only)."""
    nodes: Array = prob.nodes[:, 0] if prob.nodes.ndim == 2 else prob.nodes
    return nodes


def _eh_n_moments(prob: CompiledProblem) -> int:
    """Return the number of reference-group moment constraints (0, 1 or 2; DESIGN §15.34)."""
    lat = prob.latent
    return 2 if lat.standardize_var else (1 if lat.standardize_mean else 0)


def _eh_weights_update(prob: CompiledProblem, counts_or_w: Array) -> Array:
    """Exact EH M-step (or I-projection) for every group: constrained reference, floored focal."""
    w = jax.vmap(eh_project_floor)(counts_or_w, prob.latent.eh_min_weight)
    n_mom = _eh_n_moments(prob)
    if n_mom == 0:
        return w
    w0 = eh_constrained_weights(
        counts_or_w[0], _nodes_1d(prob), prob.latent.eh_min_weight[0], n_mom
    )
    return w.at[0].set(w0)


def _m_step_eh(
    prob: CompiledProblem,
    eta: Array,
    counts: ExpectedCounts,
    n_steps: int,
    unconstrained: bool = False,
) -> tuple[Array, Array]:
    """Item Newton M-step + exact (moment-constrained) EH M-step (DESIGN.md §4.4, §15.34)."""
    new_eta = _m_step_items(prob, eta, counts, n_steps, unconstrained)
    return new_eta, _eh_weights_update(prob, counts.n)


def m_step(
    prob: CompiledProblem,
    eta: Array,
    counts: ExpectedCounts,
    n_steps: int,
    unconstrained: bool = False,
) -> Array:
    """M-step: item-wise and group-wise masked Newton, ``vmap``-ed (DESIGN.md §4.3, §4.4, §14.2).

    Each item is one Newton block over its local vector, which holds the parameters of all
    groups with across-group equalities merged into shared slots. Item-parameter priors are added
    directly to each item's local objective so the item-wise decomposition is preserved and every
    shared parameter's prior is counted once.
    """
    new_eta = _m_step_items(prob, eta, counts, n_steps, unconstrained)
    red = prob.reduction
    if prob.latent.is_eh or (red is not None and red.n_primary < 2):
        # Bifactor latent parameters are all fixed (compile_problem checks; DESIGN.md §17.1).
        return new_eta

    lat_block = prob.latent
    lat = _expand(eta, lat_block.fixed_values, lat_block.free_index)
    if red is not None:
        # Two-tier: only the correlations among primary factors can be free (§17.2).
        def solve_reduced(n: Array, x: Array, free: Array, lo: Array, hi: Array) -> Array:
            return newton_ascent(
                lambda p: _reduced_latent_q(red, n, p),
                x,
                free,
                lo,
                hi,
                n_steps,
                unconstrained=unconstrained,
            )

        lat_reduced = jax.vmap(solve_reduced)(
            counts.n, lat, lat_block.free_index >= 0, lat_block.lower, lat_block.upper
        )
        return _scatter(new_eta, lat_reduced, lat_block.free_index)
    deg = _latent_kind(lat_block)

    def solve_group(n: Array, x: Array, free: Array, lo: Array, hi: Array) -> Array:
        return newton_ascent(
            lambda p: _latent_q(prob.nodes, prob.node_log_base, n, p, deg),
            x,
            free,
            lo,
            hi,
            n_steps,
            unconstrained=unconstrained,
        )

    lat_new = jax.vmap(solve_group)(
        counts.n, lat, lat_block.free_index >= 0, lat_block.lower, lat_block.upper
    )
    return _scatter(new_eta, lat_new, lat_block.free_index)


class EMResult(NamedTuple):
    """Output of :func:`em` and :func:`squarem`.

    History arrays have length ``max_iter``; entries past ``n_iter`` are NaN (for float arrays),
    0 (for ``em_eval_history``), or False (for ``safeguard_history``).
    """

    eta: Array
    eh_weights: Array
    n_iter: Array
    converged: Array
    loglik: Array
    log_posterior: Array
    loglik_history: Array
    log_posterior_history: Array
    change_history: Array
    n_em_evaluations: Array
    em_eval_history: Array
    safeguard_history: Array
    n_safeguards: Array


_SQUAREM_MAX_BACKTRACK = 10
"""Maximum halvings of ``alpha`` towards ``-1`` before falling back to ``theta_2``."""


def _eta_bounds(prob: CompiledProblem, eta: Array) -> tuple[Array, Array]:
    """Per-coordinate ``(lower, upper)`` bounds of ``eta`` from the M-step blocks (§15.12)."""
    lo = jnp.full_like(eta, -jnp.inf)
    hi = jnp.full_like(eta, jnp.inf)
    for fam in prob.families:
        lo = _scatter(lo, fam.slot_lower, fam.slot_free_index)
        hi = _scatter(hi, fam.slot_upper, fam.slot_free_index)
    lo = _scatter(lo, prob.latent.lower, prob.latent.free_index)
    hi = _scatter(hi, prob.latent.upper, prob.latent.free_index)
    return lo, hi


def _with_eh_weights(prob: CompiledProblem, eh_weights: Array) -> CompiledProblem:
    """Return ``prob`` with ``latent.eh_weights`` replaced (static fields unchanged)."""
    return replace(prob, latent=replace(prob.latent, eh_weights=eh_weights))


def with_responses(prob: CompiledProblem, **changes: Any) -> CompiledProblem:
    """Return ``prob`` with response arrays replaced (``y``, ``group``, ``counts``, ``blocks``).

    Every family's static ``complete`` flag is reset to ``False``, since the new responses may
    have missing entries (DESIGN.md §15.20). Use this instead of ``replace(prob, y=...)``.
    """
    families = tuple(replace(fam, complete=False) for fam in prob.families)
    return replace(prob, families=families, **changes)


def em(
    prob: CompiledProblem,
    eta0: Array,
    *,
    tol: float,
    max_iter: int,
    n_newton: int,
    unconstrained: bool = False,
) -> EMResult:
    """Bock–Aitkin EM in a ``lax.while_loop`` (DESIGN.md §4.6, §4.7).

    ``loglik_history[t]`` and ``log_posterior_history[t]`` are evaluated at the parameters entering
    iteration ``t``; ``change_history[t]`` is the maximum absolute parameter change made by
    iteration ``t``. EM is monotone in the log posterior (which equals the log-likelihood without
    priors).
    """
    nan_hist = jnp.full((max_iter,), jnp.nan, dtype=eta0.dtype)
    eval_hist = jnp.zeros((max_iter,), dtype=jnp.int32)
    safe_hist = jnp.zeros((max_iter,), dtype=jnp.bool_)
    one = jnp.asarray(1, dtype=jnp.int32)
    is_eh = prob.latent.is_eh

    def cond(state: tuple[Array, Array, Array, Array, Array, Array, Array, Array]) -> Array:
        _, _, it, change, _, _, _, _ = state
        return (it < max_iter) & (change >= tol)

    def body(
        state: tuple[Array, Array, Array, Array, Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
        eta, eh_w, it, _, ll_hist, lp_hist, ch_hist, ev_hist = state
        prob_t = _with_eh_weights(prob, eh_w) if is_eh else prob
        counts, ll = e_step(prob_t, eta)
        lp = ll + log_prior(prob_t, eta)
        if is_eh:
            new_eta, new_w = _m_step_eh(prob_t, eta, counts, n_newton, unconstrained)
            change = jnp.maximum(
                jnp.max(jnp.abs(new_eta - eta), initial=0.0),
                jnp.max(jnp.abs(new_w - eh_w), initial=0.0),
            )
        else:
            new_eta = m_step(prob, eta, counts, n_newton, unconstrained)
            new_w = eh_w
            change = jnp.max(jnp.abs(new_eta - eta), initial=0.0)
        next_it = it + one
        return (
            new_eta,
            new_w,
            next_it,
            change,
            ll_hist.at[it].set(ll),
            lp_hist.at[it].set(lp),
            ch_hist.at[it].set(change),
            ev_hist.at[it].set(next_it),
        )

    init = (
        eta0,
        prob.latent.eh_weights,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(jnp.inf, eta0.dtype),
        nan_hist,
        nan_hist,
        nan_hist,
        eval_hist,
    )
    eta, eh_w, n_iter, change, ll_hist, lp_hist, ch_hist, ev_hist = jax.lax.while_loop(
        cond, body, init
    )
    prob_fin = _with_eh_weights(prob, eh_w) if is_eh else prob
    final_ll = marginal_loglik(prob_fin, eta)
    return EMResult(
        eta=eta,
        eh_weights=eh_w,
        n_iter=n_iter,
        converged=change < tol,
        loglik=final_ll,
        log_posterior=final_ll + log_prior(prob_fin, eta),
        loglik_history=ll_hist,
        log_posterior_history=lp_hist,
        change_history=ch_hist,
        n_em_evaluations=n_iter,
        em_eval_history=ev_hist,
        safeguard_history=safe_hist,
        n_safeguards=jnp.asarray(0, dtype=jnp.int32),
    )


def _eh_pack(eta: Array, eh_w: Array) -> Array:
    r"""Pack ``(eta, sqrt(eh_w))`` for SQUAREM on the square-root weight scale (DESIGN.md §15.36).

    :math:`z_{gq}=\sqrt{w_{gq}}` makes the multinomial Fisher information isotropic
    (:math:`4I`), so near-empty nodes, which barely affect the likelihood, do not dominate the
    SQUAREM step length as they do on the log scale.
    """
    return jnp.concatenate([eta, jnp.sqrt(eh_w).ravel()])


def _eh_unpack(
    prob: CompiledProblem, vec: Array, n_free: int, n_groups: int, n_nodes: int
) -> tuple[Array, Array]:
    r"""Unpack a SQUAREM vector: :math:`w\propto z^2`, then project onto the feasible weights."""
    eta = vec[:n_free]
    z = vec[n_free:].reshape((n_groups, n_nodes))
    w_sq = z * z
    w = _eh_weights_update(prob, w_sq / jnp.sum(w_sq, axis=1, keepdims=True))
    n_mom = _eh_n_moments(prob)
    if n_mom > 0:
        # A candidate whose I-projection misses the moment constraints (dual optimum on the
        # boundary, DESIGN.md §15.34) is made NaN so the SQUAREM safeguard rejects it.
        nodes = _nodes_1d(prob)
        moments = jnp.stack([jnp.ones_like(nodes), nodes, nodes**2])[: n_mom + 1] @ w[0]
        target = jnp.asarray([1.0, 0.0, 1.0], dtype=w.dtype)[: n_mom + 1]
        ok = jnp.max(jnp.abs(moments - target)) < 1e-10
        w = jnp.where(ok, w, jnp.nan)
    return eta, w


def _squarem_eh(
    prob: CompiledProblem,
    eta0: Array,
    *,
    tol: float,
    max_iter: int,
    n_newton: int,
    unconstrained: bool = False,
) -> EMResult:
    """SQUAREM scheme S3 on ``(eta, sqrt(w))`` for EH models (DESIGN.md §15.36)."""
    lower, upper = _eta_bounds(prob, eta0)
    n_free = eta0.shape[0]
    n_groups = prob.n_groups
    n_nodes = prob.nodes.shape[0]

    nan_hist = jnp.full((max_iter,), jnp.nan, dtype=eta0.dtype)
    eval_hist = jnp.zeros((max_iter,), dtype=jnp.int32)
    safe_hist = jnp.zeros((max_iter,), dtype=jnp.bool_)
    zero = jnp.asarray(0, dtype=jnp.int32)
    one = jnp.asarray(1, dtype=jnp.int32)
    two = jnp.asarray(2, dtype=jnp.int32)
    three = jnp.asarray(3, dtype=jnp.int32)

    def _eval_lp_pair(eta: Array, eh_w: Array) -> Array:
        return log_posterior(_with_eh_weights(prob, eh_w), eta)

    def _em_once(eta: Array, eh_w: Array) -> tuple[Array, Array, Array, Array]:
        prob_t = _with_eh_weights(prob, eh_w)
        counts, ll = e_step(prob_t, eta)
        lp = ll + log_prior(prob_t, eta)
        new_eta, new_w = _m_step_eh(prob_t, eta, counts, n_newton, unconstrained)
        return new_eta, new_w, ll, lp

    def _is_acceptable(cand: Array, lp_ref: Array) -> Array:
        eta_c = cand[:n_free]
        z_c = cand[n_free:]
        items_ok = jnp.all(jnp.isfinite(eta_c) & (eta_c > lower) & (eta_c < upper))
        z_ok = jnp.all(jnp.isfinite(z_c)) & jnp.any(z_c != 0.0)
        in_bounds = items_ok & z_ok

        def _eval_lp(c: Array) -> Array:
            e_c, w_c = _eh_unpack(prob, c, n_free, n_groups, n_nodes)
            lp_c = _eval_lp_pair(e_c, w_c)
            return jnp.isfinite(lp_c) & (lp_c >= lp_ref)

        accepted: Array = jax.lax.cond(in_bounds, _eval_lp, lambda _: jnp.asarray(False), cand)
        return accepted

    def cond(
        state: tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array, Array, Array],
    ) -> Array:
        _, _, it, change, _, _, _, _, _, _, _ = state
        return (it < max_iter) & (change >= tol)

    def body(
        state: tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array, Array, Array]:
        eta, eh_w, it, _, n_evals, n_safe, ll_hist, lp_hist, ch_hist, ev_hist, sf_hist = state
        v0 = _eh_pack(eta, eh_w)
        eta1, w1, ll0, lp0 = _em_once(eta, eh_w)
        v1 = _eh_pack(eta1, w1)
        eta2, w2, _, _ = _em_once(eta1, w1)
        v2 = _eh_pack(eta2, w2)

        r = v1 - v0
        s = v2 - v1
        v = s - r
        r_norm = jnp.sqrt(jnp.sum(r * r))
        v_norm = jnp.sqrt(jnp.sum(v * v))
        has_v = v_norm > 0.0
        raw_alpha = jnp.where(has_v, -r_norm / jnp.where(has_v, v_norm, 1.0), -1.0)
        alpha = jnp.where(jnp.isfinite(raw_alpha), jnp.minimum(raw_alpha, -1.0), -1.0)

        def _extrapolate(_unused: None) -> tuple[Array, Array, Array]:
            lp2 = _eval_lp_pair(eta2, w2)
            lp_ref = jnp.maximum(lp2, lp0)

            def backtrack_cond(bt: tuple[Array, Array, Array, Array]) -> Array:
                k, _, ok, _ = bt
                return (~ok) & (k <= _SQUAREM_MAX_BACKTRACK)

            def backtrack_body(
                bt: tuple[Array, Array, Array, Array],
            ) -> tuple[Array, Array, Array, Array]:
                k, delta, _, _ = bt
                cand = v2 - 2.0 * delta * s + (delta * delta) * v
                ok = _is_acceptable(cand, lp_ref)
                next_delta = jnp.where(ok, delta, 0.5 * delta)
                next_k = jnp.where(ok, k, k + one)
                return next_k, next_delta, ok, cand

            init_bt = (zero, alpha + 1.0, jnp.asarray(False), v2)
            k_out, _, ok, cand = jax.lax.while_loop(backtrack_cond, backtrack_body, init_bt)
            fired = k_out > zero
            v_p = jnp.where(ok, cand, v2)
            return v_p, ok, fired

        def _no_extrapolate(_unused: None) -> tuple[Array, Array, Array]:
            return v2, jnp.asarray(True), jnp.asarray(False)

        v_prime, do_stabilize, fired = jax.lax.cond(
            alpha < -1.0, _extrapolate, _no_extrapolate, None
        )

        def _stabilize(vp: Array) -> tuple[Array, Array, Array]:
            e_p, w_p = _eh_unpack(prob, vp, n_free, n_groups, n_nodes)
            e_n, w_n, _, _ = _em_once(e_p, w_p)
            return e_n, w_n, three

        def _fallback(_unused: Array) -> tuple[Array, Array, Array]:
            return eta2, w2, two

        new_eta, new_w, evals_added = jax.lax.cond(do_stabilize, _stabilize, _fallback, v_prime)
        change = jnp.maximum(
            jnp.max(jnp.abs(new_eta - eta), initial=0.0),
            jnp.max(jnp.abs(new_w - eh_w), initial=0.0),
        )
        next_evals = n_evals + evals_added
        next_safe = n_safe + fired.astype(jnp.int32)
        return (
            new_eta,
            new_w,
            it + one,
            change,
            next_evals,
            next_safe,
            ll_hist.at[it].set(ll0),
            lp_hist.at[it].set(lp0),
            ch_hist.at[it].set(change),
            ev_hist.at[it].set(next_evals),
            sf_hist.at[it].set(fired),
        )

    init = (
        eta0,
        prob.latent.eh_weights,
        zero,
        jnp.asarray(jnp.inf, eta0.dtype),
        zero,
        zero,
        nan_hist,
        nan_hist,
        nan_hist,
        eval_hist,
        safe_hist,
    )
    eta, eh_w, n_iter, change, n_evals, n_safe, ll_hist, lp_hist, ch_hist, ev_hist, sf_hist = (
        jax.lax.while_loop(cond, body, init)
    )
    prob_fin = _with_eh_weights(prob, eh_w)
    final_ll = marginal_loglik(prob_fin, eta)
    return EMResult(
        eta=eta,
        eh_weights=eh_w,
        n_iter=n_iter,
        converged=change < tol,
        loglik=final_ll,
        log_posterior=final_ll + log_prior(prob_fin, eta),
        loglik_history=ll_hist,
        log_posterior_history=lp_hist,
        change_history=ch_hist,
        n_em_evaluations=n_evals,
        em_eval_history=ev_hist,
        safeguard_history=sf_hist,
        n_safeguards=n_safe,
    )


def squarem(
    prob: CompiledProblem,
    eta0: Array,
    *,
    tol: float,
    max_iter: int,
    n_newton: int,
    unconstrained: bool = False,
) -> EMResult:
    r"""SQUAREM-accelerated Bock–Aitkin EM (Varadhan & Roland, 2008, scheme S3; DESIGN §4.6, §4.7).

    Each iteration applies the EM map :math:`F` twice to obtain :math:`\theta_1=F(\theta_0)` and
    :math:`\theta_2=F(\theta_1)`, forms :math:`r=\theta_1-\theta_0`,
    :math:`s=\theta_2-\theta_1`, :math:`v=s-r`, and computes the S3 step length

    .. math::
        \alpha = \min\!\left(-\frac{\lVert r\rVert_2}{\lVert v\rVert_2},\,-1\right),
        \qquad
        \theta'(\alpha) = \theta_0 - 2\alpha r + \alpha^2 v
        = \theta_2 - 2\delta s + \delta^2 v \quad (\delta=\alpha+1\le 0).

    For ``EmpiricalHistogram`` models, EH weights are extrapolated on the square-root scale
    :math:`z=\sqrt w` and mapped back by :math:`w\propto z^2` plus the EH M-step projection
    (DESIGN.md §15.36).
    Monotonicity and feasibility safeguard: if :math:`\theta'(\alpha)` leaves the strict parameter
    bounds ``(lower, upper)`` or gives a non-finite or lower log posterior than
    :math:`\max(\ell_{\mathrm{post}}(\theta_2),\ell_{\mathrm{post}}(\theta_0))`, :math:`\alpha` is
    stepped back towards :math:`-1` via :math:`\alpha\leftarrow(\alpha-1)/2` (i.e.
    :math:`\delta\leftarrow\delta/2`) up to ``_SQUAREM_MAX_BACKTRACK`` times. An accepted
    :math:`\theta'` is followed by one stabilizing EM step :math:`\theta_{\text{new}}=F(\theta')`
    (3 EM map evaluations); if no backtracked candidate is acceptable,
    :math:`\theta_{\text{new}}=\theta_2` is used directly (2 EM map evaluations). Runs entirely
    inside ``lax.while_loop``.
    """
    if prob.latent.is_eh:
        return _squarem_eh(
            prob,
            eta0,
            tol=tol,
            max_iter=max_iter,
            n_newton=n_newton,
            unconstrained=unconstrained,
        )
    lower, upper = _eta_bounds(prob, eta0)
    nan_hist = jnp.full((max_iter,), jnp.nan, dtype=eta0.dtype)
    eval_hist = jnp.zeros((max_iter,), dtype=jnp.int32)
    safe_hist = jnp.zeros((max_iter,), dtype=jnp.bool_)
    zero = jnp.asarray(0, dtype=jnp.int32)
    one = jnp.asarray(1, dtype=jnp.int32)
    two = jnp.asarray(2, dtype=jnp.int32)
    three = jnp.asarray(3, dtype=jnp.int32)

    def _is_acceptable(cand: Array, lp_ref: Array) -> Array:
        in_bounds = jnp.all(jnp.isfinite(cand) & (cand > lower) & (cand < upper))

        def _eval_lp(c: Array) -> Array:
            lp_c = log_posterior(prob, c)
            return jnp.isfinite(lp_c) & (lp_c >= lp_ref)

        accepted: Array = jax.lax.cond(in_bounds, _eval_lp, lambda _: jnp.asarray(False), cand)
        return accepted

    def cond(
        state: tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array, Array],
    ) -> Array:
        _, it, change, _, _, _, _, _, _, _ = state
        return (it < max_iter) & (change >= tol)

    def body(
        state: tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array, Array]:
        eta, it, _, n_evals, n_safe, ll_hist, lp_hist, ch_hist, ev_hist, sf_hist = state
        counts0, ll0 = e_step(prob, eta)
        lp0 = ll0 + log_prior(prob, eta)
        theta1 = m_step(prob, eta, counts0, n_newton, unconstrained)
        counts1, _ = e_step(prob, theta1)
        theta2 = m_step(prob, theta1, counts1, n_newton, unconstrained)

        r = theta1 - eta
        s = theta2 - theta1
        v = s - r
        r_norm = jnp.sqrt(jnp.sum(r * r))
        v_norm = jnp.sqrt(jnp.sum(v * v))
        has_v = v_norm > 0.0
        raw_alpha = jnp.where(has_v, -r_norm / jnp.where(has_v, v_norm, 1.0), -1.0)
        alpha = jnp.where(jnp.isfinite(raw_alpha), jnp.minimum(raw_alpha, -1.0), -1.0)

        def _extrapolate(_unused: None) -> tuple[Array, Array, Array]:
            lp2 = log_posterior(prob, theta2)
            lp_ref = jnp.maximum(lp2, lp0)

            def backtrack_cond(bt: tuple[Array, Array, Array, Array]) -> Array:
                k, _, ok, _ = bt
                return (~ok) & (k <= _SQUAREM_MAX_BACKTRACK)

            def backtrack_body(
                bt: tuple[Array, Array, Array, Array],
            ) -> tuple[Array, Array, Array, Array]:
                k, delta, _, _ = bt
                cand = theta2 - 2.0 * delta * s + (delta * delta) * v
                ok = _is_acceptable(cand, lp_ref)
                next_delta = jnp.where(ok, delta, 0.5 * delta)
                next_k = jnp.where(ok, k, k + one)
                return next_k, next_delta, ok, cand

            init_bt = (zero, alpha + 1.0, jnp.asarray(False), theta2)
            k_out, _, ok, cand = jax.lax.while_loop(backtrack_cond, backtrack_body, init_bt)
            fired = k_out > zero
            theta_p = jnp.where(ok, cand, theta2)
            return theta_p, ok, fired

        def _no_extrapolate(_unused: None) -> tuple[Array, Array, Array]:
            return theta2, jnp.asarray(True), jnp.asarray(False)

        theta_prime, do_stabilize, fired = jax.lax.cond(
            alpha < -1.0, _extrapolate, _no_extrapolate, None
        )

        def _stabilize(t: Array) -> tuple[Array, Array]:
            c, _ = e_step(prob, t)
            return m_step(prob, t, c, n_newton, unconstrained), three

        def _fallback(_unused: Array) -> tuple[Array, Array]:
            return theta2, two

        new, evals_added = jax.lax.cond(do_stabilize, _stabilize, _fallback, theta_prime)
        change = jnp.max(jnp.abs(new - eta), initial=0.0)
        next_evals = n_evals + evals_added
        next_safe = n_safe + fired.astype(jnp.int32)
        return (
            new,
            it + one,
            change,
            next_evals,
            next_safe,
            ll_hist.at[it].set(ll0),
            lp_hist.at[it].set(lp0),
            ch_hist.at[it].set(change),
            ev_hist.at[it].set(next_evals),
            sf_hist.at[it].set(fired),
        )

    init = (
        eta0,
        zero,
        jnp.asarray(jnp.inf, eta0.dtype),
        zero,
        zero,
        nan_hist,
        nan_hist,
        nan_hist,
        eval_hist,
        safe_hist,
    )
    eta, n_iter, change, n_evals, n_safe, ll_hist, lp_hist, ch_hist, ev_hist, sf_hist = (
        jax.lax.while_loop(cond, body, init)
    )
    final_ll = marginal_loglik(prob, eta)
    return EMResult(
        eta=eta,
        eh_weights=prob.latent.eh_weights,
        n_iter=n_iter,
        converged=change < tol,
        loglik=final_ll,
        log_posterior=final_ll + log_prior(prob, eta),
        loglik_history=ll_hist,
        log_posterior_history=lp_hist,
        change_history=ch_hist,
        n_em_evaluations=n_evals,
        em_eval_history=ev_hist,
        safeguard_history=sf_hist,
        n_safeguards=n_safe,
    )


_HESSIAN_CHUNK_SIZE = 8
"""Number of basis directions differentiated together in :func:`information_hessian`."""
_OAKES_CHUNK_ELEMENTS = 4_000_000
"""Target element budget (~32 MB in float64) for the ``(Q, B, D)`` Oakes cross-term workspace."""
_OAKES_MAX_CHUNK = 1024
"""Maximum number of response patterns processed per chunk in :func:`information_oakes`."""


def information_hessian(prob: CompiledProblem, eta: Array) -> Array:
    r"""Observed posterior information :math:`-\nabla^2_\eta[\ell(\eta)+\log\pi(\eta)]` (§4.8).

    Evaluates Hessian-vector products of :func:`log_posterior` in fixed-size batches of standard
    basis vectors via ``jax.linearize`` and ``lax.scan`` so peak memory stays bounded on large
    problems (DESIGN.md §15.20). Reduces to :math:`-\nabla^2_\eta\ell(\eta)` when no priors are
    present.
    """
    if eta.shape[0] == 0:
        return jnp.zeros((0, 0), dtype=eta.dtype)
    return -_chunked_jacobian(jax.grad(lambda e: log_posterior(prob, e)), eta)


def _chunked_jacobian(fn: Callable[[Array], Array], eta: Array) -> Array:
    """Jacobian of ``fn: R^n -> R^n`` at ``eta`` by batched forward-mode JVPs (§15.20).

    ``jax.linearize`` is applied once and the standard basis is pushed through in fixed-size
    batches with ``lax.scan``, so peak memory stays bounded. Row ``i`` is the derivative along
    the ``i``-th basis vector (symmetric for the Hessian-like uses here).
    """
    n_free = eta.shape[0]
    _, hvp = jax.linearize(fn, eta)
    chunk = min(_HESSIAN_CHUNK_SIZE, n_free)
    n_chunks = (n_free + chunk - 1) // chunk
    pad = n_chunks * chunk - n_free
    eye = jnp.eye(n_free, dtype=eta.dtype)
    if pad:
        eye = jnp.concatenate([eye, jnp.zeros((pad, n_free), dtype=eta.dtype)], axis=0)
    chunks = eye.reshape(n_chunks, chunk, n_free)

    def _batch(_carry: None, basis: Array) -> tuple[None, Array]:
        return None, jax.vmap(hvp)(basis)

    _, cols = jax.lax.scan(_batch, None, chunks)
    jac: Array = cols.reshape(n_chunks * chunk, n_free)[:n_free]
    return jac


def _oakes_own_hessian(prob: CompiledProblem, eta: Array, counts: ExpectedCounts) -> Array:
    r"""Complete-data and prior Hessian block of Oakes' identity (DESIGN.md §15.20)."""
    n_free = eta.shape[0]
    h_own = jnp.zeros((n_free, n_free), dtype=eta.dtype)
    for fam, r in zip(prob.families, counts.r, strict=True):
        u0 = _expand(eta, jnp.zeros(fam.slot_free_index.shape, eta.dtype), fam.slot_free_index)
        r_items = jnp.swapaxes(r, 0, 1)  # (J_f, G, K_f, Q)
        local = jnp.swapaxes(fam.local_index, 0, 1)  # (J_f, G, P_f)
        fixed = jnp.swapaxes(fam.fixed_values, 0, 1)

        def item_hess(
            rj: Array, loc: Array, fix: Array, u: Array, k: ItemKernel = fam.kernel
        ) -> Array:
            def q_item(v: Array) -> Array:
                xi = _local_to_xi(v, loc, fix)
                return jnp.sum(jax.vmap(lambda rg, pg: _item_q(k, prob.nodes, rg, pg))(rj, xi))

            out: Array = jax.hessian(q_item)(u)
            return out

        h_items = jax.vmap(item_hess)(r_items, local, fixed, u0)  # (J_f, S_f, S_f)
        idx = jnp.where(fam.slot_free_index >= 0, fam.slot_free_index, n_free)
        h_own = h_own.at[idx[:, :, None], idx[:, None, :]].add(h_items, mode="drop")

    lat_block = prob.latent
    if not lat_block.is_eh:
        lat = _expand(eta, lat_block.fixed_values, lat_block.free_index)
        deg = _latent_kind(lat_block)

        def group_hess(n: Array, x: Array) -> Array:
            out: Array = jax.hessian(
                lambda p: _latent_q(prob.nodes, prob.node_log_base, n, p, deg)
            )(x)
            return out

        h_lat = jax.vmap(group_hess)(counts.n, lat)  # (G, L, L)
        lat_idx = jnp.where(lat_block.free_index >= 0, lat_block.free_index, n_free)
        h_own = h_own.at[lat_idx[:, :, None], lat_idx[:, None, :]].add(h_lat, mode="drop")

    def prior_d2(x: Array, kind: Array, p1: Array, p2: Array) -> Array:
        out: Array = jax.grad(jax.grad(lambda v: eval_log_prior(v, kind, p1, p2)))(x)
        return out

    d2_prior = jax.vmap(prior_d2)(eta, prob.priors.kind, prob.priors.param1, prob.priors.param2)
    return h_own + jnp.diag(d2_prior)


class _OakesTables(NamedTuple):
    """Centered score tables shared by both forms of Oakes' cross term (§15.20, §26)."""

    fam_d: list[tuple[FamilyBlock, Array]]  # per family ``(G, J_f, K_f, Q, S_f)``
    fam_cols: list[Array]  # per family ``(G, J_f, K'_f, Q, S_f)`` columns of ``L_q``
    fam_const: list[Array | None]  # per complete family ``(G, J_f, Q, S_f)``, row 0 of ``L_q``
    fam_mean: list[Array]  # per family ``(G, J_f, K_f, S_f)`` centering subtracted from fam_d
    d_logw: Array  # ``(G, Q, L)``
    lat_mean: Array  # ``(G, L)`` centering subtracted from d_logw
    fam_fids: list[Array]  # per family ``(J_f, S_f)`` free index, ``n_free`` if fixed
    lat_fids: Array  # ``(G, L)`` free index, ``n_free`` if fixed
    n_cols: int  # ``D = 1 + sum_f J_f K'_f``, ``K'_f = K_f - 1`` for complete families


def _oakes_tables(prob: CompiledProblem, eta: Array, counts: ExpectedCounts) -> _OakesTables:
    r"""Kernel and latent score tables, centered at each group's expected-count weights.

    For a family with no missing response, :math:`\mathbb 1[y_{pj}=0]=1-\sum_{k>0}
    \mathbb 1[y_{pj}=k]`, so :math:`d_{j,y_{pj},q}=d_{j0q}+\sum_{k>0}\mathbb 1[y_{pj}=k]\,
    (d_{jkq}-d_{j0q})`: the category-0 column is dropped, :math:`d_{j0q}` moves to the constant
    row 0 of :math:`L_q` and the other columns become :math:`d_{jkq}-d_{j0q}` (§15.20).
    """
    n_free = eta.shape[0]
    G = prob.n_groups
    Q = prob.nodes.shape[0]
    # Normalize with a guarded positive denominator (a clamp at 1 would leave weights that do not
    # sum to 1 when a total is fractional); an empty group falls back to uniform weights.
    n_g_tot = jnp.sum(counts.n, axis=1, keepdims=True)
    q_w_g = jnp.where(
        n_g_tot > 0, counts.n / jnp.where(n_g_tot > 0, n_g_tot, 1.0), 1.0 / counts.n.shape[1]
    )  # (G, Q)

    fam_d: list[tuple[FamilyBlock, Array]] = []
    fam_cols: list[Array] = []
    fam_const: list[Array | None] = []
    fam_mean: list[Array] = []
    D = 1  # column 0 carries the per-group latent score (and d_{j0q} of complete families)
    for fam, r in zip(prob.families, counts.r, strict=True):
        u0 = _expand(eta, jnp.zeros(fam.slot_free_index.shape, eta.dtype), fam.slot_free_index)
        local = jnp.swapaxes(fam.local_index, 0, 1)
        fixed = jnp.swapaxes(fam.fixed_values, 0, 1)

        def item_jac(loc: Array, fix: Array, u: Array, k: ItemKernel = fam.kernel) -> Array:
            def log_prob_of_slots(v: Array) -> Array:
                xi = _local_to_xi(v, loc, fix)
                return jax.vmap(lambda p: k.log_prob(prob.nodes, p))(xi)

            out: Array = jax.jacfwd(log_prob_of_slots)(u)
            return out

        d_lp = jnp.transpose(jax.vmap(item_jac)(local, fixed, u0), (1, 0, 3, 2, 4))  # (G,J,K,Q,S)
        r_tot = jnp.sum(r, axis=-1, keepdims=True)  # (G, J, K, 1)
        r_w = jnp.where(r_tot > 0, r / jnp.where(r_tot > 0, r_tot, 1.0), q_w_g[:, None, None, :])
        d_mean = jnp.sum(d_lp * r_w[..., None], axis=3)  # (G, J, K, S)
        d_lp = d_lp - d_mean[:, :, :, None, :]
        fam_d.append((fam, d_lp))
        fam_mean.append(d_mean)
        if fam.complete:
            fam_const.append(d_lp[:, :, 0])
            fam_cols.append(d_lp[:, :, 1:] - d_lp[:, :, :1])
        else:
            fam_const.append(None)
            fam_cols.append(d_lp)
        D += fam.columns.shape[0] * fam_cols[-1].shape[2]

    lat_block = prob.latent
    if lat_block.is_eh:
        d_logw = jnp.zeros((G, Q, lat_block.fixed_values.shape[1]), dtype=eta.dtype)
        lat_mean = jnp.zeros((G, lat_block.fixed_values.shape[1]), dtype=eta.dtype)
    else:
        lat = _expand(eta, lat_block.fixed_values, lat_block.free_index)
        deg = _latent_kind(lat_block)

        def latent_jac(p: Array) -> Array:
            out: Array = jax.jacfwd(
                lambda x: _latent_log_weights_group(prob.nodes, prob.node_log_base, x, deg)
            )(p)
            return out

        d_logw = jax.vmap(latent_jac)(lat)  # (G, Q, L)
        lat_mean = jnp.sum(d_logw * q_w_g[:, :, None], axis=1)  # (G, L)
        d_logw = d_logw - lat_mean[:, None, :]

    return _OakesTables(
        fam_d=fam_d,
        fam_cols=fam_cols,
        fam_const=fam_const,
        fam_mean=fam_mean,
        d_logw=d_logw,
        lat_mean=lat_mean,
        fam_fids=[
            jnp.where(fam.slot_free_index >= 0, fam.slot_free_index, n_free) for fam, _ in fam_d
        ],
        lat_fids=jnp.where(lat_block.free_index >= 0, lat_block.free_index, n_free),
        n_cols=D,
    )


def _oakes_flat_tables(tabs: _OakesTables) -> list[Array]:
    """Per family, the score tables reshaped to ``(G, Q, J_f K_f S_f)`` for the chunk products."""
    out = []
    for _, d_lp in tabs.fam_d:
        n_groups, _, _, n_nodes, _ = d_lp.shape
        out.append(jnp.transpose(d_lp, (0, 3, 1, 2, 4)).reshape(n_groups, n_nodes, -1))
    return out


def _oakes_pairs(n_cols: int) -> tuple[np.ndarray, np.ndarray]:
    """Row and column indices of the upper triangle (with diagonal) of a ``(D, D)`` matrix."""
    rows, cols = np.triu_indices(n_cols)
    return rows.astype(np.int32), cols.astype(np.int32)


def _oakes_chunk_terms(
    tabs: _OakesTables,
    eta: Array,
    d_flats_g: list[Array],
    dlw_g: Array,
    lfids_g: Array,
    y_c: Array,
    c_c: Array,
    h_c: Array,
    g: Array | int,
    with_meat: bool,
) -> tuple[Array, Array]:
    r"""Category co-occurrences and the ``T_2`` term of one chunk of one group.

    The co-occurrences :math:`C_q=\sum_p c_p h_{pq}\,x_px_p^\top` are symmetric, so only their
    upper triangle is formed, as one ``(Q, B) @ (B, D(D+1)/2)`` product: shape
    ``(Q, D(D+1)/2)`` in the order of :func:`_oakes_pairs`. ``d_flats_g``, ``dlw_g`` and
    ``lfids_g`` are that group's slices of :func:`_oakes_flat_tables`, ``tabs.d_logw`` and
    ``tabs.lat_fids``; ``c_c`` must already be zero outside the group (and on padding rows).

    With ``with_meat`` the second output is ``(2, n_free, n_free)``: ``T_2`` and the score
    cross product :math:`\sum_p c_p s_ps_p^\top` with :math:`s_p=\nabla_\eta\log L_p` the
    uncentered :math:`\bar s_p` (the centering is :math:`q`-independent and
    :math:`\sum_q h_{pq}=1`, so adding it back gives the raw score exactly; §17.5).
    """
    n_free = eta.shape[0]
    B = y_c.shape[0]
    wg = c_c[:, None] * h_c  # (B, Q)
    x_parts = [jnp.ones((B, 1), dtype=eta.dtype)]
    s_bar = jnp.zeros((B, n_free), dtype=eta.dtype)
    shift = jnp.zeros((B, n_free), dtype=eta.dtype)  # centering of s_bar, with_meat only
    for (fam, _), fids, d_flat, d_mean in zip(
        tabs.fam_d, tabs.fam_fids, d_flats_g, tabs.fam_mean, strict=True
    ):
        K_f = fam.kernel.n_categories
        J_f = fam.columns.shape[0]
        S_f = fam.slot_free_index.shape[1]
        y_f = y_c[:, fam.columns]
        oh = jax.nn.one_hot(y_f, K_f, dtype=eta.dtype)
        if fam.complete:
            oh = oh[:, :, 1:]  # category 0 is implied by the constant column
        x_parts.append(oh.reshape(B, -1))
        hd = (h_c @ d_flat).reshape(B, J_f, K_f, S_f)
        y_safe = jnp.maximum(y_f, 0)
        picked = jnp.take_along_axis(hd, y_safe[:, :, None, None], axis=2)[:, :, 0, :]
        picked = jnp.where((y_f >= 0)[:, :, None], picked, 0.0)
        s_bar = s_bar.at[:, fids.ravel()].add(picked.reshape(B, -1), mode="drop")
        if with_meat:
            mean_j = d_mean[g][jnp.arange(J_f)[None, :], y_safe]  # (B, J_f, S_f)
            mean_j = jnp.where((y_f >= 0)[:, :, None], mean_j, 0.0)
            shift = shift.at[:, fids.ravel()].add(mean_j.reshape(B, -1), mode="drop")
    x_c = jnp.concatenate(x_parts, axis=1)  # (B, D)
    iu, ju = _oakes_pairs(tabs.n_cols)
    c_gq = wg.T @ (x_c[:, iu] * x_c[:, ju])  # (Q, D(D+1)/2)
    s_bar = s_bar.at[:, lfids_g].add(h_c @ dlw_g, mode="drop")
    t2_g = (s_bar * c_c[:, None]).T @ s_bar
    if not with_meat:
        return c_gq, t2_g
    s_raw = shift.at[:, lfids_g].add(tabs.lat_mean[g], mode="drop") + s_bar
    meat_g = (s_raw * c_c[:, None]).T @ s_raw
    return c_gq, jnp.stack([t2_g, meat_g])


def _oakes_t1(tabs: _OakesTables, eta: Array, g: Array | int, c_tri: Array) -> Array:
    r""":math:`T_1=\sum_q L_{gq}^\top C_{gq} L_{gq}` of group ``g`` from its co-occurrences.

    ``c_tri`` is the ``(Q, D(D+1)/2)`` upper triangle accumulated by :func:`_oakes_chunk_terms`.
    """
    n_free = eta.shape[0]
    D = tabs.n_cols
    Q = c_tri.shape[0]
    iu, ju = _oakes_pairs(D)
    c_mat = jnp.zeros((Q, D, D), dtype=eta.dtype).at[:, iu, ju].set(c_tri)
    c_mat = c_mat.at[:, ju, iu].set(c_tri)
    dlw_g = tabs.d_logw[g]
    lfids_g = tabs.lat_fids[g]

    def accumulate_t1(t1_acc: Array, q_idx: Array) -> tuple[Array, None]:
        l_q = jnp.zeros((D, n_free), dtype=eta.dtype)
        l_q = l_q.at[0, lfids_g].add(dlw_g[q_idx], mode="drop")
        col = 1
        for fam_cols, const, fids in zip(tabs.fam_cols, tabs.fam_const, tabs.fam_fids, strict=True):
            _, J_f, K_x, _, _ = fam_cols.shape
            n_cols = J_f * K_x
            if const is not None:
                l_q = l_q.at[0, fids].add(const[g, :, q_idx, :], mode="drop")
            d_idx = (col + jnp.arange(n_cols)).reshape(J_f, K_x)
            l_q = l_q.at[d_idx[:, :, None], fids[:, None, :]].add(
                fam_cols[g, :, :, q_idx, :], mode="drop"
            )
            col += n_cols
        return t1_acc + l_q.T @ (c_mat[q_idx] @ l_q), None

    t1, _ = jax.lax.scan(
        accumulate_t1,
        jnp.zeros((n_free, n_free), dtype=eta.dtype),
        jnp.arange(Q),
    )
    return t1


def _oakes_chunk_size(tabs: _OakesTables, n_nodes: int, n_rows: int) -> int:
    """Largest power of two not above ``n_rows`` that keeps a chunk's workspace bounded.

    The workspace is the ``(B, D(D+1)/2)`` pair products and the ``(B, Q)`` weights.
    """
    n_pairs = tabs.n_cols * (tabs.n_cols + 1) // 2
    target_b = max(1, _OAKES_CHUNK_ELEMENTS // max(1, n_pairs + n_nodes))
    return 1 << (min(n_rows, target_b, _OAKES_MAX_CHUNK).bit_length() - 1)


def _oakes_cross_hessian(
    prob: CompiledProblem, eta: Array, counts: ExpectedCounts, h: Array, *, with_meat: bool = False
) -> tuple[Array, Array | None]:
    r"""Oakes cross term :math:`\partial^2 Q(\eta\mid\eta')/\partial\eta\,\partial\eta'^\top`.

    Writing :math:`Q(\eta\mid\eta')=\sum_p c_p\sum_q h_{pq}(\eta')\,\psi_{pq}(\eta)` with
    :math:`h_{pq}(\eta')=\mathrm{softmax}_q(\psi_{pq}(\eta'))`, differentiating through the
    E-step gives (Oakes, 1999; Louis, 1982; DESIGN.md §15.20)

    .. math::
        \left.\frac{\partial^2 Q(\eta\mid\eta')}
            {\partial\eta\,\partial\eta'^\top}\right|_{\eta'=\eta}
        = \sum_{p=1}^P c_p\left[
            \sum_{q=1}^Q h_{pq}\,s_{pq}s_{pq}^\top - \bar s_p\bar s_p^\top
          \right], \qquad
        s_{pq}=\nabla_\eta\psi_{pq}(\eta),\quad \bar s_p=\sum_q h_{pq}s_{pq}.

    Every kernel and latent derivative comes from ``jax.jacfwd``. Subtracting a
    :math:`q`-independent weighted mean from each category's score table leaves
    :math:`s_{pq}-\bar s_p` algebraically invariant while preventing cancellation in the
    difference. Pattern chunks skip zero-count padding rows via ``lax.cond`` and never
    materialize ``(P, Q, n_free)`` or ``(G, Q, D, n_free)`` intermediates.

    Returns the cross term and, with ``with_meat``, the score cross product
    :math:`\sum_p c_p s_ps_p^\top` from the same pass (else ``None``).
    """
    if prob.blocks is not None:
        return _oakes_cross_hessian_blocked(prob, eta, counts, h, with_meat=with_meat)
    n_free = eta.shape[0]
    G = prob.n_groups
    Q = prob.nodes.shape[0]
    P = prob.y.shape[0]
    tabs = _oakes_tables(prob, eta, counts)
    n_pairs = tabs.n_cols * (tabs.n_cols + 1) // 2
    sq_shape = (2, n_free, n_free) if with_meat else (n_free, n_free)

    B = _oakes_chunk_size(tabs, Q, P)
    pad = (-P) % B
    if pad:
        y_pad = jnp.concatenate(
            [prob.y, jnp.zeros((pad, prob.y.shape[1]), dtype=prob.y.dtype)], axis=0
        )
        g_pad = jnp.concatenate([prob.group, jnp.zeros((pad,), dtype=prob.group.dtype)], axis=0)
        c_pad = jnp.concatenate([prob.counts, jnp.zeros((pad,), dtype=prob.counts.dtype)], axis=0)
        h_pad = jnp.concatenate([h, jnp.zeros((pad, Q), dtype=h.dtype)], axis=0)
    else:
        y_pad, g_pad, c_pad, h_pad = prob.y, prob.group, prob.counts, h

    n_chunks = y_pad.shape[0] // B
    y_chunks = y_pad.reshape(n_chunks, B, -1)
    g_chunks = g_pad.reshape(n_chunks, B)
    c_chunks = c_pad.reshape(n_chunks, B)
    h_chunks = h_pad.reshape(n_chunks, B, Q)

    h_cross = jnp.zeros((n_free, n_free), dtype=eta.dtype)
    meat = jnp.zeros((n_free, n_free), dtype=eta.dtype)
    d_flats = _oakes_flat_tables(tabs)
    for g in range(G):  # G is small here: many groups take the blocked path (§26)
        d_flats_g = [d[g] for d in d_flats]

        def scan_body(
            acc: tuple[Array, Array],
            xs: tuple[Array, Array, Array, Array],
            g_idx: int = g,
            tables_g: list[Array] = d_flats_g,
        ) -> tuple[tuple[Array, Array], None]:
            c_acc, t2_acc = acc
            y_c, g_c, c_c, h_c = xs
            mask_g = (g_c == g_idx) & (c_c > 0)

            def do_chunk(_unused: None) -> tuple[Array, Array]:
                cg = jnp.where(mask_g, c_c, 0.0)
                c_gq, t2_g = _oakes_chunk_terms(
                    tabs,
                    eta,
                    tables_g,
                    tabs.d_logw[g_idx],
                    tabs.lat_fids[g_idx],
                    y_c,
                    cg,
                    h_c,
                    g_idx,
                    with_meat,
                )
                return c_acc + c_gq, t2_acc + t2_g

            next_acc: tuple[Array, Array] = jax.lax.cond(
                jnp.any(mask_g), do_chunk, lambda _: acc, None
            )
            return next_acc, None

        init_acc = (
            jnp.zeros((Q, n_pairs), dtype=eta.dtype),
            jnp.zeros(sq_shape, dtype=eta.dtype),
        )
        (c_q, sq_g), _ = jax.lax.scan(scan_body, init_acc, (y_chunks, g_chunks, c_chunks, h_chunks))
        if with_meat:
            sq_g, meat_g = sq_g[0], sq_g[1]
            meat = meat + meat_g
        h_cross = h_cross + (_oakes_t1(tabs, eta, g, c_q) - sq_g)

    return h_cross, (meat if with_meat else None)


def _oakes_cross_hessian_blocked(
    prob: CompiledProblem, eta: Array, counts: ExpectedCounts, h: Array, *, with_meat: bool = False
) -> tuple[Array, Array | None]:
    r"""Oakes' cross term as one ``lax.scan`` over single-group blocks (DESIGN.md §26).

    Each block adds its co-occurrences to the running group's :math:`C_q` and its
    :math:`\bar s_p\bar s_p^\top` term to :math:`T_2`; the last block of a group adds
    :math:`T_1=\sum_q L_{gq}^\top C_q L_{gq}` and resets :math:`C`. Blocks larger than the
    chunk budget are processed in power-of-two sub-chunks. The traced program does not depend
    on the number of groups.
    """
    blk = prob.blocks
    assert blk is not None
    n_free = eta.shape[0]
    Q = prob.nodes.shape[0]
    tabs = _oakes_tables(prob, eta, counts)
    n_pairs = tabs.n_cols * (tabs.n_cols + 1) // 2
    sq_shape = (2, n_free, n_free) if with_meat else (n_free, n_free)
    y_pad, c_pad = _padded_rows(prob)
    h_pad = jnp.concatenate([h, jnp.zeros((1, Q), dtype=h.dtype)], axis=0)
    B = blk.rows.shape[1]
    sub = _oakes_chunk_size(tabs, Q, B)
    n_sub = B // sub
    zero_c = jnp.zeros((Q, n_pairs), dtype=eta.dtype)
    zero_t = jnp.zeros((n_free, n_free), dtype=eta.dtype)
    zero_sq = jnp.zeros(sq_shape, dtype=eta.dtype)
    d_flats = _oakes_flat_tables(tabs)

    def body(
        carry: tuple[Array, Array, Array], xs: tuple[Array, Array, Array]
    ) -> tuple[tuple[Array, Array, Array], None]:
        c_acc, t1_acc, t2_acc = carry
        rows, g, last = xs
        # The group's tables are gathered once per block, not per sub-chunk.
        d_flats_g = [d[g] for d in d_flats]
        dlw_g = tabs.d_logw[g]
        lfids_g = tabs.lat_fids[g]

        def sub_body(acc: tuple[Array, Array], sub_rows: Array) -> tuple[tuple[Array, Array], None]:
            c_c = c_pad[sub_rows]

            def do_chunk(state: tuple[Array, Array]) -> tuple[Array, Array]:
                c_gq, t2_g = _oakes_chunk_terms(
                    tabs,
                    eta,
                    d_flats_g,
                    dlw_g,
                    lfids_g,
                    y_pad[sub_rows],
                    c_c,
                    h_pad[sub_rows],
                    g,
                    with_meat,
                )
                return state[0] + c_gq, state[1] + t2_g

            # Sub-chunks of padding rows only (count 0) are skipped.
            out: tuple[Array, Array] = jax.lax.cond(
                jnp.any(c_c > 0), do_chunk, lambda state: state, acc
            )
            return out, None

        (c_acc, t2_acc), _ = jax.lax.scan(sub_body, (c_acc, t2_acc), rows.reshape(n_sub, sub))

        def flush(state: tuple[Array, Array]) -> tuple[Array, Array]:
            c_mat, t1 = state
            return zero_c, t1 + _oakes_t1(tabs, eta, g, c_mat)

        c_acc, t1_acc = jax.lax.cond(last, flush, lambda state: state, (c_acc, t1_acc))
        return (c_acc, t1_acc, t2_acc), None

    (_, t1, sq), _ = jax.lax.scan(body, (zero_c, zero_t, zero_sq), (blk.rows, blk.group, blk.last))
    if with_meat:
        return t1 - sq[0], sq[1]
    return t1 - sq, None


def information_oakes(prob: CompiledProblem, eta: Array) -> Array:
    r"""Observed (posterior) information by Oakes' identity (Oakes, 1999; DESIGN.md §4.8, §15.18).

    .. math::
        I(\hat\eta)=-\Big[\frac{\partial^2Q_{\mathrm{post}}(\eta\mid\eta')}{\partial\eta\,\partial\eta^\top}
        +\frac{\partial^2Q_{\mathrm{post}}(\eta\mid\eta')}{\partial\eta\,\partial\eta'^\top}\Big]_{\eta=\eta'=\hat\eta}

    where :math:`Q_{\mathrm{post}}(\eta\mid\eta')=Q(\eta\mid\eta')+\log\pi(\eta)`. Reuses a
    single E-step, exploits the item/latent/prior block structure of the first term, and evaluates
    the cross term via pattern-chunked category co-occurrences and quadrature-node contractions
    without materializing ``(P, Q, n_free)`` intermediates. Adding :math:`\log\pi(\eta)`
    contributes :math:`-\nabla^2_\eta\log\pi(\hat\eta)` and keeps the exact identity with
    :func:`information_hessian`.
    """
    if eta.shape[0] == 0:
        return jnp.zeros((0, 0), dtype=eta.dtype)
    if prob.reduction is not None:
        return _information_oakes_total(prob, eta)
    counts, _, h = _e_step_details(prob, eta)
    h_own = _oakes_own_hessian(prob, eta, counts)
    h_cross, _ = _oakes_cross_hessian(prob, eta, counts, h)
    info: Array = -(h_own + h_cross)
    return info


def information_oakes_and_cross_product(prob: CompiledProblem, eta: Array) -> tuple[Array, Array]:
    r""":func:`information_oakes` and the cross-product information from one pattern pass.

    The second output is
    :math:`B_{\mathrm{post}}=\sum_p c_p\,s_ps_p^\top-\nabla^2_\eta\log\pi(\hat\eta)`, the
    same as :func:`flexpyirt._robust_se.information_cross_product`: the pattern scores
    :math:`s_p=\sum_q h_{pq}\nabla_\eta\psi_{pq}` are the uncentered :math:`\bar s_p` of
    Oakes' cross term (DESIGN.md §15.20, §17.5). This is the bread and meat of the unweighted,
    unclustered ``se="sandwich"``. Models with a dimension reduction (bifactor, two-tier) take the
    separate passes.
    """
    if prob.reduction is not None or eta.shape[0] == 0:
        from flexpyirt._robust_se import information_cross_product

        return information_oakes(prob, eta), information_cross_product(prob, eta)
    counts, _, h = _e_step_details(prob, eta)
    h_own = _oakes_own_hessian(prob, eta, counts)
    h_cross, meat = _oakes_cross_hessian(prob, eta, counts, h, with_meat=True)
    assert meat is not None
    d2_prior = jax.vmap(
        lambda x, kind, p1, p2: jax.grad(jax.grad(lambda v: eval_log_prior(v, kind, p1, p2)))(x)
    )(eta, prob.priors.kind, prob.priors.param1, prob.priors.param2)
    info: Array = -(h_own + h_cross)
    cross: Array = meat - jnp.diag(d2_prior)
    return info, cross


def _information_oakes_total(prob: CompiledProblem, eta: Array) -> Array:
    r"""Oakes' identity as a total derivative through the E-step (DESIGN.md §17.1).

    With :math:`F(\eta)=\nabla_\eta Q_{\mathrm{post}}(\eta\mid\eta')|_{\eta'=\eta}` (the
    complete-data gradient with the E-step weights computed at the same point),

    .. math::
        \frac{\mathrm dF}{\mathrm d\eta}
        =\frac{\partial^2Q_{\mathrm{post}}}{\partial\eta\,\partial\eta^\top}
        +\frac{\partial^2Q_{\mathrm{post}}}{\partial\eta\,\partial\eta'^\top}
        =-I(\hat\eta)

    (Oakes, 1999), evaluated by chunked forward-mode autodiff. Used for bifactor models, whose
    posterior is factorized over block grids rather than one joint grid, so the structured
    cross term of :func:`_oakes_cross_hessian` does not apply.
    """

    def complete_grad(e: Array) -> Array:
        counts, _, _ = _e_step_details(prob, e)
        out: Array = jax.grad(
            lambda x: expected_complete_loglik(prob, x, counts) + log_prior(prob, x)
        )(e)
        return out

    jac = _chunked_jacobian(complete_grad, eta)
    info: Array = -0.5 * (jac + jac.T)
    return info


def _bifactor_eap(prob: CompiledProblem, eta: Array) -> tuple[Array, Array]:
    r"""EAP and posterior SD of every factor of a bifactor model (DESIGN.md §17.1).

    .. math::
        \hat\theta_{p0}=\sum_g\theta_g h_{pg},\qquad
        \hat\theta_{ps}=\sum_{g,t}\theta_t h^{(s)}_{pgt}
    """
    red = prob.reduction
    assert red is not None
    h, h_block, _ = _bifactor_posterior(prob, eta)
    x = red.nodes_1d
    spec = jnp.sum(h_block[:, : red.n_specific], axis=2)  # (P, S, n) specific marginals
    mean_p = h @ red.primary_nodes  # (P, n_primary); the primary grid for two-tier (§17.2)
    var_p = h @ (red.primary_nodes**2) - mean_p**2
    mean_s = spec @ x
    var_s = spec @ (x**2) - mean_s**2
    mean = jnp.concatenate([mean_p, mean_s], axis=1)
    var = jnp.concatenate([var_p, var_s], axis=1)
    return mean, jnp.sqrt(jnp.maximum(var, 0.0))


def eap(prob: CompiledProblem, eta: Array) -> tuple[Array, Array]:
    r"""EAP and posterior SD of each pattern, using its group's latent distribution as prior.

    .. math::
        \hat{\boldsymbol\theta}_p=\sum_q\boldsymbol\theta_q h_{pq},\qquad
        \mathrm{SD}_{pm}=\Big(\sum_q\theta_{qm}^2 h_{pq}-\hat\theta_{pm}^2\Big)^{1/2}

    Returns arrays of shape ``(P,)`` when ``D = 1`` and ``(P, D)`` when ``D > 1``. Bifactor
    models use the factorized posterior (:func:`_bifactor_eap`).
    """
    if prob.reduction is not None:
        return _bifactor_eap(prob, eta)
    h = jax.nn.softmax(_log_joint(prob, eta), axis=1)
    mean = h @ prob.nodes
    var = h @ (prob.nodes**2) - mean**2
    sd = jnp.sqrt(jnp.maximum(var, 0.0))
    if prob.nodes.ndim == 2 and prob.nodes.shape[1] == 1:
        return mean[:, 0], sd[:, 0]
    return mean, sd
