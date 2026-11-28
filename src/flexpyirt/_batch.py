"""Batched estimation of many models of one design in one compiled kernel (DESIGN.md §28).

Functional core for the workflows that fit many models on the same data structure: the
likelihood-ratio refits of DIF tests, multi-start estimation and (internally) resampled pattern
counts. The models are :class:`~flexpyirt._core.CompiledProblem` instances with one tree
structure; :func:`stack_problems` pads the per-model shapes that can differ (the free-parameter
vector ``eta`` and the local slots of each family) and keeps every leaf that is identical across
the models (responses, nodes, ...) shared, unstacked. :func:`run_batch` then runs
:func:`~flexpyirt._core.em` or :func:`~flexpyirt._core.squarem` over the stacked leaves and
the starting values in one compiled kernel: ``vmap``-ed on GPU / TPU, a sequential ``lax.map``
on CPU (DESIGN.md §28.4).

Per-model convergence freeze: ``jax.vmap`` of a ``lax.while_loop`` whose predicate depends on
the batched state runs the body while *any* model is unconverged and keeps every finished
model's carry unchanged (a ``select`` on its own predicate). A finished model therefore keeps its
parameters, iteration count and histories exactly as its own loop would have left them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from flexpyirt import _core
from flexpyirt._core import CompiledProblem, EMResult, FamilyBlock, PriorBlock
from flexpyirt._newton import _to_natural, _to_unconstrained, _transform_kind
from flexpyirt.priors import PRIOR_NONE

__all__ = [
    "BATCH_BYTES",
    "StackedProblems",
    "chunk_size",
    "perturbed_starts",
    "run_batch",
    "stack_problems",
]

BATCH_BYTES = 256 * 2**20
"""Working-set budget of one batched chunk (bytes); sets how many models run together."""


@dataclass(frozen=True)
class StackedProblems:
    """Models of one design, ready for :func:`run_batch`.

    Attributes
    ----------
    shared : tuple of Array
        Leaves identical in every model, in tree order.
    stacked : tuple of Array
        Leaves that differ, each with a leading model axis ``M``.
    batched : tuple of bool
        Per leaf of the problem tree, whether it is in ``stacked`` (``True``) or ``shared``.
    treedef : PyTreeDef
        Tree structure of the (harmonized) problems.
    eta0 : (M, n_max) ndarray
        Starting values, zero-padded to the longest ``eta``.
    n_free : (M,) ndarray of int
        Length of each model's own ``eta``.
    """

    shared: tuple[Array, ...]
    stacked: tuple[Array, ...]
    batched: tuple[bool, ...]
    treedef: Any
    eta0: np.ndarray
    n_free: np.ndarray

    @property
    def n_models(self) -> int:
        """Number of models ``M``."""
        return int(self.eta0.shape[0])


def _pad_last(values: Array, size: int, fill: float | int) -> Array:
    """Pad the last axis of ``values`` to ``size`` with ``fill``."""
    extra = size - values.shape[-1]
    if extra == 0:
        return values
    widths = [(0, 0)] * (values.ndim - 1) + [(0, extra)]
    return jnp.pad(values, widths, constant_values=fill)


def _harmonize_family(fam: FamilyBlock, n_slots: int, invariant: tuple[bool, ...]) -> FamilyBlock:
    """Pad ``fam``'s local slots to ``n_slots`` (unused slots) and set its pooled items."""
    return replace(
        fam,
        slot_free_index=_pad_last(fam.slot_free_index, n_slots, -1),
        slot_lower=_pad_last(fam.slot_lower, n_slots, -jnp.inf),
        slot_upper=_pad_last(fam.slot_upper, n_slots, jnp.inf),
        invariant=invariant,
    )


def _harmonize_priors(priors: PriorBlock, n_max: int) -> PriorBlock:
    """Pad the prior block to ``n_max`` free parameters with no prior."""
    return PriorBlock(
        kind=_pad_last(priors.kind, n_max, PRIOR_NONE),
        param1=_pad_last(priors.param1, n_max, 0.0),
        param2=_pad_last(priors.param2, n_max, 1.0),
    )


def _same_leaf(a: Array, b: Array) -> bool:
    """Whether two leaves are the same array (identity, or equal shape, dtype and values)."""
    if a is b:
        return True
    a_np, b_np = np.asarray(a), np.asarray(b)
    return (
        a_np.shape == b_np.shape
        and a_np.dtype == b_np.dtype
        and bool(np.array_equal(a_np, b_np, equal_nan=a_np.dtype.kind == "f"))
    )


def stack_problems(
    problems: Sequence[CompiledProblem], eta0s: Sequence[np.ndarray]
) -> StackedProblems | None:
    r"""Stack models of one design for :func:`run_batch`, or ``None`` if they cannot be stacked.

    The per-model shapes are made common without changing any model's objective:

    - ``eta`` is zero-padded to the longest :math:`n_{\max}`; the padding coordinates are read by
      no ``free_index``, so no step ever writes them and they add 0 to the parameter change and
      to the SQUAREM norms; the prior block is padded with "no prior".
    - Each family's local slots are padded to the largest slot count with unused slots
      (``slot_free_index = -1``, bounds :math:`\pm\infty`), which the masked Newton step keeps
      fixed (DESIGN.md §14.2).
    - An item is pooled over groups (``FamilyBlock.invariant``, §26.9) only if it is invariant in
      every model; the per-group path gives the same M-step up to summation order.

    Models whose tree structures still differ (for example different components of equality
    constraints across items) return ``None``; the caller fits them one by one.
    """
    if len(problems) != len(eta0s) or not problems:
        raise ValueError("stack_problems needs one starting vector per problem.")
    n_max = max(int(np.asarray(e).shape[0]) for e in eta0s)
    n_fam = len(problems[0].families)
    if any(len(p.families) != n_fam for p in problems):
        return None
    harmonized: list[CompiledProblem] = []
    families_per_model = [p.families for p in problems]
    new_families_per_model: list[list[FamilyBlock]] = [[] for _ in problems]
    for f in range(n_fam):
        fams = [fs[f] for fs in families_per_model]
        n_slots = max(int(fam.slot_free_index.shape[-1]) for fam in fams)
        flags = [fam.invariant for fam in fams]
        if any(len(fl) != len(flags[0]) for fl in flags) or not flags[0]:
            invariant: tuple[bool, ...] = ()
        else:
            invariant = tuple(bool(all(col)) for col in zip(*flags, strict=True))
        for m, fam in enumerate(fams):
            new_families_per_model[m].append(_harmonize_family(fam, n_slots, invariant))
    for m, prob in enumerate(problems):
        harmonized.append(
            replace(
                prob,
                families=tuple(new_families_per_model[m]),
                priors=_harmonize_priors(prob.priors, n_max),
            )
        )
    flat = [jax.tree_util.tree_flatten(p) for p in harmonized]
    treedef = flat[0][1]
    if any(td != treedef for _, td in flat):
        return None
    leaves_per_model = [lv for lv, _ in flat]
    shared: list[Array] = []
    stacked: list[Array] = []
    batched: list[bool] = []
    for leaves in zip(*leaves_per_model, strict=True):
        first = leaves[0]
        if all(_same_leaf(first, other) for other in leaves[1:]):
            shared.append(first)
            batched.append(False)
        else:
            if any(np.shape(other) != np.shape(first) for other in leaves[1:]):
                return None
            stacked.append(jnp.stack([jnp.asarray(v) for v in leaves]))
            batched.append(True)
    eta0 = np.zeros((len(problems), n_max))
    n_free = np.zeros(len(problems), dtype=np.int64)
    for m, e in enumerate(eta0s):
        e_np = np.asarray(e, dtype=float)
        eta0[m, : e_np.shape[0]] = e_np
        n_free[m] = e_np.shape[0]
    return StackedProblems(
        shared=tuple(shared),
        stacked=tuple(stacked),
        batched=tuple(batched),
        treedef=treedef,
        eta0=eta0,
        n_free=n_free,
    )


def chunk_size(problem: CompiledProblem, n_models: int, budget: int = BATCH_BYTES) -> int:
    """Number of models run together so a chunk's E-step working set stays near ``budget``.

    Per model: the log joint and posterior ``(P, Q)`` twice plus the expected item counts
    :math:`G\\sum_f J_fK_fQ` (flat E-step), or the blocked E-step's own budget (§26.10).
    The chunk is then balanced so every chunk has (nearly) the same size.
    """
    n_rows = int(problem.y.shape[0])
    n_nodes = int(problem.nodes.shape[0])
    jk = sum(int(f.columns.shape[0]) * f.kernel.n_categories for f in problem.families)
    counts = problem.n_groups * jk * n_nodes
    if problem.blocks is not None:
        per_model = _core._E_STEP_BATCH_BYTES + 8 * counts
    else:
        per_model = 8 * (2 * n_rows * n_nodes + counts)
    cap = max(1, min(n_models, budget // max(per_model, 1)))
    n_chunks = -(-n_models // cap)
    return -(-n_models // n_chunks)


def _batched_runner(
    shared: tuple[Array, ...],
    stacked: tuple[Array, ...],
    eta0: Array,
    tol: Array,
    *,
    treedef: Any,
    batched: tuple[bool, ...],
    accelerate: bool,
    max_iter: int,
    n_newton: int,
    unconstrained: bool,
    parallel: bool,
) -> EMResult:
    """:func:`_core.em` / :func:`_core.squarem` over the stacked leaves and starts.

    ``parallel=True``: ``vmap`` (all models step together; finished ones are frozen).
    ``parallel=False``: ``lax.map`` (one model after another, each to its own convergence,
    all in the same compiled kernel).
    """
    runner = _core.squarem if accelerate else _core.em

    def one(stacked_leaves: tuple[Array, ...], start: Array) -> EMResult:
        it_shared = iter(shared)
        it_stacked = iter(stacked_leaves)
        leaves = [next(it_stacked) if b else next(it_shared) for b in batched]
        prob = jax.tree_util.tree_unflatten(treedef, leaves)
        return runner(
            prob, start, tol=tol, max_iter=max_iter, n_newton=n_newton, unconstrained=unconstrained
        )

    if parallel:
        return jax.vmap(one)(stacked, eta0)
    return jax.lax.map(lambda xs: one(*xs), (stacked, eta0))


_run_chunk = jax.jit(
    _batched_runner,
    static_argnames=(
        "treedef",
        "batched",
        "accelerate",
        "max_iter",
        "n_newton",
        "unconstrained",
        "parallel",
    ),
)


def _take(stack: StackedProblems, idx: np.ndarray) -> tuple[tuple[Array, ...], Array]:
    """Stacked leaves and starting values of the models ``idx``."""
    sel = jnp.asarray(idx)
    return tuple(leaf[sel] for leaf in stack.stacked), jnp.asarray(stack.eta0[idx])


def run_batch(
    stack: StackedProblems,
    *,
    tol: float,
    max_iter: int,
    n_newton: int,
    accelerate: bool,
    unconstrained: bool = False,
    batch_size: int | None = None,
    problem: CompiledProblem | None = None,
    parallel: bool | None = None,
) -> list[EMResult]:
    """Run EM (or SQUAREM) for every stacked model; returns one NumPy :class:`EMResult` each.

    ``parallel`` (default: ``True`` on GPU / TPU, ``False`` on CPU) chooses between ``vmap``
    over the models of a chunk (they step together; DESIGN.md §28.4) and a sequential
    ``lax.map`` inside the same compiled kernel (each model stops at its own convergence, so no
    work is spent on frozen models; the whole set is one chunk). Parallel chunks have a common
    size (``batch_size``, or :func:`chunk_size` of ``problem``); the last chunk is padded by
    repeating its last model, so every chunk reuses one compiled kernel. Each returned ``eta``
    is cut back to its model's own length.
    """
    n_models = stack.n_models
    if parallel is None:
        parallel = jax.default_backend() != "cpu"
    if not parallel:
        batch_size = n_models
    elif batch_size is None:
        batch_size = n_models if problem is None else chunk_size(problem, n_models)
    batch_size = max(1, min(batch_size, n_models))
    results: list[EMResult] = []
    tol_arr = jnp.asarray(tol, dtype=jnp.float64)
    for start in range(0, n_models, batch_size):
        idx = np.arange(start, min(start + batch_size, n_models))
        n_real = idx.shape[0]
        if n_real < batch_size:
            idx = np.concatenate([idx, np.full(batch_size - n_real, idx[-1])])
        stacked, eta0 = _take(stack, idx)
        out = _run_chunk(
            stack.shared,
            stacked,
            eta0,
            tol_arr,
            treedef=stack.treedef,
            batched=stack.batched,
            accelerate=accelerate,
            max_iter=max_iter,
            n_newton=n_newton,
            unconstrained=unconstrained,
            parallel=parallel,
        )
        host = jax.tree_util.tree_map(np.asarray, out)
        for k in range(n_real):
            one = EMResult(*(np.asarray(leaf)[k] for leaf in host))
            n_m = int(stack.n_free[idx[k]])
            results.append(one._replace(eta=one.eta[:n_m]))
    return results


def perturbed_starts(
    problem: CompiledProblem, eta0: np.ndarray, *, n_starts: int, scale: float, key: Array
) -> np.ndarray:
    r"""Starting values for multi-start estimation, ``(n_starts, n_free)``.

    Row 0 is ``eta0`` itself. Row :math:`m\ge1` perturbs every free parameter on the scale on
    which its bounds vanish (DESIGN.md §15.47: identity if unbounded,
    :math:`\operatorname{logit}((x-l)/(u-l))` if both bounds are finite, :math:`\log(x-l)` or
    :math:`\log(u-x)` with one bound):

    .. math::
        z^{(m)} = z(\eta^{(0)}) + s\,\varepsilon^{(m)},\qquad
        \varepsilon^{(m)}\sim N(0, I),\qquad \eta^{(m)} = z^{-1}(z^{(m)}),

    so every start stays strictly inside the bounds. A coordinate whose transformed start is not
    finite (a start on a bound) is not perturbed.
    """
    e0 = jnp.asarray(eta0, dtype=jnp.float64)
    out = np.tile(np.asarray(eta0, dtype=float), (n_starts, 1))
    if n_starts <= 1 or e0.shape[0] == 0:
        return out
    lo, hi = _core._eta_bounds(problem, e0)
    kind = _transform_kind(jnp.ones(e0.shape, dtype=jnp.bool_), lo, hi)
    z0 = _to_unconstrained(e0, kind, lo, hi)
    noise = jax.random.normal(key, (n_starts - 1, e0.shape[0]), dtype=jnp.float64)
    z = z0[None, :] + scale * noise
    eta = jax.vmap(lambda v: _to_natural(v, kind, lo, hi))(z)
    ok = np.isfinite(np.asarray(z0))[None, :] & np.isfinite(np.asarray(eta))
    out[1:] = np.where(ok, np.asarray(eta), out[1:])
    return out
