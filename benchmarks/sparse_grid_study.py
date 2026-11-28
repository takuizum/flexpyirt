r"""Design study: Smolyak sparse grids vs other rules for D-dimensional IRT (DESIGN.md §12, item 6).

Not a benchmark of flexpyirt. Reproduces the tables of the study:

1. point counts and signed weights of Gauss–Hermite Smolyak grids (GQN) for D = 2..10;
2. per-pattern accuracy of the marginal log-likelihood
   :math:`\log\int L_p(\theta)\phi(\theta)d\theta` for simple-structure 2PL tests with
   independent factors (the exact value factorizes into 1-D integrals), for Smolyak GQN/KPN,
   product Gauss–Hermite, product equispaced, Halton QMC and adaptive Gauss–Hermite;
3. direct marginal ML fits of a correlated confirmatory 2PL (3-D and 5-D) with each rule, against
   a fine product Gauss–Hermite reference. Signed weights cannot go through the BA-EM E-step, so
   these fits maximize :math:`\sum_p c_p\log\sum_q w_q L_p(\mathbf L\mathbf z_q)` with L-BFGS and
   JAX gradients (scratch code; not flexpyirt's estimator).

KPN (nested Genz–Keister) rules come from R ``SparseGrid``:

    Rscript validation/r/sparse_grid_reference.R /tmp/kpn
    uv run python benchmarks/sparse_grid_study.py --kpn-dir /tmp/kpn

Runs in a few minutes (the 5-D reference fit takes about a minute). The mirt rows of the fit table
come from ``benchmarks/sparse_grid_mirt.R`` on the data written by ``--write-3d-data``.
"""

from __future__ import annotations

import argparse
import itertools
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit, logsumexp
from scipy.stats import norm, qmc

from benchmarks.sparse_grid import gauss_hermite_rule, smolyak_normal

jax.config.update("jax_enable_x64", True)

Rule = tuple[np.ndarray, np.ndarray]


def product_rule(n_points: int, n_factors: int, kind: str = "gauss_hermite") -> Rule:
    """Return a product rule for N(0, I): Gauss–Hermite, or equispaced on [-6, 6] with φ weights."""
    if kind == "gauss_hermite":
        x, w = gauss_hermite_rule(n_points)
    else:
        x = np.linspace(-6.0, 6.0, n_points)
        w = norm.pdf(x) / norm.pdf(x).sum()
    nodes = np.array(list(itertools.product(x, repeat=n_factors)))
    weights = np.prod(np.array(list(itertools.product(w, repeat=n_factors))), axis=1)
    return nodes, weights


def halton_rule(n_points: int, n_factors: int) -> Rule:
    """Return unscrambled Halton points mapped by Φ^{-1} (first point dropped), equal weights."""
    u = qmc.Halton(n_factors, scramble=False).random(n_points + 1)[1:]
    return norm.ppf(u), np.full(n_points, 1.0 / n_points)


def kpn_rule(kpn_dir: Path, n_factors: int, level: int) -> Rule:
    """Read an R SparseGrid KPN rule written by validation/r/sparse_grid_reference.R."""
    table = np.loadtxt(kpn_dir / f"kpn_{n_factors}_{level}.csv", delimiter=",", skiprows=1)
    return table[:, :n_factors], table[:, n_factors]


# --- 1. point counts -------------------------------------------------------------------------


def point_counts() -> None:
    """Print Smolyak GQN point counts, negative weights and the product rule of the same degree."""
    print("D L degree Q_smolyak Q_product neg_share sum|w| Tchakaloff_bound")
    for n_factors in range(2, 11):
        for level in (3, 4, 5):
            _, w = smolyak_normal(n_factors, level)
            degree = 2 * level - 1
            print(
                n_factors,
                level,
                degree,
                len(w),
                level**n_factors,
                f"{np.mean(w < 0):.2f}",
                f"{np.abs(w).sum():.0f}",
                math.comb(n_factors + degree, degree),
            )


# --- 2. per-pattern accuracy -----------------------------------------------------------------


def _log_lik_at(y: np.ndarray, a: np.ndarray, d: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """Return log L_p(θ_q) of 2PL patterns ``y (N, J)`` at ``nodes (Q, D)``, shape (N, Q)."""
    z = nodes @ a.T + d
    out: np.ndarray = y @ log_expit(z).T + (1 - y) @ log_expit(-z).T
    return out


def simple_structure_test(
    rng: np.random.Generator, n_factors: int, per_factor: int, n_persons: int = 500
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate independent-factor simple-structure 2PL data and the exact log marginal."""
    n_items = n_factors * per_factor
    a = np.zeros((n_items, n_factors))
    for m in range(n_factors):
        a[m * per_factor : (m + 1) * per_factor, m] = rng.uniform(1.0, 2.0, per_factor)
    d = rng.normal(0.0, 1.0, n_items)
    theta = rng.standard_normal((n_persons, n_factors))
    y = (rng.random((n_persons, n_items)) < expit(theta @ a.T + d)).astype(float)
    x, w = gauss_hermite_rule(201)
    exact = np.zeros(n_persons)
    for m in range(n_factors):
        cols = slice(m * per_factor, (m + 1) * per_factor)
        ll = _log_lik_at(y[:, cols], a[cols, m : m + 1], d[cols], x[:, None])
        exact += logsumexp(ll + np.log(w), axis=1)
    return y, a, d, exact


def adaptive_gauss_hermite(
    y: np.ndarray, a: np.ndarray, d: np.ndarray, n_points: int
) -> np.ndarray:
    """Return log marginals by adaptive Gauss–Hermite (Schilling & Bock, 2005).

    Nodes are centred at each pattern's posterior mode and scaled by the Cholesky factor of the
    inverse negative Hessian of the log posterior.
    """
    n_persons, n_factors = y.shape[0], a.shape[1]
    theta = np.zeros((n_persons, n_factors))
    eye = np.eye(n_factors)
    for _ in range(30):
        p = expit(theta @ a.T + d)
        grad = (y - p) @ a - theta
        hess = -np.einsum("nj,jk,jl->nkl", p * (1 - p), a, a) - eye
        theta = theta - np.linalg.solve(hess, grad[..., None])[..., 0]
    p = expit(theta @ a.T + d)
    hess = -np.einsum("nj,jk,jl->nkl", p * (1 - p), a, a) - eye
    chol = np.linalg.cholesky(np.linalg.inv(-hess))
    z, wz = product_rule(n_points, n_factors)
    out = np.empty(n_persons)
    for i in range(n_persons):
        nodes = theta[i] + z @ chol[i].T
        log_w = (
            np.log(wz)
            - 0.5 * (nodes**2).sum(axis=1)
            + 0.5 * (z**2).sum(axis=1)
            + np.log(np.diag(chol[i])).sum()
        )
        out[i] = logsumexp(_log_lik_at(y[i : i + 1], a, d, nodes)[0] + log_w)
    return out


def pattern_accuracy(kpn_dir: Path | None) -> None:
    """Print per-pattern log-marginal errors and the share of non-positive signed marginals."""
    rng = np.random.default_rng(1)
    print("D items/factor rule Q mean|err| max|err| share(L<=0)")
    for n_factors, per_factor in [(3, 5), (3, 10), (5, 5), (5, 10)]:
        y, a, d, exact = simple_structure_test(rng, n_factors, per_factor)
        levels = range(3, 8) if n_factors == 3 else range(3, 7)
        rules: list[tuple[str, Rule]] = [
            (f"GQN-Smolyak L={lv}", smolyak_normal(n_factors, lv)) for lv in levels
        ]
        if kpn_dir is not None:
            kpn_levels = range(3, 9) if n_factors == 3 else range(3, 8)
            rules += [
                (f"KPN-Smolyak L={lv}", kpn_rule(kpn_dir, n_factors, lv)) for lv in kpn_levels
            ]
        sizes = [5, 7, 9, 15] if n_factors == 3 else [3, 5, 7]
        rules += [(f"product GH n={n}", product_rule(n, n_factors)) for n in sizes]
        n_eq = 15 if n_factors == 3 else 7
        rules.append((f"product equispaced n={n_eq}", product_rule(n_eq, n_factors, "equispaced")))
        rules += [(f"Halton QMC Q={q}", halton_rule(q, n_factors)) for q in (1000, 5000)]
        for name, (nodes, weights) in rules:
            value, sign = logsumexp(
                _log_lik_at(y, a, d, nodes), b=weights[None, :], axis=1, return_sign=True
            )
            ok = sign > 0
            err = np.abs(value[ok] - exact[ok])
            print(
                n_factors,
                per_factor,
                name,
                len(weights),
                f"{err.mean():.1e}",
                f"{err.max():.1e}",
                f"{1 - ok.mean():.3f}",
            )
        for n in (2, 3):
            err = np.abs(adaptive_gauss_hermite(y, a, d, n) - exact)
            print(
                n_factors,
                per_factor,
                f"adaptive GH n={n}",
                n**n_factors,
                f"{err.mean():.1e}",
                f"{err.max():.1e}",
                "0.000",
            )


# --- 3. direct marginal ML fits --------------------------------------------------------------


def simulate_cfa(
    n_factors: int, per_factor: int, n_persons: int, rho: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a simple-structure 2PL with equicorrelated factors; return patterns, counts, map."""
    rng = np.random.default_rng(seed)
    n_items = n_factors * per_factor
    a = rng.uniform(1.0, 2.0, n_items)
    d = rng.normal(0.0, 1.0, n_items)
    factor = np.repeat(np.arange(n_factors), per_factor)
    sigma = np.full((n_factors, n_factors), rho)
    np.fill_diagonal(sigma, 1.0)
    theta = rng.standard_normal((n_persons, n_factors)) @ np.linalg.cholesky(sigma).T
    y = (rng.random((n_persons, n_items)) < expit(a * theta[:, factor] + d)).astype(float)
    patterns, counts = np.unique(y, axis=0, return_counts=True)
    return patterns, counts.astype(float), factor


def direct_fit(
    y: np.ndarray, counts: np.ndarray, factor: np.ndarray, n_factors: int, rule: Rule
) -> tuple[np.ndarray, float, float, bool]:
    """Maximize the signed-rule marginal likelihood; return (a, d, rho), logL, seconds, ok."""
    n_items = y.shape[1]
    lower = np.tril_indices(n_factors, -1)
    loading = jnp.asarray(np.eye(n_factors)[factor])
    yj, cj = jnp.asarray(y), jnp.asarray(counts)
    zj, wj = jnp.asarray(rule[0]), jnp.asarray(rule[1])

    def corr_factor(raw: jax.Array) -> jax.Array:
        rows = jnp.eye(n_factors).at[lower].set(raw)
        return rows / jnp.linalg.norm(rows, axis=1, keepdims=True)  # L with diag(L L') = 1

    def neg_loglik(x: jax.Array) -> jax.Array:
        a, d = x[:n_items], x[n_items : 2 * n_items]
        theta = zj @ corr_factor(x[2 * n_items :]).T  # nodes moved by the Cholesky factor
        lin = theta @ (loading * a[:, None]).T + d
        ll = yj @ jax.nn.log_sigmoid(lin).T + (1 - yj) @ jax.nn.log_sigmoid(-lin).T
        # NaN as soon as a pattern's signed sum is <= 0 (log undefined).
        return -jnp.sum(cj * jax.scipy.special.logsumexp(ll, b=wj[None, :], axis=1))

    value_and_grad = jax.jit(jax.value_and_grad(neg_loglik))
    n_pairs = n_factors * (n_factors - 1) // 2
    x0 = np.concatenate([np.ones(n_items), np.zeros(n_items), np.zeros(n_pairs)])
    value_and_grad(x0)
    start = time.perf_counter()
    res = minimize(
        lambda x: tuple(np.asarray(v) for v in value_and_grad(x)),
        x0,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 2000, "gtol": 1e-7},
    )
    seconds = time.perf_counter() - start
    lc = np.asarray(corr_factor(jnp.asarray(res.x[2 * n_items :])))
    est = np.concatenate([res.x[: 2 * n_items], (lc @ lc.T)[lower]])
    ok = bool(res.success) and bool(np.isfinite(res.fun))
    return est, -float(res.fun), seconds, ok


def fits(kpn_dir: Path | None) -> None:
    """Print direct-ML fit differences from a fine product Gauss–Hermite reference."""
    cases = [(3, 5, 0.5, 21, [5, 7, 15], range(3, 9)), (5, 5, 0.4, 7, [3, 5], range(3, 8))]
    for n_factors, per_factor, rho, ref_n, sizes, levels in cases:
        y, counts, factor = simulate_cfa(n_factors, per_factor, 2000, rho, seed=7)
        n_items = y.shape[1]
        ref, ll_ref, sec, _ = direct_fit(
            y, counts, factor, n_factors, product_rule(ref_n, n_factors)
        )
        print(
            f"--- D={n_factors} J={n_items} N=2000 rho={rho} patterns={len(counts)}; reference "
            f"product GH {ref_n}^{n_factors}={ref_n**n_factors} logL={ll_ref:.4f} ({sec:.1f}s)"
        )
        rules: list[tuple[str, Rule]] = [
            (f"product GH n={n}", product_rule(n, n_factors)) for n in sizes
        ]
        rules += [(f"GQN-Smolyak L={lv}", smolyak_normal(n_factors, lv)) for lv in levels]
        if kpn_dir is not None:
            rules += [(f"KPN-Smolyak L={lv}", kpn_rule(kpn_dir, n_factors, lv)) for lv in levels]
        rules += [(f"Halton QMC Q={q}", halton_rule(q, n_factors)) for q in (1000, 5000)]
        for name, rule in rules:
            est, ll, sec, ok = direct_fit(y, counts, factor, n_factors, rule)
            if not ok:
                print(f"{name} Q={len(rule[1])} FAILED: signed marginal <= 0 for some pattern")
                continue
            diff = np.abs(est - ref)
            print(
                f"{name} Q={len(rule[1])} max|da|={diff[:n_items].max():.1e} "
                f"max|dd|={diff[n_items : 2 * n_items].max():.1e} "
                f"max|drho|={diff[2 * n_items :].max():.1e} dlogL={ll - ll_ref:+.2e} "
                f"({sec:.1f}s)"
            )


def main() -> None:
    """Run the study."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kpn-dir", type=Path, default=None)
    parser.add_argument("--part", choices=["counts", "accuracy", "fits", "all"], default="all")
    parser.add_argument(
        "--write-3d-data",
        type=Path,
        default=None,
        help="write the 3-D fit data (one row per person) to this CSV for sparse_grid_mirt.R",
    )
    args = parser.parse_args()
    if args.write_3d_data is not None:
        y, counts, _ = simulate_cfa(3, 5, 2000, 0.5, seed=7)
        rows = np.repeat(y, counts.astype(int), axis=0)
        np.savetxt(args.write_3d_data, rows, fmt="%d", delimiter=",")
        return
    if args.part in ("counts", "all"):
        point_counts()
    if args.part in ("accuracy", "all"):
        pattern_accuracy(args.kpn_dir)
    if args.part in ("fits", "all"):
        fits(args.kpn_dir)


if __name__ == "__main__":
    main()
