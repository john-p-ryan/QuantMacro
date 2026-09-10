"""
Two-asset household model solved via Graves' two-step method.

Combines the Endogenous Grid Method (EGM) for the liquid asset dimension
with a grid search over the illiquid asset, exploiting the equivalence:

    V_tilde(k, b, z; k') = V_NA(k', b*, z)

where b* = b + (R_k / R_b)(k - k') - g(k, k') / R_b.

This reduces computational cost from O(N_k^2 * N_b^2 * N_z) per Bellman
update (double grid search) to O(N_k^2 * N_b * N_z) with cheaper per-point
work via spline evaluation instead of inner-loop enumeration.

Reference: two_step.md in this directory.
"""
#%%
import numpy as np
from scipy.interpolate import PchipInterpolator
from dataclasses import dataclass
import time


# ============================================================
# Parameters and grid construction
# ============================================================
#%%
@dataclass
class Params:
    """Model parameters for the two-asset household problem."""
    beta: float = 0.96      # discount factor
    sigma: float = 2.0      # CRRA coefficient
    R_k: float = 1.04       # gross return on illiquid asset
    R_b: float = 1.01       # gross return on liquid asset
    b_min: float = 0.0      # borrowing constraint on liquid asset
    k_min: float = 0.0      # lower bound on illiquid asset
    chi: float = 0.1       # adjustment cost scale


def make_grids(p: Params, *, n_k=30, n_b=50, k_max=10.0, b_max=10.0):
    """Create asset grids and a 2-state Markov income process."""
    k_grid = np.linspace(p.k_min, k_max, n_k)
    b_grid = np.linspace(p.b_min, b_max, n_b)
    z_vals = np.array([0.25, 1.0])
    Pi = np.array([[0.8, 0.2],
                   [0.2, 0.8]])
    return k_grid, b_grid, z_vals, Pi


# ============================================================
# Utility functions (vectorized)
# ============================================================

_EPS = 1e-10


def u(c, sigma):
    """CRRA utility, safe for c <= 0."""
    safe = np.maximum(c, _EPS)
    if sigma == 1.0:
        val = np.log(safe)
    else:
        val = (safe ** (1.0 - sigma) - 1.0) / (1.0 - sigma)
    return np.where(c > 0, val, -1e10)


def u_prime(c, sigma):
    """Marginal utility."""
    return np.maximum(c, _EPS) ** (-sigma)


def u_prime_inv(mu, sigma):
    """Inverse marginal utility."""
    return np.maximum(mu, _EPS) ** (-1.0 / sigma)


def adj_cost(k, kp, chi):
    """Quadratic adjustment cost for the illiquid asset."""
    return chi * (k - kp) ** 2


def _safe_eval(spl, x):
    """Evaluate PCHIP with flat extrapolation (clamp to boundary values).

    Prevents the cubic polynomial from running away outside the knot range,
    which can destabilise the value-function iteration.
    """
    x_clamped = np.clip(x, spl.x[0], spl.x[-1])
    return spl(x_clamped)


# ============================================================
# Step 1: No-Adjust sub-problem via EGM
# ============================================================

def _egm_no_adjust(V, c_full, p, k_grid, b_grid, z_vals, Pi, b_floor):
    """
    Solve the no-adjust sub-problem for every (k, z) pair using EGM on
    the liquid asset b.

    The no-adjust problem (k' = k, g(k,k) = 0) is:
        V_NA(k, b, z) = max_{c, b'} u(c) + beta * E_z'[V(k, b', z')]
        s.t.  c + b' = (R_k - 1)*k + R_b*b + z,   b' >= b_min

    The Euler equation u'(c) = beta * R_b * E[u'(c')] is inverted via EGM:
    assume b' on the grid, solve for (c, b) endogenously, then interpolate
    back to the exogenous b grid.

    Returns
    -------
    V_NA_spl : list[list[PchipInterpolator]]
        V_NA_spl[ik][iz](b) gives V_NA at (k_grid[ik], b, z_vals[iz]).
    bp_NA_spl : list[list[PchipInterpolator]]
        bp_NA_spl[ik][iz](b) gives optimal b' in the no-adjust problem.
    """
    n_k, n_b, n_z = len(k_grid), len(b_grid), len(z_vals)
    sigma, beta, R_k, R_b, b_min = p.sigma, p.beta, p.R_k, p.R_b, p.b_min

    N_CON = 30  # extra grid points in the constrained region

    V_NA_spl = [[None] * n_z for _ in range(n_k)]
    bp_NA_spl = [[None] * n_z for _ in range(n_k)]

    for ik in range(n_k):
        k = k_grid[ik]
        income_k = (R_k - 1.0) * k   # illiquid interest income

        for iz in range(n_z):
            z = z_vals[iz]

            # ── EGM: treat b' as exogenous (on b_grid) ──────────────
            EMU = beta * R_b * (u_prime(c_full[ik], sigma) @ Pi[iz])
            c_endo = u_prime_inv(EMU, sigma)
            b_endo = (c_endo + b_grid - income_k - z) / R_b
            EV = beta * (V[ik] @ Pi[iz])
            V_NA_endo = u(c_endo, sigma) + EV

            # ── Constrained extension: b' = b_min for b below b_endo[0] ──
            # Only include feasible points (c > 0).
            b_feas_min = (b_min - income_k - z) / R_b + 1e-8
            b_con_lo = max(b_floor, b_feas_min)
            EV_bmin = beta * (Pi[iz] @ V[ik, 0])

            if b_con_lo < b_endo[0] - 1e-10:
                b_con = np.linspace(b_con_lo, b_endo[0] - 1e-10, N_CON)
                c_con = income_k + R_b * b_con + z - b_min
                V_NA_con = u(c_con, sigma) + EV_bmin
                bp_con = np.full(N_CON, b_min)
            else:
                b_con = np.empty(0)
                V_NA_con = np.empty(0)
                bp_con = np.empty(0)

            # ── Assemble knots and build PCHIP splines ───────────────
            b_all = np.concatenate([b_con, b_endo])
            V_all = np.concatenate([V_NA_con, V_NA_endo])
            bp_all = np.concatenate([bp_con, b_grid])

            order = np.argsort(b_all)
            b_all, V_all, bp_all = b_all[order], V_all[order], bp_all[order]
            keep = np.concatenate([[True], np.diff(b_all) > 1e-12])
            b_all, V_all, bp_all = b_all[keep], V_all[keep], bp_all[keep]

            V_NA_spl[ik][iz] = PchipInterpolator(b_all, V_all, extrapolate=False)
            bp_NA_spl[ik][iz] = PchipInterpolator(b_all, bp_all, extrapolate=False)

    return V_NA_spl, bp_NA_spl


# ============================================================
# Step 2: Maximize over k' using the two-step equivalence
# ============================================================

def _maximise_over_kprime(V_NA_spl, bp_NA_spl, p, k_grid, b_grid, z_vals):
    """
    For each state (k, b, z), find the optimal illiquid asset choice k' by
    evaluating V_tilde(k, b, z; k') = V_NA(k', b*, z) over the k-grid.

    Returns
    -------
    V_new, c_new, k_pol, b_pol : arrays of shape (n_k, n_b, n_z)
    """
    n_k, n_b, n_z = len(k_grid), len(b_grid), len(z_vals)
    R_k, R_b, chi, b_min = p.R_k, p.R_b, p.chi, p.b_min

    V_new = np.full((n_k, n_b, n_z), -np.inf)
    kp_idx = np.zeros((n_k, n_b, n_z), dtype=np.intp)

    for iz in range(n_z):
        z = z_vals[iz]
        for ikp in range(n_k):
            kp = k_grid[ikp]
            gcosts = adj_cost(k_grid, kp, chi)

            b_star = (b_grid[None, :]
                      + (R_k / R_b) * (k_grid[:, None] - kp)
                      - gcosts[:, None] / R_b)

            resources = (R_k - 1.0) * kp + R_b * b_star + z
            feasible = resources > b_min + _EPS

            vals = np.full((n_k, n_b), -np.inf)
            if np.any(feasible):
                vals[feasible] = _safe_eval(V_NA_spl[ikp][iz], b_star[feasible])

            better = vals > V_new[:, :, iz]
            V_new[:, :, iz] = np.where(better, vals, V_new[:, :, iz])
            kp_idx[:, :, iz] = np.where(better, ikp, kp_idx[:, :, iz])

    # ── Recover k', b', and c policies ───────────────────────────
    k_pol = k_grid[kp_idx]
    b_pol = np.empty_like(k_pol)

    for iz in range(n_z):
        for ikp in range(n_k):
            mask = kp_idx[:, :, iz] == ikp
            if not np.any(mask):
                continue
            kp = k_grid[ikp]
            iks, ibs = np.nonzero(mask)
            bstar = (b_grid[ibs]
                     + (R_k / R_b) * (k_grid[iks] - kp)
                     - adj_cost(k_grid[iks], kp, chi) / R_b)
            b_pol[iks, ibs, iz] = np.clip(
                _safe_eval(bp_NA_spl[ikp][iz], bstar),
                b_min, b_grid[-1]
            )

    # Consumption from the full budget constraint
    c_new = np.empty_like(V_new)
    for iz in range(n_z):
        k2 = k_grid[:, None]
        b2 = b_grid[None, :]
        kp2 = k_pol[:, :, iz]
        bp2 = b_pol[:, :, iz]
        c_new[:, :, iz] = (R_k * k2 + R_b * b2 + z_vals[iz]
                           - adj_cost(k2, kp2, chi) - kp2 - bp2)
    c_new = np.maximum(c_new, _EPS)

    return V_new, c_new, k_pol, b_pol


# ============================================================
# Main solver
# ============================================================

def solve(p: Params, k_grid, b_grid, z_vals, Pi,
          *, tol=1e-6, max_iter=500, damping=0.0, verbose=True):
    """
    Solve the two-asset model using Graves' two-step method.

    Parameters
    ----------
    p : Params
    k_grid, b_grid : 1-d arrays (illiquid and liquid asset grids)
    z_vals : 1-d array (income states)
    Pi : 2-d array (Markov transition matrix for z)
    tol : float (convergence tolerance on sup|V_new - V|)
    max_iter : int
    damping : float in [0, 1). Fraction of old V to blend in each step.
        Helps convergence when discrete k' choice causes oscillation.
        Set to 0.3-0.5 for fine k-grids.
    verbose : bool

    Returns
    -------
    V : value function, shape (n_k, n_b, n_z)
    k_pol : illiquid asset policy (levels), shape (n_k, n_b, n_z)
    b_pol : liquid asset policy (levels), shape (n_k, n_b, n_z)
    c_pol : consumption policy, shape (n_k, n_b, n_z)
    """
    n_k, n_b, n_z = len(k_grid), len(b_grid), len(z_vals)

    # Initial guess: consume all current-period income, save nothing
    c_full = np.empty((n_k, n_b, n_z))
    for iz in range(n_z):
        c_full[:, :, iz] = np.maximum(
            (p.R_k - 1.0) * k_grid[:, None] + p.R_b * b_grid[None, :] + z_vals[iz],
            _EPS
        )
    V = u(c_full, p.sigma) / (1.0 - p.beta)

    # Floor for the extended b domain used by V_NA splines.
    b_floor = (b_grid[0]
               + (p.R_k / p.R_b) * (k_grid[0] - k_grid[-1])
               - adj_cost(k_grid[-1], k_grid[0], p.chi) / p.R_b
               - 3.0)

    t0 = time.perf_counter()
    for it in range(max_iter):
        V_NA_spl, bp_NA_spl = _egm_no_adjust(
            V, c_full, p, k_grid, b_grid, z_vals, Pi, b_floor
        )
        V_new, c_new, k_pol, b_pol = _maximise_over_kprime(
            V_NA_spl, bp_NA_spl, p, k_grid, b_grid, z_vals
        )

        # Optional damping to smooth discrete-k' oscillation
        if damping > 0:
            V_new = (1.0 - damping) * V_new + damping * V

        err = np.max(np.abs(V_new - V))
        if verbose and it % 10 == 0:
            print(f"  iter {it:4d}   |ΔV| = {err:.6e}")

        if err < tol:
            elapsed = time.perf_counter() - t0
            if verbose:
                print(f"  Converged at iter {it} ({elapsed:.1f}s, err = {err:.2e})")
            return V_new, k_pol, b_pol, c_new

        V, c_full = V_new, c_new

    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"  Stopped at iter {max_iter} ({elapsed:.1f}s, err = {err:.2e})")
    return V, k_pol, b_pol, c_full


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    p = Params()
    k_grid, b_grid, z_vals, Pi = make_grids(
        p, n_k=250, n_b=90, k_max=25.0, b_max=9.0
    )

    print("Solving two-asset model via Graves' two-step method...")
    V, k_pol, b_pol, c_pol = solve(
        p, k_grid, b_grid, z_vals, Pi,
        tol=1e-6, max_iter=500, damping=0.4
    )
    print("Done.\n")

    print(f"  V range:     [{V.min():.4f}, {V.max():.4f}]")
    print(f"  c range:     [{c_pol.min():.6f}, {c_pol.max():.4f}]")
    print(f"  k' range:    [{k_pol.min():.4f}, {k_pol.max():.4f}]")
    print(f"  b' range:    [{b_pol.min():.4f}, {b_pol.max():.4f}]")

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        ib_slices = [0, len(b_grid) // 2, len(b_grid) - 1]
        labels = [f"b = {b_grid[ib]:.1f}" for ib in ib_slices]

        for ax, (title, pol) in zip(axes, [("k' policy", k_pol),
                                            ("b' policy", b_pol),
                                            ("Consumption", c_pol)]):
            for ib, lab in zip(ib_slices, labels):
                for iz in range(len(z_vals)):
                    ls = "-" if iz == 0 else "--"
                    zlab = f"z={z_vals[iz]}" if ib == ib_slices[0] else ""
                    ax.plot(k_grid, pol[:, ib, iz], ls,
                            label=f"{lab}, {zlab}" if zlab else lab)
            ax.set_xlabel("k")
            ax.set_title(title)
        axes[0].legend(fontsize=7, ncol=2)
        plt.tight_layout()
        plt.savefig("two_step_policies.png", dpi=150)
        plt.show()
        print("  Policy plots saved to two_step_policies.png")
    except ImportError:
        print("  (matplotlib not available — skipping plots)")

# %%
