"""
Two-asset household model solved via Graves' two-step method
with continuous illiquid-asset choice (interpolated VFI).

Identical to two_step.py except that Step 2 (the outer maximisation over k')
allows k' to take any value in [k_min, k_max].  After evaluating V_tilde on
the discrete k-grid to locate the best interval, a golden-section refinement
with bilinear interpolation of V_NA (linear in k, PCHIP in b) pins down the
continuous optimum.

The EGM-based no-adjust sub-problem (Step 1) and the outer VFI loop are
unchanged.

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
    chi: float = 0.05       # adjustment cost scale


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
_PENALTY = -1e10


def u(c, sigma):
    """CRRA utility, safe for c <= 0."""
    safe = np.maximum(c, _EPS)
    if sigma == 1.0:
        val = np.log(safe)
    else:
        val = (safe ** (1.0 - sigma) - 1.0) / (1.0 - sigma)
    return np.where(c > 0, val, _PENALTY)


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
# Step 1: No-Adjust sub-problem via EGM  (unchanged)
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
# Step 2: Continuous maximisation over k'
# ============================================================

def _interp_across_k(kp, b_star, iz, spl_2d, k_grid):
    """
    Evaluate spl_2d[·][iz](b_star) at continuous kp via bilinear
    interpolation: linear between bracketing k-grid points, PCHIP in b.

    Parameters
    ----------
    kp : 1-d array – continuous k' values
    b_star : 1-d array – b values at which to query the splines
    iz : int – income-state index
    spl_2d : list[list[PchipInterpolator]] – spl_2d[ik][iz](b)
    k_grid : 1-d array

    Returns
    -------
    vals : 1-d array
    """
    n_k = len(k_grid)
    j = np.searchsorted(k_grid, kp, side='right') - 1
    j = np.clip(j, 0, n_k - 2)

    dk = k_grid[j + 1] - k_grid[j]
    w = np.clip((kp - k_grid[j]) / np.maximum(dk, 1e-15), 0.0, 1.0)

    N = len(kp)
    v_lo = np.full(N, _PENALTY)
    v_hi = np.full(N, _PENALTY)

    unique_j = np.unique(j)
    for jj in unique_j:
        mask = j == jj
        v_lo[mask] = _safe_eval(spl_2d[jj][iz], b_star[mask])
        v_hi[mask] = _safe_eval(spl_2d[jj + 1][iz], b_star[mask])

    return (1.0 - w) * v_lo + w * v_hi


def _eval_vtilde_batch(kp, k_vals, b_vals, iz, V_NA_spl, p, k_grid, z_vals):
    """
    Evaluate V_tilde(k, b, z; kp) for a batch of states at continuous kp.

    Uses the two-step equivalence V_tilde(k,b,z;k') = V_NA(k', b*, z) with
    b* = b + (R_k/R_b)(k - k') - g(k,k')/R_b, and bilinear interpolation
    of V_NA across (k, b).
    """
    b_star = (b_vals
              + (p.R_k / p.R_b) * (k_vals - kp)
              - adj_cost(k_vals, kp, p.chi) / p.R_b)

    vals = _interp_across_k(kp, b_star, iz, V_NA_spl, k_grid)

    # Feasibility: resources at (kp, b_star, z) must cover b_min
    resources = (p.R_k - 1.0) * kp + p.R_b * b_star + z_vals[iz]
    vals[resources <= p.b_min + _EPS] = _PENALTY

    return vals


def _golden_section_max(eval_fn, a, b, n_iter=40):
    """
    Vectorised golden-section search to maximise eval_fn over [a, b].

    Parameters
    ----------
    eval_fn : callable  (1-d array) -> (1-d array)
    a, b : 1-d arrays – lower and upper bounds
    n_iter : int – number of refinement iterations

    Returns
    -------
    x_opt, f_opt : 1-d arrays
    """
    GR = (np.sqrt(5) + 1.0) / 2.0

    c = b - (b - a) / GR
    d = a + (b - a) / GR
    fc = eval_fn(c)
    fd = eval_fn(d)

    for _ in range(n_iter):
        fc_old = fc.copy()
        fd_old = fd.copy()
        c_old = c.copy()
        d_old = d.copy()

        right = fd_old >= fc_old    # keep [c, b] when True

        # Narrow the interval
        a = np.where(right, c_old, a)
        b = np.where(right, b, d_old)

        # New interior points
        c = b - (b - a) / GR
        d = a + (b - a) / GR

        # Reuse one evaluation per iteration:
        #   right: d_new ≈ d_old (golden ratio), so fd reused; eval c_new
        #  ~right: c_new ≈ c_old (golden ratio), so fc reused; eval d_new
        need_pts = np.where(right, c, d)
        need_vals = eval_fn(need_pts)

        fc = np.where(right, need_vals, fc_old)
        fd = np.where(right, fd_old, need_vals)

    x_opt = (a + b) / 2.0
    return x_opt, eval_fn(x_opt)


def _maximise_over_kprime(V_NA_spl, bp_NA_spl, p, k_grid, b_grid, z_vals):
    """
    For each state (k, b, z), find the optimal continuous k' by:

      1. Evaluating V_tilde on the discrete k-grid (same as two_step.py).
      2. Refining via golden-section search in the bracket around the best
         grid point, with bilinear interpolation of V_NA.

    Returns
    -------
    V_new, c_new, k_pol, b_pol : arrays of shape (n_k, n_b, n_z)
        k_pol now contains continuous k' values (not restricted to k_grid).
    """
    n_k, n_b, n_z = len(k_grid), len(b_grid), len(z_vals)
    R_k, R_b, chi, b_min = p.R_k, p.R_b, p.chi, p.b_min

    V_new = np.full((n_k, n_b, n_z), -np.inf)
    k_pol = np.zeros((n_k, n_b, n_z))
    b_pol = np.zeros((n_k, n_b, n_z))

    # Pre-build flat index arrays for the vectorised golden section
    ik_flat = np.repeat(np.arange(n_k), n_b)       # (n_k * n_b,)
    ib_flat = np.tile(np.arange(n_b), n_k)          # (n_k * n_b,)
    k_state = k_grid[ik_flat]
    b_state = b_grid[ib_flat]

    for iz in range(n_z):
        z = z_vals[iz]

        # ── Phase 1: evaluate V_tilde on the discrete k' grid ────
        V_tilde_grid = np.full((n_k, n_b, n_k), -np.inf)

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

            V_tilde_grid[:, :, ikp] = vals

        # Best discrete index for each (k, b) state
        best_idx = np.argmax(V_tilde_grid, axis=2)     # (n_k, n_b)
        best_flat = best_idx.ravel()                    # (n_k * n_b,)

        # ── Phase 2: golden-section refinement ────────────────────
        # Search in [k_grid[best-1], k_grid[best+1]], clipped to grid range.
        lo_idx = np.clip(best_flat - 1, 0, n_k - 1)
        hi_idx = np.clip(best_flat + 1, 0, n_k - 1)
        gs_a = k_grid[lo_idx]
        gs_b = k_grid[hi_idx]

        def eval_fn(kp_vals):
            return _eval_vtilde_batch(
                kp_vals, k_state, b_state, iz,
                V_NA_spl, p, k_grid, z_vals
            )

        kp_opt, v_opt = _golden_section_max(eval_fn, gs_a, gs_b, n_iter=40)

        # Safety: keep the grid optimum if golden section did worse
        # (can happen at non-smooth boundaries or infeasible regions).
        v_grid_best = V_tilde_grid[ik_flat, ib_flat, best_flat]
        kp_grid_best = k_grid[best_flat]

        use_grid = v_grid_best > v_opt
        kp_opt = np.where(use_grid, kp_grid_best, kp_opt)
        v_opt = np.where(use_grid, v_grid_best, v_opt)

        V_new[:, :, iz] = v_opt.reshape(n_k, n_b)
        k_pol[:, :, iz] = kp_opt.reshape(n_k, n_b)

        # ── Recover b' policy at the continuous k'* ──────────────
        b_star_opt = (b_state
                      + (R_k / R_b) * (k_state - kp_opt)
                      - adj_cost(k_state, kp_opt, chi) / R_b)

        bp_opt = _interp_across_k(kp_opt, b_star_opt, iz, bp_NA_spl, k_grid)
        bp_opt = np.clip(bp_opt, b_min, b_grid[-1])
        b_pol[:, :, iz] = bp_opt.reshape(n_k, n_b)

    # ── Consumption from the full budget constraint ───────────────
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
    Solve the two-asset model using Graves' two-step method with
    continuous k' choice (interpolated VFI).

    Parameters
    ----------
    p : Params
    k_grid, b_grid : 1-d arrays (illiquid and liquid asset grids)
    z_vals : 1-d array (income states)
    Pi : 2-d array (Markov transition matrix for z)
    tol : float (convergence tolerance on sup|V_new - V|)
    max_iter : int
    damping : float in [0, 1). Fraction of old V to blend in each step.
        Helps convergence when the EGM-driven cycling between value and
        consumption policies causes oscillation.  Set to 0.3-0.5 for
        typical grids.
    verbose : bool

    Returns
    -------
    V : value function, shape (n_k, n_b, n_z)
    k_pol : illiquid asset policy (continuous levels), shape (n_k, n_b, n_z)
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

        # Optional damping
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

    print("Solving two-asset model (continuous k', interpolated VFI)...")
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
        plt.savefig("two_step_interp_policies.png", dpi=150)
        plt.show()
        print("  Policy plots saved to two_step_interp_policies.png")
    except ImportError:
        print("  (matplotlib not available — skipping plots)")

# %%
