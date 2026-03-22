# QuantMacro
A Julia repository of quantitative macroeconomic models and computational methods. The goal is clean, efficient, and modular implementations that are useful both as a reference and as a foundation for research extensions.

---

## Repository Structure

```
QuantMacro/
├── README.md
├── Project.toml
├── Manifest.toml
│
├── models/
│   ├── neoclassical/
|   |   ├── deterministic
|   |   └── stochastic
│   ├── incomplete_markets/
│   │   ├── huggett/
│   │   │   ├── grid_search
│   │   │   └── egm
│   │   └── aiyagari/
│   │       ├── grid_search
│   │       ├── interpolated_vfi
│   │       └── egm
│   ├── lifecycle/
│   │       ├── conesa_krueger
│   │       ├── denardi_bequests
│   ├── firm_dynamics/
|   |   ├── hopenhayn
|   |   └── ericson_pakes
|   └──agg_uncertainty/
│       ├── krusell_smith
│       ├── reiter
│       ├── boppart_krusell_mitman
│       └── sequence_space_jacobians
│
├── metrics/
|   ├── sim_method_moments
|   ├── indirect_inference
|   └── sim_max_likelihood
│
└── tools/
    └── spline
```



---

## Models

### Neoclassical Growth:

### Incomplete Markets: Huggett (1993) & Aiyagari (1994)

Located in `models/incomplete_markets/`. Both models feature households facing uninsurable idiosyncratic income risk and a borrowing constraint. The two models differ in their closure: Huggett uses a bond market in zero net supply, while Aiyagari embeds households in a general equilibrium with a capital firm who uses capital and labor inputs.

There are three main solution methods, in increasing sophistication:

| Method | Folder | Description |
|---|---|---|
| Grid search VFI | `grid_search/` | Brute-force maximization over a discretized action space |
| Interpolated VFI | `interpolated_vfi/` | Policy function iteration with continuous interpolation of the value function |
| Endogenous Grid Method | `egm/` | Carroll (2006) EGM; constructs the policy function analytically by inverting the Euler equation on an endogenous grid |

The EGM is substantially faster and more accurate than grid search for these models and should be the default for serious use. One should note that a non-grid solution method (continuous savings choice by the household) necessitates a different solution method for the distribution of agents. In this case, we use Young (2010) method of histograms.

### Lifecycle Models: 
Located in `models/lifecycle/`.
#### Conesa & Krueger (1999)

An OLG economy with a lifecycle earnings profile, social security, and endogenous labor supply. Households solve a finite-horizon consumption-savings problem and the model is used to study the aggregate and distributional consequences of social security reform. Solved by backward induction from the terminal age.

Key ingredients:
- Deterministic age-efficiency profile
- Stochastic idiosyncratic productivity (discretized Markov chain)
- Mandatory retirement and pay-as-you-go social security
- General equilibrium (capital and labor market clearing)

#### De Nardi (2004)

WIP. Lifecycle model with intergenerational links. Warm glow bequest motive with 

### Firm Dynamics: 
Located in `models/firm_dynamics/`.

#### Hopenhayn (1992)

A stationary equilibrium model of perfectly competitive firms with entry, exit, and size dynamics driven by idiosyncratic productivity shocks. Firms solve an optimal stopping problem; a free-entry condition pins down the equilibrium wage. The model generates a non-degenerate firm size distribution.

Key ingredients:
- Stationary distribution of incumbent firms over productivity
- Entry and exit with sunk entry costs and fixed operating costs
- Labor market clearing

#### Ericson & Pakes (1995)

Firm dynamics model similar to Hopenhayn, but exploring cases of market power. The version solved is more similar to the Hopenhayn code for purposes of comparison. We solve perfect competition, monopoly, and duopoly versions. 

### Aggregate Uncertainty:
Located in `models/agg_uncertainty/`.

#### Krusell & Smith (1998)

The canonical incomplete-markets model with aggregate risk. Households face both idiosyncratic and aggregate productivity shocks and cannot insure against either. The aggregate state is infinite-dimensional (the full wealth distribution), so the model is solved with the Krusell-Smith algorithm: households forecast aggregate capital using a low-dimensional polynomial rule, and the rule is updated iteratively until it is consistent with simulated equilibrium dynamics.

Key ingredients:
- Aggregate TFP shocks (2-state Markov chain)
- Idiosyncratic employment shocks correlated with the aggregate state
- Log-linear forecasting rule for the capital stock
- Simulation and regression loop to achieve approximate equilibrium

---
The next three methods all approximate the dynamics of a (potentially large-scale) heterogeneous-agent model around a steady state to incorporate aggregate uncertainty. They can be used with the Krusell-Smith model or other HA models when the size of shocks are small. For example, they break down if agents want to insure themselves against aggregate shocks.

#### Reiter (2009)

Reiter's method combines a global solution for the cross-sectional distribution in steady state with a local (perturbation) solution for aggregate dynamics. The distribution is approximated on a fine grid and the full nonlinear system is linearized using automatic differentiation, yielding a large but sparse state-space system that can be reduced via standard methods.

#### Boppart, Krusell & Mitman (2018)

The BKM method computes impulse responses to aggregate shocks by solving a deterministic transition path. It exploits the fact that, to first order, the response of a HA model to an aggregate shock can be computed as a deterministic transition, which avoids the need to solve for a full stochastic equilibrium.

#### Sequence Space Jacobians — Auclert et al. (2021)

The sequence space Jacobian (SSJ) method works directly with the model's equilibrium conditions expressed as mappings between infinite sequences of aggregate variables. Jacobians of these mappings are computed via automatic differentiation and convolution, and the linearized impulse responses are obtained by solving a linear system in sequence space. Supports efficient computation of first-order dynamics for large HA models and can be extended to handle multiple blocks in a modular way.

---

## Tools



### Splines — `tools/spline/`

Custom spline interpolation routines in a unified interface (`evaluate_spline`, `evaluate_spline_derivative`, `evaluate_spline_antiderivative`). Implements `LinearSpline`, `CubicSpline` (natural, clamped, or not-a-knot boundary conditions), `PchipSpline` (monotonicity-preserving; recommended default for VFI), and `BilinearSpline` (2D tensor-product grid). ForwardDiff-compatible variants (`safe_spline`, `safe_pchip`) are provided for use with automatic differentiation. Also includes a `make_grid` utility with a density parameter to concentrate grid points near the lower bound.

---

## Future Extensions:

**Models**
- **Two asset models** - Kaplan and Violante (2014). Sebastian Graves' two step method. 
- **Heterogeneous Agent New Keynesian models**
- **Continuous time methods** - Moll et. al (2022)

**Computational Methods**
- **Howard Policy Improvement** — acceleration of VFI via policy function iteration
- **Multigrid VFI** — coarse-to-fine grid strategies for faster convergence



---

## References

- Aiyagari, S. R. (1994). Uninsured idiosyncratic risk and aggregate saving. *Quarterly Journal of Economics*, 109(3), 659–684.
- Auclert, A., Bardóczy, B., Rognlie, M., & Straub, L. (2021). Using the sequence‐space Jacobian to solve and estimate heterogeneous‐agent models. *Econometrica*, 89(5), 2375–2408.
- Boppart, T., Krusell, P., & Mitman, K. (2018). Exploiting MIT shocks in heterogeneous-agent economies. *Journal of Economic Dynamics and Control*, 89, 68–92.
- Carroll, C. D. (2006). The method of endogenous gridpoints for solving dynamic stochastic optimization problems. *Economics Letters*, 91(3), 312–320.
- Conesa, J. C., & Krueger, D. (1999). Social security reform with heterogeneous agents. *Review of Economic Dynamics*, 2(4), 757–795.
- Hopenhayn, H., & Rogerson, R. (1993). Job turnover and policy evaluation: a general equilibrium analysis. *Journal of Political Economy*, 101(5), 915–938.
- Huggett, M. (1993). The risk-free rate in heterogeneous-agent incomplete-insurance economies. *Journal of Economic Dynamics and Control*, 17(5–6), 953–969.
- Krusell, P., & Smith, A. A. (1998). Income and wealth heterogeneity in the macroeconomy. *Journal of Political Economy*, 106(5), 867–896.
- Reiter, M. (2009). Solving heterogeneous-agent models by projection and perturbation. *Journal of Economic Dynamics and Control*, 33(3), 649–665.