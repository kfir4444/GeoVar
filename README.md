# GeoVar: Variational Geodesic Initialization for Transition State Search

GeoVar is a computational chemistry tool designed to generate high-quality Transition State (TS) guesses without relying on expensive quantum mechanical (QM) Hessian calculations. It formulates the reaction path finding as a variational problem in the internal coordinate space, minimizing a Riemannian path length action subject to chemical constraints.

## Mathematical Formulation

### 1. Reaction Path Parametrization

The reaction path $\mathbf{q}(RC)$ connects the reactant configuration $\mathbf{q}_R$ (at $RC=-1$) to the product configuration $\mathbf{q}_P$ (at $RC=1$). The path is defined as a set of parametric curves for each active internal coordinate $q_i$.

For a coordinate $i$ moving from $q_{start}$ to $q_{end}$:

$$
q_i(RC) = q_{start} + (q_{end} - q_{start}) \cdot S(RC; t_{0,i}, k_i)
$$

where $S(RC)$ is a normalized switching function such that $S(-1)=0$ and $S(1)=1$.

#### Curve Types

The type of curve depends on the displacement magnitude $\Delta q_i = |q_{end} - q_{start}|$:

1.  **Normalized Sigmoid** (for $\Delta q_i \ge 0.5 \text{ \AA}$):
    $$
    S_{\sigma}(RC) \propto \frac{1}{1 + e^{-k_i(RC - t_{0,i})}}
    $$
    This models a standard monotonic transition typical of bond breaking/forming.

2.  **Normalized Gaussian Bump** (for $\Delta q_i < 0.5 \text{ \AA}$):
    $$
    S_{G}(RC) \propto \text{erf}(k_i(RC - t_{0,i}))
    $$
    This models subtle reorganization where the coordinate might not change monotonically or stays relatively close to the initial value but needs flexibility.

3.  **Linear Interpolation** (for Spectator Atoms):
    $$
    \mathbf{q}_{spec}(RC) = \mathbf{q}_{spec, R} + \frac{RC + 1}{2} (\mathbf{q}_{spec, P} - \mathbf{q}_{spec, R})
    $$

### 2. Variational Action (The Loss Function)

We define the "Action" $\mathcal{S}$ as the approximation of the Riemannian length of the path in the configuration space. Minimizing this action yields the geodesic path (shortest path) between reactants and products on the potential energy surface approximation.

$$
\mathcal{L}_{action} = \int_{-1}^{1} \sum_{i \in \text{Active}} \left( \frac{dq_i}{dRC} \right)^2 dRC
$$

This integral is computed numerically using a grid of 30 points. The term $\left( \frac{dq_i}{dRC} \right)^2$ represents the kinetic energy-like contribution of the coordinate change. Minimizing this ensures the path is "smooth" and avoids unnecessary detours.

### 3. Chemical Constraints (Heuristic Penalty)

To ensure the generated Transition State guess (at $RC=0$) is chemically plausible, we introduce penalty terms:

$$
\mathcal{L}_{total} = w_{action}\mathcal{L}_{action} + w_{chem}\mathcal{L}_{valency} + w_{steric}\mathcal{L}_{steric}
$$

#### Valency Conservation
We enforce that the sum of bond orders (valency) at the TS ($RC=0$) matches the ideal valency of the atoms (calculated from the Reactant geometry). The bond order is estimated using Pauling's formula:

$$
BO_{ij} = \exp\left( \frac{r_{eq} - r_{ij}}{0.3} \right)
$$

$$
\mathcal{L}_{valency} = \sum_{a \in \text{Active}} \left( \sum_{b} BO_{ab}(RC=0) - V_{target, a} \right)^2
$$

#### Steric Repulsion
We impose a soft-sphere repulsion penalty to prevent atoms from passing through each other:

$$
\mathcal{L}_{steric} = \sum_{a,b} \max\left( 0, 0.8(R_{vdw,a} + R_{vdw,b}) - r_{ab} \right)^2
$$

### 4. Optimization

The problem is solved by optimizing the set of curve parameters $\{\mathbf{t}_0, \mathbf{k}\}$ (2 parameters per active coordinate) using **Differential Evolution**, a global optimization algorithm.

*   **Variables:** $t_{0,i} \in [-0.9, 0.9]$, $k_i \in [0.5, 15.0]$.
*   **Optimizer:** Scipy's `differential_evolution`.
*   **Strategy:** `best1bin` with population size 25.

The global optimization is followed by a local polishing step using **L-BFGS-B** to refine the minimum.

> **Note on Convergence:** It is common for the global optimizer to reach the maximum iteration limit (`maxiter=300`) without satisfying the strict convergence tolerance (`tol=0.01`). In most cases, the resulting geometry (loss ~50-60) is a sufficiently good starting point for subsequent QM Transition State optimizations.

## Usage

```bash
uv run python src/geovar/cli.py reactant.xyz product.xyz ts_guess.xyz --verbose
```

## Dependencies
*   `numpy`, `scipy`: Numerical engine and optimization.
*   `rdkit`: Chemical topology and bond perception.
*   `tqdm`: Progress tracking.
