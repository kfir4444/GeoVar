# GeoVar: Variational Geodesic Initialization for Transition State Search

GeoVar is a computational chemistry tool designed to generate high-quality Transition State (TS) guesses without relying on expensive quantum mechanical (QM) Hessian calculations. It formulates the reaction path finding as a variational problem in the internal coordinate space, minimizing a Riemannian path length action subject to chemical constraints.

## Installation

GeoVar is managed by `uv`.

```bash
# 1. Clone the repository
git clone https://github.com/your-org/geovar.git
cd geovar

# 2. Sync dependencies
uv sync
```

## Usage

Execute the following command to run the tool:

```bash
uv run geovar reactant.xyz product.xyz ts_guess.xyz --verbose
```

To generate a visualization of the reaction path:

```bash
uv run geovar reactant.xyz product.xyz ts_guess.xyz --plot
```

## Mathematical Formulation

### 1. Reaction Path Parametrization

The reaction path $q(RC)$ connects the reactant configuration at $RC=-1$ to the product configuration at $RC=1$. The path is defined as a set of parametric curves for each active internal coordinate $q_i$.

For a coordinate $i$:

$$
q_i(RC) = q_{start} + (q_{end} - q_{start}) \cdot S(RC; t_{0,i}, k_i)
$$

where $S(RC)$ is a normalized switching function such that $S(-1)=0$ and $S(1)=1$.

#### Curve Types

1. **Normalized Sigmoid** (for $\Delta q_i \ge 0.5\,\mathrm{\AA}$):

    Models standard monotonic transitions (bond breaking/forming).

    ```math
    S_{\sigma}(RC) \propto \frac{1}{1 + e^{-k_i (RC - t_{0,i})}}
    ```

2. **Normalized Gaussian Bump** (for $\Delta q_i < 0.5\,\mathrm{\AA}$):

    Models subtle reorganization where coordinates may not change monotonically.

    ```math
    S_{G}(RC) \propto \operatorname{erf}\!\left(k_i (RC - t_{0,i})\right)
    ```


3.  **Linear Interpolation** (Spectator Atoms):
    Atoms identified as spectators (background environment) are interpolated linearly to maintain structural integrity relative to the active site.

### 2. Variational Action (Loss Function)

We define the "Action" $\mathcal{S}$ as the approximation of the Riemannian length of the path. Minimizing this action yields the geodesic path.

$$ 
\mathcal{L}_{action} = \int_{-1}^{1} \sum_{i \in \text{Active}} \left( \frac{dq_i}{dRC} \right)^2 dRC
$$ 

### 3. Chemical Constraints

To ensure the generated Transition State guess (at $RC=0$) is chemically plausible, penalty terms are added to the objective function:

$$ 
\mathcal{L}_{total} = w_{action}\mathcal{L}_{action} + w_{chem}\mathcal{L}_{valency} + w_{steric}\mathcal{L}_{steric}
$$ 

*   **Valency Conservation:** Enforces that the sum of bond orders at the TS matches the target valency of the atoms (derived from the Reactant).
*   **Steric Repulsion:** Imposes a soft-sphere repulsion penalty to prevent non-bonded atoms from overlapping.

### 4. Optimization

The problem is solved by optimizing the set of curve parameters {$\\mathbf{t}_0, \\mathbf{k}$} (2 parameters per active coordinate) using **L-BFGS-B** with random restarts to avoid local minima.

*   **Variables:** $t_{0,i} \in [-1.5, 1.5]$, $k_i \in [0.5, 15.0]$.
*   **Method:** Gradient-based optimization using analytical derivatives of the path action and penalties.

## Dependencies

*   `numpy` >= 2.4.0
*   `scipy` >= 1.16.3
*   `rdkit` >= 2025.9.3
*   `matplotlib` >= 3.10.8 (for plotting)
*   `tqdm` >= 4.67.1