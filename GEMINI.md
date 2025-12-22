# **Project Specification: GeoVar**

**Title:** Variational Geodesic Initialization for Automated Transition State Search
**Stack:** Python 3.12+, `uv` (manager), `numpy`, `scipy`, `rdkit`, `networkx`.

## **1. High-Level Concept**

GeoVar is a computational chemistry tool that generates high-quality Transition State (TS) guesses without performing expensive quantum mechanical (QM) calculations.

* **Method:** It defines the reaction path as a set of parametric curves (Sigmoids/Gaussians) in Internal Coordinate space.
* **Optimization:** Instead of solving differential equations, it uses a Genetic Algorithm (Differential Evolution) to optimize the curve parameters (t_0, k).
* **Loss Function:** It minimizes the **Riemannian Path Length** (Variational Geodesic Action) subject to chemical constraints (Valency conservation, Steric avoidance).

## **2. Directory Structure**

The project uses a standard `src`-layout managed by `uv`.

```text
GeoVar/
├── pyproject.toml       # Deps: numpy, scipy, rdkit, geometric, networkx
├── README.md
├── src/
│   └── geovar/
│       ├── __init__.py
│       ├── curves.py    # Analytic curve math
│       ├── geom.py      # RDKit topology, Kabsch alignment, Active Set
│       ├── physics.py   # Chemical heuristics (Bond orders, Force constants)
│       ├── loss.py      # Variational integral & Penalty functions
│       ├── opt.py        # Optimization solver.
│       └── cli.py       # Entry point
└── tests/               # Unit tests for curves and integration

```

## **3. Module Specifications**

### **A. `src/geovar/curves.py` (Math Engine)**

* **Class `NormalizedSigmoid**`:
* **Input:** `q_start`, `q_end`, `t0` (shift \in [-0.9, 0.9]), `k` (steepness).
* **Logic:** A logistic function strictly normalized to pass through `q_start` at RC=-1 and `q_end` at RC=1.
* **Methods:**
* `value(rc)`: Returns interpolated coordinate.
* `deriv(rc)`: Returns analytic 1st derivative (dq/dRC).
* `second_deriv(rc)`: Returns analytic 2nd derivative (for smoothness checks).


* **Class `GaussianBump**`:
* **Input:** `q_start`, `q_end`, `t0` (shift \in [-0.9, 0.9]), `k` (steepness).
* **Logic:** A Gaussian function strictly normalized to pass through `q_start` at RC=-1 and `q_end` at RC=1.
* **Methods:**
* `value(rc)`: Returns interpolated coordinate.
* `deriv(rc)`: Returns analytic 1st derivative (dq/dRC).
* `second_deriv(rc)`: Returns analytic 2nd derivative (for smoothness checks).


* **Class `Linear**`:
* **Input:** `q_start`, `q_end`, `t0` (shift \in [-0.9, 0.9]), `k` (steepness), for the spectator coordinates.
* **Logic:** A linear function strictly normalized to pass through `q_start` at RC=-1 and `q_end` at RC=1.
* **Methods:**
* `value(rc)`: Returns interpolated coordinate.
* `deriv(rc)`: Returns analytic 1st derivative (dq/dRC).
* `second_deriv(rc)`: Returns analytic 2nd derivative (for smoothness checks).




### **B. `src/geovar/geom.py` (Topology & Alignment)**

* **Dependency:** `rdkit`, `scipy.spatial.transform`.
* **Function `identify_active_indices(r_xyz, p_xyz)**`:
* Uses RDKit `rdDetermineBonds` to infer bond orders for Reactant (R) and Product (P).
* **Logic:**
1. **Core:** Atoms where Bond Order changes (|BO_R - BO_P| > 0.1).
2. **Shell 1:** Immediate neighbors of Core atoms in both R and P.


* Returns `active_indices` (set) and `spectator_indices` (set).


* **Function `align_product_to_reactant(r_xyz, p_xyz, spectator_indices)**`:
* Performs **Kabsch Alignment** (RMSD minimization) using *only* the spectator atoms as anchors.
* Ensures the "background" is static while the reaction center moves.



### **C. `src/geovar/physics.py` (Chemical Rules)**

* **Function `calculate_bond_order(r_ij)**`:
* Implements Pauling’s Formula: BO = \exp((r_{eq} - r) / 0.3).


* **Function `check_steric_clash(active_xyz, spectator_xyz)**`:
* Calculates soft-sphere repulsion. Returns penalty if r < 0.8 \times (vdw_1 + vdw_2).



### **D. `src/geovar/loss.py` (The Objective Function)**

* **Function `variational_action(genes, grid)**`:
* **Input:** Genes (t_0, k for each active coordinate), Integration Grid (linspace -1 to 1).
* **Math:** Calculates \mathcal{L} = \int_{-1}^{1} \sum w_i (\frac{dq_i}{dRC})^2 dRC.
* Uses analytic derivatives from `curves.py`.


* **Function `heuristic_penalty(genes)**`:
* Constructs geometry at RC=0.
* Calculates `ValencyError` (Deviation from ideal bond order sums) + `StericPenalty`.


* **Total Loss:** `Action + (Weight_Chem * ValencyError) + (Weight_Steric * StericPenalty)`.

### **E. `src/geovar/opt.py` (The Optimizer)**

* **Dependency:** `scipy.optimize`.
* **Logic:**
* Encodes the path as a flat array of parameters (2 per active coordinate).
* Bounds: t_0 \in [-0.9, 0.9], k \in [0.5, 15.0].
* Optimizes to minimize `total_loss` defined in `loss.py`.



## **4. Workflow Summary**

1. **Input:** Reads Reactant and Product XYZ files.
2. **Analyze:** Identifies the "Active Set" of atoms (bonds breaking/forming + neighbors).
3. **Align:** Rotates Product to match Reactant's spectator atoms.
4. **Optimize:** Runs GA to find the optimal curve parameters (t_0, k) that minimize the variational action while conserving valency.
5. **Output:** Generates the Transition State guess geometry (RC=0) via constrained geometric reconstruction.
