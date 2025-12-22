import numpy as np
from scipy.optimize import differential_evolution
from geovar.loss import PathObjective

def optimize_path(atoms, r_coords, p_coords, active_indices, spectator_indices, verbose=False):
    """
    Runs the Differential Evolution optimization to find the optimal path parameters.
    Returns:
        ts_coords: (N, 3) array of the Transition State guess (RC=0)
        result: Scipy optimization result object
    """
    
    objective = PathObjective(r_coords, p_coords, active_indices, spectator_indices, atoms)
    
    n_active = len(active_indices)
    # 3 coords per atom, 2 params per coord => 6 params per atom
    n_params = n_active * 6
    
    if n_active == 0:
        # Trivial case: No movement?
        # Should not happen if identified correctly.
        return r_coords, None

    # Bounds: t0 in [-0.9, 0.9], k in [0.5, 15.0]
    bounds = []
    for _ in range(n_active * 3):
        bounds.append((-0.9, 0.9)) # t0
        bounds.append((0.5, 15.0)) # k
        
    if verbose:
        print(f"Starting optimization with {n_params} parameters (Active atoms: {n_active})...")

    # Run DE
    # reducing workers to 1 to avoid multiprocessing issues in some envs, 
    # but -1 (all cpus) is better for performance.
    result = differential_evolution(
        objective.total_loss,
        bounds,
        strategy='best1bin',
        maxiter=300, 
        popsize=25,
        tol=0.01,
        mutation=(0.5, 1),
        recombination=0.7,
        disp=verbose,
        workers=1 
    )
    
    # Reconstruct TS Geometry at RC=0
    best_genes = result.x
    ts_active_coords = objective.get_active_geometry(best_genes, 0.0)
    ts_spectator_coords = objective.get_spectator_geometry(0.0)
    
    ts_full_coords = np.array(r_coords, copy=True) 
    # Ensure spectator alignment is respected (though they shouldn't have moved)
    ts_full_coords[spectator_indices] = ts_spectator_coords
    ts_full_coords[active_indices] = ts_active_coords
    
    return ts_full_coords, result
