import numpy as np
from scipy.optimize import minimize
from geovar.loss import PathObjective
from tqdm import tqdm

def optimize_path(atoms, r_coords, p_coords, active_indices, spectator_indices, verbose=False):
    """
    Runs L-BFGS-B optimization with analytical gradients to find the optimal path.
    Uses multiple random restarts to avoid local minima.
    
    Returns:
        ts_coords: (N, 3) array of the Transition State guess (RC=0)
        result: Scipy optimization result object (best one)
    """
    
    objective = PathObjective(r_coords, p_coords, active_indices, spectator_indices, atoms)
    
    if verbose:
        objective.log_curve_selection()
    
    n_active = len(active_indices)
    n_params = n_active * 6
    
    if n_active == 0:
        return r_coords, None, objective

    # Bounds: t0 in [-1.5, 1.5], k in [0.5, 15.0]
    bounds = []
    for _ in range(n_active * 3):
        bounds.append((-1.5, 1.5)) # t0
        bounds.append((0.5, 15.0)) # k
        
    if verbose:
        print(f"Starting Gradient-Based Optimization (L-BFGS-B) with {n_params} parameters...")

    best_result = None
    best_loss = float('inf')
    
    # Restarts strategy
    n_restarts = 25
    
    # 1. Neutral Start (t0=0, k=2.0)
    x0_neutral = np.zeros(n_params)
    # Reshape to (N, 3, 2)
    x0_reshaped = x0_neutral.reshape((n_active, 3, 2))
    x0_reshaped[:, :, 0] = 0.0 # t0
    x0_reshaped[:, :, 1] = 2.0 # k
    starts = [x0_neutral]
    
    # 2. Random Starts
    for _ in range(n_restarts - 1):
        x_rand = np.zeros(n_params)
        for i in range(n_params):
            low, high = bounds[i]
            x_rand[i] = np.random.uniform(low, high)
        starts.append(x_rand)
        
    pbar = tqdm(total=len(starts), disable=not verbose, desc="Restarts")
    
    for x0 in starts:
        try:
            res = minimize(
                objective.total_loss,
                x0,
                method='L-BFGS-B',
                jac=objective.gradient,
                bounds=bounds,
                options={'disp': False, 'ftol': 1e-9, 'maxiter': 5000}
            )
            
            if res.fun < best_loss:
                best_loss = res.fun
                best_result = res
                
        except Exception as e:
            if verbose:
                print(f"Optimization failed for a start: {e}")
        
        pbar.update(1)
        
    pbar.close()
    
    if best_result is None:
        raise RuntimeError("Optimization failed for all restarts.")
        
    if verbose:
        print(f"Optimization Success: {best_result.success}")
        print(f"Final Loss: {best_result.fun}")
    
    # Reconstruct TS Geometry at RC=0
    best_genes = best_result.x
    ts_active_coords = objective.get_active_geometry(best_genes, 0.0)
    ts_spectator_coords = objective.get_spectator_geometry(0.0)
    
    ts_full_coords = np.array(r_coords, copy=True) 
    # Ensure spectator alignment is respected
    ts_full_coords[spectator_indices] = ts_spectator_coords
    ts_full_coords[active_indices] = ts_active_coords
    
    return ts_full_coords, best_result, objective
