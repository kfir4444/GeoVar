import numpy as np

# Simplified dictionary for common organic elements
# Covalent radii (Single bond) from Pyykko/Alvarez
# VdW radii from Bondi/Mantina
ATOM_DATA = {
    'H': {'cov': 0.32, 'vdw': 1.20},
    'C': {'cov': 0.75, 'vdw': 1.70},
    'N': {'cov': 0.71, 'vdw': 1.55},
    'O': {'cov': 0.63, 'vdw': 1.52},
    'F': {'cov': 0.64, 'vdw': 1.47},
    'P': {'cov': 1.11, 'vdw': 1.80},
    'S': {'cov': 1.03, 'vdw': 1.80},
    'Cl': {'cov': 0.99, 'vdw': 1.75},
    'Br': {'cov': 1.14, 'vdw': 1.85},
    'I': {'cov': 1.33, 'vdw': 1.98},
    # Fallbacks
    'X': {'cov': 0.70, 'vdw': 1.70} 
}

def get_cov_radius(symbol):
    return ATOM_DATA.get(symbol, ATOM_DATA['X'])['cov']

def get_vdw_radius(symbol):
    return ATOM_DATA.get(symbol, ATOM_DATA['X'])['vdw']

def calculate_bond_order(distance, atom1_symbol, atom2_symbol):
    """
    Pauling's Formula: BO = exp((r_eq - r) / 0.3)
    r_eq is estimated as sum of single-bond covalent radii.
    """
    r1 = get_cov_radius(atom1_symbol)
    r2 = get_cov_radius(atom2_symbol)
    r_eq = r1 + r2
    return np.exp((r_eq - distance) / 0.3)

def check_steric_clash(active_coords, active_symbols, spectator_coords, spectator_symbols):
    """
    Calculates soft-sphere repulsion penalty.
    Penalty if r < 0.8 * (vdw1 + vdw2).
    
    Args:
        active_coords: (N, 3) array
        active_symbols: list of N strings
        spectator_coords: (M, 3) array
        spectator_symbols: list of M strings
    
    Returns:
        total_penalty: float
    """
    total_penalty = 0.0
    
    if len(spectator_coords) == 0 or len(active_coords) == 0:
        return 0.0
    
    # Simple pairwise check
    # Optimization: This is O(N*M). For small molecules this is fine.
    for i, r_act in enumerate(active_coords):
        sym_act = active_symbols[i]
        vdw_act = get_vdw_radius(sym_act)
        
        for j, r_spec in enumerate(spectator_coords):
            sym_spec = spectator_symbols[j]
            vdw_spec = get_vdw_radius(sym_spec)
            
            # Distance
            dist = np.linalg.norm(r_act - r_spec)
            
            threshold = 0.8 * (vdw_act + vdw_spec)
            
            if dist < threshold:
                # Quadratic penalty for smoothness
                total_penalty += (threshold - dist)**2

    return total_penalty

def get_steric_gradient(active_coords, active_symbols, spectator_coords, spectator_symbols):
    """
    Returns gradient of steric penalty w.r.t active_coords.
    Shape: (N, 3)
    """
    grad = np.zeros_like(active_coords)
    
    if len(spectator_coords) == 0 or len(active_coords) == 0:
        return grad
        
    for i, r_act in enumerate(active_coords):
        sym_act = active_symbols[i]
        vdw_act = get_vdw_radius(sym_act)
        
        for j, r_spec in enumerate(spectator_coords):
            sym_spec = spectator_symbols[j]
            vdw_spec = get_vdw_radius(sym_spec)
            
            diff = r_act - r_spec
            dist = np.linalg.norm(diff)
            
            threshold = 0.8 * (vdw_act + vdw_spec)
            
            if dist < threshold and dist > 1e-6:
                # E = (thresh - dist)^2
                # dE/d_dist = -2 * (thresh - dist)
                # d_dist/d_r_act = (r_act - r_spec) / dist
                dE_dist = -2.0 * (threshold - dist)
                dist_grad = diff / dist
                grad[i] += dE_dist * dist_grad
                
    return grad

def get_valency_gradient(active_coords, active_indices, full_coords, atoms, target_valencies):
    """
    Returns gradient of valency penalty w.r.t active_coords.
    Shape: (N_active, 3)
    
    active_indices: list of global indices corresponding to active_coords rows.
    full_coords: (N_total, 3) array containing BOTH active and spectator coords.
                 (Assumes active_coords are already updated in full_coords)
    """
    grad = np.zeros_like(active_coords)
    
    # Valency Error E = Sum_i (V_i - V_target_i)^2
    # Only iterate over ACTIVE atoms i, as we only optimize their positions?
    # NO. The penalty considers deviation of valency for ACTIVE atoms.
    # But spectator atoms might also change valency if active atoms move close to them.
    # However, standard implementation usually constrains active atoms' valency.
    # Let's stick to active atoms' valency deviation for now.
    
    for i_local, i_global in enumerate(active_indices):
        # Calculate current valency V_i and its gradient w.r.t r_i
        # V_i = Sum_j BO_ij
        
        atom_i = atoms[i_global]
        r_i = active_coords[i_local]
        
        # We need V_i to compute the prefactor 2(V_i - V_target)
        # Re-calculating V_i here (inefficient but safe) or pass it?
        # Let's re-calculate to be self-contained.
        
        v_sum = 0.0
        # Gradients of V_i w.r.t r_i (force on i due to bonds)
        dv_dri = np.zeros(3)
        
        # Loop over ALL other atoms j to compute V_i
        for j_global in range(len(full_coords)):
            if i_global == j_global:
                continue
            
            atom_j = atoms[j_global]
            r_j = full_coords[j_global]
            diff = r_i - r_j
            dist = np.linalg.norm(diff)
            
            if dist < 4.0 and dist > 1e-6:
                bo = calculate_bond_order(dist, atom_i, atom_j)
                v_sum += bo
                
                # d(BO)/d(dist) = BO * (-1/0.3)
                dbo_ddist = bo * (-1.0 / 0.3)
                ddist_dri = diff / dist
                
                dv_dri += dbo_ddist * ddist_dri
        
        # Contribution to gradient from atom i's valency term
        # E_i = (V_i - V_tgt)^2
        # dE/dr_i = 2(V_i - V_tgt) * dV_i/dr_i
        
        diff_val = v_sum - target_valencies[i_local]
        grad[i_local] += 2.0 * diff_val * dv_dri
        
        # Note: If we included spectators in the sum of (V - V_tgt)^2, we'd need
        # to add contributions where 'j' is a spectator whose valency changes due to 'i' moving.
        # But typically we focus on the reacting center.
        
    return grad
