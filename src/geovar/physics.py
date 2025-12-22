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
