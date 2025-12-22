import numpy as np
from geovar.curves import NormalizedSigmoid, GaussianBump
from geovar.physics import calculate_bond_order, check_steric_clash

class PathObjective:
    def __init__(self, r_coords, p_coords, active_indices, spectator_indices, atoms, 
                 w_action=1.0, w_chem=10.0, w_steric=5.0):
        self.r_coords = r_coords
        self.p_coords = p_coords
        self.active_indices = sorted(list(active_indices))
        self.spectator_indices = sorted(list(spectator_indices))
        self.atoms = atoms
        
        # Weights
        self.w_action = w_action
        self.w_chem = w_chem
        self.w_steric = w_steric
        
        # Extract active subset for R and P
        self.active_r = r_coords[self.active_indices]
        self.active_p = p_coords[self.active_indices]
        
        # Spectators are static
        self.spectator_coords = r_coords[self.spectator_indices]
        self.spectator_atoms = [atoms[i] for i in self.spectator_indices]
        self.active_atom_symbols = [atoms[i] for i in self.active_indices]

        # Calculate Target Valencies from Reactant Geometry
        # We assume Reactant is a valid structure
        self.target_valencies = self._calculate_valencies(r_coords)

    def log_curve_selection(self):
        """Prints the selected curve type for each active coordinate."""
        print("Curve Selection per Active Coordinate:")
        n_active = len(self.active_indices)
        dims = ['x', 'y', 'z']
        
        for i in range(n_active):
            atom_idx = self.active_indices[i]
            atom_sym = self.atoms[atom_idx]
            print(f"  Atom {atom_idx} ({atom_sym}):")
            for dim in range(3):
                q_start = self.active_r[i, dim]
                q_end = self.active_p[i, dim]
                dist = abs(q_end - q_start)
                
                curve_type = "GaussianBump" if dist < 0.5 else "NormalizedSigmoid"
                print(f"    {dims[dim]}: dist={dist:.4f} -> {curve_type}")

    def _calculate_valencies(self, coords):
        # Calculate sum of Bond Orders for each ACTIVE atom
        # We consider bonds with ALL atoms (active + spectator)
        valencies = []
        for i_idx, i_global in enumerate(self.active_indices):
            v_sum = 0.0
            atom_i = self.atoms[i_global]
            r_i = coords[i_global]
            
            # Loop over all other atoms
            for j_global in range(len(coords)):
                if i_global == j_global:
                    continue
                
                atom_j = self.atoms[j_global]
                r_j = coords[j_global]
                dist = np.linalg.norm(r_i - r_j)
                
                # Cutoff for bond order calculation to save time (e.g. 2.5 A)
                # But BO decays exponentially, so it's safe-ish.
                # Let's apply a generous cutoff 3.0A
                if dist < 4.0:
                    bo = calculate_bond_order(dist, atom_i, atom_j)
                    v_sum += bo
            valencies.append(v_sum)
        return np.array(valencies)

    def decode_genes(self, genes):
        """
        genes: (N_active * 3 * 2) flat array
        returns list of curve objects (length 3 * N_active)
        """
        n_active = len(self.active_indices)
        params = genes.reshape((n_active, 3, 2))
        
        curve_objs = []
        for i in range(n_active):
            for dim in range(3):
                q_start = self.active_r[i, dim]
                q_end = self.active_p[i, dim]
                t0, k = params[i, dim]
                
                dist = abs(q_end - q_start)
                
                # Logic:
                # "close and active" -> Gaussian
                # "otherwise" -> Sigmoid
                # (Spectators are handled as Linear externally/implicitly)
                
                if dist < 0.5:
                    c = GaussianBump(q_start, q_end, t0, k)
                else:
                    c = NormalizedSigmoid(q_start, q_end, t0, k)
                    
                curve_objs.append(c)
        return curve_objs

    def variational_action(self, genes, grid=None):
        if grid is None:
            grid = np.linspace(-1, 1, 30) # 30 points integration
        
        curves = self.decode_genes(genes)
        d_rc = grid[1] - grid[0]
        
        # shape: (num_curves, num_grid_points)
        derivers = np.array([ [c.deriv(rc) for rc in grid] for c in curves ])
        
        # Sum squared derivatives (energy)
        # Sum over curves, then sum over grid
        integrand = np.sum(derivers**2, axis=0) 
        action = np.sum(integrand) * d_rc
        
        return action

    def get_active_geometry(self, genes, rc):
        curves = self.decode_genes(genes)
        active_flat = np.array([c.value(rc) for c in curves])
        return active_flat.reshape((len(self.active_indices), 3))

    def get_spectator_geometry(self, rc):
        """
        Returns spectator coordinates at a given RC using Linear interpolation.
        RC is in [-1, 1].
        """
        # Map RC [-1, 1] to t [0, 1]
        t = (rc + 1.0) * 0.5
        
        # Linear Interpolation: R + (P - R) * t
        # self.spectator_coords is R (from __init__)
        # We need P for spectators too.
        
        # In __init__, we extracted:
        # self.spectator_coords = r_coords[self.spectator_indices]
        # We should also have stored spectator_p if we want to interpolate.
        
        spectator_r = self.spectator_coords
        spectator_p = self.p_coords[self.spectator_indices]
        
        return spectator_r + (spectator_p - spectator_r) * t

    def heuristic_penalty(self, genes):
        # Geometry at RC=0
        active_coords_rc0 = self.get_active_geometry(genes, 0.0)
        spectator_coords_rc0 = self.get_spectator_geometry(0.0)
        
        # 1. Steric Penalty (Active vs Spectator)
        steric = check_steric_clash(active_coords_rc0, self.active_atom_symbols,
                                    spectator_coords_rc0, self.spectator_atoms)
        
        # 2. Valency Penalty
        full_coords = np.zeros_like(self.r_coords)
        full_coords[self.spectator_indices] = spectator_coords_rc0
        full_coords[self.active_indices] = active_coords_rc0
        
        current_valencies = self._calculate_valencies(full_coords)
        valency_error = np.sum((current_valencies - self.target_valencies)**2)
        
        return self.w_chem * valency_error + self.w_steric * steric

    def total_loss(self, genes):
        return self.variational_action(genes) + self.heuristic_penalty(genes)
