import unittest
import numpy as np
from geovar.physics import (
    calculate_bond_order, 
    check_steric_clash, 
    get_steric_gradient, 
    get_valency_gradient
)

class TestPhysics(unittest.TestCase):
    def test_bond_order(self):
        # C-C single bond is roughly 1.54 A.
        bo = calculate_bond_order(1.54, 'C', 'C')
        self.assertTrue(0.8 < bo < 1.0)
        
        # C=C double bond is roughly 1.34 A.
        bo_double = calculate_bond_order(1.34, 'C', 'C')
        self.assertTrue(1.5 < bo_double < 2.0)

    def test_steric_clash(self):
        active = np.array([[0.0, 0.0, 0.0]])
        active_sym = ['H']
        spectator = np.array([[1.0, 0.0, 0.0]])
        spectator_sym = ['H']
        
        p = check_steric_clash(active, active_sym, spectator, spectator_sym)
        self.assertGreater(p, 0.0)

    def test_steric_gradient(self):
        """Numerical verification of the steric gradient. Accuracy is paramount."""
        active = np.array([[0.0, 0.0, 0.0]])
        active_sym = ['H']
        spectator = np.array([[1.0, 0.0, 0.0]]) # Very close
        spectator_sym = ['H']
        
        # Analytical
        grad_analytic = get_steric_gradient(active, active_sym, spectator, spectator_sym)
        
        # Numerical
        h = 1e-5
        grad_numeric = np.zeros_like(grad_analytic)
        for i in range(3):
            active_plus = active.copy()
            active_plus[0, i] += h
            p_plus = check_steric_clash(active_plus, active_sym, spectator, spectator_sym)
            
            active_minus = active.copy()
            active_minus[0, i] -= h
            p_minus = check_steric_clash(active_minus, active_sym, spectator, spectator_sym)
            
            grad_numeric[0, i] = (p_plus - p_minus) / (2 * h)
            
        np.testing.assert_array_almost_equal(grad_analytic, grad_numeric, decimal=5)

    def test_valency_gradient(self):
        """Numerical verification of the valency gradient. Failure is not an option."""
        # Simple H2 system
        atoms = ['H', 'H']
        active_indices = [0] # Only atom 0 is active
        active_coords = np.array([[0.0, 0.0, 0.0]])
        full_coords = np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
        target_valencies = np.array([1.0])
        
        def calc_valency_penalty(coords_active):
            # Helper to calculate penalty for numerical grad
            tmp_full = full_coords.copy()
            tmp_full[0] = coords_active[0]
            
            # V = Sum BO
            v = 0.0
            dist = np.linalg.norm(tmp_full[0] - tmp_full[1])
            bo = calculate_bond_order(dist, 'H', 'H')
            v = bo
            
            return (v - target_valencies[0])**2

        # Analytical
        grad_analytic = get_valency_gradient(active_coords, active_indices, full_coords, atoms, target_valencies)
        
        # Numerical
        h = 1e-6
        grad_numeric = np.zeros_like(grad_analytic)
        for i in range(3):
            coords_plus = active_coords.copy()
            coords_plus[0, i] += h
            p_plus = calc_valency_penalty(coords_plus)
            
            coords_minus = active_coords.copy()
            coords_minus[0, i] -= h
            p_minus = calc_valency_penalty(coords_minus)
            
            grad_numeric[0, i] = (p_plus - p_minus) / (2 * h)
            
        np.testing.assert_array_almost_equal(grad_analytic, grad_numeric, decimal=5)

if __name__ == '__main__':
    unittest.main()
