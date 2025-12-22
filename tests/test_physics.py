import unittest
import numpy as np
from geovar.physics import calculate_bond_order, check_steric_clash

class TestPhysics(unittest.TestCase):
    def test_bond_order(self):
        # C-C single bond is roughly 1.54 A.
        # r_eq = 0.75 + 0.75 = 1.50
        # BO = exp((1.50 - 1.54) / 0.3) = exp(-0.04/0.3) ~= exp(-0.133) ~= 0.87
        bo = calculate_bond_order(1.54, 'C', 'C')
        self.assertTrue(0.8 < bo < 1.0)
        
        # C=C double bond is roughly 1.34 A.
        # BO = exp((1.50 - 1.34) / 0.3) = exp(0.16/0.3) ~= exp(0.53) ~= 1.7
        bo_double = calculate_bond_order(1.34, 'C', 'C')
        self.assertTrue(1.5 < bo_double < 2.0)

    def test_steric_clash(self):
        # H-H vdw is 1.2. Threshold = 0.8 * 2.4 = 1.92
        
        active = np.array([[0.0, 0.0, 0.0]])
        active_sym = ['H']
        
        # Far away
        spectator = np.array([[10.0, 0.0, 0.0]])
        spectator_sym = ['H']
        
        p = check_steric_clash(active, active_sym, spectator, spectator_sym)
        self.assertEqual(p, 0.0)
        
        # Close (distance 1.0 < 1.92)
        spectator_close = np.array([[1.0, 0.0, 0.0]])
        p_close = check_steric_clash(active, active_sym, spectator_close, spectator_sym)
        self.assertGreater(p_close, 0.0)
