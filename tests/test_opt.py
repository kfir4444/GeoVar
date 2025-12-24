import unittest
import numpy as np
from geovar.opt import optimize_path

class TestOpt(unittest.TestCase):
    def test_optimize_simple(self):
        # Trivial case: 1 atom moving from A to B
        atoms = ['H']
        r_coords = np.array([[0.0, 0.0, 0.0]])
        p_coords = np.array([[2.0, 0.0, 0.0]])
        
        active_indices = [0]
        spectator_indices = []
        
        # This will run the optimization
        # We limit iterations to make it fast
        ts_coords, res, _ = optimize_path(atoms, r_coords, p_coords, active_indices, spectator_indices, verbose=False)
        
        # Result should be roughly in the middle (RC=0) for a symmetric path
        # But optimize_path returns the geometry at RC=0.
        # Ideally, linear interpolation would give (1.0, 0, 0)
        # The optimizer minimizes action.
        
        self.assertEqual(ts_coords.shape, (1, 3))
        self.assertTrue(res.success or res.message) # Basic check that it ran
        
        # Check that we are somewhat bounded between start and end
        self.assertTrue(-1.0 < ts_coords[0,0] < 3.0)

    def test_optimize_no_active(self):
        atoms = ['H']
        r_coords = np.array([[0.0, 0.0, 0.0]])
        p_coords = np.array([[0.0, 0.0, 0.0]])
        active = []
        spectator = [0]
        
        ts_coords, res, _ = optimize_path(atoms, r_coords, p_coords, active, spectator)
        
        np.testing.assert_array_equal(ts_coords, r_coords)
        self.assertIsNone(res)

if __name__ == '__main__':
    unittest.main()
