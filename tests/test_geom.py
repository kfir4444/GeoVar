import unittest
import numpy as np
import os
import tempfile
import sys

try:
    from rdkit import Chem
except ImportError:
    raise ImportError("RDKit is required for these tests. Please install RDKit to proceed.")

from geovar.geom import read_xyz, write_xyz, align_product_to_reactant, identify_active_indices

class TestGeom(unittest.TestCase):
    def setUp(self):
        self.test_atoms = ['C', 'O']
        self.test_coords = np.array([
            [0.0, 0.0, 0.0],
            [1.2, 0.0, 0.0]
        ])
        
    def test_io_xyz(self):
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.xyz') as tmp:
            write_xyz(tmp.name, self.test_atoms, self.test_coords, comment="Test")
            tmp_path = tmp.name
            
        try:
            atoms, coords = read_xyz(tmp_path)
            self.assertEqual(atoms, self.test_atoms)
            np.testing.assert_array_almost_equal(coords, self.test_coords)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_alignment_identity(self):
        r_coords = self.test_coords
        p_coords = self.test_coords.copy()
        spectators = {0, 1}
        aligned_p = align_product_to_reactant(r_coords, p_coords, spectators)
        np.testing.assert_array_almost_equal(aligned_p, r_coords)

    def test_alignment_translation(self):
        r_coords = self.test_coords
        p_coords = r_coords + 1.0
        spectators = {0, 1}
        aligned_p = align_product_to_reactant(r_coords, p_coords, spectators)
        np.testing.assert_array_almost_equal(aligned_p, r_coords)

    def test_identify_active_isomerization(self):
        """Test HCN -> HNC isomerization. All atoms should be active."""
        atoms = ['H', 'C', 'N']
        r_coords = np.array([
            [-1.06, 0.0, 0.0], # H
            [0.0, 0.0, 0.0],   # C
            [1.15, 0.0, 0.0]    # N
        ])
        p_coords = np.array([
            [2.16, 0.0, 0.0],  # H (now bonded to N)
            [0.0, 0.0, 0.0],   # C
            [1.17, 0.0, 0.0]   # N
        ])
        
        active, spectators = identify_active_indices(atoms, r_coords, p_coords)
        
        self.assertIn(0, active)
        self.assertIn(1, active)
        self.assertIn(2, active)
        self.assertEqual(len(spectators), 0)

    def test_identify_active_with_spectators(self):
        """Test a system with a clear spectator (Argon)."""
        atoms = ['H', 'C', 'N', 'Ar']
        r_coords = np.array([
            [-1.06, 0.0, 0.0], [0.0, 0.0, 0.0], [1.15, 0.0, 0.0], # HCN
            [10.0, 10.0, 10.0]                                   # Ar
        ])
        p_coords = np.array([
            [2.16, 0.0, 0.0], [0.0, 0.0, 0.0], [1.17, 0.0, 0.0],  # HNC
            [10.0, 10.0, 10.0]                                   # Ar
        ])
        
        active, spectators = identify_active_indices(atoms, r_coords, p_coords)
        
        # HCN atoms should be active
        self.assertIn(0, active)
        self.assertIn(1, active)
        self.assertIn(2, active)
        # Argon should be a spectator
        self.assertIn(3, spectators)

if __name__ == '__main__':
    unittest.main()