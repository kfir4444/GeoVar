import unittest
import numpy as np
import os
import tempfile

from geovar.geom import read_xyz, write_xyz, align_product_to_reactant

class TestGeom(unittest.TestCase):
    def setUp(self):
        self.test_atoms = ['C', 'O']
        self.test_coords = np.array([
            [0.0, 0.0, 0.0],
            [1.2, 0.0, 0.0]
        ])
        
    def test_io_xyz(self):
        # Create a temp file
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
        # Aligning identical structures should yield 0 shift
        r_coords = self.test_coords
        p_coords = self.test_coords.copy()
        
        # Spectators: all of them
        spectators = {0, 1}
        
        aligned_p = align_product_to_reactant(r_coords, p_coords, spectators)
        np.testing.assert_array_almost_equal(aligned_p, r_coords)

    def test_alignment_translation(self):
        r_coords = self.test_coords
        # Shift P by (1, 1, 1)
        p_coords = r_coords + 1.0
        
        spectators = {0, 1}
        aligned_p = align_product_to_reactant(r_coords, p_coords, spectators)
        
        # Should be shifted back
        np.testing.assert_array_almost_equal(aligned_p, r_coords)

    def test_alignment_rotation(self):
        # Rotate P by 90 degrees around Z
        # (1.2, 0, 0) -> (0, 1.2, 0)
        r_coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]
        ])
        
        # Rotated 90 deg around Z + Translated
        p_coords = np.array([
            [5.0, 5.0, 5.0],       # Center shifted
            [5.0, 6.0, 5.0]        # (0, 1, 0) relative to center
        ])
        
        spectators = {0, 1}
        aligned_p = align_product_to_reactant(r_coords, p_coords, spectators)
        
        # After alignment, it should match R (since structure is rigid/identical internally)
        np.testing.assert_array_almost_equal(aligned_p, r_coords, decimal=5)

    def test_alignment_subset(self):
        # Only align based on atom 0 (point) - this usually just translates
        r_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        p_coords = np.array([[10.0, 10.0, 10.0], [12.0, 10.0, 10.0]]) 
        
        spectators = {0}
        aligned_p = align_product_to_reactant(r_coords, p_coords, spectators)
        
        # Atom 0 should match R atom 0
        np.testing.assert_array_almost_equal(aligned_p[0], r_coords[0])
        
        # Atom 1 was distance 2.0 away in P, so it should still be 2.0 away from Atom 0
        dist = np.linalg.norm(aligned_p[1] - aligned_p[0])
        self.assertAlmostEqual(dist, 2.0)

if __name__ == '__main__':
    unittest.main()