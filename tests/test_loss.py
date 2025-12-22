import unittest
import numpy as np
from geovar.loss import PathObjective

class TestLoss(unittest.TestCase):
    def setUp(self):
        self.atoms = ['C', 'C']
        # Active: Both
        self.active_indices = [0, 1]
        self.spectator_indices = []
        
        self.r_coords = np.array([
            [0.0, 0.0, 0.0],
            [1.54, 0.0, 0.0]
        ])
        
        self.p_coords = np.array([
            [0.0, 0.0, 0.0],
            [1.34, 0.0, 0.0] # Shorter bond
        ])
        
        self.objective = PathObjective(self.r_coords, self.p_coords, 
                                       self.active_indices, self.spectator_indices, 
                                       self.atoms)

    def test_initialization(self):
        self.assertEqual(len(self.objective.active_r), 2)
        
    def test_variational_action(self):
        # 2 atoms * 3 coords * 2 params = 12 genes
        genes = np.zeros(12) 
        # t0=0, k=1.0 for all (interleaved)
        # Reshape to (2, 3, 2)
        # index 0: t0, index 1: k
        # so set odd indices to 1.0
        genes[1::2] = 1.0 
        
        action = self.objective.variational_action(genes)
        self.assertGreater(action, 0.0)

    def test_heuristic_penalty(self):
        genes = np.zeros(12)
        genes[1::2] = 1.0
        
        # Should be penalty due to valency change or geometry
        loss = self.objective.heuristic_penalty(genes)
        # It's hard to predict exact value, but it should run
        self.assertTrue(isinstance(loss, float))
