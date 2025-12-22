import unittest
import numpy as np
from geovar.curves import NormalizedSigmoid, GaussianBump, Linear

class TestCurves(unittest.TestCase):
    def test_sigmoid_endpoints(self):
        # Test that start and end points are respected
        curve = NormalizedSigmoid(q_start=-5.0, q_end=5.0, t0=0.0, k=5.0)
        self.assertAlmostEqual(curve.value(-1.0), -5.0)
        self.assertAlmostEqual(curve.value(1.0), 5.0)

    def test_gaussian_endpoints(self):
        curve = GaussianBump(q_start=0.0, q_end=10.0, t0=0.2, k=2.0)
        self.assertAlmostEqual(curve.value(-1.0), 0.0)
        self.assertAlmostEqual(curve.value(1.0), 10.0)

    def test_linear_endpoints(self):
        curve = Linear(q_start=1.0, q_end=2.0)
        self.assertAlmostEqual(curve.value(-1.0), 1.0)
        self.assertAlmostEqual(curve.value(1.0), 2.0)
        self.assertAlmostEqual(curve.value(0.0), 1.5)

    def test_sigmoid_deriv(self):
        # Numerical derivative check
        curve = NormalizedSigmoid(q_start=0, q_end=1, t0=0, k=1)
        rc = 0.5
        h = 1e-5
        numeric_deriv = (curve.value(rc + h) - curve.value(rc - h)) / (2 * h)
        analytic_deriv = curve.deriv(rc)
        self.assertAlmostEqual(numeric_deriv, analytic_deriv, places=5)

    def test_gaussian_deriv(self):
        curve = GaussianBump(q_start=0, q_end=1, t0=0.1, k=3)
        rc = -0.2
        h = 1e-5
        numeric_deriv = (curve.value(rc + h) - curve.value(rc - h)) / (2 * h)
        analytic_deriv = curve.deriv(rc)
        self.assertAlmostEqual(numeric_deriv, analytic_deriv, places=5)

    def test_linear_deriv(self):
        curve = Linear(q_start=0, q_end=10)
        # Derivative should be constant 5 (delta q / delta rc = 10 / 2 = 5)
        self.assertEqual(curve.deriv(0.0), 5.0)
        self.assertEqual(curve.deriv(0.5), 5.0)
        self.assertEqual(curve.second_deriv(0.0), 0.0)

    def test_sigmoid_second_deriv(self):
        curve = NormalizedSigmoid(q_start=0, q_end=1, t0=0, k=2)
        rc = 0.1
        h = 1e-5
        numeric_d2 = (curve.deriv(rc + h) - curve.deriv(rc - h)) / (2 * h)
        analytic_d2 = curve.second_deriv(rc)
        self.assertAlmostEqual(numeric_d2, analytic_d2, places=4)

    def test_gaussian_second_deriv(self):
        curve = GaussianBump(q_start=0, q_end=1, t0=-0.3, k=4)
        rc = 0.2
        h = 1e-5
        numeric_d2 = (curve.deriv(rc + h) - curve.deriv(rc - h)) / (2 * h)
        analytic_d2 = curve.second_deriv(rc)
        self.assertAlmostEqual(numeric_d2, analytic_d2, places=4)

if __name__ == '__main__':
    unittest.main()
