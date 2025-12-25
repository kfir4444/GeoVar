import unittest
from geovar.curves import NormalizedSigmoid, ErrorFunction, Linear

class TestCurves(unittest.TestCase):
    def test_sigmoid_endpoints(self):
        # Test that start and end points are respected
        curve = NormalizedSigmoid(q_start=-5.0, q_end=5.0, t0=0.0, k=5.0)
        self.assertAlmostEqual(curve.value(-1.0), -5.0)
        self.assertAlmostEqual(curve.value(1.0), 5.0)

    def test_error_function_endpoints(self):
        curve = ErrorFunction(q_start=0.0, q_end=10.0, t0=0.2, k=2.0)
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

    def test_error_function_deriv(self):
        curve = ErrorFunction(q_start=0, q_end=1, t0=0.1, k=3)
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

    def test_error_function_second_deriv(self):
        curve = ErrorFunction(q_start=0, q_end=1, t0=-0.3, k=4)
        rc = 0.2
        h = 1e-5
        numeric_d2 = (curve.deriv(rc + h) - curve.deriv(rc - h)) / (2 * h)
        analytic_d2 = curve.second_deriv(rc)
        self.assertAlmostEqual(numeric_d2, analytic_d2, places=4)

    def test_sigmoid_param_derivs(self):
        """Test derivatives of q(rc) w.r.t parameters t0 and k."""
        q_start, q_end = 0.0, 1.0
        t0, k = 0.1, 2.0
        rc = 0.3
        
        curve = NormalizedSigmoid(q_start, q_end, t0, k)
        analytic_grads = curve.param_derivs(rc)
        
        # Check d/dt0
        h = 1e-5
        c_plus = NormalizedSigmoid(q_start, q_end, t0 + h, k)
        c_minus = NormalizedSigmoid(q_start, q_end, t0 - h, k)
        num_dt0 = (c_plus.value(rc) - c_minus.value(rc)) / (2*h)
        self.assertAlmostEqual(num_dt0, analytic_grads[0], places=4)
        
        # Check d/dk
        c_plus = NormalizedSigmoid(q_start, q_end, t0, k + h)
        c_minus = NormalizedSigmoid(q_start, q_end, t0, k - h)
        num_dk = (c_plus.value(rc) - c_minus.value(rc)) / (2*h)
        self.assertAlmostEqual(num_dk, analytic_grads[1], places=4)

    def test_error_function_param_derivs(self):
        """Test derivatives of q(rc) w.r.t parameters t0 and k."""
        q_start, q_end = 0.0, 1.0
        t0, k = -0.2, 3.0
        rc = 0.0
        
        curve = ErrorFunction(q_start, q_end, t0, k)
        analytic_grads = curve.param_derivs(rc)
        
        # Check d/dt0
        h = 1e-5
        c_plus = ErrorFunction(q_start, q_end, t0 + h, k)
        c_minus = ErrorFunction(q_start, q_end, t0 - h, k)
        num_dt0 = (c_plus.value(rc) - c_minus.value(rc)) / (2*h)
        self.assertAlmostEqual(num_dt0, analytic_grads[0], places=4)
        
        # Check d/dk
        c_plus = ErrorFunction(q_start, q_end, t0, k + h)
        c_minus = ErrorFunction(q_start, q_end, t0, k - h)
        num_dk = (c_plus.value(rc) - c_minus.value(rc)) / (2*h)
        self.assertAlmostEqual(num_dk, analytic_grads[1], places=4)

    def test_sigmoid_mixed_param_derivs(self):
        """Test derivatives of dq/dRC w.r.t parameters t0 and k."""
        q_start, q_end = 0.0, 1.0
        t0, k = 0.0, 1.5
        rc = 0.5
        
        curve = NormalizedSigmoid(q_start, q_end, t0, k)
        analytic_grads = curve.mixed_param_derivs(rc)
        
        # Check d(dq/dRC)/dt0
        h = 1e-5
        c_plus = NormalizedSigmoid(q_start, q_end, t0 + h, k)
        c_minus = NormalizedSigmoid(q_start, q_end, t0 - h, k)
        num_dt0 = (c_plus.deriv(rc) - c_minus.deriv(rc)) / (2*h)
        self.assertAlmostEqual(num_dt0, analytic_grads[0], places=4)
        
        # Check d(dq/dRC)/dk
        c_plus = NormalizedSigmoid(q_start, q_end, t0, k + h)
        c_minus = NormalizedSigmoid(q_start, q_end, t0, k - h)
        num_dk = (c_plus.deriv(rc) - c_minus.deriv(rc)) / (2*h)
        self.assertAlmostEqual(num_dk, analytic_grads[1], places=4)

    def test_error_function_mixed_param_derivs(self):
        """Test derivatives of dq/dRC w.r.t parameters t0 and k."""
        q_start, q_end = 0.0, 1.0
        t0, k = 0.1, 2.5
        rc = -0.3
        
        curve = ErrorFunction(q_start, q_end, t0, k)
        analytic_grads = curve.mixed_param_derivs(rc)
        
        # Check d(dq/dRC)/dt0
        h = 1e-5
        c_plus = ErrorFunction(q_start, q_end, t0 + h, k)
        c_minus = ErrorFunction(q_start, q_end, t0 - h, k)
        num_dt0 = (c_plus.deriv(rc) - c_minus.deriv(rc)) / (2*h)
        self.assertAlmostEqual(num_dt0, analytic_grads[0], places=4)
        
        # Check d(dq/dRC)/dk
        c_plus = ErrorFunction(q_start, q_end, t0, k + h)
        c_minus = ErrorFunction(q_start, q_end, t0, k - h)
        num_dk = (c_plus.deriv(rc) - c_minus.deriv(rc)) / (2*h)
        self.assertAlmostEqual(num_dk, analytic_grads[1], places=4)
        
    def test_extreme_parameters(self):
        # High steepness
        curve = NormalizedSigmoid(0, 1, 0, 20.0)
        self.assertAlmostEqual(curve.value(0), 0.5, places=2)
        
        # Extreme t0 (near boundary)
        curve = ErrorFunction(0, 1, 0.9, 1.0)
        self.assertAlmostEqual(curve.value(-1), 0.0)
        self.assertAlmostEqual(curve.value(1), 1.0)

if __name__ == '__main__':
    unittest.main()