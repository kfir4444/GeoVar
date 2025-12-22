import numpy as np
from scipy.special import erf

class NormalizedSigmoid:
    """
    A sigmoid curve strictly normalized to start at q_start (-1) and end at q_end (1).
    Params:
      t0: Inflection point (time shift) [-0.9, 0.9]
      k:  Steepness [0.5, 20.0]
    """
    def __init__(self, q_start, q_end, t0, k):
        self.q_start = q_start
        self.q_end = q_end
        self.t0 = t0
        self.k = k
        
        # Pre-calculate normalization constants to ensure q(-1)=start, q(1)=end
        self.s_min = self._raw_sigmoid(-1.0)
        self.s_max = self._raw_sigmoid(1.0)
        self.denom = self.s_max - self.s_min

    def _raw_sigmoid(self, rc):
        # Standard logistic function
        return 1.0 / (1.0 + np.exp(-self.k * (rc - self.t0)))

    def _raw_deriv(self, rc):
        # Derivative of raw sigmoid: k * S * (1-S)
        s = self._raw_sigmoid(rc)
        return self.k * s * (1.0 - s)

    def value(self, rc):
        # Normalized value
        s_raw = self._raw_sigmoid(rc)
        s_norm = (s_raw - self.s_min) / self.denom
        return self.q_start + (self.q_end - self.q_start) * s_norm

    def deriv(self, rc):
        # Analytical derivative dq/dRC
        d_raw = self._raw_deriv(rc)
        d_norm = d_raw / self.denom
        return (self.q_end - self.q_start) * d_norm
    
    def second_deriv(self, rc):
        # Analytical second derivative d²q/dRC²
        s = self._raw_sigmoid(rc)
        d_raw = self._raw_deriv(rc)
        d2_raw = self.k * d_raw * (1.0 - 2.0 * s)
        d2_norm = d2_raw / self.denom
        return (self.q_end - self.q_start) * d2_norm


class GaussianBump:
    """
    A Gaussian Error Function (CDF) curve strictly normalized to start at q_start (-1) and end at q_end (1).
    Params:
      t0: Inflection point (time shift) [-0.9, 0.9]
      k:  Steepness [0.5, 20.0]
    """
    def __init__(self, q_start, q_end, t0, k):
        self.q_start = q_start
        self.q_end = q_end
        self.t0 = t0
        self.k = k
        
        # Pre-calculate normalization constants
        self.s_min = self._raw_func(-1.0)
        self.s_max = self._raw_func(1.0)
        self.denom = self.s_max - self.s_min

    def _raw_func(self, rc):
        # Gaussian Error Function
        return erf(self.k * (rc - self.t0))

    def _raw_deriv(self, rc):
        # Derivative of erf(u) where u = k(rc - t0)
        # d/dx erf(x) = 2/sqrt(pi) * exp(-x^2)
        u = self.k * (rc - self.t0)
        return (2.0 * self.k / np.sqrt(np.pi)) * np.exp(-u**2)

    def value(self, rc):
        s_raw = self._raw_func(rc)
        s_norm = (s_raw - self.s_min) / self.denom
        return self.q_start + (self.q_end - self.q_start) * s_norm

    def deriv(self, rc):
        d_raw = self._raw_deriv(rc)
        d_norm = d_raw / self.denom
        return (self.q_end - self.q_start) * d_norm
    
    def second_deriv(self, rc):
        # d/dRC (d_raw)
        u = self.k * (rc - self.t0)
        d_raw = self._raw_deriv(rc)
        d2_raw = -2.0 * self.k * u * d_raw
        d2_norm = d2_raw / self.denom
        return (self.q_end - self.q_start) * d2_norm


class Linear:
    """
    A linear interpolation curve.
    Params:
      t0, k: Ignored (kept for compatibility)
    """
    def __init__(self, q_start, q_end, t0=0, k=1):
        self.q_start = q_start
        self.q_end = q_end
    
    def value(self, rc):
        # Map [-1, 1] to [0, 1]
        t = (rc + 1.0) * 0.5
        return self.q_start + (self.q_end - self.q_start) * t

    def deriv(self, rc):
        # d/dRC = 0.5 * (q_end - q_start)
        return (self.q_end - self.q_start) * 0.5
    
    def second_deriv(self, rc):
        # 0
        return 0.0 * rc
