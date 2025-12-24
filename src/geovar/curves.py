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
        if abs(self.denom) < 1e-12:
            self.denom = 1e-12

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

    def param_derivs(self, rc):
        """
        Returns derivatives of q(rc) w.r.t [t0, k].
        Returns: np.array([dq/dt0, dq/dk])
        """
        dq = self.q_end - self.q_start
        s_val = self._raw_sigmoid(rc)
        
        ds_raw_val = s_val * (1.0 - s_val)
        
        ds_dt0 = ds_raw_val * self.k
        ds_dk  = ds_raw_val * (self.t0 - rc)
        
        s_min_val = self.s_min
        ds_min_raw = s_min_val * (1.0 - s_min_val)
        ds_min_dt0 = ds_min_raw * self.k
        ds_min_dk  = ds_min_raw * (self.t0 - (-1.0))
        
        s_max_val = self.s_max
        ds_max_raw = s_max_val * (1.0 - s_max_val)
        ds_max_dt0 = ds_max_raw * self.k
        ds_max_dk  = ds_max_raw * (self.t0 - 1.0)
        
        dD_dt0 = ds_max_dt0 - ds_min_dt0
        dD_dk  = ds_max_dk  - ds_min_dk
        
        numerator_t0 = self.denom * (ds_dt0 - ds_min_dt0) - (s_val - self.s_min) * dD_dt0
        grad_t0 = dq * numerator_t0 / (self.denom**2)
        
        numerator_dk = self.denom * (ds_dk - ds_min_dk) - (s_val - self.s_min) * dD_dk
        grad_dk = dq * numerator_dk / (self.denom**2)
        
        return np.array([grad_t0, grad_dk])

    def mixed_param_derivs(self, rc):
        """
        Returns derivatives of dq/dRC w.r.t [t0, k].
        """
        dq = self.q_end - self.q_start
        
        s_val = self._raw_sigmoid(rc)
        ds_val = self.k * s_val * (1.0 - s_val) 
        
        s_min_val = self.s_min
        ds_min_raw = s_min_val * (1.0 - s_min_val)
        ds_min_dt0 = ds_min_raw * self.k
        ds_min_dk  = ds_min_raw * (self.t0 + 1.0)
        
        s_max_val = self.s_max
        ds_max_raw = s_max_val * (1.0 - s_max_val)
        ds_max_dt0 = ds_max_raw * self.k
        ds_max_dk  = ds_max_raw * (self.t0 - 1.0)
        
        dD_dt0 = ds_max_dt0 - ds_min_dt0
        dD_dk  = ds_max_dk  - ds_min_dk
        
        d_Sprime_dt0 = (self.k**2) * s_val * (1.0 - s_val) * (1.0 - 2.0 * s_val)
        
        dS_dk = s_val * (1.0 - s_val) * (self.t0 - rc)
        d_Sprime_dk = s_val * (1.0 - s_val) + self.k * (1.0 - 2.0 * s_val) * dS_dk
        
        num_t0 = self.denom * d_Sprime_dt0 - ds_val * dD_dt0
        grad_t0 = dq * num_t0 / (self.denom**2)
        
        num_dk = self.denom * d_Sprime_dk - ds_val * dD_dk
        grad_dk = dq * num_dk / (self.denom**2)
        
        return np.array([grad_t0, grad_dk])


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
        if abs(self.denom) < 1e-12:
            self.denom = 1e-12

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

    def param_derivs(self, rc):
        """
        Returns derivatives of q(rc) w.r.t [t0, k].
        """
        dq = self.q_end - self.q_start
        u = self.k * (rc - self.t0)
        pre = 2.0 / np.sqrt(np.pi)
        exp_u2 = np.exp(-u**2)
        
        ds_dt0 = pre * exp_u2 * (-self.k)
        ds_dk  = pre * exp_u2 * (rc - self.t0)
        
        u_min = self.k * (-1.0 - self.t0)
        exp_min = np.exp(-u_min**2)
        ds_min_dt0 = pre * exp_min * (-self.k)
        ds_min_dk  = pre * exp_min * (-1.0 - self.t0)
        
        u_max = self.k * (1.0 - self.t0)
        exp_max = np.exp(-u_max**2)
        ds_max_dt0 = pre * exp_max * (-self.k)
        ds_max_dk  = pre * exp_max * (1.0 - self.t0)
        
        dD_dt0 = ds_max_dt0 - ds_min_dt0
        dD_dk  = ds_max_dk  - ds_min_dk
        
        s_val = self._raw_func(rc)
        
        num_t0 = self.denom * (ds_dt0 - ds_min_dt0) - (s_val - self.s_min) * dD_dt0
        grad_t0 = dq * num_t0 / (self.denom**2)
        
        num_dk = self.denom * (ds_dk - ds_min_dk) - (s_val - self.s_min) * dD_dk
        grad_dk = dq * num_dk / (self.denom**2)
        
        return np.array([grad_t0, grad_dk])

    def mixed_param_derivs(self, rc):
        """
        Returns derivatives of dq/dRC w.r.t [t0, k].
        """
        dq = self.q_end - self.q_start
        u = self.k * (rc - self.t0)
        pre = 2.0 / np.sqrt(np.pi)
        exp_u2 = np.exp(-u**2)
        
        s_prime = pre * self.k * exp_u2
        
        d_Sprime_dt0 = pre * (self.k**2) * (2.0 * u) * exp_u2
        d_Sprime_dk = pre * exp_u2 * (1.0 - 2.0 * u**2)
        
        u_min = self.k * (-1.0 - self.t0)
        exp_min = np.exp(-u_min**2)
        ds_min_dt0 = pre * exp_min * (-self.k)
        ds_min_dk  = pre * exp_min * (-1.0 - self.t0)
        
        u_max = self.k * (1.0 - self.t0)
        exp_max = np.exp(-u_max**2)
        ds_max_dt0 = pre * exp_max * (-self.k)
        ds_max_dk  = pre * exp_max * (1.0 - self.t0)
        
        dD_dt0 = ds_max_dt0 - ds_min_dt0
        dD_dk  = ds_max_dk  - ds_min_dk
        
        num_t0 = self.denom * d_Sprime_dt0 - s_prime * dD_dt0
        grad_t0 = dq * num_t0 / (self.denom**2)
        
        num_dk = self.denom * d_Sprime_dk - s_prime * dD_dk
        grad_dk = dq * num_dk / (self.denom**2)
        
        return np.array([grad_t0, grad_dk])


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

    def param_derivs(self, rc):
        # No dependency on t0, k
        return np.array([0.0, 0.0])

    def mixed_param_derivs(self, rc):
        # No dependency
        return np.array([0.0, 0.0])