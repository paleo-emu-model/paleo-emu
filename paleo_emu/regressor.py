
from pyparsing import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, Matern, RationalQuadratic,
    ConstantKernel as C, WhiteKernel
)
from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV
import yaml
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone

class RegressorGenerator:
    """Utility for building regressors from data.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input features for regression.
    Y : array-like, shape (n_samples, n_targets)
        Target values for regression.
    model_config : object
    """

    def __init__(self, X, Y, model_config):
        """Create a RegressorGenerator."""
        self.X = X
        self.Y = Y
        self.model_config = model_config
    
    def _generate_gpr_regressor(self):
        """Generate a Gaussian Process Regressor based on config."""
        # Extract GPR config
        gpr_config = self.model_config.regressor_config.get('GPR', {})
        kernel_name = gpr_config.get('kernel_name', 'RBF')
        nugget_value = gpr_config.get('nugget_value', 1.0)
        length_scales = gpr_config.get('length_scales', 1.0)
        noise_level = gpr_config.get('noise_level', 1.0)
        n_restarts_optimizer = gpr_config.get('n_restarts_optimizer', 5)
        alpha = gpr_config.get('alpha', 1e-6)
        nu = gpr_config.get('nu', 1.5)
        constant_value = gpr_config.get('constant_value', 1.0)
        constant_value_bounds = (constant_value * 0.1, constant_value * 10.0)

        # Create kernel
        if kernel_name == "RBF":
            kernel = C(constant_value, constant_value_bounds) * RBF(length_scale=length_scales)
        elif kernel_name == "RBF_White":
            kernel = C(constant_value, constant_value_bounds) * RBF(length_scale=length_scales) + WhiteKernel(noise_level=noise_level)
        elif kernel_name == "Matern":
            kernel = C(constant_value, constant_value_bounds) * Matern(length_scale=length_scales, nu=nu)
        elif kernel_name == "Matern_White":
            kernel = C(constant_value, constant_value_bounds) * Matern(length_scale=length_scales, nu=nu) + WhiteKernel(noise_level=noise_level)
        else:
            raise ValueError(f"[ERROR] Unsupported kernel name: {kernel_name}. Create kernel manually.")

        # Create GPR regressor
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=n_restarts_optimizer,
            alpha=nugget_value,
            normalize_y=True
        )
        return gpr