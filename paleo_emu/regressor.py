"""
This module is used to build regressors for pipeline.
To be confirmed: does the encoder affect the choice of regressor?
"""

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, Matern, RationalQuadratic, ExpSineSquared,
    ConstantKernel as C, WhiteKernel
)
from lightgbm import LGBMRegressor

def build_regressor(regressor_type="GPR", kernel_name="RBF_White", encoder="PCA",fixed_hp=False):
    """
    adapt kernel selection based on encoder
    """
    if regressor_type == "GPR":
        if encoder == "PCA":
            kernels = {
                "RBF": C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)),
                "Matern_1.5": C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=1.5, length_scale_bounds=(1e-2, 1e3)),
                "Matern_0.5_White": C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=0.5, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1)),
                "RationalQuadratic": C(1.0, (1e-3, 1e3)) * RationalQuadratic(length_scale=1.0, alpha=1.0, length_scale_bounds=(1e-2, 1e3), alpha_bounds=(1e-2, 1e3)),
                "RBF_White": C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1)),
                "Matern_2.5_White": C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=2.5, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1)),
            }
        elif encoder == "VAE":
            kernels = {
                "RBF": C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e4)),
                "Matern_1.5": C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=1.5, length_scale_bounds=(1e-5, 1e3)),
                "Matern_0.5_White": C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=0.5, length_scale_bounds=(1e-5, 1e3)) +
                                    WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-8, 10)),
                "RationalQuadratic": C(1.0, (1e-3, 1e3)) *
                                    RationalQuadratic(length_scale=1.0, alpha=1.0,
                                                    length_scale_bounds=(1e-9, 1e3),
                                                    alpha_bounds=(1e-5, 1e7)),
                "RBF_White": C(1.0, (1e-3, 1e3)) *
                            RBF(length_scale=1.0, length_scale_bounds=(1e-6, 1e4)) +
                            WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-8, 10)),
                "Matern_2.5_White": C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=2.5, length_scale_bounds=(1e-5, 1e3)) +
                                    WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-8, 10))
            }
        if kernel_name not in kernels:
            raise ValueError(f"[ERROR] Kernel '{kernel_name}' not found. Available: {list(kernels.keys())}")

        if fixed_hp:
            # set fixed hyperparameters
                nkeep=15.0
                # if emu_type == "modlowice":
                # hp_values = [0.523323, 2.791735, 1.310285, 1.663824, 10.000000, 0.000000000224038]
                # if emu_type == "modhighice":
                hp_values = [1.003084, 6.907880, 7.499054, 5.460205, 0.290289, 0.050143]
                hp_values = [value * nkeep for value in hp_values]
                length_scales = hp_values[:-1]  # Extract all but the last value for length scales
                nugget_value = hp_values[-1]   # The last value is the nugget

                print(f"Length of length_scale: {len(length_scales)}")
                kernel = RBF(length_scale=length_scales)
                regressor = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=nugget_value,  # 使用 alpha 参数设置 nugget
                    optimizer=None,      # 关闭优化器
                    normalize_y=False,
                    copy_X_train=True
                )
        else:
            regressor = GaussianProcessRegressor(kernel=kernels[kernel_name], n_restarts_optimizer=5, random_state=42)

    elif regressor_type == "LGBM":
        regressor = LGBMRegressor(
            n_estimators=200,         # number of samples is small, so the number of trees cannot be too many
            learning_rate=0.05,       # reasonable step size
            num_leaves=10,            # very small number of leaves to avoid overfitting
            max_depth=3,              # limit tree depth
            subsample=0.7,            # row sampling
            colsample_bytree=0.8,     # column sampling
            reg_alpha=0.1,            # L1 regularization
            reg_lambda=1.0,           # L2 regularization
            random_state=42,
            n_jobs=-1
        )
    else:
        raise ValueError("[ERROR] regressor_type must be either 'GPR' or 'LGBM'.")

    return regressor