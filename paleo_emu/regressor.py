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

def build_regressor(model_type="GPR", kernel_name="RBF_White", encoder="PCA",fixed_hp=False):
    """
    adapt kernel selection based on encoder
    """
    if model_type == "GPR":
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
            """
            if emu_type == "modlowice":
                # hyperparameters for the modlowice emulator
                hp = pd.DataFrame({
                    'l.co2': [0.523323] * nkeep,
                    'l.esinw': [2.791735] * nkeep,
                    'l.ecosw': [1.310285] * nkeep,
                    'l.obl': [1.663824] * nkeep,
                    'l.icevol': [10.000000] * nkeep,
                    'nugget': [0.000000000224038] * nkeep
                })
            elif emu_type == "modhighice":
                # hyperparameters for the modhighice emulator
                hp = pd.DataFrame({
                    'l.co2': [1.003084] * nkeep,
                    'l.esinw': [6.907880] * nkeep,
                    'l.ecosw': [7.499054] * nkeep,
                    'l.obl': [5.460205] * nkeep,
                    'l.icevol': [0.290289] * nkeep,
                    'nugget': [0.050143] * nkeep
                })
                # compute the covariance matrix of X, also known as the kernel matrix
                R = cov_mat(lambda_, X, X)
                Rt = R + np.diag(np.full(n, lambda_['nugget'])) # add the nugget term to the diagonal of the kernel matrix
            """
            regressor = GaussianProcessRegressor(kernel=kernels[kernel_name], n_restarts_optimizer=10, random_state=42)
        else:
            regressor = GaussianProcessRegressor(kernel=kernels[kernel_name], n_restarts_optimizer=5, random_state=42)

    elif model_type == "LGBM":
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
        raise ValueError("[ERROR] model_type must be either 'GPR' or 'LGBM'.")

    return regressor