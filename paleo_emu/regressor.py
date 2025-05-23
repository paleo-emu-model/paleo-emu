
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, Matern, RationalQuadratic, ExpSineSquared,
    ConstantKernel as C, WhiteKernel
)
from lightgbm import LGBMRegressor


# ===== 模块 3：构建regressor =====
def build_regressor(model_type="GPR", kernel_name="RBF_White", encoder="PCA"):
    """
    根据model_type选择回归器
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


        regressor = GaussianProcessRegressor(kernel=kernels[kernel_name], n_restarts_optimizer=10, random_state=42) 
        #regressor = GaussianProcessRegressor(kernel=kernels[kernel_name], n_restarts_optimizer=5, random_state=42)
        

    elif model_type == "LGBM":
        # regressor = LGBMRegressor(
        #     n_estimators=500,
        #     learning_rate=0.05,
        #     num_leaves=31,
        #     max_depth=-1,
        #     subsample=0.8,
        #     colsample_bytree=0.8,
        #     random_state=42,
        #     n_jobs=-1
        # )
        regressor = LGBMRegressor(
            n_estimators=200,         # 样本少，树数量不能太多
            learning_rate=0.05,       # 合理步长
            num_leaves=10,            # 非常小的叶子数，避免过拟合
            max_depth=3,              # 限制树深度
            subsample=0.7,            # 行采样
            colsample_bytree=0.8,     # 列采样
            reg_alpha=0.1,            # L1 正则
            reg_lambda=1.0,           # L2 正则
            random_state=42,
            n_jobs=-1
        )
    else:
        raise ValueError("[ERROR] model_type must be either 'GPR' or 'LGBM'.")

    return regressor