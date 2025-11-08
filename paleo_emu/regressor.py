"""
This module is used to build regressors for pipeline.
To be confirmed: does the encoder affect the choice of regressor?
"""

from pyparsing import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, Matern, RationalQuadratic,
    ConstantKernel as C, WhiteKernel
)
from lightgbm import LGBMRegressor
import yaml

def build_regressor(cfg_path, regressor_type="GPR", encoder="PCA", fixed_regressor_hp=True, verbose=True):
        # accept either a dict (already parsed) or a path to a yaml file
    if isinstance(cfg_path, dict):
        cfg = cfg_path
    else:
        cfg_file = Path(cfg_path)
        if not cfg_file.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        with open(cfg_file, "r") as fh:
            cfg = yaml.safe_load(fh)
            
    if regressor_type == "GPR":
        if fixed_regressor_hp:
            # set fixed hyperparameters
            # Load hyperparameters from emulator.yaml
            print("[INFO] Using fixed hyperparameters range for GPR from YAML configuration.")
            print("[INFO] Dont fix the hyperparameters, because different PCs may need different hyperparameters.")
            kernel_name = cfg['GPR_config'][encoder]['kernel']
            nugget_value = cfg['GPR_config'][encoder]['nugget_value']
            length_scales = cfg['GPR_config'][encoder]['length_scales']
            noise_level = cfg['GPR_config'][encoder].get('noise_level', 1.0)
            n_restarts_optimizer = cfg['GPR_config'][encoder].get('n_restarts_optimizer', 5)
            alpha = cfg['GPR_config'][encoder].get('alpha', 1e-6)
            nu = cfg['GPR_config'][encoder].get('nu', 1.5)
            constant_value = cfg['GPR_config'][encoder].get('constant_value', 1.0)
            constant_value_bounds = (constant_value * 0.1, constant_value * 10.0) # allow small range of constant value tuning
            # coerce types
            try:
                nugget_value = float(nugget_value)
            except Exception:
                raise ValueError(f"Invalid nugget_value in config: {nugget_value!r}")

            # ensure length_scales is numeric (scalar or list)
            if isinstance(length_scales, (list, tuple)):
                length_scales = [float(x) for x in length_scales]
            else:
                length_scales = float(length_scales)

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
            
            regressor = GaussianProcessRegressor(
                kernel=kernel,
                alpha=nugget_value,  # using alpha parameter to set nugget
                optimizer="fmin_l_bfgs_b",      # still use optimizer to fine-tune hyperparameters
                n_restarts_optimizer=n_restarts_optimizer, 
                normalize_y=True,
                copy_X_train=True
            )
        elif fixed_regressor_hp == "old_R_emulator":
            print("[INFO] Using fixed hyperparameters for GPR from the old R emulator.")
            nkeep=20.0
            hp_values = []
            # Load hyperparameters from emulator.yaml
            nkeep = cfg['GPR_config']['old_R_emulator_nkeep']['nkeep']
            hp_values = cfg['GPR_config']['old_R_emulator_hyperparameters']
            hp_values = [value * nkeep for value in hp_values]
            length_scales = hp_values[:-1]  # Extract all but the last value for length scales
            nugget_value = hp_values[-1]   # The last value is the nugget
            kernel = RBF(length_scale=length_scales)
            regressor = GaussianProcessRegressor(
                kernel=kernel,
                alpha=nugget_value,  # 使用 alpha 参数设置 nugget
                optimizer=None,      # 关闭优化器
                normalize_y=False,
                copy_X_train=True
            )
        else:
            print("[INFO] Using GPR optimization with chosen range for hyperparameters.")
            # use WhiteKernel rather than alpha nugget for numerical stability
            # XY has been normalized, so ConstantKernel is not necessary here
            n_restarts_optimizer = 12
            kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-6, 1e1))
            regressor = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=n_restarts_optimizer,  # n_restarts_optimizer can be set higher for better optimization
                random_state=42,
                normalize_y=True
            )
            if verbose:
                print(f"[GPR] init kernel={regressor.kernel} | restarts={n_restarts_optimizer}")

    elif regressor_type == "LGBM":
        if fixed_regressor_hp:
            print("[INFO] Using fixed hyperparameters for LGBMRegressor.")
            # Load hyperparameters from emulator.yaml
            with open(cfg_path, 'r') as file:
                config = yaml.safe_load(file)
            lgbm_params = config['LGBM_config'][encoder]
            print(f"[INFO] Loaded LGBM hyperparameters: {lgbm_params}")
            regressor = LGBMRegressor(
                n_estimators=lgbm_params['n_estimators'],
                learning_rate=lgbm_params['learning_rate'],
                num_leaves=lgbm_params['num_leaves'],
                max_depth=lgbm_params['max_depth'],
                min_child_samples=lgbm_params['min_child_samples'],
                subsample=lgbm_params['subsample'],
                colsample_bytree=lgbm_params['colsample_bytree'],
                random_state=lgbm_params['random_state'],
                n_jobs=lgbm_params['n_jobs'],
                verbosity=-1
            )
        else:
            print("[INFO] Using default LGBMRegressor hyperparameters.")
            print("[INFO] For proper hyperparameter tuning, using run_training_LGBM_optimization.")
            regressor = LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                num_leaves=31,
                max_depth=-1,
                random_state=42,
                n_jobs=-1,
                verbosity=-1
            )

    else:
        raise ValueError("[ERROR] regressor_type must be either 'GPR' or 'LGBM'.")

    return regressor