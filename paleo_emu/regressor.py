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
from sklearn.model_selection import RandomizedSearchCV
import yaml
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone

def build_regressor(cfg_path, regressor_type="GPR", fixed_regressor_hp=True, verbose=True):
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
            print("[INFO] Different PCs should need different hyperparameters, check if we need to fix it or not.")
            kernel_name = cfg['GPR_config']['kernel']
            nugget_value = cfg['GPR_config']['nugget_value']
            length_scales = cfg['GPR_config']['length_scales']
            noise_level = cfg['GPR_config'].get('noise_level', 1.0)
            n_restarts_optimizer = cfg['GPR_config'].get('n_restarts_optimizer', 5)
            alpha = cfg['GPR_config'].get('alpha', 1e-6)
            nu = cfg['GPR_config'].get('nu', 1.5)
            constant_value = cfg['GPR_config'].get('constant_value', 1.0)
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
        import warnings
        warnings.filterwarnings("ignore", message=".*valid feature names.*", category=UserWarning)
        if fixed_regressor_hp:
            print("[INFO] Using fixed hyperparameters for LGBMRegressor.")
            # Load hyperparameters from emulator.yaml
            lgbm_params = cfg['LGBM_config'][encoder]
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
            print("[INFO] Using LGBM optimization.")
            print("[INFO] Note: it will take long time if dataset is large.")
            # load search settings from config if present
            lgbm_cfg = cfg.get('LGBM_config', {}).get(encoder, {}) if cfg else {}
            # stronger-regularization defaults: smaller trees, more min samples per leaf,
            # L1/L2 penalties, feature/row subsampling and modest learning rate.
            param_distributions = lgbm_cfg.get('param_distributions', {
                # model capacity (small because few samples)
                'num_leaves': [8, 12, 16, 24, 32],
                'max_depth': [3, 4, 6],
                # learning / ensemble size (conservative)
                'learning_rate': [0.001, 0.005, 0.01, 0.03],
                'n_estimators': [100, 200, 400, 800],
                # row/feature subsampling to reduce overfit
                'subsample': [0.6, 0.7, 0.8, 1.0],
                'colsample_bytree': [0.5, 0.6, 0.7, 0.8],
                'feature_fraction': [0.5, 0.6, 0.7, 0.8],
                'bagging_fraction': [0.6, 0.8, 1.0],
                'bagging_freq': [0, 1, 5],
                # leaf / split constraints (increase min samples per leaf)
                'min_child_samples': [10, 20, 30, 50],
                # regularization penalties (favor some L2)
                'lambda_l1': [0.0, 0.01, 0.1, 1.0],
                'lambda_l2': [0.0, 0.5, 1.0, 5.0],
                # require minimum gain to split (avoid tiny noisy splits)
                'min_gain_to_split': [0.0, 0.01, 0.05, 0.1],
            })
            # search budget & CV
            n_iter = int(lgbm_cfg.get('n_iter', 20))    # increase if you can afford time (30-100)
            cv = int(lgbm_cfg.get('cv', 4))             # with 120 samples 4 or 5 folds is reasonable
            random_state = int(lgbm_cfg.get('random_state', 42))

            base_lgb = LGBMRegressor(random_state=random_state, verbosity=-1)
            # Use RandomizedSearchCV so that when cloned for each output it will tune per‑PC on that PC's training data.
            # IMPORTANT: set n_jobs=1 here to avoid nested parallelism when outer code also parallelizes.
            search = RandomizedSearchCV(
                estimator=base_lgb,
                param_distributions=param_distributions,
                n_iter=n_iter,
                cv=cv,
                scoring='neg_mean_squared_error',
                random_state=random_state,
                n_jobs=1,
                verbose=0
            )
            regressor = search
    else:
        raise ValueError("[ERROR] regressor_type must be either 'GPR' or 'LGBM'.")

    return regressor

