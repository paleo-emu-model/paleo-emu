"""
Training module using chosen regressors, kernels, and encoders.

See config loader (_load.py / config_loader.py) for the typed config:
- PaleoEmuConfig
- _RegressorConfig
- make_kernel

The joblib artifact will contain:
- "model": best fitted EncodedTargetRegressor
           (includes encoder, scaler, and regressor)
- "grid_search": fitted GridSearchCV
- "lat_array": latitude grid used during training
- "lon_array": longitude grid used during training
"""

import os
import warnings

import joblib
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.exceptions import ConvergenceWarning
from paleo_emu.config import PaleoEmuConfig, _GPRegressorConfig,_XGBRegressorConfig, make_kernel
from paleo_emu.regressor import EncodedTargetRegressor, GPMultiOutputWithStd


class TrainingGenerator:
    """Utility for training emulators with specified encoders and regressors.

    Parameters
    ----------
    model_configuration : PaleoEmuConfig
        Typed configuration object loaded via `load_config(path)`.
    X_train : array-like, shape (n_samples, n_features)
        Input features for training.
    y_train : array-like, shape (n_samples, n_outputs)
        Target fields for training (e.g. flattened spatial fields).
    lat_array : array-like
        Latitude grid corresponding to y fields.
    lon_array : array-like
        Longitude grid corresponding to y fields.
    output_dir : str, optional
        Directory to save the joblib artifact. Defaults to current directory.

    The joblib artifact will contain:
    - "model": best fitted EncodedTargetRegressor
    - "grid_search": fitted GridSearchCV
    - "lat_array": latitude grid
    - "lon_array": longitude grid
    """

    def __init__(
        self,
        model_configuration: PaleoEmuConfig,
        X_train,
        Y_train,
        lat_array,
        lon_array,
        output_dir: str = ".",
    ):
        self.cfg: PaleoEmuConfig = model_configuration
        self.X_train = X_train
        self.Y_train = Y_train
        self.lat_array = lat_array
        self.lon_array = lon_array
        self.output_dir = output_dir

    # ----------------- helpers -----------------
    def _build_kernel_candidates(self):
        """Build a list of ARD kernels, one per kernel name in the config.

        ARD is enforced by always using a length_scale vector of shape (n_features,).
        """
        reg_cfg: _GPRegressorConfig = self.cfg.regressor_config
        n_features = self.X_train.shape[1]
        return [
            make_kernel(name, reg_cfg, n_features=n_features)
            for name in reg_cfg.kernels
        ]

    def _build_param_grid(self):
        if type(self.cfg.regressor_config) == _GPRegressorConfig:

            kernels = self._build_kernel_candidates()
            # NOTE: parameter path:
            # EncodedTargetRegressor(base_estimator=Pipeline([...]))
            # -> base_estimator (Pipeline)
            # -> "regressor" step (MultiOutputRegressor)
            # -> underlying estimator (GaussianProcessRegressor) -> "kernel"
            return {"base_estimator__regressor__estimator__kernel": kernels}
        elif type(self.cfg.regressor_config) == _XGBRegressorConfig:    
            reg_cfg: _XGBRegressorConfig = self.cfg.regressor_config
            param_grid = {
                "base_estimator__regressor__estimator__num_leaves": reg_cfg.num_leaves,
                "base_estimator__regressor__estimator__max_depth": reg_cfg.max_depth,
                "base_estimator__regressor__estimator__learning_rate": reg_cfg.learning_rate,
                "base_estimator__regressor__estimator__n_estimators": reg_cfg.n_estimators,
                "base_estimator__regressor__estimator__subsample": reg_cfg.subsample,
                "base_estimator__regressor__estimator__colsample_bytree": reg_cfg.colsample_bytree,
                "base_estimator__regressor__estimator__min_child_samples": reg_cfg.min_child_samples,
            }
            return param_grid

    def _build_regressor(self) -> MultiOutputRegressor:
        """Build a (potentially) multi-output Gaussian Process regressor."""
        if type(self.cfg.regressor_config) == _GPRegressorConfig:
            print("inferred type is Gaussian Process")

            reg_cfg: _GPRegressorConfig = self.cfg.regressor_config
            base_regressor = GaussianProcessRegressor(
                normalize_y=True,
                alpha=reg_cfg.alpha,
                n_restarts_optimizer=reg_cfg.n_restarts_optimizer,
                random_state=self.cfg.random_state,
            )

        elif type(self.cfg.regressor_config) == _XGBRegressorConfig:
            reg_cfg: _XGBRegressorConfig = self.cfg.regressor_config
            print("inferred type is XGBoost")
            base_regressor = XGBRegressor(
                num_leaves=reg_cfg.num_leaves,
                n_estimators=reg_cfg.n_estimators,
                max_depth=reg_cfg.max_depth,
                learning_rate=reg_cfg.learning_rate,
                subsample=reg_cfg.subsample,
                colsample_bytree=reg_cfg.colsample_bytree,
                min_child_samples=reg_cfg.min_child_samples,
                random_state=self.cfg.random_state,
            ) 


        # Wrap in GPMultiOutputWithStd for GP, regular MultiOutputRegressor for others
        if type(self.cfg.regressor_config) == _GPRegressorConfig:
            return GPMultiOutputWithStd(base_regressor)
        else:
            return MultiOutputRegressor(base_regressor)
    # ----------------- main training -----------------
    def run_training(self) -> str:
        """Run training and export results as a joblib file.

        Returns
        -------
        artifact_path : str
            Path to the saved joblib artifact.
        """
        if self.X_train is None or self.Y_train is None:
            raise ValueError(
                "X_train and Y_train must be provided. "
                "Automatic data loading is not implemented here."
            )

        # Build base regressor, param_grid, and pipeline (on X only)
        regressor = self._build_regressor()
        base_pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", regressor),
            ]
        )

        # Wrap with EncodedTargetRegressor so Y is encoded/decoded internally
        model = EncodedTargetRegressor(
            base_estimator=base_pipeline,
            model_config=self.cfg,
            return_encoded=False,  # we want decoded predictions in original Y space
        )

        param_grid = self._build_param_grid()
        cv_cfg = self.cfg.cv

        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv_cfg.folds,
            n_jobs=cv_cfg.n_jobs,
            scoring=cv_cfg.scoring,
        )

        # Fit on RAW Y (high-dimensional field); encoding happens inside model
        # Suppress ConvergenceWarnings during parallel GP optimization
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            grid.fit(self.X_train, self.Y_train)
        
        print("-----------------------------------")
        print("Best parameters:")
        for param, value in grid.best_params_.items():
            print(f"  {param}: {value}")
        print(f"Best CV score: {grid.best_score_:.4f}")
        print("------------------------------------")

        best_model = grid.best_estimator_

        # export with joblib
        os.makedirs(self.output_dir, exist_ok=True)
        artifact = {
            "model": best_model,                 # EncodedTargetRegressor (bare regressor)（contains scaler/encoder/regressor）
            "grid_search": grid,
            "lat_array": self.lat_array,
            "lon_array": self.lon_array,
            "mean_val": getattr(self, "mean_val", None),
            "std_val": getattr(self, "std_val", None)
        }
        artifact_name = f"{self.cfg.model_run_name}_fitted_pipeline.joblib"
        artifact_path = os.path.join(self.output_dir, artifact_name)

        joblib.dump(artifact, artifact_path)
        print(f"[INFO] Saved fitted model artifact to {artifact_path}")

        return artifact_path
