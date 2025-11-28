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

import joblib
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from paleo_emu.config import PaleoEmuConfig, _RegressorConfig, make_kernel
from paleo_emu.encoding import EncodedTargetRegressor


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
        reg_cfg: _RegressorConfig = self.cfg.regressor_config
        n_features = self.X_train.shape[1]
        return [
            make_kernel(name, reg_cfg, n_features=n_features)
            for name in reg_cfg.kernels
        ]

    def _build_param_grid(self):
        kernels = self._build_kernel_candidates()
        # NOTE: parameter path:
        # EncodedTargetRegressor(base_estimator=Pipeline([...]))
        # -> base_estimator (Pipeline)
        # -> "regressor" step (MultiOutputRegressor)
        # -> underlying estimator (GaussianProcessRegressor) -> "kernel"
        return {"base_estimator__regressor__estimator__kernel": kernels}

    def _build_regressor(self) -> MultiOutputRegressor:
        """Build a (potentially) multi-output Gaussian Process regressor."""
        reg_cfg: _RegressorConfig = self.cfg.regressor_config

        base_gpr = GaussianProcessRegressor(
            normalize_y=True,
            n_restarts_optimizer=reg_cfg.n_restarts_optimizer,
            random_state=self.cfg.random_state,
        )

        # Wrap in MultiOutputRegressor so we can handle latent_dim > 1 cleanly
        return MultiOutputRegressor(base_gpr)

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
        grid.fit(self.X_train, self.Y_train)

        print("[INFO] Best hyperparameters from GridSearchCV:")
        print(grid.best_params_)

        best_model = grid.best_estimator_

        # export with joblib
        os.makedirs(self.output_dir, exist_ok=True)
        artifact = {
            "model": best_model,
            "grid_search": grid,
            "lat_array": self.lat_array,
            "lon_array": self.lon_array,
        }

        artifact_name = f"{self.cfg.model_run_name}_fitted_pipeline.joblib"
        artifact_path = os.path.join(self.output_dir, artifact_name)

        joblib.dump(artifact, artifact_path)
        print(f"[INFO] Saved fitted model artifact to {artifact_path}")

        return artifact_path
