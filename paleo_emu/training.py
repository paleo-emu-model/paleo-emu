"""
Training module using chosen regressors, kernels, and encoders.

See config loader (_load.py / config_loader.py) for the typed config:
- PaleoEmuConfig
- _RegressorConfig
- make_kernel
"""

import os

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib  

from paleo_emu.encoders import EncoderGenerator
from paleo_emu.load_config import PaleoEmuConfig, _RegressorConfig, make_kernel


class TrainingGenerator:
    """Utility for training emulators with specified encoders and regressors.

    Parameters
    ----------
    model_configuration : PaleoEmuConfig
        Typed configuration object loaded via `load_config(path)`.
    X_train : array-like, shape (n_samples, n_features)
    Y_train : array-like, shape (n_samples,)
    output_dir : str, optional
        Directory to save the joblib artifact. Defaults to current directory.

    The joblib artifact will contain:
    - "pipeline": best fitted sklearn Pipeline (scaler + GPR)
    - "grid_search": fitted GridSearchCV
    - "decoder": decoder from EncoderGenerator
    - "mean_val": mean of Y used in encoding
    - "std_val": std of Y used in encoding
    """

    def __init__(self, model_configuration: PaleoEmuConfig, X_train=None, Y_train=None,
                 output_dir: str = "."):
        self.cfg: PaleoEmuConfig = model_configuration
        self.X_train = X_train
        self.Y_train = Y_train
        self.output_dir = output_dir

    # ----------------- helpers -----------------
    def _build_kernel_candidates(self):
        reg_cfg: _RegressorConfig = self.cfg.regressor_config
        return [make_kernel(name, reg_cfg) for name in reg_cfg.kernels]

    def _build_param_grid(self):
        kernels = self._build_kernel_candidates()
        # Pipeline step is named "regressor"
        return {"regressor__kernel": kernels}

    def _build_regressor(self) -> GaussianProcessRegressor:
        reg_cfg: _RegressorConfig = self.cfg.regressor_config
        return GaussianProcessRegressor(
            normalize_y=True,
            n_restarts_optimizer=reg_cfg.n_restarts_optimizer,
            random_state=self.cfg.random_state,
        )

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

        # 1–4. Encode Y and get decoder / normalization stats
        # EncoderGenerator likely expects dict-like config; use model_dump().
        enc = EncoderGenerator(self.Y_train, self.cfg.model_dump())
        Y_train_encoded, decoder, mean_val, std_val = enc.generate_encoder()

        # 5. Build regressor, param_grid, and pipeline
        regressor = self._build_regressor()
        param_grid = self._build_param_grid()

        model_pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", regressor),
            ]
        )

        cv_cfg = self.cfg.cv
        grid = GridSearchCV(
            estimator=model_pipeline,
            param_grid=param_grid,
            cv=cv_cfg.folds,
            n_jobs=cv_cfg.n_jobs,
            scoring=cv_cfg.scoring,
        )

        grid.fit(self.X_train, Y_train_encoded)

        print("[INFO] Best hyperparameters from GridSearchCV:")
        print(grid.best_params_)

        best_pipeline = grid.best_estimator_

        # --------- export with joblib instead of returning the tuple ----------
        os.makedirs(self.output_dir, exist_ok=True)
        artifact = {
            "pipeline": best_pipeline,
            "grid_search": grid,
            "decoder": decoder,
            "mean_val": mean_val,
            "std_val": std_val,
        }

        artifact_name = f"{self.cfg.model_run_name}_fitted_pipeline.joblib"
        artifact_path = os.path.join(self.output_dir, artifact_name)

        joblib.dump(artifact, artifact_path)
        print(f"[INFO] Saved fitted pipeline artifact to {artifact_path}")

        return artifact_path
