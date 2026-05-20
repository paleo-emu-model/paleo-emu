import os
from pathlib import Path
import unittest
import warnings

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from paleo_emu.training import TrainingGenerator
from paleo_emu.config import load_config
from paleo_emu.load import load_training_data
from paleo_emu.load import load_forcing_data
from paleo_emu.regressor import EncodedTargetRegressor  
import xarray as xr

class TestTraining(unittest.TestCase):
    def setUp(self):
        # Directory of this test file: .../tests
        here = Path(__file__).resolve().parent

        # Repo root (one level up from tests/)
        self.repo_root = here.parent

        # Path to examples directory
        self.examples_dir = self.repo_root / "examples"

    def _run_training_with_cfg(self, cfg_filename: str):
        model_cfg_path = self.repo_root / "tests" / cfg_filename

        # Use the typed loader (PaleoEmuConfig)
        cfg = load_config(str(model_cfg_path))

        # Load full training data from disk
        X_full, Y_full, _, _, lat_array, lon_array, _ = load_training_data(cfg)

        # 80/20 train–test split for performance evaluation
        X_train, X_test, Y_train, Y_test = train_test_split(
            X_full,
            Y_full,
            test_size=0.1,
            random_state=cfg.random_state,
        )

        training = TrainingGenerator(
            cfg,
            X_train,
            Y_train,
            lat_array,
            lon_array,
            output_dir=str(self.repo_root / "tests"), 
        )
        artifact_path = training.run_training()
        self.assertTrue(os.path.exists(artifact_path))

        # Return everything needed for checks
        return artifact_path, X_full, X_test, Y_test

    def _check_artifact_and_predictions(self, artifact_path, X_full, X_test, Y_test,
                                          r2_threshold=0.98, mean_delta=0.05):
        # Load the artifact
        artifact = joblib.load(artifact_path)

        # Basic keys check
        self.assertIn("model", artifact)
        model = artifact["model"]

        # Model should be an EncodedTargetRegressor
        self.assertIsInstance(model, EncodedTargetRegressor)

        # -------------------------------------------------
        # 1) Mean value check on original X field
        # -------------------------------------------------
        Y_pred_full = model.predict(X_full)
        field_mean = np.mean(Y_pred_full)
        self.assertAlmostEqual(field_mean, 5.3, delta=mean_delta)
        print(f"Mean temperature: {field_mean}")

        # -------------------------------------------------
        # 2) Performance check on 10% hold-out set
        # -------------------------------------------------
        Y_pred_test = model.predict(X_test)
        r2 = r2_score(Y_test, Y_pred_test, multioutput="uniform_average")
        print(f"Hold-out R^2: {r2}")
        self.assertGreater(r2, r2_threshold, msg=f"Hold-out R^2 too low: {r2}")

        # -------------------------------------------------
        # 3) Physical plausibility: SST range check
        # -------------------------------------------------
        pred_min = np.nanmin(Y_pred_full)
        pred_max = np.nanmax(Y_pred_full)
        self.assertGreater(pred_min, -100.0, msg=f"Predictions too cold: min={pred_min:.2f}")
        self.assertLess(pred_max, 100.0, msg=f"Predictions too warm: max={pred_max:.2f}")

    def _resolve_artifact_path(self, cfg) -> Path:
        """Resolve artifact path from config, treating relative output_dir as relative to repo root."""
        if cfg.output_dir is not None:
            artifact_dir = self.repo_root / cfg.output_dir
        else:
            artifact_dir = self.repo_root / "tests"
        artifact_filename = cfg.artifact_name if cfg.artifact_name is not None else f"{cfg.model_run_name}_fitted_pipeline.joblib"
        return artifact_dir / artifact_filename

    def _run_prediction_with_cfg(self, cfg_filename: str, scenario:str):
        model_cfg_path = self.repo_root / "tests" / cfg_filename

        # Use the typed loader (PaleoEmuConfig)
        cfg = load_config(str(model_cfg_path))

        # Resolve artifact path: use config's output_dir/artifact_name if set, else defaults
        artifact_path = self._resolve_artifact_path(cfg)
        self.assertTrue(os.path.exists(artifact_path))
        artifact = joblib.load(artifact_path)

        model = artifact["model"]
        self.assertIsInstance(model, EncodedTargetRegressor)

        # Make predictions
        X_pred = load_forcing_data(cfg, scenario=scenario)
        
        # Get predictions with variance
        Y_pred, Y_std = model.predict_with_variance(X_pred)

        return Y_pred, Y_std
        
    def _test_unique_kernels(self, cfg_filename: str):
        model_cfg_path = self.repo_root / "tests" / cfg_filename
        cfg = load_config(str(model_cfg_path))

        artifact_path = self._resolve_artifact_path(cfg)
        self.assertTrue(os.path.exists(artifact_path))
        artifact = joblib.load(artifact_path)
        model = artifact["model"]

        pipe = model.estimator_
        mor = pipe.named_steps["regressor"]

        length_scales = []
        noise_levels = []

        for j, gpr in enumerate(mor.estimators_):
            k = gpr.kernel_
            length_scales.append(k.k1.length_scale)
            noise_levels.append(k.k2.noise_level)

        def is_all_ones(x) -> bool:
            arr = np.asarray(x)
            return np.all(arr == 1)

        # Fail if *every* estimator kept initial params (== 1 everywhere)
        self.assertFalse(
            all(is_all_ones(ls) for ls in length_scales),
            msg=f"Kernel optimization likely did not run: all length_scales still 1: {length_scales}"
        )

    def test_run_training_pca_gp(self):
        """Full training run using PCA encoder config with 10% hold-out performance check."""
        artifact_path, X_full, X_test, Y_test = self._run_training_with_cfg("test_PCA_GP.yml")
        Y_pred, Y_std = self._run_prediction_with_cfg("test_PCA_GP.yml", scenario="800ka")
        self._check_artifact_and_predictions(artifact_path, X_full, X_test, Y_test)
        self._test_unique_kernels("test_PCA_GP.yml")

        # GP uncertainty check: std should be positive
        self.assertIsNotNone(Y_std, "GP model should return non-None std")
        self.assertTrue(np.all(Y_std >= 0), "GP std should be non-negative everywhere")
        self.assertGreater(np.mean(Y_std), 0, "GP mean std should be > 0")

    def test_run_training_pca_xgb(self):
        """Full training run using PCA encoder config with 10% hold-out performance check."""
        artifact_path, X_full, X_test, Y_test = self._run_training_with_cfg("test_PCA_XGB.yml")
        Y_pred, Y_std = self._run_prediction_with_cfg("test_PCA_XGB.yml", scenario="800ka")
        self._check_artifact_and_predictions(artifact_path, X_full, X_test, Y_test)


    def test_run_training_vae_gp(self):
        """VAE encoder + GP regressor — smoke test (10 epochs, checks code runs and output is physical)."""
        artifact_path, X_full, X_test, Y_test = self._run_training_with_cfg("test_VAE_GP.yml")
        Y_pred, Y_std = self._run_prediction_with_cfg("test_VAE_GP.yml", scenario="800ka")
        self._check_artifact_and_predictions(artifact_path, X_full, X_test, Y_test,
                                              r2_threshold=-1.0, mean_delta=50.0)
        self.assertIsNotNone(Y_std, "GP model should return non-None std")
        self.assertGreater(np.mean(Y_std), 0, "GP mean std should be > 0")

    def test_run_training_vae_xgb(self):
        """VAE encoder + XGBoost regressor — smoke test (10 epochs, checks code runs and output is physical)."""
        artifact_path, X_full, X_test, Y_test = self._run_training_with_cfg("test_VAE_XGB.yml")
        Y_pred, Y_std = self._run_prediction_with_cfg("test_VAE_XGB.yml", scenario="800ka")
        self._check_artifact_and_predictions(artifact_path, X_full, X_test, Y_test,
                                              r2_threshold=-1.0, mean_delta=50.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
