import os
import tempfile
from pathlib import Path
import unittest

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import xarray as xr
from pydantic import ValidationError

from paleo_emu.training import TrainingGenerator
from paleo_emu.config import PaleoEmuConfig, load_config
from paleo_emu.load import load_training_data
from paleo_emu.load import load_forcing_data
from paleo_emu.regressor import EncodedTargetRegressor
from paleo_emu.export import save_prediction


class TestForcingConfigValidation(unittest.TestCase):
    def _base_config(self, forcing_data):
        return {
            "regressor_config": {
                "kernels": ["RBF"],
                "n_restarts_optimizer": 0,
                "alpha": 1e-6,
                "whitekernel_noise_level": 1e-2,
                "whitekernel_noise_level_bounds": [1e-5, 1e1],
            },
            "cv": {"folds": 2, "n_jobs": 1},
            "random_state": 29,
            "model_run_name": "test",
            "encoder_config": {"encoder_type": "PCA", "n_components": 2},
            "training_file_path": "examples/training_data/",
            "X_input_file_name": "training_data_lowmodice_temp_formatted.res",
            "Y_input_file_name": "training_data_lowmodice_temp_formatted.nc",
            "X_column_names": ["co2", "obliquity", "esinw", "ecosw", "ice"],
            "forcing_data_path": "examples/forcing_data/",
            "forcing_data": forcing_data,
        }

    def test_single_forcing_config(self):
        cfg = PaleoEmuConfig(**self._base_config({
            "SSP585": {
                "kind": "single",
                "forcing_input": "Forcings_SSP585_since2000.res",
            }
        }))

        self.assertEqual(cfg.forcing_data["SSP585"].kind, "single")
        self.assertEqual(
            cfg.forcing_data["SSP585"].forcing_input,
            "Forcings_SSP585_since2000.res",
        )

    def test_pattern_numbered_sweep_config(self):
        cfg = PaleoEmuConfig(**self._base_config({
            "past800ka_ens": {
                "kind": "pattern",
                "forcing_input_pattern": "Forcings_past800ka_sst_member{member}.res",
                "member": {"start": 1, "end": 3, "width": 3},
            }
        }))

        sweep = cfg.forcing_data["past800ka_ens"].expanded_sweep_values()
        self.assertEqual(sweep["member"], ["001", "002", "003"])

    def test_pattern_list_sweep_config(self):
        cfg = PaleoEmuConfig(**self._base_config({
            "past800ka_var": {
                "kind": "pattern",
                "forcing_input_pattern": "Forcings_past800ka_{var}_member1.res",
                "var": ["sst", "precip", "co2"],
            }
        }))

        sweep = cfg.forcing_data["past800ka_var"].expanded_sweep_values()
        self.assertEqual(sweep["var"], ["sst", "precip", "co2"])

    def test_pattern_requires_matching_placeholders(self):
        with self.assertRaises(ValidationError):
            PaleoEmuConfig(**self._base_config({
                "bad": {
                    "kind": "pattern",
                    "forcing_input_pattern": "Forcings_past800ka_{var}.res",
                    "variable": ["sst"],
                }
            }))

    def test_pattern_requires_placeholder(self):
        with self.assertRaises(ValidationError):
            PaleoEmuConfig(**self._base_config({
                "bad": {
                    "kind": "pattern",
                    "forcing_input_pattern": "Forcings_past800ka_fixed.res",
                    "member": {"start": 1, "end": 3},
                }
            }))

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
        X_full, Y_full, _, _, lat_array, lon_array, _, _ = load_training_data(cfg)

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


    def test_save_prediction_cf_attrs(self):
        """save_prediction output is CF-1.8 compliant."""
        rng = np.random.default_rng(0)
        lat = np.linspace(-90, 90, 4)
        lon = np.linspace(0, 360, 6)
        Y_pred = rng.random((3, 4, 6))
        Y_var  = rng.random((3, 4, 6))

        with tempfile.TemporaryDirectory() as tmp:
            save_prediction(
                Y_pred, Y_var, lat, lon, tmp,
                file_name="cf_test",
                var_name="tos",
                var_attrs={"long_name": "sea surface temperature", "units": "degC",
                           "standard_name": "sea_surface_temperature"},
            )
            ds = xr.open_dataset(Path(tmp) / "cf_test.nc")

            # Convention flag
            self.assertEqual(ds.attrs.get("Conventions"), "CF-1.8")

            # Coordinate CF attributes
            self.assertEqual(ds["latitude"].attrs.get("units"), "degrees_north")
            self.assertEqual(ds["longitude"].attrs.get("units"), "degrees_east")
            self.assertEqual(ds["latitude"].attrs.get("axis"), "Y")
            self.assertEqual(ds["longitude"].attrs.get("axis"), "X")
            self.assertEqual(ds["time"].attrs.get("axis"), "T")

            # Variable attributes pulled from var_attrs
            self.assertEqual(ds["tos"].attrs.get("standard_name"), "sea_surface_temperature")
            self.assertEqual(ds["tos"].attrs.get("units"), "degC")
            self.assertEqual(ds["tos"].attrs.get("long_name"), "sea surface temperature")

            # Variance attributes
            self.assertIn("variance of", ds["variance"].attrs.get("long_name", ""))
            self.assertIn("degC", ds["variance"].attrs.get("units", ""))

            ds.close()

    def test_save_prediction_cf_attrs_no_training_attrs(self):
        """save_prediction falls back gracefully when training data has no CF attrs."""
        rng = np.random.default_rng(1)
        lat = np.array([-45.0, 0.0, 45.0])
        lon = np.array([0.0, 90.0, 180.0, 270.0])
        Y_pred = rng.random((2, 3, 4))
        Y_var  = rng.random((2, 3, 4))

        with tempfile.TemporaryDirectory() as tmp:
            # No var_name or var_attrs → pure defaults
            save_prediction(Y_pred, Y_var, lat, lon, tmp, file_name="cf_defaults")
            ds = xr.open_dataset(Path(tmp) / "cf_defaults.nc")

            self.assertEqual(ds.attrs.get("Conventions"), "CF-1.8")
            self.assertIn("prediction", ds.data_vars)
            self.assertEqual(ds["latitude"].attrs.get("units"), "degrees_north")
            self.assertEqual(ds["longitude"].attrs.get("units"), "degrees_east")

            ds.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
