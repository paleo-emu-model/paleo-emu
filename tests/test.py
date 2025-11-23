import os
from pathlib import Path
import unittest

import joblib
import numpy as np

from paleo_emu.training import TrainingGenerator
from paleo_emu.config import load_config
from paleo_emu.load import load_training_data
from paleo_emu.encoding import EncodedTargetRegressor  # ensure class is importable for joblib


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
        X_train, Y_train, _, _, lat_array, lon_array = load_training_data(cfg)

        training = TrainingGenerator(
            cfg,
            X_train,
            Y_train,
            lat_array,
            lon_array,
            output_dir=str(self.examples_dir),
        )
        artifact_path = training.run_training()
        self.assertTrue(os.path.exists(artifact_path))

        # Return data & path for further checks in individual tests
        return artifact_path, X_train

    def _check_artifact_and_predictions(self, artifact_path, X_train):
        # Load the artifact
        artifact = joblib.load(artifact_path)

        # Basic keys check
        self.assertIn("model", artifact)

        model = artifact["model"]

        # Model should be an EncodedTargetRegressor
        self.assertIsInstance(model, EncodedTargetRegressor)

        # Predict on the original X field
        Y_pred = model.predict(X_train)

        # Compute mean of predicted field
        field_mean = np.mean(Y_pred)

        self.assertAlmostEqual(field_mean, 5.28, delta=0.01)

    def test_run_training_pca(self):
        """Full training run using PCA encoder config."""
        artifact_path, X_train = self._run_training_with_cfg("test_PCA.yml")
        self._check_artifact_and_predictions(artifact_path, X_train)

    # def test_run_training_vae(self):
    #     """Full training run using VAE (learned encoder) config."""
    #     artifact_path, X_train = self._run_training_with_cfg("test_VAE.yml")
    #     self._check_artifact_and_predictions(artifact_path, X_train)


if __name__ == "__main__":
    unittest.main(verbosity=2)
