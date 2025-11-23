import os
from pathlib import Path
import unittest

from paleo_emu.training import TrainingGenerator
from paleo_emu.config import load_config
from paleo_emu.load import load_training_data


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
            output_dir=str(self.examples_dir),
        )
        artifact_path = training.run_training()
        self.assertTrue(os.path.exists(artifact_path))

    def test_run_training_pca(self):
        """Full training run using PCA encoder config."""
        self._run_training_with_cfg("test_PCA.yml")

    def test_run_training_vae(self):
        """Full training run using VAE (learned encoder) config."""
        self._run_training_with_cfg("test_VAE.yml")


if __name__ == "__main__":
    unittest.main(verbosity=2)
