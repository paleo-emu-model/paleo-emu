import os
from pathlib import Path
import unittest

from paleo_emu.training import TrainingGenerator
from paleo_emu.load_config import load_config   
from paleo_emu.load import load_training_data 

class TestTraining(unittest.TestCase):

    def __init__(self, methodName="runTest"):
        super().__init__(methodName)

        # Directory of this test file: .../tests
        here = Path(__file__).resolve().parent

        # Repo root (one level up from tests/)
        repo_root = here.parent

        # Path to examples directory
        self.examples_dir = repo_root / "examples"
        model_cfg_path = repo_root / "tests" / "test.yaml"

        # Use the typed loader (no direct yaml.safe_load)
        self.cfg = load_config(str(model_cfg_path))

        # Simple synthetic training data for the test
        self.X_train, self.Y_train, _, _, lat_array, lon_array = load_training_data(self.cfg)

    def test_run_training(self):
        training = TrainingGenerator(
            self.cfg,
            self.X_train,
            self.Y_train,
            output_dir=str(self.examples_dir),
        )
        artifact_path = training.run_training()
        # If run_training returns a path, assert it exists
        self.assertTrue(os.path.exists(artifact_path))


if __name__ == "__main__":
    unittest.main(verbosity=2)
