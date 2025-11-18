import os
from pathlib import Path
import unittest
import xarray as xr
from paleo_emu.training import run_training
from paleo_emu.prediction import run_prediction


class TestTraining(unittest.TestCase):

    def __init__(self, methodName="runTest"):
        super().__init__(methodName)

        # Directory of this test file: .../tests
        here = Path(__file__).resolve().parent

        # Repo root (one level up from tests/)
        repo_root = here.parent

        # Path to examples directory
        self.examples_dir = repo_root / "examples"

    def test_run_training(self):
        cfg_path = self.examples_dir / "training_test.yaml"
        emulatorPCAGPR = run_training(
            cfg_path=str(cfg_path),
            regressor_type="GPR",
            encoder="PCA",
            save_pipeline=True,
        )

    def test_run_prediction(self):
        model_cfg_path = (
            self.examples_dir
            / "outputs"
            / "emulator_saved"
            / "emulator_PCA+GPR_lowice_test.joblib"
        )
        forcing_cfg = self.examples_dir / "forcing.yaml"

        prediction = run_prediction(
            model_cfg=str(model_cfg_path),
            forcing_cfg=str(forcing_cfg),
            scenario="rcp85.1",
            output_dir=str(self.examples_dir / "outputs" / "prediction"),
        )

    def test_model_output(self):
        ds_path = (
            self.examples_dir
            / "outputs"
            / "prediction"
            / "PCA_GPR_forcing.yaml_prediction.nc"
        )
        ds = xr.open_dataset(ds_path)
        self.assertAlmostEqual(ds["prediction"].mean(), 5.21, delta=0.01)


if __name__ == "__main__":
    unittest.main(verbosity=2)
