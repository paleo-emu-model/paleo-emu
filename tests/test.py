import unittest
from pathlib import Path

import xarray as xr

from paleo_emu.training import run_training
from paleo_emu.prediction import run_prediction


class TestTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Run training and prediction once for the whole test class.
        All tests can then rely on the generated files existing.
        """
        # Directory of this test file: .../tests
        here = Path(__file__).resolve().parent

        # Repo root (one level up from tests/)
        repo_root = here.parent

        # Path to examples directory
        cls.examples_dir = repo_root / "examples"

        # --- Run training once ---
        cfg_path = cls.examples_dir / "training_test.yaml"
        run_training(
            cfg_path=str(cfg_path),
            regressor_type="GPR",
            encoder="PCA",
            save_pipeline=True,
        )

        # --- Run prediction once ---
        model_cfg_path = (
            cls.examples_dir
            / "outputs"
            / "emulator_saved"
            / "emulator_PCA+GPR_lowice_test.joblib"
        )
        forcing_cfg = cls.examples_dir / "forcing.yaml"
        output_dir = cls.examples_dir / "outputs" / "prediction"

        run_prediction(
            model_cfg=str(model_cfg_path),
            forcing_cfg=str(forcing_cfg),
            scenario="rcp85.1",
            output_dir=str(output_dir),
        )

    def test_run_training(self):
        """
        Check that training produced the expected model file.
        """
        model_path = (
            self.examples_dir
            / "outputs"
            / "emulator_saved"
            / "emulator_PCA+GPR_lowice_test.joblib"
        )
        self.assertTrue(
            model_path.exists(),
            msg=f"Trained model not found at {model_path}",
        )

    def test_run_prediction(self):
        """
        Check that prediction produced the expected NetCDF file.
        """
        ds_path = (
            self.examples_dir
            / "outputs"
            / "prediction"
            / "PCA_GPR_forcing.yaml_prediction.nc"
        )
        self.assertTrue(
            ds_path.exists(),
            msg=f"Prediction file not found at {ds_path}",
        )

    def test_model_output(self):
        """
        Open the prediction file and validate the mean of the 'prediction' variable.
        """
        ds_path = (
            self.examples_dir
            / "outputs"
            / "prediction"
            / "PCA_GPR_forcing.yaml_prediction.nc"
        )

        ds = xr.open_dataset(ds_path, engine="h5netcdf")
        self.assertAlmostEqual(ds["prediction"].mean(), 5.21, delta=0.01)


if __name__ == "__main__":
    unittest.main(verbosity=2)
