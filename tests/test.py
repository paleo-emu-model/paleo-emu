import os
import unittest
import xarray as xr
from paleo_emu.training import run_training_28

class TestTraining(unittest.TestCase):

    def test_training_28(self):
        emulator = "lowmod_ice"
        vae_config = {
            "latent_dim": 1024,
            "epochs": 80,
            "learning_rate": 1e-4,
            "batch_size": 128,
            "kl_weight": 0.1
        }

        file_path = os.path.join(".", "examples", "training_data")
        train_dict = {
            "lowmod_ice": {
                "file_path": file_path,
                "X_input": "training_data_lowmodice_temp_formatted.res",
                "Y_output": "training_data_lowmodice_temp_formatted.nc",
                "label": "lowmod_ice",
            },
            "highmod_ice": {
                "file_path": file_path,
                "X_input": "training_data_highmodice_temp_formatted.res",
                "Y_output": "training_data_highmodice_temp_formatted.nc",
                "label": "highmod_ice",
            },
            "highlowmod_ice": {
                "file_path": file_path,
                "X_input": "training_data_highlowmodice_temp_formatted.res",
                "Y_output": "training_data_highlowmodice_temp_formatted.nc",
                "label": "highlowmod_ice",
            },
        }

        run_training_28(
            train_dict[emulator],
            regressor_type="GPR",
            kernel="RBF",
            encoder="PCA",
            vae_config=vae_config,
            return_validation=True,
        )

    def test_model_output(self):
        file_path = os.path.join(".", "examples", "training_data")
        ds = xr.open_dataset(
            os.path.join(file_path, "training_data_lowmodice_temp_formatted.nc")
        )
        self.assertAlmostEqual(ds["var"].mean(), 5.28, delta=0.01)


if __name__ == "__main__":
    # Local-only runner
    # - Run all tests in this file:  python tests/test_training.py -v
    # - Or enable training:          RUN_TRAIN=1 python tests/test_training.py -v
    unittest.main(verbosity=2)
