import unittest
unittest.TestLoader.sortTestMethodsUsing = None
import xarray as xr
import os
from paleo_emu.training import run_training_28

class TestTest(unittest.TestCase):

    # from src.optimise import full_emulator_experiment

    def training_28_test(self):
        # define emulator training data
        emulator = "lowmod_ice"
        forcing = "rcp85.1"

        # define VAE configuration
        vae_config = {
            "latent_dim": 1024, # 32, 64, 128, 256，512， 1024
            "epochs": 80,
            "learning_rate": 1e-4, # 1e-4, 5e-5, 1e-5
            "batch_size": 128,
            "kl_weight": 0.1 # 0.1, 0.5, 1.0
        }

        file_path = os.path.join(".", "examples", "training_data")

        # define inputs
        train_dict = {
            "lowmod_ice": {
                "file_path": file_path,
                "X_input": "training_data_lowmodice_temp_formatted.res",
                "Y_output": "training_data_lowmodice_temp_formatted.nc",
                "label": "lowmod_ice"
            },
            "highmod_ice": {
                "file_path": file_path,
                "X_input": "training_data_highmodice_temp_formatted.res",
                "Y_output": "training_data_highmodice_temp_formatted.nc",
                "label": "highmod_ice"
            },
            "highlowmod_ice": {
                "file_path": file_path,
                "X_input": "training_data_highlowmodice_temp_formatted.res",
                "Y_output": "training_data_highlowmodice_temp_formatted.nc",
                "label": "highlowmod_ice"
            }
        }

        # run emulator training
        emulator = run_training_28(train_dict[emulator],
                                   regressor_type="GPR", 
                                   kernel="RBF",encoder="PCA", 
                                   vae_config=vae_config, 
                                   return_validation=True)

    def test_model_output(self):
    # import output
        file_path = os.path.join(".", "examples", "training_data")

        ds = xr.open_dataset(os.path.join(
            file_path, "training_data_lowmodice_temp_formatted.nc"))

        self.assertAlmostEqual(
            ds['var'].mean(), 5.28, delta=0.01
        )


if __name__ == '__main__':
    # Create a test suite combining all test cases in order
    suite = unittest.TestSuite()
#    suite.addTest(TestTest('training_28_test'))
    suite.addTest(TestTest('test_model_output'))
    runner = unittest.TextTestRunner()
    runner.run(suite)
