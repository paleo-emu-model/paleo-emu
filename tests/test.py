import os
import unittest
import xarray as xr
from paleo_emu.training import run_training
from paleo_emu.prediction import run_prediction

class TestTraining(unittest.TestCase):

    def __init__(self, methodName = "runTest"):
        super().__init__(methodName)
        file_path_tmp = os.path.join(".", "examples")
        self.file_path = os.getenv('GITHUB_WORKSPACE', file_path_tmp) #os.path.join(".", "examples")

    def test_run_training(self):
        cfg_path = os.path.join(self.file_path, "training_test.yaml")
        emulatorPCAGPR = run_training(cfg_path=cfg_path,
                                regressor_type="GPR", 
                                encoder="PCA", 
                                save_pipeline=True)
        
    def test_run_prediction(self):
        model_cfg_path = os.path.join(self.file_path, "outputs", "emulator_saved", "emulator_PCA+GPR_lowice_test.joblib")
        forcing_cfg = os.path.join(self.file_path, "forcing.yaml")
        prediction = run_prediction(model_cfg=model_cfg_path,
                                forcing_cfg=forcing_cfg,
                                scenario="rcp85.1", 
                                output_dir=os.path.join(self.file_path, "outputs", "prediction"))
        
    def test_model_output(self):
        ds = xr.open_dataset(
            os.path.join(self.file_path, "outputs", "prediction", "PCA_GPR_forcing.yaml_prediction.nc")
        )
        self.assertAlmostEqual(ds["prediction"].mean(), 5.21, delta=0.01)


if __name__ == "__main__":
    # Local-only runner
    # - Run all tests in this file:  python tests/test_training.py -v
    # - Or enable training:          RUN_TRAIN=1 python tests/test_training.py -v
    unittest.main(verbosity=2)
