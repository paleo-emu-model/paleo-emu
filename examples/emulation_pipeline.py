
"""
This script is used to run the emulator training pipeline for the Paleo-Emu project.
"""

# from src.optimise import full_emulator_experiment
import os,yaml
from paleo_emu.training import run_training
from paleo_emu.prediction import run_prediction
import joblib



if __name__ == "__main__":
    # run emulator training

    # cfg_path = os.path.join(os.path.dirname(__file__), "training_test.yaml")
    # emulatorPCAGPR = run_training(cfg_path=cfg_path,
    #                             regressor_type="GPR", 
    #                             encoder="PCA", 
    #                             save_pipeline=True)

# # run emulator prediction
    model_cfg_path = os.path.join(os.path.dirname(__file__), "outputs/emulator_saved/emulator_PCA+GPR_lowice_test.joblib")
    forcing_cfg = os.path.join(os.path.dirname(__file__), "forcing.yaml")
    prediction = run_prediction(model_cfg=model_cfg_path,
                                forcing_cfg=forcing_cfg,
                                scenario="rcp85.1",
                                output_dir="examples/outputs/prediction/")
