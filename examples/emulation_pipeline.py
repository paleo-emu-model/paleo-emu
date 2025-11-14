
"""
This script is used to run the emulator training pipeline for the Paleo-Emu project.
"""

# from src.optimise import full_emulator_experiment
import os,yaml
from paleo_emu.training import run_training_LGBM_optimization
from paleo_emu.training import run_training_10fold
from paleo_emu.training import run_training
from paleo_emu.prediction import run_prediction
from joblib import dump
import joblib

# python
import numpy as np




cfg_path = os.path.join(os.path.dirname(__file__), "lowmod_ice_emulator.yaml")
with open(cfg_path, "r") as fh:
    cfg = yaml.safe_load(fh)

if __name__ == "__main__":
    # run emulator training
    # training is to get the best hyperparameters, or get the trained emulator for prediction
    # emulator = run_training_all(train_dict=train_dict[emulator],
    #                            regressor_type="GPR", kernel="RBF", encoder="PCA", vae_config=vae_config, fixed_hp=True)

    # emulator = run_training_10fold(cfg, 
    #                            regressor_type="GPR", 
    #                            encoder="PCA", 
    #                            fixed_encoder_hp=False, 
    #                            fixed_regressor_hp=False, 
    #                            return_validation=True)

    # emulator = run_training_leave_one_out(cfg_path=cfg_path,
    #                                     encoder="PCA", 
    #                                     regressor_type="LGBM",
    #                                     fixed_encoder_hp=False,
    #                                     fixed_regressor_hp=False)
# 

    emulatorPCAGPR = run_training(cfg_path,
                                X_train=None,
                                Y_train=None,
                                regressor_type="GPR", 
                                encoder="PCA", 
                                fixed_encoder_hp=False, 
                                save_path="examples/outputs/emulator_saved",
                                save_name="emulator_PCA+GPR_lowice",
                                save_pipeline=True)


# run emulator prediction
    full = joblib.load("examples/outputs/emulator_saved/emulator_PCA+GPR_lowice.joblib")
    pipeline_obj = full["pipeline"]
    decoder_obj = full["decoder"]
    meta = full.get("meta", {})

    emu = {
        "pipeline_model": pipeline_obj,
        "decoder": decoder_obj,
        "mean_val": meta.get("mean_val"),
        "std_val": meta.get("std_val"),
        "spatial_shape": meta.get("spatial_shape"),
        "encoder": meta.get("encoder"),
        "regressor_type": meta.get("regressor_type"),
        "lat_array": meta.get("lat_array"),
        "lon_array": meta.get("lon_array")
    }
    forcing_cfg = {
    "forcing_data": {
        "file_path": "../Emulator_Charlie/Results/",
        # load_forcing_data 默认使用 scenario="rcp85.1"，所以这里用这个名称或在 run_prediction 中传入其他 scenario
        "rcp85.1": {
            "forcing_input": "emul_inputs_RCP85.67.res"
        }
    }
    }
    prediction = run_prediction(model_name="emulator_PCA+GPR_lowice.joblib",
                                forcing_cfg=forcing_cfg, 
                                output_dir="examples/outputs/prediction/",
                                model_path="examples/outputs/emulator_saved/")

    # training = run_training_LGBM_optimization(cfg_path=cfg_path,
    #                                     encoder="PCA", 
    #                                     regressor="LGBM",
    #                                     fixed_encoder_hp=True,
    #                                     fixed_regressor_hp=False)

    # training = run_training_GPR_optimization(cfg_path=cfg_path,
    #                                     encoder="PCA", 
    #                                     regressor="GPR",
    #                                     fixed_encoder_hp=True,
    #                                     fixed_regressor_hp=False,
    #                                     do_leaveoneout=True)
    # # Save the best model to a file
    # dump(training, "pipeline.joblib")