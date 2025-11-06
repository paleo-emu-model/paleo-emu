
"""
This script is used to run the emulator training pipeline for the Paleo-Emu project.
"""

# from src.optimise import full_emulator_experiment
import os,yaml
from paleo_emu.training import run_training_GPR_optimization
from paleo_emu.training import run_training_10fold
from paleo_emu.prediction import run_prediction

cfg_path = os.path.join(os.path.dirname(__file__), "highlowmod_ice_emulator.yaml")
with open(cfg_path, "r") as fh:
    cfg = yaml.safe_load(fh)

if __name__ == "__main__":
    # run emulator training
    # training is to get the best hyperparameters, or get the trained emulator for prediction
    # emulator = run_training_all(train_dict=train_dict[emulator],
    #                            regressor_type="GPR", kernel="RBF", encoder="PCA", vae_config=vae_config, fixed_hp=True)

    # emulator = run_training_leave_one_out(cfg,
    #                                     regressor_type="GPR", 
    #                                     encoder="PCA", 
    #                                     fixed_encoder_hp=True,
    #                                     fixed_regressor_hp=True)


    emulator = run_training_10fold(cfg,
                                        regressor_type="GPR", 
                                        encoder="PCA", 
                                        fixed_encoder_hp=True,
                                        fixed_regressor_hp=True)

    # prediction = run_prediction(emulator=emulator, forcing_cfg="rcp85.1",fixed_hp=True, output_dir="examples/outputs/prediction/")

    # training = run_training_LGBM_optimization(cfg_path=cfg_path,
    #                                     encoder="PCA", 
    #                                     regressor="LGBM",
    #                                     fixed_encoder_hp=True,
    #                                     fixed_regressor_hp=False)

    # training = run_training_GPR_optimization(cfg_path=cfg_path,
    #                                     encoder="PCA", 
    #                                     regressor="GPR",
    #                                     fixed_encoder_hp=True,
    #                                     fixed_regressor_hp=False)
    