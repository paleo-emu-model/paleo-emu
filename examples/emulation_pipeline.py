"""
This script is used to run the emulator training pipeline for the Paleo-Emu project.
"""



# from src.optimise import full_emulator_experiment
import os
from paleo_emu.training import run_training_leave_one_out

# define emulator training data
emulator = "lowmod_ice"
forcing = "rcp85.1"

seeds = 2025

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
emulator = run_training(train_dict[emulator],model_type="GPR",kernel="Matern_2.5_White",encoder="PCA", vae_config=vae_config, seed=seeds, return_validation=True)

