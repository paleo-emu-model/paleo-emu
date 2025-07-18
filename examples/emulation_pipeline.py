
# from src.optimise import full_emulator_experiment
import os
from paleo_emu.training_leave1out import run_training

emulator = "lowmod_ice"
forcing = "rcp85.1"

seeds = 2025

vae_config = {
    "latent_dim": 1024, # 32, 64, 128, 256，512， 1024
    "epochs": 80,
    "learning_rate": 1e-4, # 1e-4, 5e-5, 1e-5
    "batch_size": 128,
    "kl_weight": 0.1 # 0.1, 0.5, 1.0
}

file_path = os.path.join(".", "examples", "training_data")

# config_dict.py
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

# ===单次训练===
emulator = run_training(train_dict[emulator],model_type="GPR",kernel="Matern_2.5_White",encoder="VAE", vae_config=vae_config, seed=seeds, return_pred=True)

