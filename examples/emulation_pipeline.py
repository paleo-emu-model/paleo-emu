"""
This script is used to run the emulator training pipeline for the Paleo-Emu project.
"""

# from src.optimise import full_emulator_experiment
import os
from paleo_emu.training import run_training_all
from paleo_emu.prediction import run_prediction
# from paleo_emu.load import load_training_data

# define emulator training data
emulator = "highlowmod_ice"
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
forcing_data = {"file_path": "examples/forcing_data/",
                "forcing_input": "emul_inputs_RCP85.67.res"}

# run emulator training
emulator = run_training_all(train_dict[emulator], regressor_type="GPR", kernel="RBF", encoder="PCA", vae_config=vae_config, fixed_hp=True)

prediction = run_prediction(emulator, forcing_data, output_dir="examples/outputs/prediction/")

# from paleo_emu.optimise import optimize_hyperparameters
# hp_optimisation = optimize_hyperparameters(train_dict["highlowmod_ice"])