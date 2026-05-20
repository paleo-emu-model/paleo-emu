"""
Example: train a PCA+GP emulator and predict under scenario forcing
===================================================================
Edit ``example_PCA_GP.yml`` to change training data, forcing scenarios,
encoder/regressor settings, or output paths — no code changes needed.

To skip training and use an existing artifact, set RUN_TRAINING = False.
"""

from pathlib import Path

import joblib
import numpy as np

from paleo_emu.config import load_config
from paleo_emu.load import load_training_data, load_forcing_data
from paleo_emu.training import TrainingGenerator
from paleo_emu.export import save_prediction

here = Path(__file__).resolve().parent
cfg  = load_config(str(here / "example_PCA_GP.yml"))

RUN_TRAINING = True   # set to False to skip training and load existing artifact

# ===========================================================
# PART 1 — TRAIN
# ===========================================================
if RUN_TRAINING:
    X, Y, _, _, lat_array, lon_array, _ = load_training_data(cfg)
    training      = TrainingGenerator(cfg, X, Y, lat_array, lon_array)
    artifact_path = training.run_training()
    print(f"[TRAIN] artifact saved → {artifact_path}")

# ===========================================================
# PART 2 — PREDICT
# ===========================================================
artifact_path = here / "pretrained" / cfg.artifact_name
artifact      = joblib.load(artifact_path)
model         = artifact["model"]
lat_array     = artifact["lat_array"]
lon_array     = artifact["lon_array"]
n_lat, n_lon  = len(lat_array), len(lon_array)

output_dir = here / "outputs"
for scenario in cfg.forcing_data:
    X_forcing     = load_forcing_data(cfg, scenario=scenario)
    Y_pred, Y_std = model.predict_with_variance(X_forcing)
    Y_pred_3d     = Y_pred.reshape(-1, n_lat, n_lon)
    Y_var_3d      = (Y_std ** 2).reshape(-1, n_lat, n_lon) if Y_std is not None else np.zeros_like(Y_pred_3d)
    save_prediction(Y_pred_3d, Y_var_3d, lat_array, lon_array,
                    output_dir=str(output_dir),
                    file_name=f"example_{scenario}_prediction")
    print(f"[PREDICT] {scenario} → {output_dir}/example_{scenario}_prediction.nc")
