"""
This module is to run predictions using the trained pipeline and PCA model.
parameters:
    pipeline: trained sklearn Pipeline containing the regressor, : fitted PCA model
    forcing_cfg: configuration for loading forcing data
    other info needed: decoder, std_val, and mean_val - are all info from encoder.
    spatial_shape: (lat, lon) of the original spatial structure
return:
    Y_out: simulated prediction results, shape (n_samples, lat, lon)
"""

from paleo_emu import encoder
from paleo_emu.load import load_forcing_data
import numpy as np

def run_prediction(pipeline, decoder, forcing_cfg, spatial_shape):
    # Check if forcing_cfg is a path to an ASCII data file or a data object
    if isinstance(forcing_cfg, dict):
        # Assume it's a file path
        X_pred = load_forcing_data(forcing_cfg)
    else:
        # If it's already data, use it directly
        X_pred = forcing_cfg
    Y_pred_encoded = pipeline.predict(X_pred)
    if encoder == "PCA":
        Y_full = decoder.inverse_transform(Y_pred_encoded)
        Y_full = Y_full * std_val + mean_val
    elif encoder == "VAE":
        Y_full = decoder.decoder.predict(Y_pred_encoded)
        Y_full = Y_full * std_val + mean_val
    else:
        Y_full = Y_pred_encoded

    n = Y_full.shape[0]
    lat, lon = spatial_shape
    Y_out = Y_full.reshape(n, lat, lon)
    return Y_out

    