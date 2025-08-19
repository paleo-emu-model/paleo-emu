"""
This module is to run predictions using the trained pipeline and PCA model.
parameters:
    pipeline: trained sklearn Pipeline
    decoder: fitted PCA model
    forcing_cfg: configuration for loading forcing data
    spatial_shape: (lat, lon) of the original spatial structure
return:
    Y_out: simulated prediction results, shape (n_samples, lat, lon)
"""

from paleo_emu.load import load_forcing_data

def run_prediction(pipeline, decoder, forcing_cfg, spatial_shape):
    # used module to load forcing data
    X_pred = load_forcing_data(forcing_cfg)
    Y_pca_pred = pipeline.predict(X_pred)
    Y_full = decoder.inverse_transform(Y_pca_pred)
    n = Y_full.shape[0]
    lat, lon = spatial_shape
    Y_out = Y_full.reshape(n, lat, lon)
    return Y_out