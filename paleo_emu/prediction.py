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

from paleo_emu.load import load_forcing_data
from paleo_emu.export import save_prediction
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.multioutput import MultiOutputRegressor
import numpy as np
import joblib

def run_prediction(emulator, forcing_cfg, output_dir):
    X_pred = load_forcing_data(forcing_cfg) if isinstance(forcing_cfg, dict) else forcing_cfg

    pipeline = joblib.load(emulator["pipeline_model"])
    decoder  = joblib.load(emulator["decoder"])
    mean_val = emulator["mean_val"]
    std_val  = emulator["std_val"]
    lat, lon = emulator["spatial_shape"]
    lat_array = emulator["lat_array"]
    lon_array = emulator["lon_array"]
    enc_type  = emulator["encoder"]

    # 拆分 pipeline
    final_est = pipeline
    X_enc = X_pred
    if hasattr(pipeline, "steps"):
        steps = pipeline.steps
        if len(steps) > 1:
            for _, step in steps[:-1]:
                X_enc = step.transform(X_enc)
        final_est = steps[-1][1]

    mean_encoded = None
    var_encoded  = None
    if enc_type == "PCA":
        if isinstance(final_est, GaussianProcessRegressor):
            m, s = final_est.predict(X_enc, return_std=True)
            if m.ndim == 1:  # 单输出
                m = m[:, None]; s = s[:, None]
            mean_encoded = m
            var_encoded  = s**2
        elif isinstance(final_est, MultiOutputRegressor):
            means = []
            vars_ = []
            ok = True
            for est in final_est.estimators_:
                if not isinstance(est, GaussianProcessRegressor):
                    ok = False
                    break
                m, s = est.predict(X_enc, return_std=True)
                means.append(m); vars_.append(s**2)
            if ok:
                mean_encoded = np.stack(means, axis=1)  # (n,k)
                var_encoded  = np.stack(vars_, axis=1)

    # 若没有方差信息则正常预测均值
    if mean_encoded is None:
        Y_pred_encoded = pipeline.predict(X_pred)
    else:
        Y_pred_encoded = mean_encoded  # 使用 GPR 均值

    # 解码
    if enc_type == "PCA":
        Y_std_flat = decoder.inverse_transform(Y_pred_encoded)
    elif enc_type == "VAE":
        Y_std_flat = decoder.decoder.predict(Y_pred_encoded)
    else:
        Y_std_flat = Y_pred_encoded

    Y_raw_flat = Y_std_flat * std_val + mean_val
    n = Y_raw_flat.shape[0]
    Y_out = Y_raw_flat.reshape(n, lat, lon)

    # 方差传播
    if (var_encoded is not None) and enc_type == "PCA":
        k = var_encoded.shape[1]
        comps = decoder.components_[:k]      # (k,D)
        W2 = comps**2
        var_std_flat_all = var_encoded @ W2  # (n,D)
        var_raw_flat_all = var_std_flat_all * (std_val**2)
        Var_out = var_raw_flat_all.reshape(n, lat, lon)
    else:
        Var_out = np.full((n, lat, lon), np.nan)

    save_prediction(
        Y_out,
        Var_out,
        lat_array,
        lon_array,
        output_dir,
        file_name=f"{enc_type}_{emulator['regressor_type']}_prediction"
    )
    return Y_out, Var_out