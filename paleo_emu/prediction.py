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
import os


def run_prediction(model_name, forcing_cfg, output_dir, pipeline_path="examples/outputs/emulator_saved/", decoder_path="examples/outputs/emulator_saved/", decoder_name=None):
    """
    emulator: dict or object. If dict, may contain keys 'pipeline_model' and 'decoder' pointing to joblib files.
    pipeline_path / decoder_path: optional explicit paths to pipeline.joblib and decoder.joblib (take precedence).
    """
    X_pred = load_forcing_data(forcing_cfg) if isinstance(forcing_cfg, dict) else forcing_cfg

    pipeline_joblib_name = os.path.join(pipeline_path, model_name)

    if not os.path.exists(pipeline_joblib_name):
        raise FileNotFoundError(f"Pipeline file not found: {pipeline_joblib_name}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pipeline = joblib.load(pipeline_joblib_name)
    model = pipeline["model"]
    decoder = pipeline["decoder"]
    meta = pipeline.get("meta")

    # Normalize decoder input: accept either a dict with metadata or a raw PCA object
    pca_obj = None
    dec = {}
    if isinstance(decoder, dict):
        dec = decoder.copy()
    else:
        # treat sklearn PCA (or PCA-like) object as decoder
        if hasattr(decoder, "inverse_transform") and hasattr(decoder, "components_"):
            pca_obj = decoder
            dec["encoder"] = "PCA"
            # expose components_ and mean_ if present for fallback inverse
            dec["components_"] = getattr(decoder, "components_", None)
            dec["mean_"] = getattr(decoder, "mean_", None)
        else:
            raise ValueError("Unsupported decoder object loaded; expected dict or PCA-like object.")

    mean_val = meta("mean_val")
    std_val = meta("std_val")
    # spatial_shape = meta("spatial_shape")
    lat_array = meta("lat_array")
    lon_array = meta("lon_array")
    enc_type = meta("encoder")

    # if spatial_shape is None:
    #     raise ValueError("spatial_shape not found in decoder or emulator; cannot reshape predictions.")
    lat, lon = lat_array.shape[0], lon_array.shape[0]
    # mean_val/std_val are required to rescale back to raw units
    if mean_val is None or std_val is None:
        raise ValueError("mean_val or std_val missing in decoder/emulator; needed to rescale predictions.")

    # 拆分 model
    final_est = model
    X_enc = X_pred
    if hasattr(model, "steps"):
        steps = model.steps
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
        Y_pred_encoded = model.predict(X_pred)
    else:
        Y_pred_encoded = mean_encoded  # 使用 GPR 均值

    # 解码: support PCA dict, PCA object, or VAE decoder
    if enc_type == "PCA":
        if pca_obj is not None:
            Y_std_flat = pca_obj.inverse_transform(Y_pred_encoded)
        else:
            # decoder dict may contain a saved pca under key 'pca_obj'
            if isinstance(decoder, dict) and decoder.get("pca_obj") is not None:
                Y_std_flat = decoder["pca_obj"].inverse_transform(Y_pred_encoded)
            elif isinstance(decoder, dict) and ("components_" in decoder and "mean_" in decoder):
                comps = decoder["components_"]
                mean_ = decoder["mean_"]
                Y_std_flat = (Y_pred_encoded @ comps) + mean_
            else:
                raise ValueError("No PCA object or components_/mean_ found in decoder dict; cannot inverse_transform.")
    elif enc_type == "VAE":
        # expect decoder dict with key 'decoder' (keras model) or decoder object with .decoder
        if isinstance(decoder, dict) and decoder.get("decoder") is not None:
            Y_std_flat = decoder["decoder"].predict(Y_pred_encoded)
        elif hasattr(decoder, "decoder") and decoder.decoder is not None:
            Y_std_flat = decoder.decoder.predict(Y_pred_encoded)
        else:
            raise ValueError("VAE decoder not found in decoder object/dict.")
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
        file_name=f"{enc_type}_{emulator.get('regressor_type', 'reg')}_{os.path.basename(str(forcing_cfg))}_prediction"
    )
    return Y_out, Var_out