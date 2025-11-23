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
import pandas as pd


def run_prediction(model_cfg=None,forcing_cfg_path=None, scenario=None, output_dir=None):
    """
    emulator: dict or object. If dict, may contain keys 'pipeline_model' and 'decoder' pointing to joblib files.
    pipeline_path / decoder_path: optional explicit paths to pipeline.joblib and decoder.joblib (take precedence).
    """
    # load model config
    model_cfg = joblib.load(model_cfg) if isinstance(model_cfg, str) else model_cfg
    model_pipeline = model_cfg["model"]
    decoder = model_cfg["decoder"]
    mean_val = model_cfg.get("meta", {}).get("mean_val")
    std_val = model_cfg.get("meta", {}).get("std_val")
    lat_array = model_cfg.get("meta", {}).get("lat_array")
    lon_array = model_cfg.get("meta", {}).get("lon_array")
    encoder_type = model_cfg.get("meta", {}).get("encoder")
    regressor_type = model_cfg.get("meta", {}).get("regressor_type", "reg")
    residual_variance = model_cfg.get("meta", {}).get("residual_variance", None)
    mean_val = np.array(mean_val)
    std_val = np.array(std_val)
    lat_array = np.array(lat_array)
    lon_array = np.array(lon_array)

    X_pred = load_forcing_data(forcing_cfg_path, scenario=scenario)

    model = model_pipeline
    X_test_raw = X_pred
    X_test = np.asarray(X_test_raw, dtype=float)
    mean_val = np.asarray(mean_val, dtype=float)
    std_val = np.asarray(std_val, dtype=float)


    # **************************************************
    # calculate variance of Y_test_encoded if PCA is used
    # **************************************************
    # cannot return variance directly from pipeline.predict
    # because it is a MultiOutputRegressor wrapper instead of a single GPR model
    var_encoded = None
    mean_encoded = None
    if encoder_type == "PCA":
        if hasattr(model_pipeline, "named_steps"):
            scaler = model_pipeline.named_steps.get("scaler", None)
            reg_wrap = model_pipeline.named_steps.get("regressor", model_pipeline)
        else:
           scaler = None
           reg_wrap = model_pipeline
        X_feat = scaler.transform(X_test) if scaler is not None else X_test
        # get variance for each PCA component
        if isinstance(reg_wrap, MultiOutputRegressor):
            ests = reg_wrap.estimators_
            if all(isinstance(e, GaussianProcessRegressor) for e in ests):
                vars_ = []
                means_ = []
                for e in ests:
                    m, s = e.predict(X_feat, return_std=True)
                    means_.append(m)
                    vars_.append(s**2)
                mean_encoded = np.stack(means_, axis=1)  # (n, k)
                var_encoded  = np.stack(vars_, axis=1)   # (n, k)
                # DIAGNOSTIC: print per-PC predicted encoded stats and kernel/noise
                try:
                    import math
                    print("[DIAG] mean_encoded.shape:", mean_encoded.shape, "var_encoded.shape:", var_encoded.shape)
                    print("[DIAG] mean_encoded mean/std (first 10 PCs):",
                          np.mean(mean_encoded, axis=0)[:10],
                          np.std(mean_encoded, axis=0)[:10])
                    print("[DIAG] var_encoded mean (first 10 PCs):", np.mean(var_encoded, axis=0)[:10])
                    for idx, e in enumerate(ests[:10]):
                        kstr = str(getattr(e, "kernel_", getattr(e, "kernel", None)))
                        lml = getattr(e, "log_marginal_likelihood_value_", None)
                        print(f"[DIAG] estimator {idx}: lml={lml}, kernel_summary={kstr[:120]}")
                except Exception as _diag_e:
                    print("[DIAG] failed to print per-PC diagnostics:", _diag_e)
        elif isinstance(reg_wrap, GaussianProcessRegressor):
            m, s = reg_wrap.predict(X_feat, return_std=True)
            if m.ndim == 1: m = m[:, None]; s = s[:, None]
            mean_encoded = m
            var_encoded  = s**2

    if mean_encoded is not None:
        Y_pred_encoded = mean_encoded
    else:
        # Try to present the same feature-name layout used in training to the LGBM estimators:
        # If any underlying estimator has booster_.feature_name(), build a DataFrame with those names.
        X_for_pred = X_test
        try:
            reg = None
            if hasattr(model_pipeline, "named_steps"):
                reg = model_pipeline.named_steps.get("regressor", model_pipeline)
            else:
                reg = model_pipeline
            # locate underlying estimators (MultiOutputRegressor / custom MultiEstimator / single estimator)
            ests = getattr(reg, "estimators_", None) or getattr(reg, "estimators", None) or [reg]
            feat_names = None
            for e in ests:
                b = getattr(e, "booster_", None)
                if b is not None:
                    try:
                        feat_names = b.feature_name()
                        if feat_names:
                            break
                    except Exception:
                        feat_names = None
            if feat_names is not None:
                # if original X was DataFrame and contains the same names, reuse subset; else build DataFrame
                if isinstance(X_test_raw, pd.DataFrame):
                    if all(fn in X_test_raw.columns for fn in feat_names):
                        X_for_pred = X_test_raw.loc[:, feat_names]
                    else:
                        X_for_pred = pd.DataFrame(X_test, columns=feat_names)
                else:
                    X_for_pred = pd.DataFrame(X_test, columns=feat_names)
        except Exception:
            X_for_pred = X_test

        Y_pred_encoded = model_pipeline.predict(X_for_pred)


    # decode Y
    if encoder_type == "PCA":
        Y_pred_std = decoder.inverse_transform(Y_pred_encoded)
    elif encoder_type == "VAE":
        Y_pred_std = decoder.decoder.predict(Y_pred_encoded)
    else:
        Y_pred_std = Y_pred_encoded

    # 正确的反标准化：original = scaled * std + mean
    Y_pred_full = Y_pred_std * std_val + mean_val

    n = Y_pred_full.shape[0]
    lat, lon = lat_array.shape[0], lon_array.shape[0]
    Y_pred_out = Y_pred_full.reshape(n, lat, lon)

    # decode variance if PCA is used
    if (var_encoded is not None) and encoder_type == "PCA":
        k = var_encoded.shape[1]
        comps = decoder.components_[:k]          # (k, D)
        W2 = comps**2                            # (k, D)
        var_std_flat_all = var_encoded @ W2      # (n, D)
        if residual_variance is not None:
            # residual_variance 需与 var_std_flat_all 相加（而非覆盖）
            rv = residual_variance
            if np.isscalar(rv):
                var_std_flat_all = var_std_flat_all + rv
            else:
                try:
                    var_std_flat_all = var_std_flat_all + np.broadcast_to(np.asarray(rv), var_std_flat_all.shape)
                except Exception:
                    var_std_flat_all = var_std_flat_all + np.mean(rv)
        var_raw_flat_all = var_std_flat_all * (std_val**2)
        Y_var_out = var_raw_flat_all.reshape(n, lat, lon)
    else:
        Y_var_out = np.full((n, lat, lon), np.nan)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    save_prediction(
        Y_pred_out,
        Y_var_out,
        lat_array,
        lon_array,
        output_dir,
        file_name=f"{encoder_type}_{regressor_type}_test_prediction"
    )
    return Y_pred_out, Y_var_out