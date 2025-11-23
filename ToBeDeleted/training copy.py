"""
This module is to train models using chosen regressors, kernels, and encoders.
2 methods are used here: 2:8 validation; leave-one-out cross-validation.
leave_one_out has a recurring loop which needs i to be looped, so need to write another function for it.
2:8 validation doesn't require a function for looping, so it will only give one pipeline fitted model

procedures of training:
1. load data
2. split data
3. encode training data (giving decoder in the mean time)
4. process test Y for validation later
5. fit model (pipline)
6. validation -> predict using test X
              -> compare with test Y
"""
# training process needs to give info like pipeline contains 
# the trained model, decoder, std_val, and mean_val, which are used in the following prediction process


from tabnanny import verbose
import numpy as np
import xarray as xr
import os
import time

import tensorflow as tf

from sklearn.model_selection import cross_val_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.multioutput import MultiOutputRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import joblib
import yaml
from pathlib import Path

#from docs.source.auto_examples.plot_pipeline import X_test
from paleo_emu.load import load_training_data
from paleo_emu.regressor import build_regressor
from paleo_emu.plotting import plot_r2_map_with_latlon, plot_histogram_4_leave1out
from paleo_emu.validation import compute_r2_map
from paleo_emu.encoders import EncoderGenerator  


def run_training(cfg_path, X_train=None, Y_train=None):
    # training for given data
    """
    X_training: (n_samples, 5) the input feature matrix
    Y_training: (n_samples, lat*lon) the flattened output matrix
    """
    # Load configuration from YAML file
    if isinstance(cfg_path, (str, Path)):
        with open(cfg_path, "r") as fh:
            cfg = yaml.safe_load(fh)
    else:
        cfg = cfg_path

    try: 
        regressor_type = cfg.regressor_config.regressor_type 
    except Exception as e: 
        raise ValueError("regressor_type not found in cfg") from e
    
    try: 
        encoder = cfg.encoder_config.encoder_type
    except Exception as e: 
        raise ValueError("encoder not found in cfg") from e
    
    try: 
        save_path = cfg.save_path 
    except Exception as e: 
        raise ValueError("save_path not found in cfg") from e
    
    try:
        save_name = cfg.save_name
    except Exception as e:
        raise ValueError("save_name not found in cfg") from e

    try:
        save_pipeline = cfg.save_pipeline
    except Exception as e:
        save_pipeline = True

    if X_train is None or Y_train is None:
        print("X_train or Y_train is None, loading training data from cfg...")
        X_train, Y_train, _, _, lat_array, lon_array = load_training_data(cfg_path)

    enc = EncoderGenerator(Y_train, cfg)

    Y_train_encoded, decoder, mean_val, std_val = enc.generate_encoder()

    
    latent_dim = Y_train_encoded.shape[1]

    print(f"[DIAG] Y_train_encoded shape: {Y_train_encoded.shape}")
    print("[DIAG] Y_train_encoded per-PC mean/std (first 10):")
    for pc_idx in range(min(10, latent_dim)):
        pc_data = Y_train_encoded[:, pc_idx]
        print(f"  PC{pc_idx}: mean={np.mean(pc_data):.4e}, std={np.std(pc_data):.4e}")

    # regressor = build_regressor(
    #         cfg=cfg,
    #         verbose=verbose
    # )

    if model_config.regressir_config,regressor_type == "GPR":   
        regressor = GaussianProcessRegressor()

    reg_step = MultiOutputRegressor(regressor)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", reg_step)
    ])
    model.fit(X_train, Y_train_encoded)


    # Ensure save dir exists before writing files
    os.makedirs(save_path, exist_ok=True)
    model_joblib_name = os.path.join(save_path, f"{save_name}.joblib")

    meta = {
        "pipeline_path": model_joblib_name,
        "encoder": encoder,
        "regressor_type": regressor_type,
        "n_components_retained": int(latent_dim),
        "mean_val": mean_val.tolist(),
        "std_val": std_val.tolist(),
        "lat_array": lat_array.tolist(),
        "lon_array": lon_array.tolist()
    }

    data_to_save = {
    "model": model,
    "decoder": decoder,
    "meta": meta
    }
   # 一次性保存到文件
    joblib.dump(data_to_save, model_joblib_name)
    
    if save_pipeline:
        # Save metadata as a YAML file (meta already converted to native types)
        meta_path = os.path.join(save_path, f"{save_name}.yaml")
        with open(meta_path, "w") as fh:
            yaml.safe_dump(meta, fh)
        print(f"[INFO] Metadata saved to {meta_path}")

    return {
        "trained_pipeline": model_joblib_name,
        "decoder": model_joblib_name,
        "encoder": encoder,
        "mean_val": mean_val,
        "std_val": std_val, 
        "lat_array": lat_array,
        "lon_array": lon_array,
        "spatial_shape": (len(lat_array), len(lon_array)),
        "residual_variance": residual_variance,
        "n_components_retained": latent_dim,
        "regressor_type": regressor_type}

def return_validation_function(X_test, Y_true_flat, trained_pipeline, decoder, mean_val, std_val, spatial_shape, encoder, residual_variance):
    """
        1. encode, decode Y_test
        2. predict Y_test_predicted using trained_pipeline
        return:
            Y_pred_out: predicted Y, shape (n_samples, lat, lon)
            Y_true_out: true Y, shape (n_samples, lat, lon)
            Y_var_out: variance of predicted Y, shape (n_samples, lat, lon) - only for PCA
            r2_score: R² score of the prediction
            rmse: RMSE of the prediction
    """
    import pandas as pd
    # normalize inputs and ensure numeric arrays
    # keep original X_test (may be DataFrame) for possible feature-name reconstruction,
    # but also keep ndarray view used for numerical ops
    X_test_raw = X_test
    X_test = np.asarray(X_test_raw)
    Y_true_flat = np.asarray(Y_true_flat, dtype=float)
    mean_val = np.asarray(mean_val, dtype=float)
    std_val = np.asarray(std_val, dtype=float)

    # Actually no need to encode and decode Y_true here,
    # this extra processing steps is to ensure the consistency of the processing for Y
    # -------
    Y_true_scaled = (Y_true_flat - mean_val) / std_val
    if encoder == "PCA":
        Y_true_encoded = decoder.transform(Y_true_scaled)
    elif encoder == "VAE":
        mean_logvar = decoder.encoder.predict(Y_true_scaled)
        mean, logvar = tf.split(mean_logvar, 2, axis=1)
        Y_true_encoded = mean.numpy()
    else:
        Y_true_encoded = Y_true_flat

    # **************************************************
    # calculate variance of Y_test_encoded if PCA is used
    # **************************************************
    # cannot return variance directly from pipeline.predict
    # because it is a MultiOutputRegressor wrapper instead of a single GPR model
    var_encoded = None
    mean_encoded = None
    if encoder == "PCA":
        if hasattr(trained_pipeline, "named_steps"):
            scaler = trained_pipeline.named_steps.get("scaler", None)
            reg_wrap = trained_pipeline.named_steps.get("regressor", trained_pipeline)
        else:
           scaler = None
           reg_wrap = trained_pipeline
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
            if hasattr(trained_pipeline, "named_steps"):
                reg = trained_pipeline.named_steps.get("regressor", trained_pipeline)
            else:
                reg = trained_pipeline
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

        Y_pred_encoded = trained_pipeline.predict(X_for_pred)
    # DIAGNOSTIC: check encoded → decoded → unscale chain
    try:
        print("[DIAG] Y_pred_encoded shape/mean/std:", getattr(Y_pred_encoded, 'shape', None), np.mean(Y_pred_encoded), np.std(Y_pred_encoded))
        from sklearn.metrics import mean_squared_error
        encoded_rmse = np.sqrt(mean_squared_error(Y_true_encoded, Y_pred_encoded))
        print(f"[DIAG] Encoded-space RMSE: {encoded_rmse:.3f}")
    except Exception:
        pass

    # ---------- 插入：逐 PC 重建误差贡献 & 比例 ----------
    try:
        if encoder == "PCA" and ('Y_true_encoded' in locals()) and (Y_pred_encoded is not None):
            y_true_enc = np.asarray(Y_true_encoded).reshape(1, -1)
            y_pred_enc = np.asarray(Y_pred_encoded).reshape(1, -1)
            # full decoded (std-space -> then unscale below)
            dec_true_std = decoder.inverse_transform(y_true_enc)
            dec_pred_std = decoder.inverse_transform(y_pred_enc)
            dec_true = dec_true_std * std_val + mean_val
            dec_pred = dec_pred_std * std_val + mean_val
            total_spatial_rmse = np.sqrt(np.mean((dec_pred - dec_true)**2))
            # per-PC contributions: apply only the difference on each PC and decode
            npc = y_true_enc.shape[1]
            per_pc_rmse = []
            for pc in range(npc):
                delta_pc = np.zeros_like(y_true_enc)
                delta_pc[0, pc] = (y_pred_enc - y_true_enc)[0, pc]
                dec_delta_std = decoder.inverse_transform(delta_pc)   # std-space change
                dec_delta = dec_delta_std * std_val                    # unscaled change
                rmse_pc = np.sqrt(np.mean(dec_delta**2))
                per_pc_rmse.append(rmse_pc)
            per_pc_rmse = np.array(per_pc_rmse)
            frac = per_pc_rmse / (per_pc_rmse.sum() + 1e-12)
            print("[DIAG-PC] total spatial RMSE from encoded error:", float(total_spatial_rmse))
            print("[DIAG-PC] per-PC RMSE:", np.round(per_pc_rmse, 6))
            print("[DIAG-PC] per-PC RMSE fraction:", np.round(frac, 4))
    except Exception as _e:
        print("[DIAG-PC] per-PC contribution diag failed:", _e)
    # ---------- end 插入 ----------

    # decode Y
    if encoder == "PCA":
        Y_pred_std = decoder.inverse_transform(Y_pred_encoded)
        Y_true_std = decoder.inverse_transform(Y_true_encoded)
    elif encoder == "VAE":
        Y_pred_std = decoder.decoder.predict(Y_pred_encoded)
        Y_true_std = decoder.decoder.predict(Y_true_encoded)
    else:
        Y_pred_std = Y_pred_encoded
        Y_true_std = Y_true_encoded

    # 正确的反标准化：original = scaled * std + mean
    Y_pred_full = Y_pred_std * std_val + mean_val
    Y_true_full = Y_true_std * std_val + mean_val

    n = Y_pred_full.shape[0]
    lat, lon = spatial_shape
    Y_pred_out = Y_pred_full.reshape(n, lat, lon)
    Y_true_out = Y_true_full.reshape(n, lat, lon) 
    # explicit RMSE to avoid external dependency mismatch
    rmse = float(np.sqrt(np.mean((Y_true_full - Y_pred_full)**2)))

    # decode variance if PCA is used
    if (var_encoded is not None) and encoder == "PCA":
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

    r2_value = r2_score(Y_true_full, Y_pred_full)
    print("[DIAG] Y_true_flat global mean:", float(np.mean(Y_true_full)))
    print("[DIAG] Y_pred_flat global mean:", float(np.mean(Y_pred_full)))
    print("[DIAG] bias (mean error):", float(np.mean(Y_pred_full - Y_true_full)))
    print("[DIAG] rmse:", rmse)

    return {"Y_pred_out": Y_pred_out,
            "Y_true_out": Y_true_out,
            "Y_var_out": Y_var_out,
            "r2_score": r2_value,
            "rmse": rmse}
