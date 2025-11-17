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
from paleo_emu.encoder import encode
from paleo_emu.regressor import build_regressor
from paleo_emu.plotting import plot_r2_map_with_latlon, plot_histogram_4_leave1out
from paleo_emu.validation import compute_r2_map

def run_training(cfg_path, X_train=None, Y_train=None, regressor_type=None, encoder=None,
                 fixed_regressor_hp=None, fixed_encoder_hp=None, save_path=None,
                 save_name=None, save_pipeline=None):
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

    regressor_type = regressor_type if regressor_type is not None else cfg.get("regressor_type", "GPR")
    encoder = encoder if encoder is not None else cfg.get("encoder", "PCA")
    fixed_regressor_hp = fixed_regressor_hp if fixed_regressor_hp is not None else cfg.get("fixed_regressor_hp", False)
    fixed_encoder_hp = fixed_encoder_hp if fixed_encoder_hp is not None else cfg.get("fixed_encoder_hp", True)
    save_path = save_path if save_path is not None else cfg.get("save_path", "examples/outputs/emulator_saved")
    save_name = save_name if save_name is not None else cfg.get("save_name", "emulator_model")
    save_pipeline = save_pipeline if save_pipeline is not None else cfg.get("save_pipeline", True)

    if X_train is None or Y_train is None:
        X_train, Y_train, _, _, lat_array, lon_array = load_training_data(cfg_path)

    Y_train_encoded, decoder, mean_val, std_val, residual_variance = encode(
        Y_train,
        encoder=encoder,
        fixed_encoder_hp=fixed_encoder_hp,
        cfg_path=cfg_path
    )
    latent_dim = Y_train_encoded.shape[1]

    print(f"[DIAG] Y_train_encoded shape: {Y_train_encoded.shape}")
    print("[DIAG] Y_train_encoded per-PC mean/std (first 10):")
    for pc_idx in range(min(10, latent_dim)):
        pc_data = Y_train_encoded[:, pc_idx]
        print(f"  PC{pc_idx}: mean={np.mean(pc_data):.4e}, std={np.std(pc_data):.4e}")

    regressor = build_regressor(
            cfg_path=cfg_path,
            regressor_type=regressor_type,
            fixed_regressor_hp=fixed_regressor_hp,
            verbose=verbose
    )

    reg_step = MultiOutputRegressor(regressor)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", reg_step)
    ])
    model.fit(X_train, Y_train_encoded)


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
    # Ensure save dir exists before writing files
    os.makedirs(save_path, exist_ok=True)
    model_joblib_name = os.path.join(save_path, f"{save_name}.joblib")
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

def run_training_all(train_dict,regressor_type="GPR", kernel="RBF_White", pca_variance_ratio=0.999, encoder="PCA", vae_config=None, fixed_hp=True):
    X_train, Y_train, var_name, spatial_shape, lat_array, lon_array = load_training_data(train_dict)
    training_info = run_training(X_train, Y_train, regressor_type=regressor_type, kernel=kernel, pca_variance_ratio=pca_variance_ratio, encoder=encoder, vae_config=vae_config,fixed_hp=fixed_hp)
    trained_pipeline, decoder, mean_val, std_val, n_components, residual_variance = training_info["trained_pipeline"], training_info["decoder"], training_info["mean_val"], training_info["std_val"], training_info["n_components_retained"], training_info["residual_variance"]
    return {
        "pipeline_model": trained_pipeline,
        "decoder": decoder,
        "encoder": encoder,
        "regressor_type": regressor_type,
        "mean_val": mean_val,
        "std_val": std_val,
        "n_components_retained": n_components,
        "original_variable": var_name,
        "spatial_shape": spatial_shape,
        "lat_array": lat_array,
        "lon_array": lon_array
    }


def run_training_10fold(cfg_path,
                        regressor_type="GPR",
                        encoder="PCA",
                        fixed_encoder_hp=True,
                        fixed_regressor_hp=True,
                        return_validation=True):
    """
    Perform 10-fold cross-validation training and validation.
    """
    from sklearn.model_selection import KFold

    if isinstance(cfg_path, (str, Path)):
        with open(cfg_path, "r") as fh:
            cfg = yaml.safe_load(fh)
    else:
        cfg = cfg_path
    emulator_name = cfg.get("emulators", "highlowmod_ice")

    # Load data
    X, Y_flat, var_name, spatial_shape, lat_array, lon_array = load_training_data(cfg_path)
    n_samples = X.shape[0]

    Y_pred_full = []
    Y_true_full = []
    rmse_full = []

    kf = KFold(n_splits=20, shuffle=True, random_state=42)

    for fold, (train_indices, test_indices) in enumerate(kf.split(X)):
        print(f"[INFO] Processing fold {fold + 1}/20...")
        if hasattr(X, "iloc"):
            X_train = X.iloc[train_indices]
            X_test  = X.iloc[test_indices]
        else:
            X_train = X[train_indices]
            X_test  = X[test_indices]

        # Y_flat is likely ndarray; index directly
        Y_train_flat = Y_flat[train_indices]
        Y_test_flat  = Y_flat[test_indices]
        # Train model
        training_info = run_training(
            cfg_path,
            X_train,
            Y_train_flat,
            regressor_type=regressor_type,
            encoder=encoder,
            fixed_encoder_hp=fixed_encoder_hp,
            fixed_regressor_hp=fixed_regressor_hp
        )

        trained_pipeline = joblib.load(training_info["trained_pipeline"])
        decoder = joblib.load(training_info["decoder"])
        mean_val = training_info["mean_val"]
        std_val = training_info["std_val"]
        residual_variance = training_info.get("residual_variance", None)

        # Validation
        validation_metrics = return_validation_function(
            X_test,
            Y_test_flat,
            trained_pipeline,
            decoder,
            mean_val,
            std_val,
            spatial_shape,
            encoder,
            residual_variance
        )

        Y_pred_out, Y_true_out, rmse = (
            validation_metrics["Y_pred_out"],
            validation_metrics["Y_true_out"],
            validation_metrics["rmse"]
        )

        Y_pred_full.extend(Y_pred_out)
        Y_true_full.extend(Y_true_out)
        rmse_full.append(rmse)

    # Stack and reshape results
    Y_pred_full = np.array(Y_pred_full)
    Y_true_full = np.array(Y_true_full)
    rmse_full = np.array(rmse_full)

    n = Y_pred_full.shape[0]
    Y_pred_out_full = Y_pred_full.reshape(n, lat_array.shape[0], lon_array.shape[0])
    Y_true_out_full = Y_true_full.reshape(n, lat_array.shape[0], lon_array.shape[0])

    # Compute overall R² score
    overall_r2 = r2_score(Y_true_out_full.flatten(), Y_pred_out_full.flatten())

    
    if return_validation:
        # Plotting for validation
        r2_map = compute_r2_map(Y_true_out_full, Y_pred_out_full, lat_array, lon_array)
        plot_r2_map_with_latlon(
            r2_map,
            lat_array=lat_array,
            lon_array=lon_array,
            regressor_type=regressor_type,
            kernel="RBF_White",
            encoder=encoder,
            save_dir="outputs/logs"
        )
        print(f"[INFO] Overall R² Score: {overall_r2:.4f}")
        # Save Y_true_out_full and Y_pred_out_full as NetCDF files
        output_dir = "examples/outputs/"
        os.makedirs(output_dir, exist_ok=True)
        y_pred_path = os.path.join(output_dir, "Y_pred_out_full_"+emulator_name+"_"+encoder+"+"+regressor_type+"_10fold.nc")
        y_true_path = os.path.join(output_dir, "Y_true_out_full_"+emulator_name+"_"+encoder+"+"+regressor_type+"_10fold.nc")

        # Remove files if they already exist
        if os.path.exists(y_pred_path):
            os.remove(y_pred_path)
        if os.path.exists(y_true_path):
            os.remove(y_true_path)

        xr.Dataset({
            "mean": (["time", "lat", "lon"], Y_true_out_full),
            "latitude": (["lat"], lat_array),
            "longitude": (["lon"], lon_array)
        }).to_netcdf(y_true_path)

        xr.Dataset({
            "mean": (["time", "lat", "lon"], Y_pred_out_full),
            "latitude": (["lat"], lat_array),
            "longitude": (["lon"], lon_array)
        }).to_netcdf(y_pred_path)

        print(f"[INFO] Y_pred_out_full saved to {y_pred_path}")
        print(f"[INFO] Y_true_out_full saved to {y_true_path}")

        # Visualize the performance
        plot_r2_map_with_latlon(r2_map, lat_array=lat_array, lon_array=lon_array,
                    regressor_type=regressor_type, encoder=encoder,
                    save_dir="examples/outputs/10fold/plots")

    return {
        "r2_score": overall_r2,
        "pipeline_model": training_info["trained_pipeline"],
        "decoder": training_info["decoder"],
        "encoder": encoder,
        "n_components_retained": training_info["n_components_retained"],
        "original_variable": var_name,
        "spatial_shape": spatial_shape,
        "Y_pred_out": Y_pred_out_full,
        "Y_true_out": Y_true_out_full,
        "rmse_per_fold": rmse_full,
        "regressor_type": regressor_type
    }


def run_training_leave_one_out(cfg_path,
                               regressor_type="GPR",
                               encoder="PCA",
                               fixed_encoder_hp=True,
                               fixed_regressor_hp=False,
                               return_validation=False):
    import warnings
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    # load cfg
    if isinstance(cfg_path, (str, Path)):
        with open(cfg_path, "r") as fh:
            cfg = yaml.safe_load(fh)
    else:
        cfg = cfg_path
    emulator_name = cfg.get("emulators", "highlowmod_ice")

    X, Y_flat, var_name, spatial_shape, lat, lon = load_training_data(cfg)
    n_samples = X.shape[0]

    Y_pred_full = []
    Y_true_full = []
    Y_var_full = []
    rmse_full = []

    time_start = time.time()
    batch_size = 1  # Number of samples to leave out each time

    # original behaviour: retrain for each leave-out
    for i in range(0, n_samples, batch_size):
        test_indices = np.arange(i, min(i + batch_size, n_samples))
        train_indices = np.setdiff1d(np.arange(n_samples), test_indices)

        if hasattr(X, "iloc"):
            X_train = X.iloc[train_indices]
            X_test  = X.iloc[test_indices]
        else:
            X_train = X[train_indices]
            X_test  = X[test_indices]
        Y_train_flat = Y_flat[train_indices]
        Y_test_flat  = Y_flat[test_indices]

        training_info = run_training(cfg_path,
                                     X_train,
                                     Y_train_flat,
                                     regressor_type=regressor_type,
                                     encoder=encoder,
                                     fixed_regressor_hp=fixed_regressor_hp,
                                     fixed_encoder_hp=fixed_encoder_hp)

        trained_pipeline_path = training_info["trained_pipeline"]
        trained_decoder_path = training_info["decoder"]
        mean_val = training_info["mean_val"]
        std_val = training_info["std_val"]
        residual_variance = training_info.get("residual_variance", None)
        n_components = training_info["n_components_retained"]

        trained_pipeline = joblib.load(trained_pipeline_path)
        trained_decoder = joblib.load(trained_decoder_path)

        validation_metrics = return_validation_function(
            X_test,
            Y_test_flat,
            trained_pipeline,
            trained_decoder,
            mean_val,
            std_val,
            spatial_shape,
            encoder,
            residual_variance
        )
        Y_pred_out, Y_true_out, Y_var_out, rmse = (validation_metrics["Y_pred_out"],
                                                  validation_metrics["Y_true_out"],
                                                  validation_metrics["Y_var_out"],
                                                  validation_metrics["rmse"])
        Y_pred_full.extend(Y_pred_out)
        Y_true_full.extend(Y_true_out)
        Y_var_full.extend(Y_var_out)
        rmse_full.append(rmse)
        time_spt = time.time() - time_start
        print(f"[TIME] {min(i + batch_size, n_samples)}/{n_samples} completed in {time_spt:.2f} seconds.")
        time_start = time.time()

    # stack and reshape
    Y_pred_full = np.array(Y_pred_full)
    Y_true_full = np.array(Y_true_full)
    Y_var_full = np.array(Y_var_full)
    rmse_full = np.array(rmse_full)

    n = Y_pred_full.shape[0]
    Y_pred_out_full = Y_pred_full.reshape(n, lat.shape[0], lon.shape[0])
    Y_true_out_full = Y_true_full.reshape(n, lat.shape[0], lon.shape[0])
    Y_var_out_full = Y_var_full.reshape(n, lat.shape[0], lon.shape[0]) if encoder == "PCA" else None

    # Save NetCDFs
    output_dir = cfg.get('output_dir', 'outputs/leave_one_out/')
    os.makedirs(output_dir, exist_ok=True)
    y_pred_path = os.path.join(output_dir, f"Y_pred_out_full_{emulator_name}_{encoder}+{regressor_type}.nc")
    y_true_path = os.path.join(output_dir, f"Y_true_out_full_{emulator_name}_{encoder}+{regressor_type}.nc")
    if os.path.exists(y_pred_path):
        os.remove(y_pred_path)
    if os.path.exists(y_true_path):
        os.remove(y_true_path)

    # debug global stats before saving
    try:
        print("[DIAG] SUMMARY before saving - Y_pred_out_full mean/min/max:", np.mean(Y_pred_out_full), np.min(Y_pred_out_full), np.max(Y_pred_out_full))
        print("[DIAG] SUMMARY before saving - Y_true_out_full mean/min/max:", np.mean(Y_true_out_full), np.min(Y_true_out_full), np.max(Y_true_out_full))
    except Exception:
        pass

    xr.Dataset({
            "mean": (["time", "lat", "lon"], Y_true_out_full),
            "latitude": (["lat"], lat),
            "longitude": (["lon"], lon)
        }).to_netcdf(y_true_path)

    if encoder == "PCA":
        xr.Dataset({
            "mean": (["time", "lat", "lon"], Y_pred_out_full),
            "variance": (["time", "lat", "lon"], Y_var_out_full),
            "latitude": (["lat"], lat),
            "longitude": (["lon"], lon)
        }).to_netcdf(y_pred_path)
    else:
        xr.DataArray(Y_pred_out_full, dims=["time", "lat", "lon"]).to_netcdf(y_pred_path)

    print(f"[INFO] Y_pred_out_full saved to {y_pred_path}")
    print(f"[INFO] Y_true_out_full saved to {y_true_path}")

    if return_validation:
        plot_histogram_4_leave1out(Y_true_out_full, Y_pred_out_full)

    time_spt = time.time() - time_start
    print(f"[TIME] plot completed in {time_spt:.2f} seconds.")

    try:
        overall_r2 = r2_score(Y_true_out_full.flatten(), Y_pred_out_full.flatten())
    except Exception:
        overall_r2 = None

    return {
        "r2_score": overall_r2,
        "n_components_retained": n_components,
        "original_variable": var_name,
        "spatial_shape": spatial_shape,
        "Y_pred_out": Y_pred_out_full,
        "Y_true_out": Y_true_out_full,
        "Y_var_out": Y_var_out_full,
        "encoder_used": encoder,
        "regressor_type": regressor_type
    }

def run_training_leave_one_out_old(cfg_path, 
                               regressor_type="GPR", 
                               encoder="PCA", 
                               fixed_encoder_hp=True, 
                               fixed_regressor_hp=False, 
                               return_validation=False):
    import warnings
    # Suppress ConvergenceWarning during the execution of this function
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # if caller passed a path, load YAML to dict; if already a dict, use it directly
    if isinstance(cfg_path, (str, Path)):
        with open(cfg_path, "r") as fh:
            cfg = yaml.safe_load(fh)
    else:
        cfg = cfg_path

    X, Y_flat, var_name, spatial_shape, lat, lon = load_training_data(cfg)
    n_samples = X.shape[0]

    Y_pred_full = []
    Y_true_full = []
    Y_var_full = []
    rmse_full = []

    time_start = time.time()

    batch_size = 1  # Number of samples to leave out each time

    for i in range(0, n_samples, batch_size):
        # split data
        # ----------
        test_indices = np.arange(i, min(i + batch_size, n_samples))
        train_indices = np.setdiff1d(np.arange(n_samples), test_indices)
        
        X_train = X.iloc[train_indices]
        Y_train_flat = Y_flat[train_indices]
        X_test = X.iloc[test_indices]
        Y_test_flat = Y_flat[test_indices]

        # fit model
        # ----------
        training_info = run_training(cfg_path,
                                     X_train,
                                     Y_train_flat, 
                                     regressor_type=regressor_type, 
                                     encoder=encoder,
                                     fixed_regressor_hp=fixed_regressor_hp,
                                     fixed_encoder_hp=fixed_encoder_hp)
        
        trained_pipeline = training_info["trained_pipeline"]
        decoder = training_info["decoder"]
        mean_val = training_info["mean_val"]
        std_val = training_info["std_val"]
        residual_variance = training_info["residual_variance"]
        n_components = training_info["n_components_retained"]

        trained_pipeline = joblib.load(trained_pipeline)
        decoder = joblib.load(decoder)
        
        validation_metrics = return_validation_function(X_test, 
                                                        Y_test_flat, 
                                                        trained_pipeline, 
                                                        decoder, 
                                                        mean_val, 
                                                        std_val, 
                                                        spatial_shape, 
                                                        encoder,
                                                        residual_variance
                                                        )
        Y_pred_out, Y_true_out, Y_var_out, rmse = validation_metrics["Y_pred_out"], validation_metrics["Y_true_out"], validation_metrics["Y_var_out"], validation_metrics["rmse"]

        Y_pred_full.extend(Y_pred_out)
        Y_true_full.extend(Y_true_out)
        Y_var_full.extend(Y_var_out)
        rmse_full.append(rmse)

        time_spt = time.time() - time_start
        print(f"[TIME] {min(i + batch_size, n_samples)}/{n_samples} completed in {time_spt:.2f} seconds.")
        time_start = time.time()

    Y_pred_full = np.array(Y_pred_full)
    Y_true_full = np.array(Y_true_full)
    Y_var_full = np.array(Y_var_full)
    rmse_full = np.array(rmse_full)

    n = Y_pred_full.shape[0]
    Y_pred_out_full = Y_pred_full.reshape(n, lat.shape[0], lon.shape[0])
    Y_true_out_full = Y_true_full.reshape(n, lat.shape[0], lon.shape[0])
    Y_var_out_full = Y_var_full.reshape(n, lat.shape[0], lon.shape[0]) if encoder == "PCA" else None

    # Save Y_pred_out_full and Y_true_out_full as NetCDF files
    output_dir = cfg.get('output_dir', 'outputs/leave_one_out/')
    os.makedirs(output_dir, exist_ok=True)
    y_pred_path = os.path.join(output_dir, "Y_pred_out_full.nc")
    y_true_path = os.path.join(output_dir, "Y_true_out_full.nc")
    # Remove files if they already exist
    if os.path.exists(y_pred_path):
        os.remove(y_pred_path)
    if os.path.exists(y_true_path):
        os.remove(y_true_path)

    # debug global stats before saving
    try:
        print("[DIAG] SUMMARY before saving - Y_pred_out_full mean/min/max:", np.mean(Y_pred_out_full), np.min(Y_pred_out_full), np.max(Y_pred_out_full))
        print("[DIAG] SUMMARY before saving - Y_true_out_full mean/min/max:", np.mean(Y_true_out_full), np.min(Y_true_out_full), np.max(Y_true_out_full))
    except Exception:
        pass

    xr.Dataset({
            "mean": (["time", "lat", "lon"], Y_true_out_full),
            "latitude": (["lat"], lat),
            "longitude": (["lon"], lon)
        }).to_netcdf(y_true_path)

    if encoder == "PCA":
        xr.Dataset({
            "mean": (["time", "lat", "lon"], Y_pred_out_full),
            "variance": (["time", "lat", "lon"], Y_var_out_full),
            "latitude": (["lat"], lat),
            "longitude": (["lon"], lon)
        }).to_netcdf(y_pred_path)
    else:
        xr.DataArray(Y_pred_out_full, dims=["time", "lat", "lon"]).to_netcdf(y_pred_path)

    print(f"[INFO] Y_pred_out_full saved to {y_pred_path}")
    print(f"[INFO] Y_true_out_full saved to {y_true_path}")

    if return_validation:
        # plot
        plot_histogram_4_leave1out(Y_true_out_full, Y_pred_out_full)

    time_spt = time.time() - time_start
    print(f"[TIME] plot completed in {time_spt:.2f} seconds.")

    return {
        "r2_score": r2_score,
        "n_components_retained": n_components,
        "original_variable": var_name,
        "spatial_shape": spatial_shape,
        "Y_pred_out": Y_pred_out_full,
        "Y_true_out": Y_true_out_full,
        "Y_var_out": Y_var_out_full,
        "encoder_used": encoder,
        "regressor_type": regressor_type
    }


def run_training_LGBM_optimization(cfg_path, encoder="PCA", regressor="LGBM", fixed_encoder_hp=True, fixed_regressor_hp=False,save_log=True):
    """
    This function is to perform hyperparameter optimization for LGBMRegressor using GridSearchCV.
    It returns the best model found during the search.
    Use PCA as encoder.
    """
    import optuna
    from paleo_emu.regressor import MultiEstimator
    from sklearn.model_selection import KFold

    with open(cfg_path, 'r') as file:
        cfg = yaml.safe_load(file) or {}

    # load data
    print("[INFO] Loading training data...")
    X_train, Y_train, var_name, spatial_shape, lat_array, lon_array = load_training_data(cfg_path)

    # encode the chosen training Y
    Y_train_encoded, decoder, mean_val, std_val, residual_variance = encode(
        Y_train,
        encoder=encoder,
        fixed_encoder_hp=fixed_encoder_hp,
        cfg_path=cfg_path
    )
    latent_dim = Y_train_encoded.shape[1]
    
    print("[INFO] Starting hyperparameter optimization for LGBMRegressor...")
    # Configurable knobs
    k_folds = cfg.get("optuna_kfolds", 20)                # use 20-fold CV by default (faster than LOO)
    trials_pc0 = int(cfg.get("optuna_trials_pc0", 40))   # budget for dominant PC
    trials_other = int(cfg.get("optuna_trials_other", 10))  # budget for other PCs
    top_k_decode_eval = int(cfg.get("optuna_top_k_decode_eval", 3))  # how many top candidates to re-evaluate in decoded space
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)

    best_params_per_pc = {}
    estimators_per_pc = []
    n_outputs = latent_dim
    # helper: conservative defaults for other PCs used when evaluating decoded RMSE candidates
    default_other = {
        "num_leaves": 31,
        "max_depth": 6,
        "learning_rate": 0.01,
        "n_estimators": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
    }


    print(f"[INFO] Hyperparameter optimization for LGBMRegressor completed.")

    # build a multi-estimator using per-pc estimators
    lgbm_regressor = MultiEstimator(estimators_per_pc)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", MultiOutputRegressor(lgbm_regressor))
    ])
    pipeline.fit(X_train, Y_train_encoded)

    out = {
        "best_model": pipeline,
        "best_params_per_pc": best_params_per_pc,
    }
    return out


def run_training_GPR_optimization(cfg_path, encoder="PCA", regressor="GPR", fixed_encoder_hp=True, fixed_regressor_hp=False, save_log=True, do_leaveoneout=True, return_validation=False):
    """
    To be deleted.
    1. optimaze GPR hyperparameters for every PC.
    2. let MultiOutputRegressor to do the multi-output fitting instead of giving fixed hyperparameters for every PC (proved to cause poor performance).

    This function is to perform hyperparameter optimization for GPR using Optuna.
    If do_leaveoneout=True the trained pipeline will be used to generate leave-one-out style outputs (no retraining).
    """
    import matplotlib.pyplot as plt
    import optuna
    import optuna.visualization as vis
    from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, WhiteKernel, ConstantKernel as C
    import warnings
    from sklearn.exceptions import ConvergenceWarning

    with open(cfg_path, 'r') as file:
        cfg = yaml.safe_load(file) or {}

    # load data
    print("[INFO] Loading training data...")
    X_train, Y_train, var_name, spatial_shape, lat_array, lon_array = load_training_data(cfg_path)

    # encode the chosen training Y
    Y_train_encoded, decoder, mean_val, std_val, residual_variance = encode(
        Y_train,
        encoder=encoder,
        fixed_encoder_hp=fixed_encoder_hp,
        cfg_path=cfg_path
    )
    latent_dim = Y_train_encoded.shape[1]


    print("[INFO] Starting hyperparameter optimization for GPR...")
    print("[INFO] Find the best kernel and hyperparameters using Optuna.")

    def objective(trial):
        # Suppress ConvergenceWarning during optimization
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)

            kernel_name = "RBF_White"  # fix to RBF+White for faster optimization
            n_features = X_train.shape[1]
            length_scales_array = np.array([
                trial.suggest_float(f"length_scale_{j}", 1e-3, 1e3, log=True)
                for j in range(n_features)
            ])
            noise_level = trial.suggest_float("noise_level", 1e-1, 1e2, log=True)
            nugget_value = trial.suggest_float("nugget", 1e-6, 1e-1, log=True)
            constant_value = trial.suggest_float("constant_value", 1e-3, 1e0, log=True)
            n_restarts_optimizer = 5  # fix to 5 for faster optimization

            if kernel_name == "RBF_White":
                kernel = C(constant_value) * RBF(length_scale=length_scales_array) + WhiteKernel(noise_level=noise_level)
            elif kernel_name == "Matern_White":
                nu = trial.suggest_categorical("nu", [1.5, 2.5])
                kernel = C(constant_value) * Matern(length_scale=length_scales_array, nu=nu) + WhiteKernel(noise_level=noise_level)

            GPR = GaussianProcessRegressor(kernel=kernel, alpha=nugget_value, n_restarts_optimizer=n_restarts_optimizer, random_state=42, normalize_y=True)
            model = MultiOutputRegressor(GPR)
            try:
                scores = cross_val_score(model, X_train, Y_train_encoded, cv=10, scoring="neg_mean_squared_error", n_jobs=1)
                return scores.mean()
            except Exception:
                raise optuna.exceptions.TrialPruned()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10, n_jobs=1)

    if save_log:
        output_dir = cfg['log_path'] if 'log_path' in cfg else "examples/logs"
        os.makedirs(output_dir, exist_ok=True)
        vis.plot_optimization_history(study).write_image(os.path.join(output_dir, "gpr_optimization_history.png"))
        vis.plot_parallel_coordinate(study).write_image(os.path.join(output_dir, "gpr_parallel_coordinate.png"))
        vis.plot_param_importances(study).write_image(os.path.join(output_dir, "gpr_param_importances.png"))
        vis.plot_slice(study).write_image(os.path.join(output_dir, "gpr_slice_plot.png"))
        vis.plot_contour(study).write_image(os.path.join(output_dir, "gpr_contour_plot.png"))
        print(f"[INFO] GPR Optimization plots saved to {output_dir}")

    best = study.best_params
    kernel_name = best.pop("kernel", "RBF_White")
    n_features = X_train.shape[1]
    best_length_scales = [best.pop(f"length_scale_{j}") for j in range(n_features)]

    if kernel_name == "RBF_White":
        GPR = GaussianProcessRegressor(
            kernel=C(best["constant_value"]) * RBF(length_scales=best_length_scales) + WhiteKernel(noise_level=best["noise_level"]),
            alpha=best["nugget"], n_restarts_optimizer=1, random_state=42, normalize_y=True
        )
    elif kernel_name == "Matern_White":
        GPR = GaussianProcessRegressor(
            kernel=C(best["constant_value"]) * Matern(length_scales=best_length_scales, nu=best["nu"]) + WhiteKernel(noise_level=best["noise_level"]),
            alpha=best["nugget"], n_restarts_optimizer=1, random_state=42, normalize_y=True
        )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", MultiOutputRegressor(GPR))
    ])

    pipeline.fit(X_train, Y_train_encoded)

    print(f"[INFO] Best CV objective (neg_mse-based): {-study.best_value:.4f}" if study is not None else "[INFO] Trained with fixed hyperparams")

    best_model = pipeline

    # if requested, generate leave-one-out style outputs using the trained pipeline (no retraining)
    if do_leaveoneout:
        loo_res = run_training_leave_one_out(cfg_path,
                                            regressor_type=regressor,
                                            encoder=encoder,
                                            fixed_encoder_hp=fixed_encoder_hp,
                                            fixed_regressor_hp=False,
                                            return_validation=return_validation)
        out = {
            "best_model": best_model,
            "best_params": study.best_params if study is not None else {},
            "best_r2_score": study.best_value if study is not None else None
        }
        out.update({k: loo_res.get(k) for k in ("Y_pred_out", "Y_true_out", "Y_var_out", "r2_score")})
        return out

    print(f"[OPTUNA] best params: {study.best_params}, best score: {study.best_value:.5f}")

    return {
        "best_model": best_model,
        "best_params": study.best_params if study is not None else {},
        "best_r2_score": study.best_value if study is not None else None
    }

