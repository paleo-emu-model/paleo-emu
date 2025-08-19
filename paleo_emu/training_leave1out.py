"""
This module is a way to train models to get the best parameters using leave-one-out cross-validation.
"""

import numpy as np
import xarray as xr
import os
import time

import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from paleo_emu.load import load_training_data
from paleo_emu.encoder import encode
from paleo_emu.regressor import build_regressor
from paleo_emu.plotting import plot_r2_map_with_latlon, plot_prediction_maps_with_info
from paleo_emu.validation import compute_r2_map

# separate train and test before PCA
def run_training(train_dict, model_type="GPR", kernel="RBF_White", pca_variance_ratio=0.999, encoder="PCA", vae_config=None, seed=42,  return_validation=True):

    # 1. 加载原始数据
    X, Y_flat, var_name, spatial_shape, lat_array, lon_array = load_training_data(train_dict)
    n_samples = X.shape[0]

    Y_pred_full = []
    Y_test_full = []
    
    time_start = time.time()

    batch_size = 10  # Number of samples to leave out each time

    for i in range(0, n_samples, batch_size):
        # Leave 10 out
        test_indices = np.arange(i, min(i + batch_size, n_samples))
        train_indices = np.setdiff1d(np.arange(n_samples), test_indices)

        X_train = X[train_indices]
        Y_train_flat = Y_flat[train_indices]
        X_test = X[test_indices]
        Y_test_flat = Y_flat[test_indices]

        # 特征提取
        Y_train_encoded, decoder, mean_val, std_val = encode(
            Y_train_flat,
            encoder=encoder,
            pca_variance_ratio=pca_variance_ratio,
            seed=seed,
            vae_config=vae_config
        )

        # 测试集编码
        if encoder == "PCA":
            Y_test_scaled = (Y_test_flat - mean_val) / std_val
            Y_test_encoded = decoder.transform(Y_test_scaled)
        elif encoder == "VAE":
            Y_test_scaled = (Y_test_flat - mean_val) / std_val
            mean_logvar = decoder.encoder.predict(Y_test_scaled)
            mean, logvar = tf.split(mean_logvar, 2, axis=1)
            Y_test_encoded = mean.numpy()
        else:
            Y_test_encoded = Y_test_flat

        # 建立并训练 Pipeline
        regressor = build_regressor(model_type=model_type, kernel_name=kernel, encoder=encoder)

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", MultiOutputRegressor(regressor))
        ])

        pipeline.fit(X_train, Y_train_encoded)

        # 预测
        Y_pred_encoded = pipeline.predict(X_test)

        # 还原
        if encoder == "PCA":
            Y_pred = decoder.inverse_transform(Y_pred_encoded)
            Y_test = decoder.inverse_transform(Y_test_encoded)
            Y_pred = Y_pred * std_val + mean_val
            Y_test = Y_test * std_val + mean_val
        elif encoder == "VAE":
            Y_pred = decoder.decoder.predict(Y_pred_encoded)
            Y_test = decoder.decoder.predict(Y_test_encoded)
            Y_pred = Y_pred * std_val + mean_val
            Y_test = Y_test * std_val + mean_val
        else:
            Y_pred = Y_pred_encoded
            Y_test = Y_test_encoded

        Y_pred_full.extend(Y_pred)
        Y_test_full.extend(Y_test)

        time_spt = time.time() - time_start
        print(f"[TIME] {min(i + batch_size, n_samples)}/{n_samples} completed in {time_spt:.2f} seconds.")
        time_start = time.time()

    Y_pred_full = np.array(Y_pred_full)
    Y_test_full = np.array(Y_test_full)

    n = Y_pred_full.shape[0]
    lat, lon = spatial_shape
    Y_pred_out = Y_pred_full.reshape(n, lat, lon)
    Y_test_out = Y_test_full.reshape(n, lat, lon)
    score = r2_score(Y_test_full, Y_pred_full)
    print(f"[INFO] LOOCV R² Score: {score:.4f}")
    # Ensure output directory exists
    os.makedirs("outputs", exist_ok=True)

    # Create DataArrays
    pred_da = xr.DataArray(
        Y_pred_out,
        dims=["sample", "lat", "lon"],
        coords={"sample": np.arange(Y_pred_out.shape[0]), "lat": lat_array, "lon": lon_array},
        name="prediction"
    )
    test_da = xr.DataArray(
        Y_test_out,
        dims=["sample", "lat", "lon"],
        coords={"sample": np.arange(Y_test_out.shape[0]), "lat": lat_array, "lon": lon_array},
        name="truth"
    )

    # Save as NetCDF
    pred_da.to_netcdf(f"outputs/pred_{model_type}_{kernel}_{encoder}.nc")
    test_da.to_netcdf(f"outputs/test_{model_type}_{kernel}_{encoder}.nc")
    
    # 画图
    r2_map = compute_r2_map(Y_test_out, Y_pred_out, lat_array, lon_array)
    plot_r2_map_with_latlon(r2_map, lat_array=lat_array, lon_array=lon_array, model_type=model_type,
                            encoder=encoder, kernel=kernel, save_dir="outputs/logs")
    # for timestep in [999]:
    plot_prediction_maps_with_info(
        Y_test_out,
        Y_pred_out,
        lat_array=lat_array,
        lon_array=lon_array,
        timestep=999,
        emulator_name=model_type,
        encoder_name=encoder,
        kernel_name=kernel,
        save_folder="outputs/maps",
        title_suffix=f"Timestep"
    )
    time_spt = time.time() - time_start
    print(f"[TIME] plot completed in {time_spt:.2f} seconds.")

    return {
        "gpr_r2_score": score,
        "n_components_retained": Y_train_encoded.shape[1],
        "original_variable": var_name,
        "spatial_shape": spatial_shape,
        "Y_pred_out": Y_pred_out,
        "Y_True_out": Y_test_out,
        "encoder_used": encoder,
        "model_type": model_type
    }
