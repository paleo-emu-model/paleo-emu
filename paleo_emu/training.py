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
from sklearn.metrics import root_mean_squared_error
import joblib

from docs.source.auto_examples.plot_pipeline import X_test
from paleo_emu.load import load_training_data
from paleo_emu.encoder import encode
from paleo_emu.regressor import build_regressor
from paleo_emu.plotting import plot_r2_map_with_latlon, plot_prediction_maps_with_info, plot_histogram_4_leave1out
from paleo_emu.validation import compute_r2_map

# training for given data 
"""
X_training: (n_samples, 5) the input feature matrix
Y_training: (n_samples, lat*lon) the flattened output matrix
"""
def run_training(X_train,Y_train,regressor_type="GPR",kernel="RBF_White",pca_variance_ratio=0.999,encoder="PCA",vae_config=None):
    # encode the chosen training Y
    Y_train_encoded, decoder, mean_val, std_val = encode(
        Y_train,
        encoder=encoder,
        pca_variance_ratio=pca_variance_ratio,
        vae_config=vae_config
    )
    latent_dim = Y_train_encoded.shape[1]

    # build and train the regressor and pipeline
    regressor = build_regressor( regressor_type= regressor_type, kernel_name=kernel,encoder=encoder)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", MultiOutputRegressor(regressor))
    ])
    pipeline.fit(X_train, Y_train_encoded)
    # save the trained pipeline
    joblib.dump(pipeline, "pipeline.joblib")
    # save the decoder
    joblib.dump(decoder, "decoder.joblib")

    return {
        "trained_pipeline": "pipeline.joblib",
        "decoder": "decoder.joblib",
        "mean_val": mean_val,
        "std_val": std_val,
        "n_components_retained": latent_dim,
        "regressor_type": regressor_type,
        "kernel": kernel}

def return_validation(X_test,Y_true_flat,trained_pipeline,decoder,mean_val,std_val,spatial_shape,encoder):
    """
        1. encode, decode Y_test
        2. predict Y_test_predicted using trained_pipeline

        outputs:
        - R² Score
        - R² map
    """
    # process Y_test for validation
    # -------
    # transform validation Y 'encoded' for validation
    # here we cannot use 'encode' function directly
    # because we need to use the mean and std as used for training Y
    # -------
    # an simple way is to skip this 'encode' and the following 'decode' procedures
    # but using the Y_test_flat for comparison directly
    # this extra processing steps is to ensure the consistency of the processing for Y
    # -------
    Y_true_scaled = (Y_true_flat - mean_val) / std_val
    if encoder == "PCA":
        # Y_true_scaled = (Y_true_flat - mean_val) / std_val
        Y_true_encoded = decoder.transform(Y_true_scaled)
    elif encoder == "VAE":
        # Y_true_scaled = (Y_true_flat - mean_val) / std_val
        mean_logvar = decoder.encoder.predict(Y_true_scaled)
        mean, logvar = tf.split(mean_logvar, 2, axis=1)
        latent = mean + tf.random.normal(tf.shape(mean)) * tf.exp(logvar * 0.5)  # reparameterize
        # Y_test_encoded = latent.numpy()
        Y_true_encoded = mean.numpy()
    else:
        # if we dont use any encoder but original Y, then use Y directly
        Y_true_encoded = Y_true_flat

    Y_pred_encoded = trained_pipeline.predict(X_test)

    # decode Y
    if encoder == "PCA" or encoder == "VAE":
        Y_pred_full = decoder.inverse_transform(Y_pred_encoded)
        Y_true_full = decoder.inverse_transform(Y_true_encoded)
        Y_pred_full = Y_pred_full * std_val + mean_val
        Y_true_full = Y_true_full * std_val + mean_val
    else:
        Y_pred_full = Y_pred_encoded
        Y_true_full = Y_true_encoded

    n = Y_pred_full.shape[0]
    lat, lon = spatial_shape
    Y_pred_out = Y_pred_full.reshape(n, lat, lon)
    Y_true_out = Y_true_full.reshape(n, lat, lon)
    r2_score = r2_score(Y_true_full, Y_pred_full)
    rmse = root_mean_squared_error(Y_true_full, Y_pred_full)

    return {"Y_pred_out": Y_pred_out,
            "Y_true_out": Y_true_out,
            "r2_score": r2_score,
            "rmse": rmse}

# 20% for validation; 80% for training
# only sample once
def run_training_28(train_dict,  regressor_type="GPR", kernel="RBF_White", pca_variance_ratio=0.999, encoder="PCA", vae_config=None, return_validation=True):
    # load data
    X, Y_flat, var_name, spatial_shape, lat_array, lon_array = load_training_data(train_dict)
    # split data for training and testing
    X_train, X_test, Y_train_flat, Y_test_flat = train_test_split(X, Y_flat, test_size=0.2)
    # train model
    training_info = run_training(X_train, Y_train_flat, regressor_type=regressor_type, kernel=kernel, pca_variance_ratio=pca_variance_ratio, encoder=encoder, vae_config=vae_config)
    trained_pipeline, decoder, mean_val, std_val, n_components = training_info["trained_pipeline"], training_info["decoder"], training_info["mean_val"], training_info["std_val"], training_info["n_components_retained"]
    trained_pipeline = joblib.load(trained_pipeline)
    decoder = joblib.load(decoder)

    if return_validation:
        # compute validation metrics
        validation_metrics = return_validation(X_test, Y_test_flat, trained_pipeline, decoder, mean_val, std_val, spatial_shape, encoder)
        Y_pred_out, Y_true_out, r2_score = validation_metrics["Y_pred_out"], validation_metrics["Y_true_out"], validation_metrics["r2_score"]
        # plotting for validation
        r2_map = compute_r2_map(Y_true_out, Y_pred_out, lat_array, lon_array)
        plot_r2_map_with_latlon(r2_map, lat_array=lat_array, lon_array=lon_array,  regressor_type= regressor_type,
                                encoder=encoder, kernel=kernel, save_dir="outputs/logs")
        print(f"[INFO] R² Score: {r2_score:.4f}")
        print("[INFO] here we picked timesteps 0 1 2 3 999 for demonstration, edit the code if you want to see other timesteps")
        for timestep in [0, 1, 2, 3, 999]:
            plot_prediction_maps_with_info(Y_true_out, Y_pred_out, lat_array=lat_array, lon_array=lon_array, timestep=timestep, emulator_name= regressor_type,
                encoder_name=encoder, kernel_name=kernel, save_folder="outputs/maps", title_suffix=f"Timestep {timestep}")

    return {
        "pipeline_model": trained_pipeline,
        "decoder": decoder,
        "r2_score": r2_score,
        "n_components_retained": n_components,
        "original_variable": var_name,
        "spatial_shape": spatial_shape,
        "Y_pred_out": Y_pred_out,
        "Y_true_out": Y_true_out,
        "X_test": X_test,
        "encoder_used": encoder,
        "regressor_type":  regressor_type
    }


def run_training_leave_one_out(train_dict, regressor_type="GPR", kernel="RBF_White", pca_variance_ratio=0.999, encoder="PCA", vae_config=None,  return_validation=True):

    # 1. 加载原始数据
    X, Y_flat, var_name, spatial_shape, lat, lon = load_training_data(train_dict)
    n_samples = X.shape[0]

    Y_pred_full = []
    Y_true_full = []
    rmse_full = []

    time_start = time.time()

    batch_size = 1  # Number of samples to leave out each time

    for i in range(0, n_samples, batch_size):
        # split data
        # ----------
        test_indices = np.arange(i, min(i + batch_size, n_samples))
        train_indices = np.setdiff1d(np.arange(n_samples), test_indices)

        X_train = X[train_indices]
        Y_train_flat = Y_flat[train_indices]
        X_test = X[test_indices]
        Y_test_flat = Y_flat[test_indices]

        # fit model
        # ----------
        training_info = run_training(X_train, Y_train_flat, regressor_type=regressor_type, kernel=kernel, pca_variance_ratio=pca_variance_ratio, encoder=encoder, vae_config=vae_config)
        trained_pipeline, decoder, mean_val, std_val = training_info["trained_pipeline"], training_info["decoder"], training_info["mean_val"], training_info["std_val"]
        trained_pipeline = joblib.load(trained_pipeline)
        decoder = joblib.load(decoder)
        
        validation_metrics = return_validation(X_test, Y_test_flat, trained_pipeline, decoder, mean_val, std_val, spatial_shape, encoder)
        Y_pred_out, Y_true_out, rmse, spatial_shape = validation_metrics["Y_pred_out"], validation_metrics["Y_true_out"], validation_metrics["rmse"], validation_metrics["spatial_shape"]

        Y_pred_full.extend(Y_pred_out)
        Y_true_full.extend(Y_true_out)
        rmse_full.extend(rmse)

        time_spt = time.time() - time_start
        print(f"[TIME] {min(i + batch_size, n_samples)}/{n_samples} completed in {time_spt:.2f} seconds.")
        time_start = time.time()

    Y_pred_full = np.array(Y_pred_full)
    Y_true_full = np.array(Y_true_full)
    rmse_full = np.array(rmse_full)

    n = Y_pred_full.shape[0]
    Y_pred_out_full = Y_pred_full.reshape(n, lat, lon)
    Y_true_out_full = Y_true_full.reshape(n, lat, lon)

    # plot
    plot_histogram_4_leave1out(Y_true_out_full, Y_pred_out_full, lat, lon, save_folder="outputs/maps")

    time_spt = time.time() - time_start
    print(f"[TIME] plot completed in {time_spt:.2f} seconds.")

    return {
        "r2_score": r2_score,
        "n_components_retained": n_samples,
        "original_variable": var_name,
        "spatial_shape": spatial_shape,
        "Y_pred_out": Y_pred_out_full,
        "Y_true_out": Y_true_out_full,
        "encoder_used": encoder,
        "regressor_type": regressor_type
    }
