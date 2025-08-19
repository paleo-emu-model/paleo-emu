"""
This module is to train models using chosen regressors, kernels, and encoders.
"""

import numpy as np

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
def run_training(train_dict,  regressor_type="GPR", kernel="RBF_White", pca_variance_ratio=0.999, encoder="PCA", vae_config=None, seed=42,  return_validation=True):

    # load data
    X, Y_flat, var_name, spatial_shape, lat_array, lon_array = load_training_data(train_dict)

    # split data for training and testing
    X_train, X_test, Y_train_flat, Y_test_flat = train_test_split(X, Y_flat, test_size=0.2, random_state=seed)

    # encode the chosen training Y
    Y_train_encoded, decoder, mean_val, std_val = encode(
        Y_train_flat,
        encoder=encoder,
        pca_variance_ratio=pca_variance_ratio,
        seed=seed,
        vae_config=vae_config
    )
    latent_dim = Y_train_encoded.shape[1]

    # transform validation Y set back for validation
    if encoder == "PCA":
        Y_test_scaled = (Y_test_flat - mean_val) / std_val
        Y_test_encoded = decoder.transform(Y_test_scaled)
    elif encoder == "VAE":
        Y_test_scaled = (Y_test_flat - mean_val) / std_val
        mean_logvar = decoder.encoder.predict(Y_test_scaled)
        mean, logvar = tf.split(mean_logvar, 2, axis=1)
        latent = mean + tf.random.normal(tf.shape(mean)) * tf.exp(logvar * 0.5)  # reparameterize
        # Y_test_encoded = latent.numpy()
        Y_test_encoded = mean.numpy()
    else:
        # if we dont use any encoder but original Y, then use Y directly
        Y_test_encoded = Y_test_flat

    # build and train the regressor and pipeline
    regressor = build_regressor( model_type= regressor_type, kernel_name=kernel,encoder=encoder)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", MultiOutputRegressor(regressor))
    ])
    pipeline.fit(X_train, Y_train_encoded)

    # using the 20% holdout X for prediction then validation.
    if  return_validation:
        Y_pred_encoded = pipeline.predict(X_test)

        print(f"[debug] shape of Y_pred_encoded: {Y_pred_encoded.shape}")
        print(f"[debug] shape of Y_test_encoded: {Y_test_encoded.shape}")

        # inverse transform
        if encoder == "PCA":
            Y_pred_full = decoder.inverse_transform(Y_pred_encoded)
            Y_test_full = decoder.inverse_transform(Y_test_encoded)
            # inverse standardization
            Y_pred_full = Y_pred_full * std_val + mean_val
            Y_test_full = Y_test_full * std_val + mean_val
        elif encoder == "VAE":
            Y_pred_full = decoder.decoder.predict(Y_pred_encoded)
            Y_test_full = decoder.decoder.predict(Y_test_encoded)
            Y_pred_full = Y_pred_full * std_val + mean_val
            Y_test_full = Y_test_full * std_val + mean_val
        else:
            Y_pred_full = Y_pred_encoded
            Y_test_full = Y_test_encoded

        n = Y_pred_full.shape[0]
        lat, lon = spatial_shape
        Y_pred_out = Y_pred_full.reshape(n, lat, lon)
        Y_test_out = Y_test_full.reshape(n, lat, lon)
        score = r2_score(Y_test_full, Y_pred_full)
        print(f"[INFO] R² Score: {score:.4f}")

        print("[DEBUG] train latent mean/std:", np.mean(Y_train_encoded), np.std(Y_train_encoded))
        print("[DEBUG] test latent (true) mean/std:", np.mean(Y_test_encoded), np.std(Y_test_encoded))
        print("[DEBUG] GPR predicted latent mean/std:", np.mean(Y_pred_encoded), np.std(Y_pred_encoded))

        # plotting for validation
        r2_map = compute_r2_map(Y_test_out, Y_pred_out, lat_array, lon_array)
        plot_r2_map_with_latlon(r2_map, lat_array=lat_array, lon_array=lon_array,  regressor_type= regressor_type,
                                encoder=encoder, kernel=kernel, save_dir="outputs/logs")
        for timestep in [0, 1, 2, 3, 999]:
            plot_prediction_maps_with_info(
                Y_test_out,
                Y_pred_out,
                lat_array=lat_array,
                lon_array=lon_array,
                timestep=timestep,
                emulator_name= regressor_type,
                encoder_name=encoder,
                kernel_name=kernel,
                save_folder="outputs/maps",
                title_suffix=f"Timestep {timestep}"
            )

        return {
            "pipeline_model": pipeline,
            "feature_extraction": decoder,
            "gpr_r2_score": score,
            "n_components_retained": Y_train_encoded.shape[1],
            "original_variable": var_name,
            "spatial_shape": spatial_shape,
            "Y_pred_out": Y_pred_out,
            "Y_True_out": Y_test_out,
            "X_test": X_test,
            "encoder_used": encoder,
            " regressor_type":  regressor_type
        }
