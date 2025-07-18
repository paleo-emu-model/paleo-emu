
import numpy as np

import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


from paleo_emu.load import load_training_data
from paleo_emu.encoder import extract_features
from paleo_emu.regressor import build_regressor
from paleo_emu.plotting import plot_r2_map_with_latlon, plot_prediction_maps_with_info
from paleo_emu.validation import compute_r2_map

# separate train and test before PCA
def run_training(train_dict, model_type="GPR", kernel="RBF_White", pca_variance_ratio=0.999, encoder="PCA", vae_config=None, seed=42, return_pred=True):

    # 1. 加载原始数据
    X, Y_flat, var_name, spatial_shape, lat_array, lon_array = load_training_data(train_dict)

    # 2. 划分原始数据
    X_train, X_test, Y_train_flat, Y_test_flat = train_test_split(X, Y_flat, test_size=0.2, random_state=seed)

    # 3. 对 Y_train_flat 进行特征提取（fit和transform）
    Y_train_encoded, feature_extractor, mean_val, std_val = extract_features(
        Y_train_flat,
        encoder=encoder,
        pca_variance_ratio=pca_variance_ratio,
        seed=seed,
        vae_config=vae_config
    )
    latent_dim = Y_train_encoded.shape[1]

    # 4. 用训练好的 feature_extractor 对 Y_test_flat 进行 transform
    if encoder == "PCA":
        Y_test_scaled = (Y_test_flat - mean_val) / std_val
        Y_test_encoded = feature_extractor.transform(Y_test_scaled)
    elif encoder == "VAE":
        Y_test_scaled = (Y_test_flat - mean_val) / std_val
        mean_logvar = feature_extractor.encoder.predict(Y_test_scaled)
        mean, logvar = tf.split(mean_logvar, 2, axis=1)
        latent = mean + tf.random.normal(tf.shape(mean)) * tf.exp(logvar * 0.5)  # ✅ reparameterize
        # Y_test_encoded = latent.numpy()
        Y_test_encoded = mean.numpy()
    else:
        # 如果没有用encoder，直接用原始
        Y_test_encoded = Y_test_flat

    # 5. 建立并训练 Pipeline
    regressor = build_regressor(model_type=model_type, kernel_name=kernel,encoder=encoder)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", MultiOutputRegressor(regressor))
    ])
    pipeline.fit(X_train, Y_train_encoded)

    # 6. 预测
    if return_pred:
        Y_pred_encoded = pipeline.predict(X_test)

        print(f"[debug] shape of Y_pred_encoded: {Y_pred_encoded.shape}")
        print(f"[debug] shape of Y_test_encoded: {Y_test_encoded.shape}")

        # 还原
        if encoder == "PCA":
            Y_pred_full = feature_extractor.inverse_transform(Y_pred_encoded)
            Y_test_full = feature_extractor.inverse_transform(Y_test_encoded)
            # 反标准化
            Y_pred_full = Y_pred_full * std_val + mean_val
            Y_test_full = Y_test_full * std_val + mean_val
        elif encoder == "VAE":
            Y_pred_full = feature_extractor.decoder.predict(Y_pred_encoded)
            Y_test_full = feature_extractor.decoder.predict(Y_test_encoded)
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

        # ===============================
        # 画图
        r2_map = compute_r2_map(Y_test_out, Y_pred_out, lat_array, lon_array)
        plot_r2_map_with_latlon(r2_map, lat_array=lat_array, lon_array=lon_array, model_type=model_type,
                                encoder=encoder, kernel=kernel, save_dir="outputs/logs")
        for timestep in [0, 1, 2, 3, 999]:
            plot_prediction_maps_with_info(
                Y_test_out,
                Y_pred_out,
                lat_array=lat_array,
                lon_array=lon_array,
                timestep=timestep,
                emulator_name=model_type,
                encoder_name=encoder,
                kernel_name=kernel,
                save_folder="outputs/maps",
                title_suffix=f"Timestep {timestep}"
            )

        return {
            "pipeline_model": pipeline,
            "feature_extraction": feature_extractor,
            "gpr_r2_score": score,
            "n_components_retained": Y_train_encoded.shape[1],
            "original_variable": var_name,
            "spatial_shape": spatial_shape,
            "Y_pred_out": Y_pred_out,
            "Y_True_out": Y_test_out,
            "X_test": X_test,
            "encoder_used": encoder,
            "model_type": model_type
        }
