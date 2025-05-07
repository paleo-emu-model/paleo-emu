import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import tensorflow as tf
from tensorflow.keras import layers, models

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, Matern, RationalQuadratic, ExpSineSquared,
    ConstantKernel as C, WhiteKernel
)
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from examples.config.training_dict_formatted import training_dict as train_dict
from examples.config.prediction_dict import forcing_dict
from sklearn.metrics import r2_score
from pathlib import Path
import random
import os

def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# ===== 模块 1：加载数据 =====
def load_training_data(cfg):
    """
    加载训练数据。
    参数：
        cfg: 配置字典，包含文件路径信息。
             - cfg["file_path"]: 基础目录
             - cfg["X_input"]: X 的 .res 文件名
             - cfg["Y_output"]: Y 的 .nc 文件名
    返回：
        X: (n_samples, 5) 的输入特征矩阵
        Y_flat: 展平后的输出 (n_samples, lat*lon)
        var_name: Y 中的变量名
        spatial_shape: 原始的 (lat, lon) 形状
    """
    # 拼接路径
    base_path = Path(cfg["file_path"])
    x_path = base_path / cfg["X_input"]
    y_path = base_path / cfg["Y_output"]

    # 读取 X 数据
    df = pd.read_csv(x_path, sep=r"\s+", header=None)
    df.columns = ['co2', 'obliquity', 'esinw', 'ecosw', 'ice']
    X = df[['co2', 'esinw', 'ecosw', 'obliquity', 'ice']].to_numpy()

    # 读取 Y 数据
    ds = xr.open_dataset(y_path)
    var_name = list(ds.data_vars)[0]
    lat_name = ds[var_name].dims[1]
    lon_name = ds[var_name].dims[2]
    Y = ds[var_name].values  # (n_samples, lat, lon)
    Y_flat = Y.reshape(Y.shape[0], -1)
    lat_array = -ds[lat_name].values
    lon_array = ds[lon_name].values

    return X, Y_flat, var_name, Y.shape[1:], lat_array, lon_array

# ======== 模块2.0： VAE 定义 =========
class VAE(tf.keras.Model):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        # 编码器
        self.encoder = models.Sequential([
            layers.InputLayer(input_shape=(input_dim,)),    # input_dim=7008
            layers.Dense(4096, activation="relu"),           # 先减半，7008 → 4096
            layers.Dense(2048, activation="relu"),           # 再减半，4096 → 2048
            layers.Dense(4096, activation="relu"),           # 保持信息展开
            layers.Dense(latent_dim * 2)                           # 最后输出 mean 和 logvar，(batch_size, 4096)
        ])
        # 解码器
        self.decoder = models.Sequential([
            layers.InputLayer(input_shape=(latent_dim,)),
            layers.Dense(4096, activation="relu"),
            layers.Dense(2048, activation="relu"),
            layers.Dense(2048, activation="relu"),   # 👈 这里再加一层
            layers.Dense(1024, activation="relu"),
            layers.Dense(7008)
        ])


    def reparameterize(self, mean, logvar):
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        eps = tf.random.normal(shape=(batch, dim))
        return eps * tf.exp(logvar * 0.5) + mean

    def call(self, x):
        x_encoded = self.encoder(x)
        mean, logvar = tf.split(x_encoded, num_or_size_splits=2, axis=1)
        z = self.reparameterize(mean, logvar)
        x_decoded = self.decoder(z)
        return x_decoded, mean, logvar

def compute_vae_loss(x, x_decoded, mean, logvar):
    reconstruction_loss = tf.reduce_mean(tf.square(x - x_decoded))
    kl_loss = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mean) - tf.exp(logvar))
    return reconstruction_loss + kl_loss

# ===== 模块 2：特征提取模块（PCA / VAE） =====
def save_training_log(epoch_losses, seed, latent_dim, epochs, learning_rate, batch_size, kl_weight, log_dir="training/logs"):
    """
    保存VAE训练日志，包括：
    - loss曲线图
    - 超参数+最终loss的CSV记录
    """

    # --- 创建logs目录 ---
    os.makedirs(log_dir, exist_ok=True)

    # --- 统一格式化信息 ---
    info_str = f"seed{seed}_latent{latent_dim}_ep{epochs}_lr{learning_rate}_bs{batch_size}_kl{kl_weight}"

    # --- 保存loss曲线 ---
    loss_curve_filename = os.path.join(log_dir, f"loss_curve_{info_str}.png")

    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(epoch_losses)+1), epoch_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"VAE Loss Curve ({info_str})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(loss_curve_filename, dpi=300)
    plt.close()

    print(f"[INFO] Loss curve saved to: {loss_curve_filename}")

    # --- 保存超参数和最终loss到CSV ---
    log_file = os.path.join(log_dir, "vae_hyperparameter_log.csv")

    log_entry = {
        "seed": seed,
        "latent_dim": latent_dim,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "kl_weight": kl_weight,
        "final_loss": epoch_losses[-1]  # 最后一个epoch的loss
    }

    if not os.path.exists(log_file):
        df = pd.DataFrame([log_entry])
        df.to_csv(log_file, index=False)
    else:
        df = pd.read_csv(log_file)
        df = pd.concat([df, pd.DataFrame([log_entry])], ignore_index=True)
        df.to_csv(log_file, index=False)

    print(f"[INFO] Hyperparameter log updated: {log_file}")

# old code, keep it for now
def extract_features(Y_flat, encoder="PCA", model_type="GPR", pca_variance_ratio=0.999, seed=1024, vae_config=None):
    print(f"[INFO] Raw Y_flat min={np.min(Y_flat)}, max={np.max(Y_flat)}, mean={np.mean(Y_flat)}, std={np.std(Y_flat)}")
    
    mean_val = np.mean(Y_flat)
    std_val = np.std(Y_flat)
    Y_flat = (Y_flat - mean_val) / std_val
    print(f"[INFO] Y_flat standardized to mean ~0, std ~1")

    if encoder == "PCA":
        print("[INFO] Using PCA for feature extraction.")
        pca_model = PCA(n_components=pca_variance_ratio)
        Y_pca = pca_model.fit_transform(Y_flat)

        print(f"PCA n_components_: {pca_model.n_components_}")
        print(f"Sum explained variance: {np.sum(pca_model.explained_variance_ratio_)}")

    elif encoder == "VAE":
        print("[INFO] Using VAE for feature extraction.")

        # === 从vae_config读超参数 ===
        if vae_config is None:
            vae_config = {"latent_dim": 256, "epochs": 150, "learning_rate": 1e-4, "batch_size": 64}

        latent_dim = vae_config.get("latent_dim", 256)
        epochs = vae_config.get("epochs", 150)
        learning_rate = vae_config.get("learning_rate", 1e-4)
        batch_size = vae_config.get("batch_size", 64)
        kl_weight = vae_config.get("kl_weight", 1.0)  # 预留，如果以后加β-VAE

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        input_dim = Y_flat.shape[1]
        vae_model = VAE(input_dim, latent_dim)

        dataset = tf.data.Dataset.from_tensor_slices((Y_flat.astype('float32')))
        dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

        epoch_losses = []

        for epoch in range(epochs):
            total_loss = 0
            for step, x_batch in enumerate(dataset):
                with tf.GradientTape() as tape:
                    x_decoded, mean, logvar = vae_model(x_batch)
                    loss = compute_vae_loss(x_batch, x_decoded, mean, logvar) * kl_weight  # 预留β-VAE调整点

                grads = tape.gradient(loss, vae_model.trainable_variables)
                optimizer.apply_gradients(zip(grads, vae_model.trainable_variables))
                total_loss += loss

            avg_loss = total_loss / (step + 1)
            epoch_losses.append(avg_loss.numpy())

            if epoch % 10 == 0 or epoch == epochs-1:
                print(f"[VAE] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

        save_training_log(
            epoch_losses=epoch_losses,
            seed=seed,
            latent_dim=latent_dim,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            kl_weight=kl_weight
        )

        mean_logvar = vae_model.encoder(Y_flat)
        mean, logvar = tf.split(mean_logvar, num_or_size_splits=2, axis=1)
        eps = tf.random.normal(shape=tf.shape(mean))
        latent = mean + eps * tf.exp(0.5 * logvar)
        # if model_type == "GPR":
        #     Y_pca = mean.numpy()
        # elif model_type == "LGBM":
        #     Y_pca = latent.numpy()
        Y_pca = mean.numpy()

        pca_model = vae_model

    else:
        raise ValueError("[ERROR] encoder must be either 'PCA' or 'VAE'.")

    return Y_pca, pca_model, mean_val, std_val

# ===== 模块 3：构建regressor =====
def build_regressor(model_type="GPR", kernel_name="RBF_White", encoder="PCA"):
    """
    根据model_type选择回归器
    """
    if model_type == "GPR":
        if encoder == "PCA":
            kernels = {
                "RBF": C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)),
                "Matern_1.5": C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=1.5, length_scale_bounds=(1e-2, 1e3)),
                "Matern_0.5_White": C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=0.5, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1)),
                "RationalQuadratic": C(1.0, (1e-3, 1e3)) * RationalQuadratic(length_scale=1.0, alpha=1.0, length_scale_bounds=(1e-2, 1e3), alpha_bounds=(1e-2, 1e3)),
                "RBF_White": C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1)),
                "Matern_2.5_White": C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=2.5, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1)),
            }
        elif encoder == "VAE":
            kernels = {
                "RBF": C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e4)),
                "Matern_1.5": C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=1.5, length_scale_bounds=(1e-5, 1e3)),
                "Matern_0.5_White": C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=0.5, length_scale_bounds=(1e-5, 1e3)) +
                                    WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-8, 10)),
                "RationalQuadratic": C(1.0, (1e-3, 1e3)) *
                                    RationalQuadratic(length_scale=1.0, alpha=1.0,
                                                    length_scale_bounds=(1e-9, 1e3),
                                                    alpha_bounds=(1e-5, 1e7)),
                "RBF_White": C(1.0, (1e-3, 1e3)) *
                            RBF(length_scale=1.0, length_scale_bounds=(1e-6, 1e4)) +
                            WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-8, 10)),
                "Matern_2.5_White": C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=2.5, length_scale_bounds=(1e-5, 1e3)) +
                                    WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-8, 10))
            }
        if kernel_name not in kernels:
            raise ValueError(f"[ERROR] Kernel '{kernel_name}' not found. Available: {list(kernels.keys())}")


        regressor = GaussianProcessRegressor(kernel=kernels[kernel_name], n_restarts_optimizer=10, random_state=42) 
        #regressor = GaussianProcessRegressor(kernel=kernels[kernel_name], n_restarts_optimizer=5, random_state=42)
        

    elif model_type == "LGBM":
        # regressor = LGBMRegressor(
        #     n_estimators=500,
        #     learning_rate=0.05,
        #     num_leaves=31,
        #     max_depth=-1,
        #     subsample=0.8,
        #     colsample_bytree=0.8,
        #     random_state=42,
        #     n_jobs=-1
        # )
        regressor = LGBMRegressor(
            n_estimators=200,         # 样本少，树数量不能太多
            learning_rate=0.05,       # 合理步长
            num_leaves=10,            # 非常小的叶子数，避免过拟合
            max_depth=3,              # 限制树深度
            subsample=0.7,            # 行采样
            colsample_bytree=0.8,     # 列采样
            reg_alpha=0.1,            # L1 正则
            reg_lambda=1.0,           # L2 正则
            random_state=42,
            n_jobs=-1
        )
    else:
        raise ValueError("[ERROR] model_type must be either 'GPR' or 'LGBM'.")

    return regressor

# ===== 模块 4：加载预测forcing数据 =====
def load_forcing_data(forcing_cfg):
    """
    加载预测阶段的 forcing 输入数据。

    参数：
        forcing_cfg: dict，包含以下字段：
            - "file_path": 基础路径
            - "forcing_file": .res 文件名（预测用）

    返回：
        X_pred: shape = (n_samples, 5)，预测用输入特征
    """
    forcing_path = Path(forcing_cfg["file_path"]) / forcing_cfg["forcing_input"]
    df = pd.read_csv(forcing_path, sep=r"\s+", skiprows=1, header=None)
    df.columns = ['co2', 'obliquity', 'esinw', 'ecosw', 'ice']
    X_pred = df[['co2', 'esinw', 'ecosw', 'obliquity', 'ice']].to_numpy()
    return X_pred

# separate train and test before PCA
def run_training(train_dict, model_type="GPR", kernel="RBF_White", pca_variance_ratio=0.999, encoder="PCA", vae_config=None, seed=42, return_pred=True):
    set_seed(seed)

    # 1. 加载原始数据
    X, Y_flat, var_name, spatial_shape, lat_array, lon_array = load_training_data(train_dict)

    # 2. 划分原始数据
    X_train, X_test, Y_train_flat, Y_test_flat = train_test_split(X, Y_flat, test_size=0.2, random_state=42)

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
                                encoder=encoder, kernel=kernel, save_dir="training/logs")
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
                save_folder="training/maps",
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


# ===== 模块 5：预测 =====
def run_prediction(pipeline, pca_model, forcing_cfg, spatial_shape):
    """
    使用训练好的 pipeline 和 PCA 对预测输入进行模拟预测。

    参数：
        pipeline: 已训练的 sklearn Pipeline
        pca_model: 已拟合的 PCA 模型
        X_pred: (n_samples, 5)，预测输入
        spatial_shape: (lat, lon)，原始空间结构

    返回：
        Y_pred: 模拟预测结果，形状为 (n_samples, lat, lon)
    """
    # 内部调用加载模块
    X_pred = load_forcing_data(forcing_cfg)
    Y_pca_pred = pipeline.predict(X_pred)
    Y_full = pca_model.inverse_transform(Y_pca_pred)
    n = Y_full.shape[0]
    lat, lon = spatial_shape
    Y_out = Y_full.reshape(n, lat, lon)
    return Y_out

# ===== 模块 6：保存结果 =====
def save_prediction(Y_pred, output_dir, file_name="prediction"):
    """
    保存预测结果。
    
    参数：
        Y_pred: (n_samples, lat, lon) 的预测数组
        output_dir: 保存文件夹路径
        file_name: 文件基本名（不要加后缀）
        save_as_netcdf: 是否保存为 .nc 格式；否则保存为 .npy 格式
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存为 netCDF 格式
    n_samples, lat, lon = Y_pred.shape
    da = xr.DataArray(
        data=Y_pred,
        dims=["sample", "lat", "lon"],
        coords={
            "sample": np.arange(n_samples),
            "lat": np.arange(lat),
            "lon": np.arange(lon),
        },
        name="prediction"
    )
    ds = xr.Dataset({"prediction": da})
    save_path = output_dir / f"{file_name}.nc"
    ds.to_netcdf(save_path)
    print(f"[INFO] Prediction saved to {save_path}")

# ===== testing module =====
# ===== 主程序X1：运行训练和预测 =====
def full_emulator_experiment(train_dict, emulator_name, output_dir="outputs",seed=None, vae_config=None):
    """
    全自动尝试所有model+encoder组合，并保存结果。
    """
    # model_kernel_combinations = [
    #     ("GPR", "RBF"),
    #     ("GPR", "RBF_White"),
    #     ("GPR", "Matern_0.5_White"),
    #     ("GPR", "Matern_1.5"),
    #     ("GPR", "RationalQuadratic"),
    #     ("GPR", "Matern_2.5_White"),
    #     ("LGBM", None)  # LGBM不需要kernel
    # ]
    model_kernel_combinations = [
        ("GPR", "Matern_2.5_White"),
        ("LGBM", None)]
 
    encoders = [ "VAE", "PCA"]

    for encoder in encoders:
        for model_type, kernel in model_kernel_combinations:
            print("="*80)
            print(f"[INFO] Training model: {model_type} | Kernel: {kernel} | Encoder: {encoder}")

            emulator = run_training(
                train_dict[emulator_name],
                model_type=model_type,
                kernel=kernel if kernel else "RBF_White",  # 给LGBM随便传一个kernel（无效但占位）
                encoder=encoder,
                vae_config=vae_config,
                seed=1024,  # 固定种子
                return_pred=True
            )

            # 打印得分
            print(f"[RESULT] {model_type} + {encoder} --> Test R² Score: {emulator['gpr_r2_score']:.4f}")

            # 保存预测和真实
            pred_filename = f"{emulator_name}_{model_type}_{kernel if kernel else 'None'}_{encoder}_Ypred.nc"
            true_filename = f"{emulator_name}_{model_type}_{kernel if kernel else 'None'}_{encoder}_Ytrue.nc"

            save_prediction(emulator["Y_pred_out"], output_dir, pred_filename)
            save_prediction(emulator["Y_True_out"], output_dir, true_filename)

# ===== 模块 7：validate training and plot =====
# 计算每个格点的R²分数
def compute_r2_map(Y_true_out, Y_pred_out,lat_array, lon_array):
    """
    对每个格点计算R²分数。

    参数：
    - Y_true_out: (n_samples, lat, lon)
    - Y_pred_out: (n_samples, lat, lon)

    返回：
    - r2_map: (lat, lon)
    """
    n_samples, lat, lon = Y_true_out.shape
    r2_map = np.full((lat, lon), np.nan)
    for i in range(len(lat_array)):
        for j in range(len(lon_array)):
            y_true_series = Y_true_out[:, i, j]
            y_pred_series = Y_pred_out[:, i, j]
            if np.all(np.isfinite(y_true_series)) and np.all(np.isfinite(y_pred_series)):
                if np.std(y_true_series) > 1e-6:
                    r2 = r2_score(y_true_series, y_pred_series)
                    r2_map[i, j] = r2
                else:
                    print(f"[DEBUG] Skipping lat index {i}, lon index {j} due to low std deviation in y_true_series")
            else:
                print(f"[DEBUG] Skipping lat index {i}, lon index {j} due to non-finite values")

    return r2_map

# plot_r2_map_with_latlon函数
def plot_r2_map_with_latlon(r2_map, lat_array, lon_array, model_type, encoder, kernel, save_dir="training/maps"):
    """
    绘制带经纬度的R²空间分布图并保存。
    
    参数：
    - r2_map: (lat, lon) 格点R²值
    - lat_array: (lat,) 维度的纬度数组
    - lon_array: (lon,) 维度的经度数组
    - model_type: str
    - encoder: str
    - kernel: str
    - save_dir: 保存路径
    """

    os.makedirs(save_dir, exist_ok=True)

    # 创建网格
    Lon, Lat = np.meshgrid(lon_array, lat_array)
    # Calculate global mean R² score
    # Calculate area-weighted mean R² score
    weights = np.cos(np.radians(lat_array))  # Latitude-based weights
    weights = weights / np.sum(weights)  # Normalize weights
    global_mean_r2 = np.nansum(r2_map * weights[:, np.newaxis]) / 100.0  # Weighted mean

    fig = plt.figure(figsize=(12,6))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # 设置经纬度范围
    ax.set_global()

    # 添加地理要素
    ax.coastlines()
    
    # 绘制r2 map
    cmap = plt.get_cmap('viridis')
    im = ax.pcolormesh(Lon, Lat, r2_map, cmap=cmap, vmin=0.7, vmax=1, shading='auto', transform=ccrs.PlateCarree())

    cbar = plt.colorbar(im, orientation='horizontal', pad=0.05, aspect=50)
    cbar.set_label('R² Score')

    plt.title(f"R² Score ({model_type} | {encoder} | {kernel}) \nGlobal Mean R²: {global_mean_r2:.4f}")
    plt.tight_layout()

    # 保存
    filename = f"r2_map_{model_type}_{encoder}_{kernel}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[INFO] R² spatial map saved to: {save_path}")


def plot_prediction_maps_with_info(Y_true_out, Y_pred_out, lat_array, lon_array, timestep=0, emulator_name="E11111", encoder_name="PCA", kernel_name="RBF", vmin=None, vmax=None, save_folder="./maps", title_suffix=""):
    """
    画Y_true, Y_pred 和误差图 (Pred-True)，并且保存图片名字包含emulator, encoder, kernel等信息。

    参数：
    - Y_true_out, Y_pred_out: 输入数据
    - timestep: 要画的样本编号
    - emulator_name, encoder_name, kernel_name: 用于保存文件名
    - vmin, vmax: 色标范围，自动统一
    - save_folder: 保存文件的目录
    - title_suffix: 标题后缀
    """
    if timestep == 999:
        true_map = np.mean(Y_true_out, axis=0)
        pred_map = np.mean(Y_pred_out, axis=0)
    else:
        true_map = Y_true_out[timestep, :, :]
        pred_map = Y_pred_out[timestep, :, :]

    error_map = pred_map - true_map

    if vmin is None:
        vmin = min(true_map.min(), pred_map.min())
    if vmax is None:
        vmax = max(true_map.max(), pred_map.max())

    extent = [lon_array.min(), lon_array.max(), lat_array.min(), lat_array.max()]

    # fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), subplot_kw={'projection': ccrs.PlateCarree()})

    im0 = axs[0].imshow(true_map, transform=ccrs.PlateCarree(), extent=extent, cmap='coolwarm', vmin=-10.0, vmax=10.0)
    axs[0].set_title(f"True {title_suffix}")
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(pred_map, transform=ccrs.PlateCarree(), extent=extent, cmap='coolwarm', vmin=-10.0, vmax=10.0)
    axs[1].set_title(f"Predicted {title_suffix}")
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    im2 = axs[2].imshow(error_map, transform=ccrs.PlateCarree(), extent=extent, cmap='RdBu_r', vmin=-2.0, vmax=2.0)
    axs[2].set_title(f"bias (Pred - True) {title_suffix}")
    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    for ax in axs:
        ax.set_global()
        ax.coastlines()

    plt.tight_layout()

    # 自动构建保存路径
    os.makedirs(save_folder, exist_ok=True)
    file_name = f"{emulator_name}_{encoder_name}_{kernel_name}_sample_{timestep}.png"
    save_path = os.path.join(save_folder, file_name)

    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Map saved to {save_path}")

# ===== 主程序X2：多种子训练 =====
def save_seed_vs_score_plot(df, emulator_name, output_dir,
                            seed=None, latent_dim=None, epochs=None, learning_rate=None, batch_size=None, kl_weight=None):
    """
    保存 Seeds vs R² 分数曲线图。
    文件名包含所有超参数信息。
    """
    os.makedirs(output_dir, exist_ok=True)

    # === 统一格式化info_str ===
    if seed is not None and latent_dim is not None:
        info_str = f"seed{seed}_latent{latent_dim}_ep{epochs}_lr{learning_rate}_bs{batch_size}_kl{kl_weight}"
    else:
        info_str = None

    if info_str:
        filename = f"{emulator_name}_seed_vs_score_{info_str}.png"
    else:
        filename = f"{emulator_name}_seed_vs_score.png"

    save_path = os.path.join(output_dir, filename)

    # === 绘制图 ===
    plt.figure(figsize=(8,5))
    plt.plot(df["seed"], df["gpr_r2_score"], 'o-')
    plt.xlabel("Seed")
    plt.ylabel("R² Score")
    plt.title(f"Seeds vs R² Score ({emulator_name})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[INFO] Seeds vs Score plot saved to: {save_path}")

# === 示例调用方式（请替换为你自己的数据文件名） ===
emulator = "highlowmod_ice"
forcing = "rcp85.1"

seeds = 2025

vae_config = {
    "latent_dim": 1024, # 32, 64, 128, 256，512， 1024
    "epochs": 80,
    "learning_rate": 1e-4, # 1e-4, 5e-5, 1e-5
    "batch_size": 128,
    "kl_weight": 0.1 # 0.1, 0.5, 1.0
}

# ===无需遍历所有组合，直接指定模型和编码器===
# model_type = "GPR"  # "GPR" or "LGBM"
# kernels = {"RBF", "Matern_1.5", "Matern_0.5_White", "RationalQuadratic", "RBF_White", "Matern_2.5_White"}
# emulator = run_training(train_dict[emulator],model_type="GPR",kernel="RBF_White",encoder="VAE", latent_dim=32, return_pred=True)
# prediction = run_prediction(emulator["pipeline_model"], emulator["feature_extraction"], forcing_dict[forcing], emulator["spatial_shape"])

# ===多种随机种子训练===
#seeds = [1, 13, 42, 1024, 2025, 3999, 4991]
#best_seed = multi_seed_run_and_select_best(train_dict, emulator_name=emulator, model_type="GPR", kernel="Matern_2.5_White", encoder="VAE", vae_config=vae_config, output_dir="training/", seeds=seeds)

# ===遍历所有组合===
full_emulator_experiment(train_dict, emulator_name=emulator, output_dir="training/",seed=seeds, vae_config=vae_config)

# ===单次训练===
# emulator = run_training(train_dict[emulator],model_type="LGBM",kernel="RationalQuadratic",encoder="PCA", vae_config=vae_config, seed=seeds, return_pred=True)

