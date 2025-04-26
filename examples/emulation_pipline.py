import xarray as xr
import pandas as pd
import numpy as np

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
    Y = ds[var_name].values  # (n_samples, lat, lon)
    Y_flat = Y.reshape(Y.shape[0], -1)

    return X, Y_flat, var_name, Y.shape[1:]


# ======== 模块2.0： VAE 定义 =========
class VAE(tf.keras.Model):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        # 编码器
        self.encoder = models.Sequential([
            layers.InputLayer(input_shape=(input_dim,)),
            layers.Dense(512, activation="relu"),
            layers.Dense(256, activation="relu"),
            layers.Dense(latent_dim * 2)  # 输出 mean 和 log_var
        ])
        # 解码器
        self.decoder = models.Sequential([
            layers.InputLayer(input_shape=(latent_dim,)),
            layers.Dense(256, activation="relu"),
            layers.Dense(512, activation="relu"),
            layers.Dense(input_dim)  # 重建回输入维度
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
def extract_features(Y_flat, encoder="PCA", pca_variance_ratio=0.999,seed=42, latent_dim=32):
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
        # batch_size = 32
        # epochs = 50

        epochs = 150
        learning_rate = 1e-4
        batch_size = 64

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        input_dim = Y_flat.shape[1]
        vae_model = VAE(input_dim, latent_dim)

        dataset = tf.data.Dataset.from_tensor_slices((Y_flat.astype('float32')))
        dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

        # --- 新增：用列表保存每个epoch的loss ---
        epoch_losses = []

        for epoch in range(epochs):
            total_loss = 0
            for step, x_batch in enumerate(dataset):
                with tf.GradientTape() as tape:
                    x_decoded, mean, logvar = vae_model(x_batch)
                    loss = compute_vae_loss(x_batch, x_decoded, mean, logvar)

                grads = tape.gradient(loss, vae_model.trainable_variables)
                optimizer.apply_gradients(zip(grads, vae_model.trainable_variables))
                total_loss += loss

            avg_loss = total_loss / (step+1)
            epoch_losses.append(avg_loss.numpy())  # 把loss保存进列表

            if epoch % 10 == 0 or epoch == epochs-1:
                print(f"[VAE] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

        # --- 训练结束后绘制并保存loss曲线 ---
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,5))
        plt.plot(range(1, len(epoch_losses)+1), epoch_losses, label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"VAE Training Loss Curve (seed={seed})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"training/vae_training_loss_curve_seed{seed}.hp2.png", dpi=300)  # 保存为高清png
        plt.show()

        print("[INFO] VAE training loss curve saved to 'vae_training_loss_curve.png'")

        # --- 最后继续进行特征提取 ---
        mean_logvar = vae_model.encoder(Y_flat)
        mean, logvar = tf.split(mean_logvar, num_or_size_splits=2, axis=1)
        Y_pca = mean.numpy()
        pca_model = vae_model

    else:
        raise ValueError("[ERROR] encoder must be either 'PCA' or 'VAE'.")

    return Y_pca, pca_model, mean_val, std_val


# ===== 模块 3：构建regressor =====
def build_regressor(model_type="GPR", kernel_name="RBF_White"):
    """
    根据model_type选择回归器
    """
    if model_type == "GPR":
        kernels = {
            "RBF": C(1.0) * RBF(length_scale=1.0),
            "Matern_1.5": C(1.0) * Matern(length_scale=1.0, nu=1.5),
            "Matern_0.5_White": C(1.0) * Matern(length_scale=1.0, nu=0.5) + WhiteKernel(noise_level=1e-2),
            "RationalQuadratic": C(1.0) * RationalQuadratic(length_scale=1.0, alpha=1.0),
            "RBF_White": C(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-2),
            "Matern_2.5_White": C(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-2),
        }
        if kernel_name not in kernels:
            raise ValueError(f"[ERROR] Kernel '{kernel_name}' not found. Available: {list(kernels.keys())}")

        regressor = GaussianProcessRegressor(kernel=kernels[kernel_name], n_restarts_optimizer=5, random_state=42)

    elif model_type == "LGBM":
        regressor = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
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

# ===== 主程序：包含模块3（构建 pipeline）+ 模块4（训练与评估） =====
def run_training(train_dict, model_type="GPR", kernel="RBF_White", pca_variance_ratio=0.999, encoder="PCA", latent_dim=32,seed=42, return_pred=True):
    """
    高效版run_training，调用各模块。
    """
    set_seed(seed)  # 设置随机种子
    # 1. 加载数据
    X, Y_flat, var_name, spatial_shape = load_training_data(train_dict)

    # 2. 特征提取
    Y_pca, feature_extractor,mean_val,std_val = extract_features(Y_flat, encoder=encoder, pca_variance_ratio=pca_variance_ratio,seed=seed, latent_dim=latent_dim)

    # 3. 划分数据
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_pca, test_size=0.1, random_state=42)

    # 4. 构建回归器
    regressor = build_regressor(model_type=model_type, kernel_name=kernel)

    # 5. 建立并训练Pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", MultiOutputRegressor(regressor))
    ])
    pipeline.fit(X_train, Y_train)

    # 6. 预测和还原
    if return_pred:
        Y_pred_pca = pipeline.predict(X_test)

        if encoder == "PCA":
            Y_pred_full = feature_extractor.inverse_transform(Y_pred_pca)
            Y_test_full = feature_extractor.inverse_transform(Y_test)
        elif encoder == "VAE":
            Y_pred_full = feature_extractor.decoder.predict(Y_pred_pca)
            Y_test_full = feature_extractor.decoder.predict(Y_test)

        # 还原标准化
        Y_pred_full = Y_pred_full * std_val + mean_val
        Y_test_full = Y_test_full * std_val + mean_val

        n = Y_pred_full.shape[0]
        lat, lon = spatial_shape
        Y_pred_out = Y_pred_full.reshape(n, lat, lon)
        Y_test_out = Y_test_full.reshape(n, lat, lon)
        score = r2_score(Y_test_full, Y_pred_full)

        return {
            "pipeline_model": pipeline,
            "feature_extraction": feature_extractor,
            "gpr_r2_score": score,
            "n_components_retained": Y_pca.shape[1],
            "original_variable": var_name,
            "spatial_shape": spatial_shape,
            "Y_pred_out": Y_pred_out,
            "Y_True_out": Y_test_out,
            "X_test": X_test,
            "encoder_used": encoder,
            "model_type": model_type
        }
    else:
        score = pipeline.score(X_test, Y_test)
        return {
            "pipeline_model": pipeline,
            "feature_extraction": feature_extractor,
            "gpr_r2_score": score,
            "n_components_retained": Y_pca.shape[1],
            "original_variable": var_name,
            "spatial_shape": spatial_shape,
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

# ===== 主程序X1：运行训练和预测 =====
def full_emulator_experiment(train_dict, emulator_name, output_dir="outputs"):
    """
    全自动尝试所有model+encoder组合，并保存结果。
    """
    model_kernel_combinations = [
        ("GPR", "RBF"),
        ("GPR", "RBF_White"),
        ("GPR", "Matern_0.5_White"),
        ("GPR", "Matern_1.5"),
        ("GPR", "RationalQuadratic"),
        ("GPR", "Matern_2.5_White"),
        ("LGBM", None)  # LGBM不需要kernel
    ]

    encoders = ["PCA", "VAE"]

    for model_type, kernel in model_kernel_combinations:
        for encoder in encoders:
            print("="*80)
            print(f"[INFO] Training model: {model_type} | Kernel: {kernel} | Encoder: {encoder}")

            emulator = run_training(
                train_dict[emulator_name],
                model_type=model_type,
                kernel=kernel if kernel else "RBF_White",  # 给LGBM随便传一个kernel（无效但占位）
                encoder=encoder,
                latent_dim=32,
                return_pred=True
            )

            # 打印得分
            print(f"[RESULT] {model_type} + {encoder} --> Test R² Score: {emulator['gpr_r2_score']:.4f}")

            # 保存预测和真实
            pred_filename = f"{emulator_name}_{model_type}_{kernel if kernel else 'None'}_{encoder}_Ypred.nc"
            true_filename = f"{emulator_name}_{model_type}_{kernel if kernel else 'None'}_{encoder}_Ytrue.nc"

            save_prediction(emulator["Y_pred_out"], output_dir, pred_filename)
            save_prediction(emulator["Y_True_out"], output_dir, true_filename)

# ===== 主程序X2：多种子训练 =====
def multi_seed_run_and_select_best(train_dict, emulator_name, model_type="GPR", kernel="RBF_White",
                                   encoder="VAE", latent_dim=32, output_dir="outputs",seeds=None):
    """
    多个随机种子训练，自动记录并选出最佳seed。
    """
    if seeds is None:
        print(f"[INFO] Randomly generated seeds: {seeds}")
    else:
        print(f"[INFO] Using provided seeds: {seeds}")
    
    results = []

    os.makedirs(output_dir, exist_ok=True)

    for seed in seeds:
        print("="*80)
        print(f"[INFO] Running training with seed {seed}")

        emulator = run_training(
            train_dict[emulator_name],
            model_type=model_type,
            kernel=kernel,
            encoder=encoder,
            latent_dim=latent_dim,
            return_pred=True,
            seed=seed
        )

        final_score = emulator["gpr_r2_score"]

        # 保存预测结果
        pred_filename = f"{emulator_name}_{model_type}_{kernel}_{encoder}_seed{seed}_Ypred.nc"
        true_filename = f"{emulator_name}_{model_type}_{kernel}_{encoder}_seed{seed}_Ytrue.nc"

        save_prediction(emulator["Y_pred_out"], output_dir, pred_filename)
        save_prediction(emulator["Y_True_out"], output_dir, true_filename)

        # 保存记录
        results.append({
            "seed": seed,
            "gpr_r2_score": final_score
        })

    # === 生成summary表格 ===
    df = pd.DataFrame(results)
    df_sorted = df.sort_values(by="gpr_r2_score", ascending=False)  # R²越高越好
    df_sorted.to_csv(os.path.join(output_dir, f"{emulator_name}_multi_seed_summary.csv"), index=False)
    print(f"[INFO] Saved multi-seed summary to {output_dir}/{emulator_name}_multi_seed_summary.csv")

    # === 选出最佳seed并保存 ===
    best_seed = int(df_sorted.iloc[0]['seed'])
    best_score = df_sorted.iloc[0]['gpr_r2_score']

    print(f"[RESULT] Best seed: {best_seed} with R² score: {best_score:.4f}")

    # === 绘制seeds vs score图 ===
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,5))
    plt.plot(df["seed"], df["gpr_r2_score"], 'o-')
    plt.xlabel("Seed")
    plt.ylabel("R² Score")
    plt.title(f"Seeds vs R² Score ({emulator_name})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{emulator_name}_seed_vs_score.png"), dpi=300)
    plt.show()

    print(f"[INFO] Seeds vs Score plot saved to {output_dir}/{emulator_name}_seed_vs_score.png")

    return best_seed

# === 示例调用方式（请替换为你自己的数据文件名） ===
emulator = "lowmod_ice"
forcing = "rcp85.1"
# ===无需遍历所有组合，直接指定模型和编码器===
# model_type = "GPR"  # "GPR" or "LGBM"
# kernels = {"RBF", "Matern_1.5", "Matern_0.5_White", "RationalQuadratic", "RBF_White", "Matern_2.5_White"}
# emulator = run_training(train_dict[emulator],model_type="GPR",kernel="RBF_White",encoder="VAE", latent_dim=32, return_pred=True)
# prediction = run_prediction(emulator["pipeline_model"], emulator["feature_extraction"], forcing_dict[forcing], emulator["spatial_shape"])
# print("R² score:", emulator["gpr_r2_score"])
# save_prediction(emulator["Y_pred_out"], output_dir="training/", file_name="prediction_output")
# save_prediction(emulator["Y_True_out"], output_dir="training/", file_name="true_output")
# ===遍历所有组合===
#full_emulator_experiment(train_dict, emulator_name=emulator, output_dir="training/")

# ===多种随机种子训练===
seeds = [1, 13, 42, 1024, 2025, 3999, 4991]
best_seed = multi_seed_run_and_select_best(train_dict, emulator_name=emulator, model_type="GPR", kernel="Matern_2.5_White", encoder="VAE", latent_dim=64, output_dir="training/", seeds=seeds)
