
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os


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
