
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import os


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
