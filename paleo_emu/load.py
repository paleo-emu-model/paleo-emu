"""
This module provides functions to load training and forcing data for the paleo-EMU.
"""

import xarray as xr
import pandas as pd
from pathlib import Path

# ===== 模块 1：加载数据 =====
def load_training_data(cfg):
    """
    load training data.
    parameter:
        cfg: configuration dictionary containing file path information.
            - cfg["file_path"]: base directory
            - cfg["X_input"]: .res file name for X
    return:
        X: (n_samples, 5) the input feature matrix
        Y_flat: (n_samples, lat*lon) the flattened output matrix
        var_name: variable name in Y
        spatial_shape: original (lat, lon) shape
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
    # join paths
    base_path = Path(cfg["file_path"])
    x_path = base_path / cfg["X_input"]
    y_path = base_path / cfg["Y_output"]

    # read X
    df = pd.read_csv(x_path, sep=r"\s+", header=None)
    df.columns = ['co2', 'obliquity', 'esinw', 'ecosw', 'ice']
    X = df[['co2', 'esinw', 'ecosw', 'obliquity', 'ice']] #.to_numpy()

    # read Y
    ds = xr.open_dataset(y_path)
    var_name = list(ds.data_vars)[0]
    lat_name = ds[var_name].dims[1]
    lon_name = ds[var_name].dims[2]
    Y = ds[var_name].values  # (n_samples, lat, lon)
    Y_flat = Y.reshape(Y.shape[0], -1)
    lat_array = -ds[lat_name].values
    lon_array = ds[lon_name].values

    return X, Y_flat, var_name, Y.shape[1:], lat_array, lon_array

# ===== 模块 4：加载预测forcing数据 =====
def load_forcing_data(forcing_cfg):
    """
    load forcing data for prediction.
    parameter:
        forcing_cfg: dict containing the following fields:
            - "file_path": base directory
            - "forcing_file": .res file name for forcing input
    return:
        X_pred: shape = (n_samples, 5) the input feature matrix for prediction
    """
    forcing_path = Path(forcing_cfg["file_path"]) / forcing_cfg["forcing_input"]
    df = pd.read_csv(forcing_path, sep=r"\s+", skiprows=1, header=None)
    df.columns = ['co2', 'obliquity', 'esinw', 'ecosw', 'ice']
    X_pred = df[['co2', 'esinw', 'ecosw', 'obliquity', 'ice']] #.to_numpy()
    return X_pred
