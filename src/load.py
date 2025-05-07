

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