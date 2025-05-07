
# sklearn 风格 pipeline 封装的气候模拟训练脚本
import pandas as pd
import numpy as np
from netCDF4 import Dataset
from pathlib import Path
import os

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, Matern
from sklearn.decomposition import PCA

from examples.config.hp_config import hp, nkeep


# === Load configuration ===
from examples.config.training_dict_formatted import training_dict


class GPComponent(BaseEstimator, RegressorMixin):
    def __init__(self, hp, output_dir='emulator', label='emulator'):
        self.hp = hp  # shape: (6, nkeep)
        self.output_dir = output_dir
        self.models = []
        self.label = label

    def fit(self, X, Y):
        os.makedirs(self.output_dir, exist_ok=True)
        nkeep = Y.shape[1]

        for i in range(nkeep):
            # 提取该主成分对应的 hp
            lscales = self.hp.iloc[0:5, i].values   # 5 input features
            nugget = self.hp.iloc[5, i]

            # 构建 RBF kernel
            #kernel = ConstantKernel(1.0) * RBF(length_scale=lscales) + WhiteKernel(noise_level=nugget)
            #This is the one for testing
            kernel = ConstantKernel() * Matern(length_scale=lscales, nu=0.5) + WhiteKernel(noise_level=nugget)
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
            gp.fit(X, Y[:, i])
            self.models.append(gp)

        return self
    
    def export_to_hdf5(self, emu_path, X, Y, regress="linear"):
        import h5py
        import datetime
        
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        h5_path = output_path / (self.label + "_GPList.h5")

        if h5_path.exists():
            backup_name = h5_path.with_name(f"GPList_{datetime.datetime.now().strftime('%Y%m%d')}.h5")
            h5_path.rename(backup_name)

        with h5py.File(h5_path, "w") as file:
            for i, model in enumerate(self.models):
                group = file.create_group(f"PC_{i+1:02}")
                group.create_dataset("X", data=X)
                group.create_dataset("Y", data=Y[:, i])
                group.create_dataset("lambda", data=model.alpha_)
                group.attrs["kernel"] = str(model.kernel_)
                group.attrs["regress"] = regress

        print("[✔] Exported GP models to HDF5:", h5_path)



def load_training_data(cfg):
    """
    加载训练数据。假设：
    - X_input 是 .res 文件，相对路径
    - Y_output 是 .nc 文件，相对路径
    - file_path 是统一的目录路径（字符串）

    返回：
    - X: 输入特征矩阵 (n_samples, n_features)
    - Y: 输出变量矩阵 (n_samples, n_targets)
    """

    # === 构造完整路径 ===
    base_path = Path(cfg["file_path"])
    x_path = base_path / cfg["X_input"]
    y_path = base_path / cfg["Y_output"]

    # === 加载输入数据 (.res) ===
    print(f"[INFO] Loading input from: {x_path}")
    cont_paramdat = pd.read_csv(x_path, sep='\\s+', header=None)

    # 手动设置列名（必须确保顺序与数据一致）
    cont_paramdat.columns = ['co2', 'obliquity', 'esinw', 'ecosw', 'ice']

    print(f"[INFO] Input shape: {cont_paramdat.shape}")
    print(f"[INFO] Columns: {list(cont_paramdat.columns)}")

    X = np.stack([
        cont_paramdat["co2"],
        cont_paramdat["esinw"],
        cont_paramdat["ecosw"],
        cont_paramdat["obliquity"],
        cont_paramdat["ice"]
    ], axis=1)
    print(f"[INFO] Final X shape: {X.shape}")

    # === 加载输出数据 (.nc) ===
    print(f"[INFO] Loading output from: {y_path}")
    temp_file = Dataset(y_path, mode="r")
    varname = "var"
    temp_all = temp_file.variables[varname][:]  # shape: (samples, lat, lon)
    print(f"[INFO] Y shape before flatten: {temp_all.shape}")
    Y = temp_all.reshape((temp_all.shape[0], -1))  # flatten to 2D
    print(f"[INFO] Y shape after flatten: {Y.shape}")
    # 转换为标准 ndarray
    Y = Y.filled(np.nan) if isinstance(Y, np.ma.MaskedArray) else Y
    temp_file.close()

    return X, Y

def save_pca_components(PCs, output_path):
    # 保存 Y_pca 为 NetCDF 文件
    output_nc_path = Path(cfg["file_path"]) / f"{cfg['label']}_Y_pca.nc"
    with Dataset(output_nc_path, "w", format="NETCDF4") as ncfile:
        # 创建维度
        ncfile.createDimension("samples", Y_pca.shape[0])
        ncfile.createDimension("components", Y_pca.shape[1])

        # 创建变量
        samples_var = ncfile.createVariable("samples", "i4", ("samples",))
        components_var = ncfile.createVariable("components", "i4", ("components",))
        y_pca_var = ncfile.createVariable("Y_pca", "f4", ("samples", "components"))

        # 写入数据
        samples_var[:] = np.arange(Y_pca.shape[0])
        components_var[:] = np.arange(Y_pca.shape[1])
        y_pca_var[:, :] = Y_pca

        # 添加元数据
        ncfile.description = "PCA-transformed output data"
        ncfile.history = f"Created on {datetime.datetime.now().isoformat()}"

     print(f"[✔] Saved Y_pca to NetCDF: {output_nc_path}")


def process_training(cfg):
    print(f"[INFO] Loading training data from: {cfg['X_input']} and {cfg['Y_output']}")
    X_input, Y_output = load_training_data(cfg)

    # 对输出 Y 做 PCA
    pca_model = PCA(n_components=nkeep)
    Y_pca = pca_model.fit_transform(Y_output)

    # 构建 pipeline
    pipeline = Pipeline([
        ('scale', StandardScaler()),
        ('gp', GPComponent(hp=hp, output_dir=cfg["file_path"], label=cfg["label"]))  # GP 模型输出路径
    ])

    pipeline.fit(X_input, Y_pca)

    # ✅ 显式保存训练模型参数
    pipeline.named_steps["gp"].export_to_hdf5(Path(cfg["file_path"]), X_input, Y_pca)

    print("[✔] Sklearn pipeline completed and model exported.")
    return pipeline



if __name__ == "__main__":
    cfg = training_dict["lowmod_ice"]
    process_training(cfg)
