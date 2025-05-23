
# === Modularized version preserving all original code ===
import json
import numpy as np
import pandas as pd
import netCDF4 as nc
import xarray as xr
from pathlib import Path
from datetime import datetime

# ==================== Step 1: get path and name ====================
def find_file_recursive(base_dir, keyword):
    matches = list(Path(base_dir).rglob(f"*{keyword}*"))
    if not matches:
        raise FileNotFoundError(f"No file matching {keyword} found in {base_dir}")
    return matches[0]

def get_paths(cfg):
    input_dir = cfg["input_dir"]
    output_dir = cfg["output_dir"]
    res_path_prefixes = cfg["res_path_prefix"]
    res_files = cfg["res_file"]
    nc_path_prefixes = cfg["nc_path_prefix"]
    nc_files = cfg["nc_file"]
    # 构造完整路径并查找文件
    input_paths = [
        find_file_recursive(Path(input_dir) / Path(prefix), res)
        for prefix, res in zip(res_path_prefixes, res_files)
    ]
    output_paths = [
        find_file_recursive(Path(output_dir) / Path(prefix), nc)
        for prefix, nc in zip(nc_path_prefixes, nc_files)
    ]
    return input_paths, output_paths

def generate_output_lists(exp_ids, postfix_dict):
    return [[exp + ch for ch in postfix_dict[exp]] for exp in exp_ids]

# ==================== Step 2: get res file ====================
def read_res_file(res_path):
        return pd.read_csv(res_path, sep='\s+', header=0)

# ==================== Step 3: read nc file ====================
def read_nc_variables(nc_path, var_names, nlat=73, nlon=96):
    ds = nc.Dataset(nc_path)
    data = []
    for var in var_names:
        if var in ds.variables:
            array = ds.variables[var][:].data.reshape(nlat * nlon)
            print(f"[ℹ] Variable {var}: shape={array.shape}, NaN count={np.isnan(array).sum()}")
            data.append(array)
        else:
            print(f"[!] Variable {var} not found in {nc_path}")
            data.append(np.full(nlat * nlon, np.nan))
    ds.close()
    return np.array(data)
# ==================== Step 4: combine training data and save  ====================
def build_training_data(input_paths, output_paths, output_lists, save_res=None, save_nc=None, nlat=73, nlon=96):
    all_X, all_y = [], []
    for i in range(len(input_paths)):
        X = read_res_file(input_paths[i]).values
        y = read_nc_variables(output_paths[i], output_lists[i], nlat=nlat, nlon=nlon)
        if X.shape[0] != y.shape[0]:
            print(f"[!] Sample count mismatch: {X.shape[0]} vs {y.shape[0]}")
            continue
        all_X.append(X)
        all_y.append(y)
    X = np.vstack(all_X)
    y = np.vstack(all_y)

    # === 保存 X 为 .res ===
    if save_res:
        Path(save_res).parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(save_res, X)
        print(f"[📄] Saved X as .res to {save_res}")
    # === 保存 y 为 .nc ===
    if save_nc:
        Path(save_nc).parent.mkdir(parents=True, exist_ok=True)
        reshaped_y = y.reshape((y.shape[0], nlat, nlon))
        ds = xr.Dataset(
            {"var": (["id", "lat", "lon"], reshaped_y)},
            coords={
                "id": np.arange(y.shape[0]),
                "lat": np.linspace(-90, 90, nlat),
                "lon": np.linspace(0, 360, nlon)
            }
        )
        ds.to_netcdf(save_nc)
        print(f"[🌍] Saved y as NetCDF to {save_nc}")
    return X, y

# ==================== Step 5: save log ====================
def save_log(input_paths, output_paths, save_res,save_nc, shape_X, shape_y):
    log_dir = Path("test_log")
    log_dir.mkdir(exist_ok=True)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_log_{now}.json"
    log_content = {
        "timestamp": now,
        "input_files": [str(p) for p in input_paths],
        "output_files": [str(p) for p in output_paths],
        "save_x_res_path": str(save_res),
        "save_y_nc_path": str(save_nc),
        "X_shape": shape_X,
        "y_shape": shape_y
    }
    with open(log_file, "w") as f:
        json.dump(log_content, f, indent=2)
    print(f"[📝] Log saved to {log_file}")

    # ==================== Step 6: Run full pipeline ====================
def preprocess_data(cfg):
    """
    This function wraps the original preprocessing script,
    using a config dict to specify data path and metadata.
    """

    exp_ids = cfg["exp_ids"]
    postfix_dict = cfg["postfix_dict"]
    
    input_paths, output_paths = get_paths(cfg)
    output_lists = generate_output_lists(exp_ids, postfix_dict)
    # Replace specific terms in output_lists
    for i, output_list in enumerate(output_lists):
        output_lists[i] = [item.replace("tdab28k", "tdab").replace("tdab80k", "tdab").replace("tdab120k", "tdab") for item in output_list]
    
    print(f"[🔍] Output lists: {output_lists}")

    X, y = build_training_data(input_paths, output_paths, output_lists,
                               save_res=cfg["formatted_res"],
                               save_nc=cfg["formatted_nc"]
                            )
                            # Dynamically set save paths for .res and .nc files
    
    print(f"[✔] Training data saved to {cfg['formatted_res']} and {cfg['formatted_nc']}")
    save_log(input_paths, output_paths, cfg["formatted_res"], cfg["formatted_nc"], X.shape, y.shape)

    return X, y
    # ---- End of original script ----

# CLI / test interface
if __name__ == "__main__":
    from examples.config.training_dict_raw import dict_raw
    preprocess_data(dict_raw["highlowmod_ice"])
