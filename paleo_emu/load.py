"""
This module provides functions to load training and forcing data for the paleo-EMU.
"""
import yaml
import xarray as xr
import pandas as pd
from pathlib import Path


def load_training_data(cfg_path):
    """
    Load training data.
    cfg_path: dict, path-to-yaml, or Path object.
    Expects resolved config to provide:
      - file_path (base dir)
      - X_input (filename for .res)
      - Y_output (filename for .nc)
    Returns: X (DataFrame), Y_flat (ndarray), var_name, spatial_shape (lat,lon), lat_array, lon_array
    """
    # accept either a dict (already parsed) or a path to a yaml file
    if isinstance(cfg_path, dict):
        cfg = cfg_path
    else:
        cfg_file = Path(cfg_path)
        if not cfg_file.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        with open(cfg_file, "r") as fh:
            cfg = yaml.safe_load(fh)

    base_path = Path(cfg.get("training_file_path", "."))
    x_name = cfg.get("X_input")
    y_name = cfg.get("Y_output")

    x_path = base_path / x_name
    y_path = base_path / y_name

    if not x_path.exists():
        raise FileNotFoundError(f"X input file not found: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Y output file not found: {y_path}")

    # read X (.res whitespace-delimited)
    df = pd.read_csv(x_path, sep=r"\s+", header=None)
    # allow files with extra cols; ensure first five meaningful columns exist
    if df.shape[1] < 5:
        raise ValueError(f"Unexpected X shape {df.shape} for {x_path}")
    # name columns in expected order (if file already had header, user should adjust)
    df.columns = ['co2', 'obliquity', 'esinw', 'ecosw', 'ice'] + [f"c{i}" for i in range(df.shape[1]-5)]
    X = df[['co2', 'esinw', 'ecosw', 'obliquity', 'ice']]

    # read Y (netCDF)
    ds = xr.open_dataset(y_path)
    var_name = list(ds.data_vars)[0]
    dims = ds[var_name].dims
    if len(dims) < 3:
        raise ValueError(f"Unexpected Y dims {dims} in {y_path}")
    Y = ds[var_name].values  # (n_samples, lat, lon)
    Y_flat = Y.reshape(Y.shape[0], -1)
    lat_name = dims[1]
    lon_name = dims[2]
    lat_array = ds[lat_name].values
    lon_array = ds[lon_name].values

    return X, Y_flat, var_name, (Y.shape[1], Y.shape[2]), lat_array, lon_array

def load_forcing_data(forcing_cfg,scenario="rcp85.1"):
    """
    forcing_cfg can be dict or path or key inside emulator.yaml forcing_data section.
    Expected keys: file_path, forcing_input
    """
    cfg = forcing_cfg
    if not isinstance(cfg, dict):
        # load from yaml file
        if isinstance(forcing_cfg, (str, Path)):
            cfg_file = Path(forcing_cfg)
            if not cfg_file.exists():
                raise FileNotFoundError(f"Forcing config file not found: {forcing_cfg}")
            with open(cfg_file, "r") as fh:
                cfg = yaml.safe_load(fh)
        else:
            raise TypeError("forcing_cfg must be dict, str, or Path")
        
    # now cfg should be dict
    base = Path(cfg.get("file_path", "."))
    forcing_input = cfg.get("scenarios", {}).get(scenario, {})
    if not forcing_input:
        raise KeyError("forcing config must include forcing_input")

    forcing_path = base / forcing_input
    if not forcing_path.exists():
        raise FileNotFoundError(f"forcing file not found: {forcing_path}")

    df = pd.read_csv(forcing_path, sep=r"\s+", skiprows=1, header=None)
    if df.shape[1] < 5:
        raise ValueError(f"Unexpected forcing file shape {df.shape} for {forcing_path}")
    df.columns = ['co2', 'obliquity', 'esinw', 'ecosw', 'ice'] + [f"c{i}" for i in range(df.shape[1]-5)]
    X_pred = df[['co2', 'esinw', 'ecosw', 'obliquity', 'ice']]
    return X_pred
