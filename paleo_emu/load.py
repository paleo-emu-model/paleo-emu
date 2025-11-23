"""
This module provides functions to load training and forcing data for the paleo-EMU.
"""
import yaml
import xarray as xr
import pandas as pd
from pathlib import Path
import os

def load_training_data(model_configuration):
    """
  
    """
    X_path = os.path.join(model_configuration.get("training_file_path"), model_configuration.get("X_input_file_name"))
    Y_path = os.path.join(model_configuration.get("training_file_path"), model_configuration.get("Y_input_file_name"))  

    # read X (.res whitespace-delimited)
    df = pd.read_csv(X_path, sep=r"\s+", header=None)
    # allow files with extra cols; ensure first five meaningful columns exist
    if df.shape[1] < 5:
        raise ValueError(f"Unexpected X shape {df.shape} for {X_path}")
    
    column_names = model_configuration.get("X_column_names")
    # name columns in expected order (if file already had header, user should adjust)
    df.columns = column_names + [f"c{i}" for i in range(df.shape[1]-5)]
    X = df[column_names].values  # (n_samples, n_features)

    # read Y (netCDF)
    ds = xr.open_dataset(Y_path)
    var_name = list(ds.data_vars)[0]
    dims = ds[var_name].dims
    if len(dims) < 3:
        raise ValueError(f"Unexpected Y dims {dims} in {Y_path}")
    Y = ds[var_name].values  # (n_samples, lat, lon)
    Y_flat = Y.reshape(Y.shape[0], -1)
    lat_name = dims[1]
    lon_name = dims[2]
    lat_array = ds[lat_name].values
    lon_array = ds[lon_name].values

    return X, Y_flat, var_name, (Y.shape[1], Y.shape[2]), lat_array, lon_array

def load_forcing_data(model_cfg, scenario="rcp85.1"):
    """
    model_cfg can be dict or path or key inside emulator.yaml forcing_data section.
    Expected keys: file_path, forcing_input
    """
    cfg = model_cfg
    if not isinstance(cfg, dict):
        # load from yaml file
        if isinstance(model_cfg, (str, Path)):
            cfg_file = Path(model_cfg)
            if not cfg_file.exists():
                raise FileNotFoundError(f"Forcing config file not found: {model_cfg}")
            with open(cfg_file, "r") as fh:
                cfg = yaml.safe_load(fh)
        else:
            raise TypeError("model_cfg must be dict, str, or Path")
    
    print(cfg)
    # now cfg should be dict
    base = Path(cfg["forcing_data"]["file_path"])
    scenario_cfg = cfg["forcing_data"].get(scenario)
    forcing_input = scenario_cfg["forcing_input"]
    print("[DEBUG]", forcing_input)
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
