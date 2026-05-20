"""
This module provides functions to load training and forcing data for the paleo-EMU.
"""

from pathlib import Path

import xarray as xr
import pandas as pd
import numpy as np

from paleo_emu.config import PaleoEmuConfig 

def load_training_data(model_configuration: PaleoEmuConfig):
    """
    Load X (forcing/inputs) and Y (target fields) for training.

    Parameters
    ----------
    model_configuration : PaleoEmuConfig 

    Returns
    -------
    X : ndarray, shape (n_samples, n_features)
    Y_flat : ndarray, shape (n_samples, n_lat * n_lon)
    var_name : str
        Name of the variable in the NetCDF file.
    spatial_shape : tuple
        (n_lat, n_lon)
    lat_array : ndarray
    lon_array : ndarray
    """

    training_dir = model_configuration.training_file_path
    X_name = model_configuration.X_input_file_name
    Y_name = model_configuration.Y_input_file_name
    column_names = model_configuration.X_column_names

    training_dir = Path(training_dir)
    X_path = training_dir / X_name
    Y_path = training_dir / Y_name

    # ---- Load X (.res whitespace-delimited) ----
    df = pd.read_csv(X_path, sep=r"\s+", header=None)

    n_features = len(column_names)
    if df.shape[1] < n_features:
        raise ValueError(
            f"Unexpected X shape {df.shape} for {X_path} "
            f"(needs at least {n_features} columns)"
        )

    # Name columns: the first n_features are the meaningful ones,
    # any extra columns get generic names c0, c1, ...
    extra_cols = df.shape[1] - n_features
    df.columns = column_names + [f"c{i}" for i in range(extra_cols)]

    X = df[column_names].values  # (n_samples, n_features)

    ind_co2 = column_names.index('co2')
    if (X[:, ind_co2] <= 0).any():
        raise ValueError("CO2 column contains non-positive values; log transform requires CO2 > 0.")
    X[:, ind_co2] = np.log(X[:, ind_co2])

    # ---- Load Y (NetCDF) ----
    ds = xr.open_dataset(Y_path, engine="h5netcdf")

    # Check if there are data variables; if not, look for the first non-coordinate variable
    if len(ds.data_vars) > 0:
        var_name = list(ds.data_vars)[0]
    else:
        # Fallback: get the first coordinate variable that has multiple dimensions
        var_name = [v for v in ds.coords if len(ds[v].dims) > 1][0]

    dims = ds[var_name].dims
    if len(dims) < 3:
        raise ValueError(f"Unexpected Y dims {dims} in {Y_path}")

    Y = ds[var_name].values  # (n_samples, lat, lon)
    Y_flat = Y.reshape(Y.shape[0], -1)

    if X.shape[0] != Y_flat.shape[0]:
        raise ValueError(
            f"X and Y sample counts do not match: X has {X.shape[0]} rows "
            f"({X_path}), Y has {Y_flat.shape[0]} samples ({Y_path})."
        )

    lat_name = dims[1]
    lon_name = dims[2]
    lat_array = ds[lat_name].values
    lon_array = ds[lon_name].values

    return X, Y_flat, var_name, (Y.shape[1], Y.shape[2]), lat_array, lon_array

def load_forcing_data(model_configuration: PaleoEmuConfig, scenario="rcp85.1"):
    """
    model_cfg can be inside emulator.yaml forcing_data section.
    Expected keys: file_path, forcing_input
    """

    base = model_configuration.forcing_data_path
    scenario_cfg = model_configuration.forcing_data.get(scenario)
    if scenario_cfg is None:
        raise KeyError(f"Scenario '{scenario}' not found in config. "
                       f"Available: {list(model_configuration.forcing_data.keys())}")
    forcing_input = scenario_cfg.get("forcing_input")
    if not forcing_input:
        raise KeyError("forcing config must include 'forcing_input'")

    forcing_path = base / forcing_input
    if not forcing_path.exists():
        raise FileNotFoundError(f"forcing file not found: {forcing_path}")

    df = pd.read_csv(forcing_path, sep=r"\s+", skiprows=0, header=None)
    
    if df.shape[1] < 5:
        raise ValueError(f"Unexpected forcing file shape {df.shape} for {forcing_path}")
    

    X_headers = ['co2', 'obliquity', 'esinw', 'ecosw', 'ice']

    df.columns = X_headers + [f"c{i}" for i in range(df.shape[1]-5)]
    X_pred = df[X_headers].copy()

    ind_co2 = X_headers.index('co2')
    if (X_pred.iloc[:, ind_co2] <= 0).any():
        raise ValueError("CO2 column contains non-positive values; log transform requires CO2 > 0.")
    X_pred.iloc[:, ind_co2] = np.log(X_pred.iloc[:, ind_co2])

    return X_pred.to_numpy()
