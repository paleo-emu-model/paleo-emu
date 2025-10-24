"""
save the training log for VAE.
- loss curve plot
- hyperparameter + final loss CSV record
"""

import xarray as xr
import numpy as np
from pathlib import Path

def save_prediction(Y_pred, lat_array, lon_array, output_dir, file_name="prediction"):
    """
    Save the prediction results.
    Parameters:
        Y_pred: (n_samples, lat, lon) 
        output_dir:path to save data
        file_name: 
        save_as_netcdf: .nc or .npy format
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as netCDF format
    n_samples = Y_pred.shape[0]
    da = xr.DataArray(
        data=Y_pred,
        dims=["year", "latitude", "longitude"],
        coords={
            "year": np.arange(n_samples),
            "latitude": np.array(lat_array),
            "longitude": np.array(lon_array),
        },
        name="prediction"
    )
    ds = xr.Dataset({"prediction": da})
    save_path = output_dir / f"{file_name}.nc"
    ds.to_netcdf(save_path)
    print(f"[INFO] Prediction saved to {save_path}")
