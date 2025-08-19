"""
This script is used for emulator validation.
calculate R² scores for each grid point.

Parameters:
- Y_true_out: (n_samples, lat, lon)
- Y_pred_out: (n_samples, lat, lon)

Returns:
- r2_map: (lat, lon)
"""

import numpy as np
from sklearn.metrics import r2_score

def compute_r2_map(Y_true_out, Y_pred_out,lat_array, lon_array):
    n_samples, lat, lon = Y_true_out.shape
    r2_map = np.full((lat, lon), np.nan)
    for i in range(len(lat_array)):
        for j in range(len(lon_array)):
            y_true_series = Y_true_out[:, i, j]
            y_pred_series = Y_pred_out[:, i, j]
            if np.all(np.isfinite(y_true_series)) and np.all(np.isfinite(y_pred_series)):
                if np.std(y_true_series) > 1e-6:
                    r2 = r2_score(y_true_series, y_pred_series)
                    r2_map[i, j] = r2
                else:
                    print(f"[DEBUG] Skipping lat index {i}, lon index {j} due to low std deviation in y_true_series")
            else:
                print(f"[DEBUG] Skipping lat index {i}, lon index {j} due to non-finite values")

    return r2_map
