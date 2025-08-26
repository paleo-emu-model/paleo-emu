"""
save the training log for VAE.
- loss curve plot
- hyperparameter + final loss CSV record
"""

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os



def save_training_log(epoch_losses, latent_dim, epochs, learning_rate, batch_size, kl_weight, log_dir="training/logs"):

    os.makedirs(log_dir, exist_ok=True)

    info_str = f"latent{latent_dim}_ep{epochs}_lr{learning_rate}_bs{batch_size}_kl{kl_weight}"

    loss_curve_filename = os.path.join(log_dir, f"loss_curve_{info_str}.png")

    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(epoch_losses)+1), epoch_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"VAE Loss Curve ({info_str})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(loss_curve_filename, dpi=300)
    plt.close()

    print(f"[INFO] Loss curve saved to: {loss_curve_filename}")

    # --- save hyperparameters and final loss to CSV ---
    log_file = os.path.join(log_dir, "vae_hyperparameter_log.csv")

    log_entry = {
        "latent_dim": latent_dim,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "kl_weight": kl_weight,
        "final_loss": epoch_losses[-1]  # the loss of the final epoch
    }

    if not os.path.exists(log_file):
        df = pd.DataFrame([log_entry])
        df.to_csv(log_file, index=False)
    else:
        df = pd.read_csv(log_file)
        df = pd.concat([df, pd.DataFrame([log_entry])], ignore_index=True)
        df.to_csv(log_file, index=False)

    print(f"[INFO] Hyperparameter log updated: {log_file}")


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
