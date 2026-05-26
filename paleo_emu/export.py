"""
save the training log for VAE.
- loss curve plot
- hyperparameter + final loss CSV record
"""

import re
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import os
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError

try:
    import tomllib          # Python 3.11+ stdlib
except ImportError:
    tomllib = None

_PKG_ROOT = Path(__file__).parent.parent


def _load_pkg_meta() -> dict:
    """Read package metadata once from pyproject.toml and CITATION.cff at the repo root."""
    meta = {}

    toml_path = _PKG_ROOT / "pyproject.toml"
    if toml_path.exists():
        if tomllib is not None:
            with open(toml_path, "rb") as f:
                data = tomllib.load(f)
            project = data.get("project", {})
            meta["version"]     = project.get("version", "")
            meta["description"] = project.get("description", "")
            meta["homepage"]    = project.get("urls", {}).get("Homepage", "")
        else:
            text = toml_path.read_text()
            for field in ("version", "description"):
                m = re.search(rf'^{field}\s*=\s*"([^"]+)"', text, re.MULTILINE)
                if m:
                    meta[field] = m.group(1)
            m = re.search(r'^Homepage\s*=\s*"([^"]+)"', text, re.MULTILINE)
            if m:
                meta["homepage"] = m.group(1)

    cff_path = _PKG_ROOT / "CITATION.cff"
    if cff_path.exists():
        try:
            with open(cff_path) as f:
                cff = yaml.safe_load(f)
            authors = cff.get("authors", [])
            meta["authors"] = "; ".join(
                f"{a.get('given-names', '')} {a.get('family-names', '')}".strip()
                for a in authors
            )
        except Exception:
            pass

    return meta


_PKG_META = _load_pkg_meta()

try:
    _VERSION = version("paleo-emu")
except PackageNotFoundError:
    _VERSION = _PKG_META.get("version", "unknown")



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


def save_prediction(Y_pred, Y_var, lat_array, lon_array, output_dir, file_name="prediction",
                    var_name=None, var_attrs=None, long_name=None, units=None):
    """
    Save prediction results as a CF-1.8 compliant NetCDF file.

    Parameters
    ----------
    Y_pred : ndarray, shape (n_samples, n_lat, n_lon)
    Y_var  : ndarray, shape (n_samples, n_lat, n_lon)
    lat_array, lon_array : 1-D arrays
    output_dir : str or Path
    file_name  : str, output filename stem (no extension)
    var_name   : str, optional — variable name to use in the NetCDF (e.g. "tos").
                 Falls back to "prediction" if None.
    var_attrs  : dict, optional — CF attributes from the original training data.
                 Used as defaults for long_name and units when not explicitly provided.
    long_name  : str, optional — override for the prediction variable's long_name.
    units      : str, optional — override for the prediction variable's units.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _var_attrs  = var_attrs or {}
    _pred_name  = var_name or "prediction"
    _long_name  = long_name or _var_attrs.get("long_name") or "emulator prediction"
    _units      = units if units is not None else _var_attrs.get("units", "")
    _std_name   = _var_attrs.get("standard_name")

    n_samples = Y_pred.shape[0]
    coords = {
        "time":      np.arange(n_samples),
        "latitude":  np.array(lat_array),
        "longitude": np.array(lon_array),
    }

    pred_attrs = {"long_name": _long_name, "units": _units}
    if _std_name:
        pred_attrs["standard_name"] = _std_name

    var_units = f"({_units})2" if _units else ""
    var_long  = f"variance of {_long_name}"

    da = xr.DataArray(
        data=Y_pred,
        dims=["time", "latitude", "longitude"],
        coords=coords,
        name=_pred_name,
        attrs=pred_attrs,
    )
    dv = xr.DataArray(
        data=Y_var,
        dims=["time", "latitude", "longitude"],
        coords=coords,
        name="variance",
        attrs={"long_name": var_long, "units": var_units},
    )

    ds = xr.Dataset({_pred_name: da, "variance": dv})

    # CF coordinate attributes
    ds["latitude"].attrs.update({
        "standard_name": "latitude",
        "units": "degrees_north",
        "axis": "Y",
    })
    ds["longitude"].attrs.update({
        "standard_name": "longitude",
        "units": "degrees_east",
        "axis": "X",
    })
    ds["time"].attrs.update({
        "long_name": "time step index",
        "axis": "T",
    })

    # CF global attributes
    today     = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    _homepage = _PKG_META.get("homepage", "https://github.com/paleo-emu-model/paleo-emu")
    ds.attrs["Conventions"]  = "CF-1.8"
    ds.attrs["history"]      = f"Created by paleo-emu v{_VERSION} on {today}"
    ds.attrs["institution"]  = "University of Bristol"
    ds.attrs["DOI"]          = "https://zenodo.org/records/20327110"
    ds.attrs["source"]       = f"paleo-emu ML emulator ({_homepage})"
    if _PKG_META.get("description"):
        ds.attrs["title"]        = _PKG_META["description"]
    if _PKG_META.get("creator_name"):
        ds.attrs["creator_name"] = _PKG_META["creator_name"]

    save_path = output_dir / f"{file_name}.nc"
    ds.to_netcdf(save_path, engine="h5netcdf")
    print(f"[INFO] Prediction saved to {save_path}")
