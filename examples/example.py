"""
Train a PCA+GP emulator, predict under scenario forcing, and plot.

Config: example_PCA_GP.yml

For the full API reference run in Python:
    from paleo_emu import PaleoEmuPlotter
    help(PaleoEmuPlotter)        # overview and usage styles
    help(PaleoEmuPlotter.map)    # spatial map — all parameters + examples
    help(PaleoEmuPlotter.timeseries)
    help(PaleoEmuPlotter.uncertainty_map)
    help(PaleoEmuPlotter.zonal_mean)
    help(PaleoEmuPlotter.delta_map)
    help(PaleoEmuPlotter.mapgif)
"""

from paleo_emu import PaleoEmuRunner, plotter

# --- train and predict ---
runner = PaleoEmuRunner("example_PCA_GP.yml")

RUN_TRAINING = False   # set to True to retrain from scratch
if RUN_TRAINING:
    runner.train()

runner.predict("SSP585")

# --- plot (using module-level singleton) ---
plotter.map("SSP585", cfg="example_PCA_GP.yml", zonal_mean=True,save_name="ssp585_map", fmt="png",save_dir="examples/figures/")
plotter.timeseries("SSP585", cfg="example_PCA_GP.yml")
