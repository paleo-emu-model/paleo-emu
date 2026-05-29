"""
Train a PCA+GP emulator, predict under scenario forcing, and plot.

Config: example_PCA_GP.yml

For the full API reference run in Python:
    from paleo_emu.plotter import PaleoEmuPlotter
    help(PaleoEmuPlotter)        # overview and usage styles
    help(PaleoEmuPlotter.map)    # spatial map — all parameters + examples
    help(PaleoEmuPlotter.timeseries)
    help(PaleoEmuPlotter.uncertainty_map)
    help(PaleoEmuPlotter.zonal_mean)
    help(PaleoEmuPlotter.delta_map)
    help(PaleoEmuPlotter.mapgif)
"""

from paleo_emu.runner import PaleoEmuRunner
from paleo_emu.plotter import PaleoEmuPlotter

# --- train and predict ---
runner = PaleoEmuRunner("example_PCA_GP.yml")

RUN_TRAINING = False   # set to True to retrain from scratch
if RUN_TRAINING:
    runner.train()

runner.predict("SSP585")

# --- plot ---
plotter = PaleoEmuPlotter("example_PCA_GP.yml")
plotter.map("SSP585", cfg="example_PCA_GP.yml",time=2, zonal_mean=True,save_name="ssp585_map_3ka", fmt="png",save_dir="examples/figures/")
plotter.timeseries("SSP585", cfg="example_PCA_GP.yml")
plotter.uncertainty_map("SSP585", cfg="example_PCA_GP.yml")
plotter.mapgif("SSP585", cfg="example_PCA_GP.yml", save_name="ssp585_map", save_dir="examples/figures/")
