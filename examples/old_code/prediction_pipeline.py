"""
prediction_pipeline_module.py

Structured pipeline version of the original prediction script.
Each function represents a modular step in the workflow.
"""

import pandas as pd
import numpy as np
import netCDF4 as nc
import h5py
import os
import datetime
import GP_P as GP_P
import compute_covariance as cov_matrix
from pathlib import Path

### these are for plotting
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import HTML
from mpl_toolkits.basemap import Basemap

### here we dont do the standardization any more. therefore we dont need to read in the 5variabelX file.


# define a function to read the h5 data recursively into a dictionary
#----------------------------------------------------------------------
def load_h5_as_dict(h5_group):
    data = {}
    for key, item in h5_group.items():
        if isinstance(item, h5py.Group):
            # If the item is a group, recurse into it
            data[key] = load_h5_as_dict(item)
        elif isinstance(item, h5py.Dataset):
            # If the item is a dataset, load it as a NumPy array
            data[key] = np.array(item)
    
    # Load attributes as well, if any
    for attr_key, attr_value in h5_group.attrs.items():
        data[attr_key] = attr_value  # Store attributes as dictionary entries

    return data 

def load_training_result(cfg_emulator):
    input_GP_path = cfg_emulator["file_path"]+cfg_emulator["label"]+ "_GPList.h5"
    input_pca_path = cfg_emulator["file_path"]+cfg_emulator["label"]+ "_emul_in_pca.nc"
    with h5py.File(input_GP_path, "r") as GPList:
        EM_Cali = load_h5_as_dict(GPList)
    # Load the PCA components 
    nc_pca = nc.Dataset(input_pca_path)
    PCs = {var: nc_pca.variables[var][:] for var in nc_pca.variables}
    print('finish reading emulator in PCA components')

    return EM_Cali, PCs

def load_forcing(cfg, emulator):
    # Read the input data (orbital parameters and CO2 values) contains the values across all years
    x_full = pd.read_csv(prediction_input, sep='\s+', header=None)  # Read table

    # Get the number of experiments
    nexp = x_full.shape[0] - 1 # Get the number of years
    index_nexp = np.arange(nexp) # Create an index for the number of years

    ### update 250219: Normalize using mean and std instead of sum

    # Normalize input variables. Starts from 1 to skip the header
    var1_raw = x_full.iloc[1:,0].values.astype(float)                # the absolute value of CO2
    var1 = (np.log(var1_raw) - co2_mean) / co2_std                                # For CO2, we need to get the log of it first and then Normalize `V1` by `co2_sum`
                                                                    
    var2 = (x_full.iloc[1:,1].values.astype(float) - obliquity_mean) / obliquity_std    # Normalize `V2` by `obliquity_sum`
    var3 = (x_full.iloc[1:,2].values.astype(float) - esinw_mean    ) / esinw_std        # Normalize `V3` by `esinw_sum`
    var4 = (x_full.iloc[1:,3].values.astype(float) - ecosw_mean    ) / ecosw_std        # Normalize `V4` by `ecosw_sum`
    var5 = (x_full.iloc[1:,4].values.astype(float) - ice_mean      ) / ice_std          # Normalize ice volume by `ice_sum`

    # make vector of all input variables at the current year
    x = np.array([var1, var2, var3, var4, var5])

def predict(cfg, emulator):
    tmp_var = GP_P.GP_P(EM_Cali, PCs, x[:,0])
    predic_var = {key: [] for key in tmp_var.keys()}

    for n in range(nexp):
        predic_var_tmp = GP_P.GP_P(EM_Cali, PCs, x[:,n])
        for key in predic_var.keys():
            predic_var[key].append(predic_var_tmp[key])

    print('Predicted results include:',predic_var.keys())

def save_results(cfg, emulator):
    if os.path.exists(prediction_output):
    now = datetime.datetime.now().strftime("%Y%m%d")
    os.rename(prediction_output, prediction_output[:-3] + "_" + now + ".nc")

    ncnew = nc.Dataset(prediction_output, "w")

    lat = PCs["lat"]
    lon = PCs["lon"]

    # Define dimensions
    ncnew.createDimension("lon", len(lon))s
    ncnew.createDimension("lat", len(lat))
    ncnew.createDimension("time", nexp)

    # Define variables:'mean', 'var', 'means', 'variances'
    mean = ncnew.createVariable("mean", "f4", ("time", "lat", "lon"))
    mean.units = "1"
    mean.long_name = "Predicted Variable Mean field"

    var = ncnew.createVariable("variance", "f4", ("time", "lat", "lon"))
    var.units = "1"
    var.long_name = "Predicted Variable variance field"

    # Define coordinate variables
    latitudes = ncnew.createVariable("latitude", "f4", ("lat",))
    longitudes = ncnew.createVariable("longitude", "f4", ("lon",))
    time = ncnew.createVariable("time", "i4", ("time",))

    # Assign attributes to coordinate variables
    latitudes.units = "degrees_north"
    latitudes.long_name = "Latitude"
    longitudes.units = "degrees_east"
    longitudes.long_name = "Longitude"
    time.units = "1"
    time.long_name = "Time"
    # Add global attributes
    ncnew.description = "This NetCDF file contains predicted data based on emulator outputs."

    ncnew.history = f"Created on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    ncnew.source = "Input forcing is: "+prediction_input

    # Write data to coordinate variables
    latitudes[:] = lat
    longitudes[:] = lon
    time[:] = index_nexp

    # Write data to variables
    mean[:, :, :] = predic_var["mean"]
    var[:, :, :] = predic_var["var"]

    # Close the NetCDF file
    ncnew.close()

def run_prediction(cfg, cfg_emulator):
    load_data_for_standardization(cfg, cfg_emulator)
    load_training_result(cfg, cfg_emulator)
    prepare_inputs(cfg, cfg_emulator)
    predict(cfg, cfg_emulator)
    plot_results(cfg, cfg_emulator)
