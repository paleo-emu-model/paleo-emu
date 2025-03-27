# %% [markdown]
# % 这个脚本建于已经read并write了所有temp和precip数据的R脚本之后，直接读取所有（100y）的模型输出数据。这一步可以被更改。
# # 添加必要package
# # test to ensure it's the same as Charlies' 20250219. Lines with modifications are marked with '#250219'

# %%

import pandas as pd
import numpy as np
import netCDF4 as nc
import h5py
import os
import datetime
# some user defined functions
import GP_P as GP_P
import compute_covariance as cov_matrix

### defined forcing data path in the prediction/run.ipynb
#prediction_input="/Users/bo20541/Library/CloudStorage/OneDrive-UniversityofBristol/TONIC-Oligocene/Emulator_Charlie/Emulator/2015_Bristol_5D_v001/orig/Input/2018-08-01 Final report/test_data.1.res"
#prediction_output="../../prediction/temp_prediction.nc"

# %% [markdown]
# # Read emulator 
# read the sum of 5 parameters in calibration to normalize the new input
# read the GP list from calibration 
# read the PCA from calibration

# %%
# Open the 5 variables sum for standardization
#################################################
nc_data = nc.Dataset("../emulator/emul_in_X_5variables_mean_std.nc")
var_names = list(nc_data.variables.keys())
for var_name in var_names:
    globals()[var_name] = nc_data.variables[var_name][:]
 
print("the 5 variables sum used to standardlize input forcings are:")
print(var_names)


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
#--------------------------------------------------------------------
# Load the HDF5 file and convert it into a dictionary
#######################################################
with h5py.File("../emulator/GPList.h5", "r") as GPList:
    EM_Cali = load_h5_as_dict(GPList)

print('finish reading emulator in GP List (GP regression information)')
print('We have the following PCs:')
print(EM_Cali.keys())
print('For each PC in GPList, the GP info used are:')
print(EM_Cali['PC_01'].keys())

# Check if the regression type is not linear
if EM_Cali['PC_01']['regress'] != 'linear':
    raise ValueError("The regression type is not linear. Please check the Clibration.ipynb and add the corresponding callable function.")


nkeep = len(EM_Cali.keys()) # number of PCAs

# Load the PCA components 
##########################
# Load the PCA components from a NetCDF file
nc_pca = nc.Dataset("../emulator/emul_in_Y_PCA.nc")
PCs = {var: nc_pca.variables[var][:] for var in nc_pca.variables}
print('finish reading emulator in PCA components')
print('For each PCA we have these info:')
print(PCs.keys())


# %% [markdown]
# # Read Input Data for Prediction
# read the new 5 parameters: orbital parameters, CO2 and ice sheet parameter

# %%

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
print('We have such many forcings to be predicted:', x.shape)
print("The 5 input variable for prediction are:")
print(x_full.head())
print('... ...')

# Print the head of normalized variables
print("Normalized CO2 values (var1):", var1[:5])
print("Normalized obliquity values (var2):", var2[:5])
print("Normalized esinw values (var3):", var3[:5])
print("Normalized ecosw values (var4):", var4[:5])
print("Normalized ice volume values (var5):", var5[:5])


# %% [markdown]
# ## Visulisaztion of the input data to help understanding 

# %%
import matplotlib.pyplot as plt

plt.scatter(index_nexp, np.exp((var1*co2_std)+co2_mean), label='CO2')
plt.xlabel('year')
plt.ylabel('CO2')
plt.title('An example of input CO2 values')
plt.show()


# %% [markdown]
# # Prediction

# %%

# get an example of the keys in the GP_P output
tmp_var = GP_P.GP_P(EM_Cali, PCs, x[:,0])
predic_var = {key: [] for key in tmp_var.keys()}

for n in range(nexp):
    predic_var_tmp = GP_P.GP_P(EM_Cali, PCs, x[:,n])
    for key in predic_var.keys():
        predic_var[key].append(predic_var_tmp[key])

print('Predicted results include:',predic_var.keys())



# %% [markdown]
# ## *optional: visulization of the results

# %%

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import HTML
from mpl_toolkits.basemap import Basemap

# 创建 figure 和轴对象
fig, (ax, ax3, ax2) = plt.subplots(3, 1, figsize=(10, 12))

# 经纬度网格
X, Y = np.meshgrid(PCs['lon'], PCs['lat'])

# Calculate weights based on latitude for weighted average
weights = np.cos(np.deg2rad(PCs['lat']))
weighted_mean = np.sum(predic_var['mean'] * weights[None, :, None], axis=1) / np.sum(weights)
global_avg = np.mean(weighted_mean, axis=1)


# 创建 Basemap（用于绘制海陆分界线）
m = Basemap(projection='cyl', lon_0=180, resolution='c', ax=ax)
m.drawcoastlines()

# 初始绘图
contour = ax.contourf(X, Y, predic_var['mean'][0], cmap='coolwarm', levels=np.linspace(-20, 20, 41))
cbar = plt.colorbar(contour, ax=ax, orientation='vertical')
cbar.set_label('predic_var (°C)')
contour.collections

# 初始化函数
def init():
    ax.set_title('Time Step: 0')
    return contour.collections

# 更新函数
def update(frame):
    ax.clear()
    m.drawcoastlines()
    
    # 更新等高线图
    contour = ax.contourf(X, Y, predic_var['mean'][frame], cmap='coolwarm', levels=np.linspace(-20, 20, 41))
    ax.set_title(f'Time Step: {frame}\n'
                 f'CO2: {x_full.iloc[frame+1,0]}, '
                 f'Obliquity: {x_full.iloc[frame+1,1]}, '
                 f'esinw: {x_full.iloc[frame+1,2]}, '
                 f'ecosw: {x_full.iloc[frame+1,3]}, '
                 f'Ice: {x_full.iloc[frame+1,4]}')
    
    # 更新散点图
    ax2.clear()
    ax2.scatter(index_nexp, global_avg, c='blue', label='Global Avg Temp')
    ax2.scatter(index_nexp[frame], global_avg[frame], c='red')  # 添加红点
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Global Avg Temperature Anomaly (°C)')
    ax2.set_title('Global Average predic_var over Time')
    ax2.legend()

    # 更新 CO2 变化图
    ax3.clear()
    ax3.scatter(index_nexp, np.exp((var1 * co2_std) + co2_mean), label='CO2', color='green')
    ax3.scatter(index_nexp[frame], np.exp((var1[frame] * co2_std) + co2_mean), c='red')  # 添加红点
    ax3.set_xlabel('Year')
    ax3.set_ylabel('CO2')
    ax3.set_title('CO2 Concentration over Time')
    ax3.legend()
    
    return contour.collections

# 创建动画
if nexp > 300:
    nframe = 300
else:
    nframe = nexp
ani = FuncAnimation(fig, update, frames=range(nframe), init_func=init, blit=False)

# 保存动画为 GIF 文件
if os.path.exists(prediction_output[:-3]+".gif"):
    now = datetime.datetime.now().strftime("%Y%m%d")
    os.rename(prediction_output[:-3]+".gif", prediction_output[:-3] + "_" + now + ".gif")

ani.save(prediction_output[:-3]+".gif", writer=PillowWriter(fps=10))


# %% [markdown]
# # Write the results into NetCDF

# %%

# Check if the NetCDF file already exists, if so, backup it
if os.path.exists(prediction_output):
    now = datetime.datetime.now().strftime("%Y%m%d")
    os.rename(prediction_output, prediction_output[:-3] + "_" + now + ".nc")

ncnew = nc.Dataset(prediction_output, "w")

lat = PCs["lat"]
lon = PCs["lon"]

# Define dimensions
ncnew.createDimension("lon", len(lon))
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


