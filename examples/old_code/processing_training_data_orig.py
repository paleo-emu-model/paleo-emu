# %% [markdown]
# % this script is used to calibrate the emulator, dont need to change anything, just run it will be fine.

# %% [markdown]
# % 这个脚本建于已经read并write了所有temp和precip数据的R脚本之后，直接读取所有（100y）的模型输出数据。这一步可以被更改。
# % Update 250219: 在后期prediction中，需要对X input 的5个变量做标准化，原本使用了CO2_sum等，现在改为使用CO2.mean 以及 CO2.std
# # 添加必要package

# %%
#%autoreload
#%reload_ext autoreload
import pandas as pd
import numpy as np
import xarray as xr
from netCDF4 import Dataset
import os
import h5py
# some user defined functions
from core_code.pca import PCA
from core_code.GP_C import GP_C
import core_code.compute_covariance as cov_mat


data_path = 'training_data/'
#os.chdir('../' + path)
emu_path = 'emulator/'
if not os.path.exists(emu_path):
    os.makedirs(emu_path)

# %% [markdown]
# # 定义hyper parameters

# %%
# number of Principal Components to take into account
nkeep = 15

# length scale and nugget hyperparameter values
# higher length scale means smoother functions, higher nugget means more noise

hp = pd.DataFrame({
    'l.co2': [0.523323] * nkeep,
    'l.esinw': [2.791735]* nkeep,
    'l.ecosw': [1.310285] * nkeep,
    'l.obl': [1.663824] * nkeep,
    'l.icevol': [10.000000] * nkeep,
    'nugget': [0.000000000224038] * nkeep
})


# The way the pe_c routine takes hp is not okay: it requires a matrix shaped this way
hp = hp.T

print("we keep ",nkeep," PCs, ")
print("the hyper parameter shape:",hp.shape)
print("every PC has the same set of HP, an example:")
print(hp.iloc[:, 0])


# %% [markdown]
# #  读取五个变量并且normalize -地球轨道，CO2，ice

# %% [markdown]
# ##  读取五个变量

# %%
# 读取文件
cont_paramdat = pd.read_csv(data_path+'emul_in_X.res', sep='\s+', header=0)

# 获取数据维度
cont_param_dim = cont_paramdat.shape

# 打印列名以检查是否存在 
print(cont_paramdat.columns)

# 提取各列
co2 = cont_paramdat['co2']
obliquity = cont_paramdat['obliquity']
esinw = cont_paramdat['esinw']
ecosw = cont_paramdat['ecosw']
ice = cont_paramdat['ice']


####!! Here's for standardization only to fit with Charlie's code
cont_paramdats = pd.read_csv(data_path+'emul_in_X_Char.res', sep='\s+', header=0)
cont_param_dims = cont_paramdats.shape
co2s = cont_paramdats['co2']
obliquitys = cont_paramdats['obliquity']
esinws = cont_paramdats['esinw']
ecosws = cont_paramdats['ecosw']
ices = cont_paramdats['ice']

# %% [markdown]
# ## normalized数据并新建一个DataFrame保存

# %%

import xarray as xr
import datetime


### log the CO2 and standardize all the parameters -----
### updata 16th Jan 2025: log the CO2, and standardize all the parameters instead of just dividing by the sum
# Log transform the CO2 data
co2 = np.log(co2)
co2s = np.log(co2s)

##!!!note here, in order to fits Charlies's code, we need to introduce both mod and modlow X to standardize the data
##!!!need to check the code in the future emulator training.
# Standardize all the parameters
paras_normalized = pd.DataFrame({
    'co2_ppm_norm': (co2 - co2s.mean()) / co2s.std(),
    'obliquity_norm': (obliquity - obliquitys.mean()) / obliquitys.std(),
    'esinw_norm': (esinw - esinws.mean()) / esinws.std(),
    'ecosw_norm': (ecosw - ecosws.mean()) / ecosws.std(),
    'ice_norm': (ice - ices.mean()) / ices.std()
})

paras_normalized.info()

# ##------ old code ------ 
# obliquity_sum = obliquity.sum()
# esinw_sum = esinw.sum()
# ecosw_sum = ecosw.sum()
# co2_sum = co2.sum()
# ice_sum = ice.sum()

# paras_normalized = pd.DataFrame(columns=[ 'co2_ppm_norm', 'esinw_norm', 'ecosw_norm', 'obliquity_norm', 'ice_norm'])
# paras_normalized['co2_ppm_norm'] = co2 / co2_sum
# paras_normalized['esinw_norm'] = esinw / esinw_sum
# paras_normalized['ecosw_norm'] = ecosw / ecosw_sum
# paras_normalized['obliquity_norm'] = obliquity / obliquity_sum
# paras_normalized['ice_norm'] = ice / ice_sum

# paras_normalized.info()

# # Create a dataset with the sums of the parameters
# data = {
#     'co2_sum': co2_sum,
#     'obliquity_sum': obliquity_sum,
#     'esinw_sum': esinw_sum,
#     'ecosw_sum': ecosw_sum,
#     'ice_sum': ice_sum
# }

# Save the mean and std of CO2 and other parameters
data = {
    'co2_mean': co2s.mean(),
    'co2_std': co2s.std(),
    'obliquity_mean': obliquitys.mean(),
    'obliquity_std': obliquitys.std(),
    'esinw_mean': esinws.mean(),
    'esinw_std': esinws.std(),
    'ecosw_mean': ecosws.mean(),
    'ecosw_std': ecosws.std(),
    'ice_mean': ices.mean(),
    'ice_std': ices.std()
}

# Convert the dictionary to an xarray Dataset
ds = xr.Dataset(data)

# backup the file if it already exists
if os.path.exists(emu_path+"emul_in_X_5variables_mean_std.nc"):
    now = datetime.datetime.now().strftime("%Y%m%d")
    os.rename(emu_path+"emul_in_X_5variables_mean_std.nc", emu_path+"emul_in_X_5variables_mean_std_" + now + ".nc")

# Save the dataset to a NetCDF file，用作后面的prediction的inputd normalize
ds.to_netcdf(emu_path+'emul_in_X_5variables_mean_std.nc')

print(paras_normalized.head())

# %% [markdown]
# # 读取模型输出数据（temp，precip) 12x100x201x201 - month x exp x lat x lon

# %% [markdown]
# ## 读取emul_in_Y数据

# %%

# Open the NetCDF file
temp_file = Dataset(data_path+'emul_in_Y.nc', mode='r')  # 'r' for read mode
# Print the file information
#print("Variables:", temp_file.variables.keys())
# Extract data from a specific variable (e.g., 'temperature')
temp_all = temp_file.variables['var'][:,:,:]
lat = temp_file.variables['latitude'][:]
lon = temp_file.variables['longitude'][:]
#print("temperature data shape:",temp_tmp.shape)
# Close the file
temp_file.close()

print("temperature data shape:",temp_all.shape)

# %% [markdown]
# # 进行PCA处理
#  产生几个components（可以理解为维度）

# %%

# Perform PCA for Y data and store the results
# backup the file if it already exists
if os.path.exists(emu_path+"emul_in_Y_PCA.nc"):
    now = datetime.datetime.now().strftime("%Y%m%d")
    os.rename(emu_path+"emul_in_Y_PCA.nc", emu_path+"emul_in_Y_PCA_" + now + ".nc")

temp_pca = PCA(temp_all[ :, :, :].transpose(1, 2, 0), nkeep)

# Convert the PCA results to an xarray Dataset
pca_data = xr.Dataset({
    'mean': (['lat', 'lon'], temp_pca['mean']),
    'PCA': (['lat', 'lon','pc'], temp_pca['PCA']),
    'scaled_amps': (['nexp', 'pc'], temp_pca['scaled_amps']),
    'amps': (['nexp', 'pc'], temp_pca['amps']),
    'd': (['nexp'], temp_pca['d']),
    'lat': (['lat'], lat),
    'lon': (['lon'], lon)
})

# Save the PCA results to a NetCDF file
pca_data.to_netcdf(emu_path+'emul_in_Y_PCA.nc')

print("PCA results saved to emul_in_Y_PCA.nc")
print('the information of the PCA results is:', temp_pca.keys())

# mean: mean of each grid
# PCA: the original PCA results 主成分
# scaled_amps: scaled amplitude of each PC。
# amps: amplitude of each PC 每个月份数据的主成分得分。
# d: 

# %% [markdown]
# # 进行GP处理

# %% [markdown]
# ### 对于cov_mat function的画图说明（以助于理解）

# %%
# ## test the covariance matrix
# %autoreload 2
# import compute_covariance as cov_mat
# import matplotlib.pyplot as plt

# ## need to be modified in the cov_mat function
# R = cov_mat.cov_mat(hp.iloc[:, 0], paras_normalized, paras_normalized)

# # a example to visualize the function of cov_mat
# x = np.arange(1, len(paras_normalized['co2_ppm_norm']) + 1)

# # First figure
# plt.figure()
# sns.regplot(x=x, y=paras_normalized['co2_ppm_norm'])
# plt.title('a example of normalized co2')
# plt.show()

# if 'RR' in locals():
#     plt.figure()
#     plt.contourf(x, x, RR[:, :, 3], cmap='seismic')
#     plt.colorbar()
#     plt.plot(x, x, 'k--', label='X=Y')  # Add X=Y line
#     plt.title('transformed covariance matrix of CO2')
#     plt.xlabel('X-axis')
#     plt.ylabel('Y-axis')
#     plt.legend()
#     plt.show()
# else:
#     print("Variable 'RR' is not read from cov_mat.")

# # Third figure
# plt.figure()
# plt.contourf(x, x, R[:, :], cmap='seismic')
# plt.colorbar()
# plt.plot(x, x, 'k--', label='X=Y')  # Add X=Y line
# plt.title('transformed covariance of 5 parameters')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.legend()
# plt.show()


# %% [markdown]
# ## GP - Calibration
#  get the E_list which contains mean, PCA ...

# %%

# define the regress method
regress = 'linear'


# backup the file if it already exists
if os.path.exists(emu_path+"GPList.h5"):
    now = datetime.datetime.now().strftime("%Y%m%d")
    os.rename(emu_path+"GPList.h5", emu_path+"GPList_" + now + ".h5")

# Print a summary of temp_pca
# Create a new group for each 'nkeep' in the HDF5 file
with h5py.File(emu_path+"GPList.h5", "w") as file: 
    # loop of nkeep of PCAs
    # amps[n,nkeep] 
    for i in range(nkeep):
        GPList = GP_C(paras_normalized, temp_pca['amps'][:, i], hp.iloc[:, i], regress=regress)
        group = file.create_group(f"PC_{i+1:02}")
        for key, value in GPList.items():
            if isinstance(value, np.ndarray):
                group.create_dataset(key, data=value)  # Save matrix
            else:
                group.attrs[key] = str(value)  # Convert non-array types to strings before saving as attributes
            group.attrs['regress'] = regress  # Store regress as a string

# check the keys of the dictionary
print(GPList.keys())
print(GPList['lambda'])
print("shape of X is",GPList['X'].shape)
print("shape of Y is",GPList['Y'].shape)


