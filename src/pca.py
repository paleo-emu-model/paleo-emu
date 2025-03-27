import numpy as np
# Define a function that performs PCA on a 3D array
# Input: A 3D array with dimensions (nr, nlon, nlat), nr is the number of experiments

def PCA(A,nkeep):
    # Dimensions of the input array
    nlon, nlat, nr = A.shape
    
    # Reshape A to a 2D matrix with dimensions (nlon * nlat, nr)
    A_reshaped = A.reshape(nlon * nlat, nr)

    # Compute row-wise means, ignoring NaNs
    means = np.nanmean(A_reshaped, axis=1)

    # Subtract means from each row
    Am = A_reshaped - means[:, np.newaxis]

    # Find rows where means are not NaN
    o = ~np.isnan(means)

    # Perform Singular Value Decomposition on rows with valid data
    U, s, Vt = np.linalg.svd(Am[o, :], full_matrices=False)
    #print("shape of U: ", U.shape)
    #print("shape of s: ", s.shape)
    #print("shape of Vt: ", Vt.shape)
    # Create an output array with NaNs, then fill valid rows with PCA results
    O = np.full_like(A_reshaped, np.nan)
    O[o, :] = U

    # Reshape O back to the original 3D shape
    O_reshaped = O.reshape(nlon, nlat, nr)

    # Reshape means back to the original spatial dimensions
    means_reshaped = means.reshape(nlon, nlat)

    scaled_amps = Vt.T * s

    # rvm : residual variance estimated  at each grid point
    remaining_d_squared = np.square(s[nkeep:])  # 排除前 nkeep 个元素并平方
    total_remaining_d_squared = np.sum(remaining_d_squared)  # 求和
    valid_count = np.count_nonzero(~np.isnan(A_reshaped[:, 0]))  # 统计非 NaN 值数量
    rvm = total_remaining_d_squared / valid_count / nr  

    # Return results as a dictionary
    return {
        "mean": means_reshaped,
        "PCA": O_reshaped,
        "amps": Vt.T,  # Transpose of V (right singular vectors)
        "d": s,         # Singular values
        "scaled_amps": scaled_amps,
        "rvm": rvm
    }
