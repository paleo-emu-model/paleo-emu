import numpy as np

def cov_mat(lambda_, X1, X2):
    m = X1.shape[1]
    theta = lambda_[0:m]       # 第 1:m 行（注意 Python 的索引从 0 开始）
    nk = len(theta)
    nx, ny = X1.shape[0], X2.shape[0] 
    # Initialize RR
    RR = np.zeros((nx, ny, nk))
    
    # Convert X1 and X2 to numpy arrays if they are not already
    if not isinstance(X1, np.ndarray):
        X1 = X1.to_numpy()
    if not isinstance(X2, np.ndarray):
        X2 = X2.to_numpy()
    
    
    # Compute scaled differences
    for k in range(nk):
        RR[:, :, k] = np.subtract.outer(X1[:, k], X2[:, k]) / theta.iloc[k]


    # Compute sum of squared differences and apply covariance function
    # the value of R(i,j) is the similarity between X(i) and X(j)
    R = np.exp(-np.sum(RR ** 2, axis=2))
    

    return R
