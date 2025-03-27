import numpy as np
from core_code.compute_covariance import cov_mat
import netCDF4 as nc

def GP_C(X, Y, lambda_, regress):
    """
    Generalized Polynomial Calibration (GP_C) function.
    Parameters:
    -----------
    X : array-like
        Input feature matrix (n_samples, n_features).
    Y : array-like
        Target values (n_samples, 1).
    lambda_ : dict
        Dictionary containing hyperparameters for the covariance matrix.
        - 'nugget': float, nugget term added to the diagonal of the kernel matrix.
        - 'epsilon': float, optional, regularization parameter for penalized REML.
    regress : str or callable
        Regression model to use. If a string, it should be "linear". If a callable, it should be a function that takes an array and returns an array.
    Returns:
    --------
    EM_Cali : dict
        Dictionary containing the results of the calibration.
        - 'betahat': array, optimal regression coefficients.
        - 'sigma_hat_2': array, estimated variance.
        - 'R': array, covariance matrix of X.
        - 'Rt': array, covariance matrix of X with nugget term added.
        - 'muX': array, transformed X using the regression model.
        - 'X': array, input feature matrix.
        - 'Y': array, target values.
        - 'lambda': dict, input hyperparameters.
        - 'e': array, residuals.
        - 'funcmu': callable, regression function used.
        - 'R1tX': array, intermediate matrix used in GLS computation.
        - 'log_REML': float, log restricted maximum likelihood.
        - 'log_pen_REML': float, penalized log restricted maximum likelihood.
        - 'covar': callable, covariance function used.
        - 'nbrr': int, degrees of freedom.
    """
    
    def get_funcmu(name):
        if name == "linear":
            return lambda X: [1] + list(X)  # add one more line to the matrix
        else:
            raise ValueError("invalid regression model")

    # if the regression is "linear", then x is x, 
    # otherwise, it can be wrote as a function 
    if isinstance(regress, str):
        funcmu = get_funcmu(regress)  # Use predefined function
    elif callable(regress):
        funcmu = regress  # Use user-defined function
    else:
        raise ValueError("regress must either be a string or a callable function")

    # Ensure X and Y are 2D numpy arrays
    if not isinstance(Y, np.ndarray):
        Y = np.array(Y)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
    if not isinstance(X, np.ndarray):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

    # make sure the shape of X and Y are n(experiment number) number of features)
    muX = np.apply_along_axis(funcmu, 1, X) # apply the linear function 
    
    if muX.shape[0] == 1:
        muX = muX.T

    nq = muX.shape[1] # 

    n = X.shape[0] # number of experiments, 100
    nbr = n - nq   # 自由度？
    nbrr = n - nq - 2 
    covar=np.exp

    # compute the covariance matrix of X, also known as the kernel matrix
    R = cov_mat(lambda_, X, X)
    
    Rt = R + np.diag(np.full(n, lambda_['nugget'])) # add the nugget term to the diagonal of the kernel matrix

    # GLS（广义最小二乘）相关计算
    R1tX = np.linalg.solve(Rt, muX) # R1tX = R^(-1) * muX
    dummy1 = muX.T @ R1tX # muX^T * R1tX
    K = np.linalg.solve(dummy1, muX.T) # GLS的系数矩阵
    betahat = K @ np.linalg.solve(Rt, Y) # 最优回归系数

    # 计算残差
    dummy2 = Y - muX @ betahat
    e = np.linalg.solve(Rt, dummy2)

    # 计算方差
    sigma_hat_2 = (dummy2.T @ e) / nbrr
    sigma_hat_2 = np.atleast_2d(sigma_hat_2)

    # 计算REML 
    M = (lambda_['nugget'] ** 2) * (dummy2.T @ np.linalg.solve(Rt.T @ Rt, dummy2)) / n
    Minfty = (dummy2.T @ dummy2) / n
    log_REML = -0.5 * (nbr * np.log(np.diag(sigma_hat_2)) + np.log(np.linalg.det(Rt)) + np.log(np.linalg.det(muX.T @ R1tX)))

    if 'epsilon' not in lambda_:
        lambda_['epsilon'] = 1

    log_pen_REML = log_REML - 2 * (M / Minfty / lambda_['epsilon'])

    EM_Cali = {                                                     
        'betahat': betahat,
        'sigma_hat_2': sigma_hat_2,
        'R': R,
        'Rt': Rt,
        'muX': muX,
        'X': X,
        'Y': Y,
        'lambda': lambda_.to_dict(),
        'e': e,
        'funcmu': funcmu,
        'R1tX': R1tX,
        'log_REML': log_REML,
        'log_pen_REML': log_pen_REML,
        'covar': covar,
        'nbrr': nbrr
    }
    return EM_Cali