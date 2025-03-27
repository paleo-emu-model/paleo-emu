import numpy as np
import pandas as pd
import ast
from compute_covariance import cov_mat

def GP_P(EM_Cali, PCs, x):
    """
    Combines Gaussian Process predictions and PCA reconstruction into a single function.
    
    Args:
        EM_Cali (list): A list of dictionaries representing Gaussian Process components.
        PCs (dict): A dictionary containing PCA data (e.g., 'PCA', 'd', 'mean', 'rvm').
        x (array): Input matrix or vector for predictions.
        
    Returns:
        dict: Contains reconstructed mean and variance fields, and the individual means and variances.
    """
    x = np.array(x)

    if x.ndim == 1:
        x = x[:, np.newaxis]

    # Initialize containers for results
    means = []
    variances = []

    # Loop through GPList and process each component
    for key, GPC in EM_Cali.items():
        # Extract elements from PC
        X = GPC['X']
        Y = GPC['Y']
        R = GPC['R']
        e = GPC['e']
        Rt = GPC['Rt']
        R1tX = GPC['R1tX']
        covar = GPC['covar']
        lambda_ = GPC['lambda']
        betahat = GPC['betahat']
        sigma_hat_2 = GPC['sigma_hat_2']
        regress = GPC['regress']
        muX = GPC['muX']

        ### process some components
        
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


        # Compute mean structure for input
        mux = np.apply_along_axis(funcmu, 0, x) 
        if funcmu([1]) == 1:
            mux = mux.T

        nx = x.shape[1]
        n = X.shape[0]
        nq = len(betahat)
        nbrr = n - nq - 2

        # Convert string to dictionary# Convert string to dictionary
        lambda_dict = ast.literal_eval(lambda_)
        lambda_series = pd.Series(lambda_dict)

        # Compute covariance vector/matrix
        r = cov_mat(lambda_series, X, x.T)

        # Compute mean and Gaussian components
        yp_mean = np.dot(mux.T, betahat.reshape(6, 1))
        yp_gaus = np.dot(r.T, e)
        yp = yp_mean + yp_gaus

        rr_nuggetfree = 1
        rr = rr_nuggetfree + lambda_dict['nugget']
        dummy1 = np.dot(muX.T, R1tX)

        # Compute diagonal variance (can be extended if calc_var=True is required)
        Sp_diag = np.zeros(nx)
        Sp_diag_nuggetfree = np.zeros(nx)

        for j in range(nx):
            rj = r[:, j]
            P = (mux[j] - np.dot(rj.T, R1tX))
            cxx0 = -np.dot(rj.T, np.linalg.solve(Rt, rj)) + np.dot(P, np.linalg.solve(dummy1, P.T))
            cxx = rr + cxx0
            cxx_nuggetfree = rr_nuggetfree + cxx0
            Sp_diag[j] = sigma_hat_2 * cxx
            Sp_diag_nuggetfree[j] = sigma_hat_2 * cxx_nuggetfree

        # Store results for current component
        means.append(yp)
        variances.append(Sp_diag)
    
    # Convert results to arrays， and remove extra dimensions
    means = np.squeeze(np.array(means))
    variances = np.squeeze(np.array(variances))

    # PCA Reconstruction for Mean Field
    sv = range(len(means-1))
    # sliced the nkeep number of PCs
    PCA_slices = PCs['PCA'][:, :, sv]
    
    scaled_means = PCA_slices * (means[:] * PCs['d'][sv][:])
    mean_field = np.sum(scaled_means, axis=2) + PCs['mean']

    # PCA Reconstruction for Variance Field
    scaled_variances = (PCA_slices**2) * (variances[:] * (PCs['d'][sv]**2)[:])
    var_field = np.sum(scaled_variances, axis=2)
    if 'rvm' in PCs:
        var_field += PCs['rvm']

    # Return results
    return {
        'mean': mean_field,
        'var': var_field,
        'means': means,
        'variances': variances
    }
