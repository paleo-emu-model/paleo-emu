"""
This module is to train models using chosen regressors, kernels, and encoders.
2 methods are used here: 2:8 validation; leave-one-out cross-validation.
leave_one_out has a recurring loop which needs i to be looped, so need to write another function for it.
2:8 validation doesn't require a function for looping, so it will only give one pipeline fitted model

procedures of training:
1. load data
2. split data
3. encode training data (giving decoder in the mean time)
4. process test Y for validation later
5. fit model (pipline)
6. validation -> predict using test X
              -> compare with test Y
"""
# training process needs to give info like pipeline contains 
# the trained model, decoder, std_val, and mean_val, which are used in the following prediction process


from tabnanny import verbose
import numpy as np
import xarray as xr
import os
import time

import tensorflow as tf


from sklearn.multioutput import MultiOutputRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import joblib
from sklearn.gaussian_process.kernels import  (RBF, Matern, ConstantKernel as C, WhiteKernel)

from paleo_emu import regressor
from paleo_emu.encoder import EncoderGenerator  


class TrainingGenerator:
    """Utility for training emulators with specified encoders and regressors.

    Parameters
    ----------
    cfg: a configuration object loaded from a YAML file.
    X_train : array-like, shape (n_samples, n_features), optional
        Input features for training. If None, data will be loaded from cfg_path.
    Y_train : array-like, shape (n_samples, n_targets), optional
        Target values for training. If None, data will be loaded from cfg_path.

    Returns
    -------
    fitted_pipeline : object
        The trained sklearn Pipeline containing the scaler and regressor.
    """

    def __init__(self, model_configuration, X_train=None, Y_train=None):
        """Create a TrainingGenerator."""
        self.model_configuration = model_configuration
        self.X_train = X_train
        self.Y_train = Y_train


    def run_training(self):

        print("X_train or Y_train is None, loading training data from model_configuration...")
        enc = EncoderGenerator(self.Y_train, self.model_configuration)
        Y_train_encoded, decoder, mean_val, std_val = enc.generate_encoder()

        if self.model_configuration["regressor_config"]["regressor_type"] == "GPR":   
            # setup a switch for different kernel types
            # if k1__k1__ is 1.0/0.0, then use/notuse RBF 
            # if k1__k2__ is 1.0/0.0, then use/notuse Matern
            # if k3__ is 1.0/0.0, then use/notuse WhiteKernel
            if self.model_configuration["regressor_config"]["kernel_type"] == "RBF":
                kernel = C(1.0) * RBF(length_scale=1.0)
                param_grid={
                    'regressor__kernel__k2__length_scale' : self.model_configuration["regressor_config"]["parameter_grid"]["rbf_length_scale"],
                }


            regressor = GaussianProcessRegressor(kernel=kernel)

        model_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", regressor)
        ])

        model = GridSearchCV(
            estimator=model_pipeline,
            param_grid=param_grid,
            cv=self.model_configuration["regressor_config"].get('cv', 5)
        )

        model.fit(self.X_train, Y_train_encoded)
        
        print(model.best_params_)

        if self.model_configuration["save_pipeline"]:
        # Save metadata as a YAML file (meta already converted to native types)
            pipeline_path = os.path.join(self.model_configuration["save_path"], "fitted_pipeline.joblib")

            with open(pipeline_path, "wb") as f:
                joblib.dump(model, f)
            print(f"[INFO] Fitted pipeline saved to {pipeline_path}")

