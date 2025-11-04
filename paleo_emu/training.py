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


from lightgbm import LGBMRegressor
import numpy as np
import xarray as xr
import os
import time

import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error
import joblib

#from docs.source.auto_examples.plot_pipeline import X_test
from paleo_emu.load import load_training_data
from paleo_emu.encoder import encode
from paleo_emu.regressor import build_regressor
from paleo_emu.plotting import plot_r2_map_with_latlon, plot_prediction_maps_with_info, plot_histogram_4_leave1out
from paleo_emu.validation import compute_r2_map



def _kernel_diag(gpr: GaussianProcessRegressor, bounds_tol=0.05):
    k = gpr.kernel_
    lml = gpr.log_marginal_likelihood_value_
    info = {"lml": lml, "hit_bounds": False}
    # 提取 length_scale
    try:
        if hasattr(k, "k1") and hasattr(k, "k2"):
            # 展开组合核 (Constant * (RBF + White)) 等
            parts = [k.k1, k.k2]
        else:
            parts = [k]
        length_scales = []
        bounds = []
        for p in parts:
            if hasattr(p, "length_scale"):
                ls = p.length_scale
                length_scales.append(ls)
                if hasattr(p, "length_scale_bounds"):
                    bounds.append(p.length_scale_bounds)
        flat_ls = []
        flat_lb = []
        flat_ub = []
        for ls, b in zip(length_scales, bounds):
            ls_arr = ls if hasattr(ls, "__len__") else [ls]
            lb, ub = b
            lb_arr = lb if hasattr(lb, "__iter__") else [lb]*len(ls_arr)
            ub_arr = ub if hasattr(ub, "__iter__") else [ub]*len(ls_arr)
            flat_ls.extend(ls_arr)
            flat_lb.extend(lb_arr)
            flat_ub.extend(ub_arr)
        hit = False
        for v, lo, hi in zip(flat_ls, flat_lb, flat_ub):
            span = hi - lo
            if span > 0:
                if (v - lo) / span < bounds_tol or (hi - v) / span < bounds_tol:
                    hit = True
                    break
        info["hit_bounds"] = hit
        info["n_length_scales"] = len(flat_ls)
    except Exception:
        pass
    return info

def diagnose_pcs(pca_model, reg_wrap):
    from sklearn.gaussian_process import GaussianProcessRegressor
    if not (hasattr(reg_wrap, "estimators_") and hasattr(pca_model, "explained_variance_ratio_")):
        return
    ratios = pca_model.explained_variance_ratio_
    rows = []
    for i, est in enumerate(reg_wrap.estimators_):
        if isinstance(est, GaussianProcessRegressor):
            k = str(est.kernel_)
            noise = None
            const = None
            if "noise_level=" in k:
                try: noise = float(k.split("noise_level=")[1].split(")")[0])
                except: pass
            if "ConstantKernel(" in k:
                try: const = float(k.split("ConstantKernel(")[1].split("**2")[0])
                except: pass
            rows.append((i, ratios[i], est.log_marginal_likelihood_value_, const, noise))
    print("PC | ratio     | LML       | const     | noise")
    for i,r,lml,c,n in rows:
        print(f"{i:2d} | {r:9.6f} | {lml:9.3f} | {c} | {n}")
    best = max(r[2] for r in rows)
    weak = [i for i,r,lml,c,n in rows if lml < best - 80]
    if weak:
        print(f"[INFO] weak PCs (LML < best-80): {weak}")
    
    
def run_training(X_train,Y_train,regressor_type="GPR",kernel="RBF_White",pca_variance_ratio=0.999,encoder="PCA",vae_config=None,fixed_hp=False):
    # training for given data 
    """
    X_training: (n_samples, 5) the input feature matrix
    Y_training: (n_samples, lat*lon) the flattened output matrix
    """
    # encode the chosen training Y
    Y_train_encoded, decoder, mean_val, std_val, residual_variance = encode(
        Y_train,
        encoder=encoder,
        pca_variance_ratio=pca_variance_ratio,
        vae_config=vae_config,
        fixed_hp=fixed_hp
    )
    latent_dim = Y_train_encoded.shape[1]

    regressor = build_regressor(
        regressor_type=regressor_type, 
        kernel_name=kernel, 
        encoder=encoder, 
        fixed_hp=False)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", MultiOutputRegressor(regressor))
    ])

    pipeline.fit(X_train, Y_train_encoded)

    # **************diagnostics********************
    if regressor_type == "GPR":
        reg_wrap = pipeline.named_steps.get("regressor")
        if isinstance(reg_wrap, MultiOutputRegressor):
            lmls = []
            hit_bounds = 0
            for est in reg_wrap.estimators_:
                if isinstance(est, GaussianProcessRegressor):
                    d = _kernel_diag(est)
                    lmls.append(d["lml"])
                    hit_bounds += int(d["hit_bounds"])
            if lmls:
                print(f"[GPR-DIAG] outputs={len(lmls)} "
                      f"LML(mean)={np.mean(lmls):.3f} LML(max)={np.max(lmls):.3f} "
                      f"hit_bounds={hit_bounds}/{len(lmls)}")
        elif isinstance(reg_wrap, GaussianProcessRegressor):
            d = _kernel_diag(reg_wrap)
            print(f"[GPR-DIAG] single LML={d['lml']:.3f} hit_bounds={d['hit_bounds']}")
        if encoder == "PCA":
            diagnose_pcs(decoder, reg_wrap)
    # **********************************

    # save the trained pipeline
    joblib.dump(pipeline, "pipeline.joblib")
    # save the decoder
    joblib.dump(decoder, "decoder.joblib")
    
    return {
        "trained_pipeline": "pipeline.joblib",
        "decoder": "decoder.joblib",
        "encoder": encoder,
        "mean_val": mean_val,
        "std_val": std_val,
        "residual_variance": residual_variance,
        "n_components_retained": latent_dim,
        "regressor_type": regressor_type,
        "kernel": kernel}

def return_validation_function(X_test,Y_true_flat,trained_pipeline,decoder,mean_val,std_val,spatial_shape,encoder,residual_variance):
    """
        1. encode, decode Y_test
        2. predict Y_test_predicted using trained_pipeline
        return:
            Y_pred_out: predicted Y, shape (n_samples, lat, lon)
            Y_true_out: true Y, shape (n_samples, lat, lon)
            Y_var_out: variance of predicted Y, shape (n_samples, lat, lon) - only for PCA
            r2_score: R² score of the prediction
            rmse: RMSE of the prediction
    """
    # Actually no need to encode and decode Y_true here,
    # this extra processing steps is to ensure the consistency of the processing for Y
    # -------
    Y_true_scaled = (Y_true_flat - mean_val) / std_val
    if encoder == "PCA":
        Y_true_encoded = decoder.transform(Y_true_scaled)
    elif encoder == "VAE":
        mean_logvar = decoder.encoder.predict(Y_true_scaled)
        mean, logvar = tf.split(mean_logvar, 2, axis=1)
        Y_true_encoded = mean.numpy()
    else:
        Y_true_encoded = Y_true_flat

    # **************************************************
    # calculate variance of Y_test_encoded if PCA is used
    # **************************************************
    # cannot return variance directly from pipeline.predict
    # because it is a MultiOutputRegressor wrapper instead of a single GPR model
    var_encoded = None
    mean_encoded = None
    if encoder == "PCA":
        if hasattr(trained_pipeline, "named_steps"):
            scaler = trained_pipeline.named_steps.get("scaler", None)
            reg_wrap = trained_pipeline.named_steps.get("regressor", trained_pipeline)
        else:
           scaler = None
           reg_wrap = trained_pipeline
        X_feat = scaler.transform(X_test) if scaler is not None else X_test
        # get variance for each PCA component
        if isinstance(reg_wrap, MultiOutputRegressor):
            ests = reg_wrap.estimators_
            if all(isinstance(e, GaussianProcessRegressor) for e in ests):
                vars_ = []
                means_ = []
                for e in ests:
                    m, s = e.predict(X_feat, return_std=True)
                    means_.append(m)
                    vars_.append(s**2)
                mean_encoded = np.stack(means_, axis=1)  # (n, k)
                var_encoded  = np.stack(vars_, axis=1)   # (n, k)
        elif isinstance(reg_wrap, GaussianProcessRegressor):
            m, s = reg_wrap.predict(X_feat, return_std=True)
            if m.ndim == 1: m = m[:, None]; s = s[:, None]
            mean_encoded = m
            var_encoded  = s**2

    if mean_encoded is not None:
        Y_pred_encoded = mean_encoded
    else:
        Y_pred_encoded = trained_pipeline.predict(X_test)

    # decode Y
    if encoder == "PCA":
        Y_pred_std = decoder.inverse_transform(Y_pred_encoded)
        Y_true_std = decoder.inverse_transform(Y_true_encoded)
    elif encoder == "VAE":
        Y_pred_std = decoder.decoder.predict(Y_pred_encoded)
        Y_true_std = decoder.decoder.predict(Y_true_encoded)
    else:
        Y_pred_std = Y_pred_encoded
        Y_true_std = Y_true_encoded

    Y_pred_full = Y_pred_std * std_val + mean_val
    Y_true_full = Y_true_std * std_val + mean_val

    n = Y_pred_full.shape[0]
    lat, lon = spatial_shape
    Y_pred_out = Y_pred_full.reshape(n, lat, lon)
    Y_true_out = Y_true_full.reshape(n, lat, lon) 
    rmse = root_mean_squared_error(Y_true_full, Y_pred_full)

    # decode variance if PCA is used
    if (var_encoded is not None) and encoder == "PCA":
        k = var_encoded.shape[1]
        comps = decoder.components_[:k]          # (k, D)
        W2 = comps**2                            # (k, D)
        var_std_flat_all = var_encoded @ W2      # (n, D)
        if residual_variance is not None:
            var_std_flat_all += residual_variance           # (n, D), add residual variance
        var_raw_flat_all = var_std_flat_all * (std_val**2)
        Y_var_out = var_raw_flat_all.reshape(n, lat, lon)
    else:
        Y_var_out = np.full((n, lat, lon), np.nan)

    r2_value = r2_score(Y_true_full, Y_pred_full)

    return {"Y_pred_out": Y_pred_out,
            "Y_true_out": Y_true_out,
            "Y_var_out": Y_var_out,
            "r2_score": r2_value,
            "rmse": rmse}

def run_training_all(train_dict,regressor_type="GPR", kernel="RBF_White", pca_variance_ratio=0.999, encoder="PCA", vae_config=None, fixed_hp=True):
    X_train, Y_train, var_name, spatial_shape, lat_array, lon_array = load_training_data(train_dict)
    training_info = run_training(X_train, Y_train, regressor_type=regressor_type, kernel=kernel, pca_variance_ratio=pca_variance_ratio, encoder=encoder, vae_config=vae_config,fixed_hp=fixed_hp)
    trained_pipeline, decoder, mean_val, std_val, n_components, residual_variance = training_info["trained_pipeline"], training_info["decoder"], training_info["mean_val"], training_info["std_val"], training_info["n_components_retained"], training_info["residual_variance"]
    return {
        "pipeline_model": trained_pipeline,
        "decoder": decoder,
        "encoder": encoder,
        "regressor_type": regressor_type,
        "mean_val": mean_val,
        "std_val": std_val,
        "n_components_retained": n_components,
        "original_variable": var_name,
        "spatial_shape": spatial_shape,
        "lat_array": lat_array,
        "lon_array": lon_array
    }


# 20% for validation; 80% for training
# only sample once
def run_training_28(train_dict,  regressor_type="GPR", kernel="RBF_White", pca_variance_ratio=0.999, encoder="PCA", vae_config=None, return_validation=True):
    # load data
    X, Y_flat, var_name, spatial_shape, lat_array, lon_array = load_training_data(train_dict)
    # split data for training and testing
    X_train, X_test, Y_train_flat, Y_test_flat = train_test_split(X, Y_flat, test_size=0.2)
    # train model
    training_info = run_training(X_train, Y_train_flat, regressor_type=regressor_type, kernel=kernel, pca_variance_ratio=pca_variance_ratio, encoder=encoder, vae_config=vae_config)
    trained_pipeline, decoder, mean_val, std_val, n_components, residual_variance = training_info["trained_pipeline"], training_info["decoder"], training_info["mean_val"], training_info["std_val"], training_info["n_components_retained"], training_info["residual_variance"]
    trained_pipeline = joblib.load(trained_pipeline)
    decoder = joblib.load(decoder)

    if return_validation:
        # compute validation metrics
        validation_metrics = return_validation_function(X_test, Y_test_flat, trained_pipeline, decoder, mean_val, std_val, spatial_shape, encoder, residual_variance)
        Y_pred_out, Y_true_out, r2_score = validation_metrics["Y_pred_out"], validation_metrics["Y_true_out"], validation_metrics["r2_score"]
        # plotting for validation
        r2_map = compute_r2_map(Y_true_out, Y_pred_out, lat_array, lon_array)
        plot_r2_map_with_latlon(r2_map, lat_array=lat_array, lon_array=lon_array,  regressor_type= regressor_type,
                                encoder=encoder, kernel=kernel, save_dir="outputs/logs")
        print(f"[INFO] R² Score: {r2_score:.4f}")
        print("[INFO] here we picked timesteps 0 1 2 3 999 for demonstration, edit the code if you want to see other timesteps")
        for timestep in [0, 1, 2, 3, 999]:
            plot_prediction_maps_with_info(Y_true_out, Y_pred_out, lat_array=lat_array, lon_array=lon_array, timestep=timestep, emulator_name= regressor_type,
                encoder_name=encoder, kernel_name=kernel, save_folder="outputs/maps", title_suffix=f"Timestep {timestep}")

    return {
        "pipeline_model": trained_pipeline,
        "decoder": decoder,
        "encoder": encoder,
        "r2_score": r2_score,
        "n_components_retained": n_components,
        "original_variable": var_name,
        "spatial_shape": spatial_shape,
        "Y_pred_out": Y_pred_out,
        "Y_true_out": Y_true_out,
        "X_test": X_test,
        "encoder_used": encoder,
        "regressor_type":  regressor_type
    }


def run_training_leave_one_out(train_dict, regressor_type="GPR", kernel="RBF_White", pca_variance_ratio=0.995, encoder="PCA", vae_config=None,  return_validation=False):

    # 1. 加载原始数据
    X, Y_flat, var_name, spatial_shape, lat, lon = load_training_data(train_dict)
    n_samples = X.shape[0]

    Y_pred_full = []
    Y_true_full = []
    Y_var_full = []
    rmse_full = []

    time_start = time.time()

    batch_size = 1  # Number of samples to leave out each time

    for i in range(0, n_samples, batch_size):
        # split data
        # ----------
        test_indices = np.arange(i, min(i + batch_size, n_samples))
        train_indices = np.setdiff1d(np.arange(n_samples), test_indices)
        
        X_train = X.iloc[train_indices]
        Y_train_flat = Y_flat[train_indices]
        X_test = X.iloc[test_indices]
        Y_test_flat = Y_flat[test_indices]

        # fit model
        # ----------
        training_info = run_training(X_train, 
                                     Y_train_flat, 
                                     regressor_type=regressor_type, 
                                     kernel=kernel, 
                                     pca_variance_ratio=pca_variance_ratio, 
                                     encoder=encoder, 
                                     vae_config=vae_config,
                                     fixed_hp=False)
        
        trained_pipeline = training_info["trained_pipeline"]
        decoder = training_info["decoder"]
        mean_val = training_info["mean_val"]
        std_val = training_info["std_val"]
        residual_variance = training_info["residual_variance"]
        n_components = training_info["n_components_retained"]

        trained_pipeline = joblib.load(trained_pipeline)
        decoder = joblib.load(decoder)
        
        validation_metrics = return_validation_function(X_test, 
                                                        Y_test_flat, 
                                                        trained_pipeline, 
                                                        decoder, 
                                                        mean_val, 
                                                        std_val, 
                                                        spatial_shape, 
                                                        encoder,
                                                        residual_variance
                                                        )
        Y_pred_out, Y_true_out, Y_var_out, rmse = validation_metrics["Y_pred_out"], validation_metrics["Y_true_out"], validation_metrics["Y_var_out"], validation_metrics["rmse"]

        Y_pred_full.extend(Y_pred_out)
        Y_true_full.extend(Y_true_out)
        Y_var_full.extend(Y_var_out)
        rmse_full.append(rmse)

        time_spt = time.time() - time_start
        print(f"[TIME] {min(i + batch_size, n_samples)}/{n_samples} completed in {time_spt:.2f} seconds.")
        time_start = time.time()

    Y_pred_full = np.array(Y_pred_full)
    Y_true_full = np.array(Y_true_full)
    Y_var_full = np.array(Y_var_full)
    rmse_full = np.array(rmse_full)

    n = Y_pred_full.shape[0]
    Y_pred_out_full = Y_pred_full.reshape(n, lat.shape[0], lon.shape[0])
    Y_true_out_full = Y_true_full.reshape(n, lat.shape[0], lon.shape[0])
    Y_var_out_full = Y_var_full.reshape(n, lat.shape[0], lon.shape[0]) if encoder == "PCA" else None

    # Save Y_pred_out_full and Y_true_out_full as NetCDF files
    output_dir = "examples/outputs/leave_one_out/"
    os.makedirs(output_dir, exist_ok=True)
    y_pred_path = os.path.join(output_dir, "Y_pred_out_full.nc")
    y_true_path = os.path.join(output_dir, "Y_true_out_full.nc")
    # Remove files if they already exist
    if os.path.exists(y_pred_path):
        os.remove(y_pred_path)
    if os.path.exists(y_true_path):
        os.remove(y_true_path)

    xr.Dataset({
            "mean": (["time", "lat", "lon"], Y_true_out_full),
            "latitude": (["lat"], lat),
            "longitude": (["lon"], lon)
        }).to_netcdf(y_true_path)

    if encoder == "PCA":
        xr.Dataset({
            "mean": (["time", "lat", "lon"], Y_pred_out_full),
            "variance": (["time", "lat", "lon"], Y_var_out_full),
            "latitude": (["lat"], lat),
            "longitude": (["lon"], lon)
        }).to_netcdf(y_pred_path)
    else:
        xr.DataArray(Y_pred_out_full, dims=["time", "lat", "lon"]).to_netcdf(y_pred_path)

    print(f"[INFO] Y_pred_out_full saved to {y_pred_path}")
    print(f"[INFO] Y_true_out_full saved to {y_true_path}")

    if return_validation:
        # plot
        plot_histogram_4_leave1out(Y_true_out_full, Y_pred_out_full)

    time_spt = time.time() - time_start
    print(f"[TIME] plot completed in {time_spt:.2f} seconds.")

    return {
        "r2_score": r2_score,
        "n_components_retained": n_components,
        "original_variable": var_name,
        "spatial_shape": spatial_shape,
        "Y_pred_out": Y_pred_out_full,
        "Y_true_out": Y_true_out_full,
        "Y_var_out": Y_var_out_full,
        "encoder_used": encoder,
        "regressor_type": regressor_type
    }

# 10 fold cross-validation
# 10% for validation; 90% for training
def run_training_10fold(train_dict,  regressor_type="GPR", kernel="RBF_White", pca_variance_ratio=0.999, encoder="PCA", vae_config=None, return_validation=True):
    # load data
    X, Y_flat, var_name, spatial_shape, lat_array, lon_array = load_training_data(train_dict)
    # split data for training and testing
    X_train, X_test, Y_train_flat, Y_test_flat = train_test_split(X, Y_flat, test_size=0.2)
    # train model
    training_info = run_training(X_train, Y_train_flat, regressor_type=regressor_type, kernel=kernel, pca_variance_ratio=pca_variance_ratio, encoder=encoder, vae_config=vae_config)
    trained_pipeline, decoder, mean_val, std_val, n_components = training_info["trained_pipeline"], training_info["decoder"], training_info["mean_val"], training_info["std_val"], training_info["n_components_retained"]
    trained_pipeline = joblib.load(trained_pipeline)
    decoder = joblib.load(decoder)

    if return_validation:
        # compute validation metrics
        validation_metrics = return_validation_function(X_test, Y_test_flat, trained_pipeline, decoder, mean_val, std_val, spatial_shape, encoder)
        Y_pred_out, Y_true_out, r2_score = validation_metrics["Y_pred_out"], validation_metrics["Y_true_out"], validation_metrics["r2_score"]
        # plotting for validation
        r2_map = compute_r2_map(Y_true_out, Y_pred_out, lat_array, lon_array)
        plot_r2_map_with_latlon(r2_map, lat_array=lat_array, lon_array=lon_array,  regressor_type= regressor_type,
                                encoder=encoder, kernel=kernel, save_dir="outputs/logs")
        print(f"[INFO] R² Score: {r2_score:.4f}")
        print("[INFO] here we picked timesteps 0 1 2 3 999 for demonstration, edit the code if you want to see other timesteps")
        for timestep in [0, 1, 2, 3, 999]:
            plot_prediction_maps_with_info(Y_true_out, Y_pred_out, lat_array=lat_array, lon_array=lon_array, timestep=timestep, emulator_name= regressor_type,
                encoder_name=encoder, kernel_name=kernel, save_folder="outputs/maps", title_suffix=f"Timestep {timestep}")

    return {
        "pipeline_model": trained_pipeline,
        "decoder": decoder,
        "encoder": encoder,
        "r2_score": r2_score,
        "n_components_retained": n_components,
        "original_variable": var_name,
        "spatial_shape": spatial_shape,
        "Y_pred_out": Y_pred_out,
        "Y_true_out": Y_true_out,
        "X_test": X_test,
        "encoder_used": encoder,
        "regressor_type":  regressor_type
    }

def run_training_LGBM_optimization(train_dict, pca_variance_ratio=0.999, encoder="PCA", vae_config=None):
    """
    This function is to perform hyperparameter optimization for LGBMRegressor using GridSearchCV.
    It returns the best model found during the search.
    Use PCA as encoder.
    """
    import optuna
    import optuna.visualization as vis
    from lightgbm import LGBMRegressor
    from sklearn.model_selection import cross_val_score
    import matplotlib.pyplot as plt

    # load data
    print("[INFO] Loading training data...")
    X_train, Y_train, var_name, spatial_shape, lat_array, lon_array = load_training_data(train_dict)

    # encode the chosen training Y
    Y_train_encoded, decoder, mean_val, std_val, residual_variance = encode(
        Y_train,
        encoder=encoder,
        pca_variance_ratio=pca_variance_ratio,
        vae_config=vae_config,
        fixed_hp=False
    )
    latent_dim = Y_train_encoded.shape[1]

    print("[INFO] Starting hyperparameter optimization for LGBMRegressor...")
    lgbm_regressor = build_regressor(
        regressor_type="LGBM", 
        kernel_name=None, 
        encoder=encoder, 
        fixed_hp=False)

    def objective(trial):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 128, 512),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-2, 0.05),
            "n_estimators": trial.suggest_int("n_estimators", 300, 700),
            "subsample": trial.suggest_uniform("subsample", 0.4, 0.8),
            "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 1, 10),
        }
        model = MultiOutputRegressor(LGBMRegressor(**params, random_state=42, n_jobs=1))
        # 使用负 MSE（越大越好），少量折数以加速
        scores = cross_val_score(model, X_train, Y_train_encoded, cv=4, scoring="neg_mean_squared_error")
        return scores.mean()

    n_jobs = 1  # Set to 1 to avoid nested parallelism issues
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30, n_jobs=n_jobs)
    print(f"[OPTUNA] best params: {study.best_params}, best score: {study.best_value:.5f}")
    best_params = study.best_params
    lgbm_regressor = LGBMRegressor(**best_params, random_state=42, n_jobs=n_jobs)
    # Visualization of optimization results
    output_dir = "examples/outputs/optimization_plots"
    os.makedirs(output_dir, exist_ok=True)

    vis.plot_optimization_history(study).write_image(os.path.join(output_dir, "optimization_history.png"))
    vis.plot_parallel_coordinate(study).write_image(os.path.join(output_dir, "parallel_coordinate.png"))
    vis.plot_param_importances(study).write_image(os.path.join(output_dir, "param_importances.png"))
    vis.plot_slice(study).write_image(os.path.join(output_dir, "slice_plot.png"))
    vis.plot_contour(study).write_image(os.path.join(output_dir, "contour_plot.png"))

    print(f"[INFO] Optimization plots saved to {output_dir}")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", MultiOutputRegressor(lgbm_regressor))
    ])

    pipeline.fit(X_train, Y_train_encoded)

    print(f"[INFO] Best parameters found: {study.best_params}")
    print(f"[INFO] Best R² score from CV: {-study.best_value:.4f}")

    best_model = pipeline

    return {
        "best_model": best_model,
        "best_params": study.best_params,
        "best_r2_score": study.best_value
    }