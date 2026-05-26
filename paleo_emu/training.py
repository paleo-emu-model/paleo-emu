"""
Training module using chosen regressors, kernels, and encoders.

See config loader (_load.py / config_loader.py) for the typed config:
- PaleoEmuConfig
- _RegressorConfig
- make_kernel

The joblib artifact will contain:
- "model": best fitted EncodedTargetRegressor
           (includes encoder, scaler, and regressor)
- "grid_search": fitted GridSearchCV
- "lat_array": latitude grid used during training
- "lon_array": longitude grid used during training
"""

import os
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.exceptions import ConvergenceWarning
from paleo_emu.config import PaleoEmuConfig, _GPRegressorConfig, _XGBRegressorConfig, make_kernel
from paleo_emu.regressor import EncodedTargetRegressor, GPMultiOutputWithStd


class TrainingGenerator:
    """Utility for training emulators with specified encoders and regressors.

    Parameters
    ----------
    model_configuration : PaleoEmuConfig
        Typed configuration object loaded via `load_config(path)`.
    X_train : array-like, shape (n_samples, n_features)
        Input features for training.
    y_train : array-like, shape (n_samples, n_outputs)
        Target fields for training (e.g. flattened spatial fields).
    lat_array : array-like
        Latitude grid corresponding to y fields.
    lon_array : array-like
        Longitude grid corresponding to y fields.
    output_dir : str, optional
        Directory to save the joblib artifact. Defaults to current directory.

    The joblib artifact will contain:
    - "model": best fitted EncodedTargetRegressor
    - "grid_search": fitted GridSearchCV
    - "lat_array": latitude grid
    - "lon_array": longitude grid
    """

    def __init__(
        self,
        model_configuration: PaleoEmuConfig,
        X_train,
        Y_train,
        lat_array,
        lon_array,
        output_dir: str = ".",
        var_name: str | None = None,
        var_attrs: dict | None = None,
        diag_dir: Path | None = None,
    ):
        self.cfg: PaleoEmuConfig = model_configuration
        self.X_train = X_train
        self.Y_train = Y_train
        self.lat_array = lat_array
        self.lon_array = lon_array
        self.output_dir = output_dir
        self.var_name = var_name
        self.var_attrs = var_attrs or {}
        self.diag_dir = diag_dir

    # ----------------- helpers -----------------
    def _build_kernel_candidates(self):
        """Build a list of ARD kernels, one per kernel name in the config.

        ARD is enforced by always using a length_scale vector of shape (n_features,).
        """
        reg_cfg: _GPRegressorConfig = self.cfg.regressor_config
        n_features = self.X_train.shape[1]
        return [
            make_kernel(name, reg_cfg, n_features=n_features)
            for name in reg_cfg.kernels
        ]

    def _build_param_grid(self):
        if isinstance(self.cfg.regressor_config, _GPRegressorConfig):
            kernels = self._build_kernel_candidates()
            # NOTE: parameter path:
            # EncodedTargetRegressor(base_estimator=Pipeline([...]))
            # -> base_estimator (Pipeline)
            # -> "regressor" step (MultiOutputRegressor)
            # -> underlying estimator (GaussianProcessRegressor) -> "kernel"
            return {"base_estimator__regressor__estimator__kernel": kernels}
        elif isinstance(self.cfg.regressor_config, _XGBRegressorConfig):
            reg_cfg: _XGBRegressorConfig = self.cfg.regressor_config
            param_grid = {
                "base_estimator__regressor__estimator__max_depth": reg_cfg.max_depth,
                "base_estimator__regressor__estimator__learning_rate": reg_cfg.learning_rate,
                "base_estimator__regressor__estimator__n_estimators": reg_cfg.n_estimators,
                "base_estimator__regressor__estimator__subsample": reg_cfg.subsample,
                "base_estimator__regressor__estimator__colsample_bytree": reg_cfg.colsample_bytree,
                "base_estimator__regressor__estimator__min_child_weight": reg_cfg.min_child_weight,
            }
            return param_grid

    def _build_regressor(self) -> MultiOutputRegressor:
        """Build a (potentially) multi-output Gaussian Process regressor."""
        if isinstance(self.cfg.regressor_config, _GPRegressorConfig):
            reg_cfg: _GPRegressorConfig = self.cfg.regressor_config
            base_regressor = GaussianProcessRegressor(
                normalize_y=True,
                alpha=reg_cfg.alpha,
                n_restarts_optimizer=reg_cfg.n_restarts_optimizer,
                random_state=self.cfg.random_state,
            )

        elif isinstance(self.cfg.regressor_config, _XGBRegressorConfig):
            reg_cfg: _XGBRegressorConfig = self.cfg.regressor_config
            base_regressor = XGBRegressor(
                n_estimators=reg_cfg.n_estimators[0],  # Use first value as default
                max_depth=reg_cfg.max_depth[0],
                learning_rate=reg_cfg.learning_rate[0],
                subsample=reg_cfg.subsample[0],
                colsample_bytree=reg_cfg.colsample_bytree[0],
                min_child_weight=reg_cfg.min_child_weight[0],
                random_state=self.cfg.random_state,
                verbosity=0,  # Suppress XGBoost's own logging
            ) 


        # Wrap in GPMultiOutputWithStd for GP, regular MultiOutputRegressor for others
        if isinstance(self.cfg.regressor_config, _GPRegressorConfig):
            return GPMultiOutputWithStd(base_regressor)
        else:
            return MultiOutputRegressor(base_regressor)
    # ----------------- main training -----------------
    def _run_training(self) -> str:
        """Run training and export results as a joblib file.

        Returns
        -------
        artifact_path : str
            Path to the saved joblib artifact.
        """
        if self.X_train is None or self.Y_train is None:
            raise ValueError(
                "X_train and Y_train must be provided. "
                "Automatic data loading is not implemented here."
            )

        # Build base regressor, param_grid, and pipeline (on X only)
        regressor = self._build_regressor()
        base_pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", regressor),
            ]
        )

        # Wrap with EncodedTargetRegressor so Y is encoded/decoded internally
        model = EncodedTargetRegressor(
            base_estimator=base_pipeline,
            model_config=self.cfg,
            return_encoded=False,  # we want decoded predictions in original Y space
        )

        param_grid = self._build_param_grid()
        cv_cfg = self.cfg.cv

        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv_cfg.folds,
            n_jobs=cv_cfg.n_jobs,
            scoring=cv_cfg.scoring,
        )

        # Fit on RAW Y (high-dimensional field); encoding happens inside model
        # Suppress ConvergenceWarnings during parallel GP optimization
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            grid.fit(self.X_train, self.Y_train)
        
        print("-----------------------------------")
        print("Best parameters:")
        for param, value in grid.best_params_.items():
            print(f"  {param}: {value}")
        print(f"Best CV score: {grid.best_score_:.4f}")
        print("------------------------------------")

        best_model = grid.best_estimator_

        # export with joblib
        output_dir = str(self.cfg.output_dir) if self.cfg.output_dir is not None else self.output_dir
        os.makedirs(output_dir, exist_ok=True)
        artifact = {
            "model": best_model,
            "grid_search": grid,
            "lat_array": self.lat_array,
            "lon_array": self.lon_array,
            "mean_val": best_model.mean_val_,
            "std_val": best_model.std_val_,
            "nan_mask": best_model.nan_mask_,
            "var_name": self.var_name,
            "var_attrs": self.var_attrs,
        }
        artifact_name = (
            self.cfg.artifact_name
            if self.cfg.artifact_name is not None
            else f"{self.cfg.model_run_name}_fitted_pipeline.joblib"
        )
        artifact_path = os.path.join(output_dir, artifact_name)

        joblib.dump(artifact, artifact_path)
        print(f"[INFO] Saved fitted model artifact to {artifact_path}")

        if self.diag_dir is not None:
            self._save_diagnostics(grid, best_model)

        return artifact_path

    def _save_diagnostics(self, grid, best_model) -> None:
        """Save training diagnostics to self.diag_dir / figures/."""
        import pandas as pd
        import matplotlib.pyplot as plt
        import xarray as xr
        import cartopy.crs as ccrs
        import cartopy.util as cutil
        from sklearn.decomposition import PCA
        from paleo_emu.encoders import _VAE
        from paleo_emu.validation import compute_r2_map

        diag_dir = Path(self.diag_dir)
        fig_dir = diag_dir / "figures"
        diag_dir.mkdir(parents=True, exist_ok=True)
        fig_dir.mkdir(parents=True, exist_ok=True)

        # --- CV results ---
        pd.DataFrame(grid.cv_results_).to_csv(diag_dir / "cv_results.csv", index=False)
        print(f"[DIAG] cv_results.csv → {diag_dir}")

        # --- R² map (train-set predictions vs. truth) ---
        n_lat, n_lon = len(self.lat_array), len(self.lon_array)
        Y_pred_flat = best_model.predict(self.X_train)
        Y_pred_3d = Y_pred_flat.reshape(-1, n_lat, n_lon)
        Y_true_3d = np.asarray(self.Y_train).reshape(-1, n_lat, n_lon)
        r2_map = compute_r2_map(Y_true_3d, Y_pred_3d, self.lat_array, self.lon_array)

        xr.DataArray(
            r2_map, dims=["latitude", "longitude"],
            coords={"latitude": self.lat_array, "longitude": self.lon_array},
            name="r2", attrs={"long_name": "R² score (training set)", "units": "1"},
        ).to_netcdf(diag_dir / "r2_map.nc")

        lon = np.array(self.lon_array)
        lat = np.array(self.lat_array)
        if lon.size > 1 and np.isclose(lon[-1], 360.0, atol=1.0) and np.isclose(lon[0], 0.0, atol=1.0):
            r2_map = r2_map[..., :-1]
            lon = lon[:-1]
        r2_cyc, lon_cyc = cutil.add_cyclic_point(r2_map, coord=lon)
        Lon, Lat = np.meshgrid(lon_cyc, lat)

        fig, ax = plt.subplots(figsize=(10, 5),
                               subplot_kw={"projection": ccrs.PlateCarree()})
        im = ax.pcolormesh(Lon, Lat, r2_cyc, vmin=0.8, vmax=1, cmap="RdYlGn",
                           shading="auto", transform=ccrs.PlateCarree())
        ax.set_global()
        ax.coastlines(linewidth=0.6)
        plt.colorbar(im, ax=ax, label="R²", shrink=0.7)
        
        ax.set_title("R² map (training set)", fontsize=12)
        plt.tight_layout()
        fig.savefig(fig_dir / "r2_map.png", dpi=300)
        plt.close(fig)
        print(f"[DIAG] r2_map.nc + r2_map.png → {diag_dir}/figures/")

        # --- encoder-specific diagnostics ---
        enc = best_model.encoder_model_
        if isinstance(enc, PCA):
            var_ratio = enc.explained_variance_ratio_
            cumulative = np.cumsum(var_ratio)
            n_comp = len(var_ratio)
            pd.DataFrame({
                "component": range(1, n_comp + 1),
                "explained_variance_ratio": var_ratio,
                "cumulative_variance_ratio": cumulative,
            }).to_csv(diag_dir / "pca_variance.csv", index=False)

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(range(1, n_comp + 1), var_ratio, label="Individual")
            ax.plot(range(1, n_comp + 1), cumulative, "r-o", markersize=4,
                    label="Cumulative")
            ax.set_xlabel("PC", fontsize=12)
            ax.set_ylabel("Explained variance ratio", fontsize=12)
            ax.set_title("PCA explained variance", fontsize=12)
            ax.legend(fontsize=12)
            ax.grid(True)
            plt.tight_layout()
            fig.savefig(fig_dir / "pca_variance.png", dpi=300)
            plt.close(fig)
            print(f"[DIAG] pca_variance.csv + pca_variance.png → {diag_dir}/figures/")

        elif isinstance(enc, _VAE):
            import tensorflow as tf  # noqa: F401 — needed for enc(...)
            Y_valid = np.asarray(self.Y_train)[:, ~best_model.nan_mask_]
            eps = 1e-99
            Y_norm = (Y_valid - best_model.mean_val_) / best_model.std_val_ + eps
            x_decoded, mean_lat, logvar_lat = enc(Y_norm.astype("float32"))

            recon_err = np.mean((Y_norm - x_decoded.numpy()) ** 2, axis=1)
            xr.DataArray(
                recon_err, dims=["sample"],
                name="reconstruction_mse",
                attrs={"long_name": "Per-sample VAE reconstruction MSE"},
            ).to_netcdf(diag_dir / "vae_reconstruction_error.nc")

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(recon_err, "b-")
            ax.set_xlabel("Sample", fontsize=12)
            ax.set_ylabel("MSE", fontsize=12)
            ax.set_title("VAE reconstruction error (training set)", fontsize=12)
            ax.grid(True)
            plt.tight_layout()
            fig.savefig(fig_dir / "vae_reconstruction_error.png", dpi=300)
            plt.close(fig)

            mean_np = mean_lat.numpy()
            logvar_np = logvar_lat.numpy()
            kl_per_dim = 0.5 * np.mean(
                np.exp(logvar_np) + mean_np ** 2 - 1 - logvar_np, axis=0
            )
            pd.DataFrame({
                "latent_dim": range(len(kl_per_dim)),
                "kl_divergence": kl_per_dim,
                "mean_of_mean": np.mean(mean_np, axis=0),
                "std_of_mean": np.std(mean_np, axis=0),
                "mean_of_logvar": np.mean(logvar_np, axis=0),
            }).to_csv(diag_dir / "vae_latent_stats.csv", index=False)
            print(f"[DIAG] VAE diagnostics → {diag_dir}")

        # --- regressor-specific diagnostics ---
        inner_reg = best_model.estimator_["regressor"]
        if isinstance(inner_reg, GPMultiOutputWithStd):
            rows = []
            for i, est in enumerate(inner_reg.estimators_):
                row: dict = {"latent_component": i}
                try:
                    row["log_marginal_likelihood"] = est.log_marginal_likelihood_value_
                    kernel = est.kernel_
                    row["noise_level"] = kernel.k2.noise_level
                    ls = kernel.k1.length_scale
                    if hasattr(ls, "__len__"):
                        for fname, lsv in zip(self.cfg.X_column_names, ls):
                            row[f"length_scale_{fname}"] = lsv
                    else:
                        row["length_scale"] = float(ls)
                except AttributeError:
                    pass
                rows.append(row)
            pd.DataFrame(rows).to_csv(diag_dir / "gp_kernel_params.csv", index=False)
            print(f"[DIAG] gp_kernel_params.csv → {diag_dir}")

        else:
            try:
                importances = np.array([
                    est.feature_importances_ for est in inner_reg.estimators_
                ])
                mean_imp = np.mean(importances, axis=0)
                feature_names = list(self.cfg.X_column_names)
                df_imp = pd.DataFrame({
                    "feature": feature_names,
                    "mean_importance": mean_imp,
                })
                for i, imp_row in enumerate(importances):
                    df_imp[f"component_{i}"] = imp_row
                df_imp.sort_values("mean_importance", ascending=False).to_csv(
                    diag_dir / "xgb_feature_importance.csv", index=False
                )

                sorted_idx = np.argsort(mean_imp)
                fig, ax = plt.subplots(
                    figsize=(8, max(4, len(feature_names) * 0.5))
                )
                ax.barh(
                    [feature_names[j] for j in sorted_idx],
                    mean_imp[sorted_idx],
                )
                ax.set_xlabel("Mean feature importance", fontsize=12)
                ax.set_title(
                    "XGBoost feature importance\n(mean across latent components)",
                    fontsize=12,
                )
                plt.tight_layout()
                fig.savefig(fig_dir / "xgb_feature_importance.png", dpi=300)
                plt.close(fig)
                print(f"[DIAG] xgb_feature_importance.csv + .png → {diag_dir}")
            except AttributeError:
                pass
