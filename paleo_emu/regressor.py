import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.decomposition import PCA
from sklearn.multioutput import MultiOutputRegressor

from paleo_emu.encoders import EncoderGenerator, _VAE
from paleo_emu.config import PaleoEmuConfig


class GPMultiOutputWithStd(MultiOutputRegressor):
    """
    Wrapper around MultiOutputRegressor that supports return_std=True.
    
    This enables uncertainty quantification for multi-output GP models
    by extracting standard deviations from individual estimators.
    """
    
    def predict(self, X, return_std=False):
        """
        Predict with optional uncertainty estimates.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
        return_std : bool, default=False
            If True, return predictions and standard deviations.
            
        Returns
        -------
        y_pred : array of shape (n_samples, n_outputs)
            Predictions.
        y_std : array of shape (n_samples, n_outputs), optional
            Standard deviations (only if return_std=True).
        """
        predictions = super().predict(X)
        
        if return_std:
            # Extract std from each individual estimator
            std_list = []
            for estimator in self.estimators_:
                try:
                    _, std = estimator.predict(X, return_std=True)
                    std_list.append(std)
                except TypeError:
                    # Estimator doesn't support return_std
                    std_list.append(np.full(X.shape[0], np.nan))
            
            y_std = np.column_stack(std_list)
            return predictions, y_std
        
        return predictions


class EncodedTargetRegressor(BaseEstimator, RegressorMixin):
    """
    Meta-estimator that:
      - encodes y using EncoderGenerator (PCA or VAE, depending on config)
      - fits any sklearn regressor on the encoded targets
      - decodes predictions back to the original y space.

    Parameters
    ----------
    base_estimator : sklearn-like regressor
        Any estimator with fit(X, y) and predict(X).
        Can be a bare estimator (e.g. GaussianProcessRegressor)
        or a Pipeline (e.g. Pipeline([("scaler", ...), ("regressor", ...)])).

    model_config : PaleoEmuConfig
        The same config you pass to EncoderGenerator / TrainingGenerator.

    return_encoded : bool, default=False
        If True, predict() returns encoded y instead of decoded y.
        (Useful if you sometimes want latent-space predictions.)
    """

    def __init__(self, base_estimator, model_config: PaleoEmuConfig,
                 return_encoded: bool = False):
        self.base_estimator = base_estimator
        self.model_config = model_config
        self.return_encoded = return_encoded

    # ----------------- core sklearn API -----------------
    def fit(self, X, y):
        """Fit encoder (PCA or VAE) on y, then fit base_estimator on encoded y."""
        # store y shape for sanity/debug
        
        y_arr = np.asarray(y)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        self.n_outputs_ = y_arr.shape[1]

        # build encoder and encode y
        enc_gen = EncoderGenerator(y_arr, self.model_config)
        Y_encoded, encoder_model, mean_val, std_val = enc_gen.generate_encoder()

        # keep encoder bits for later decoding
        self.encoder_model_ = encoder_model   # PCA instance or _VAE instance
        self.mean_val_ = mean_val
        self.std_val_ = std_val

        # ensure 2D target for fitting
        Y_encoded = np.asarray(Y_encoded)
        if Y_encoded.ndim == 1:
            Y_encoded = Y_encoded.reshape(-1, 1)

        # NaN mask: True where any training sample is NaN (e.g. land grid points)
        self.nan_mask_ = np.any(~np.isfinite(y_arr), axis=0)

        # clone and fit base estimator
        self.estimator_ = clone(self.base_estimator)
        self.estimator_.fit(X, Y_encoded)
        return self

    def predict(self, X):
        """
        Predict in encoded space with base_estimator, then decode to original y.
        
        Returns
        -------
        y_pred : array, shape (n_samples, n_outputs)
            Predictions in original y space (or encoded space if return_encoded=True).
            
        Notes
        -----
        This method only returns predictions (standard sklearn API).
        To get uncertainty estimates, use predict_with_variance() instead.
        """
        check_is_fitted = getattr(
            self, "estimator_", None
        ) is not None and getattr(self, "encoder_model_", None) is not None
        if not check_is_fitted:
            raise RuntimeError("EncodedTargetRegressor is not fitted yet.")

        
        y_enc_pred = self.estimator_.predict(X)

        y_enc_pred = np.asarray(y_enc_pred)
        if y_enc_pred.ndim == 1:
            y_enc_pred = y_enc_pred.reshape(-1, 1)

        if self.return_encoded:
            # Return encoded predictions only
            return y_enc_pred

        # Decode to original space and return
        y_pred = self._decode(y_enc_pred)
        if np.any(self.nan_mask_):
            y_pred[:, self.nan_mask_] = np.nan
        return y_pred

    def predict_with_variance(self, X):
        """
        Predict with uncertainty estimates (for GP-based models only).
        
        Returns
        -------
        y_pred : array, shape (n_samples, n_outputs)
            Predictions in original y space.
        y_std : array, shape (n_samples, n_outputs) or None
            Standard deviation in original y space (only for GP models).
            For non-GP models, returns None.
        """
        check_is_fitted = getattr(
            self, "estimator_", None
        ) is not None and getattr(self, "encoder_model_", None) is not None
        if not check_is_fitted:
            raise RuntimeError("EncodedTargetRegressor is not fitted yet.")

        # Try to get predictions and std from base estimator
        try:
            # This works for both direct GP and GPMultiOutputWithStd wrapper
            y_enc_pred, y_enc_std = self.estimator_.predict(X, return_std=True)
        except (TypeError, AttributeError):
            # Fallback for models that don't support return_std (e.g., XGBoost)
            y_enc_pred = self.estimator_.predict(X)
            y_enc_std = None

        y_enc_pred = np.asarray(y_enc_pred)
        if y_enc_pred.ndim == 1:
            y_enc_pred = y_enc_pred.reshape(-1, 1)

        if self.return_encoded:
            # Return encoded predictions with std
            return y_enc_pred, y_enc_std

        y_pred = self._decode(y_enc_pred)
        if np.any(self.nan_mask_):
            y_pred[:, self.nan_mask_] = np.nan

        # Decode std to original space if available
        if y_enc_std is not None:
            # Variance propagation through PCA: for each sample and spatial location,
            # var = sum over PCs: (component[pc] * std_latent[sample, pc])^2
            if isinstance(self.encoder_model_, PCA):
                components = self.encoder_model_.components_.T  # (7008, 13)
                # Vectorized: (800, 1, 13) * (1, 7008, 13) -> (800, 7008, 13)
                contributions = (y_enc_std[:, np.newaxis, :] * components[np.newaxis, :, :]) ** 2
                variance = np.sum(contributions, axis=2)  # (800, 7008)
                y_std_latent_space = np.sqrt(variance)
            else:
                # For VAE, apply decoder to std (approximation)
                z = y_enc_std.astype("float32")
                y_std_latent_space = self.encoder_model_.decoder(z).numpy()
            
            # Apply normalization correction: Var(Y) = Var(Y_norm) * std^2
            eps = 1e-99
            y_std_decoded = (y_std_latent_space - eps) * self.std_val_
        else:
            y_std_decoded = None

        if y_std_decoded is not None and np.any(self.nan_mask_):
            y_std_decoded[:, self.nan_mask_] = np.nan

        return y_pred, y_std_decoded

    def predict_encoded(self, X):
        """Return predictions in latent (encoded) space."""
        check_is_fitted = getattr(
            self, "estimator_", None
        ) is not None and getattr(self, "encoder_model_", None) is not None
        if not check_is_fitted:
            raise RuntimeError("EncodedTargetRegressor is not fitted yet.")

        y_enc_pred = self.estimator_.predict(X)
        return np.asarray(y_enc_pred)
    
    # ----------------- helpers -----------------
    def _decode(self, y_enc):
        """
        Decode encoded predictions back to original Y space,
        handling both PCA and VAE encoders plus normalization.
        """
        # 1) inverse through encoder model
        if isinstance(self.encoder_model_, PCA):
            # PCA inverse_transform operates in the normalized space
            Y_norm = self.encoder_model_.inverse_transform(y_enc)

        elif isinstance(self.encoder_model_, _VAE):
            # VAE decoder takes latent z and returns normalized reconstruction
            # y_enc should be float32 tensor for TF
            z = y_enc.astype("float32")
            Y_norm = self.encoder_model_.decoder(z).numpy()

        else:
            raise TypeError(
                f"Unsupported encoder model type: {type(self.encoder_model_)}"
            )

        # 2) undo normalization from EncoderGenerator
        # EncoderGenerator used: Y_norm = (Y - mean)/std + 1e-99
        # So: Y = (Y_norm - 1e-99) * std + mean
        eps = 1e-99
        Y = (Y_norm - eps) * self.std_val_ + self.mean_val_

        return Y
