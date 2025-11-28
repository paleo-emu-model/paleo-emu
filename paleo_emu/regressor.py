import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.decomposition import PCA

from paleo_emu.encoders import EncoderGenerator, _VAE  # adjust import if needed
from paleo_emu.config import PaleoEmuConfig


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

        # clone and fit base estimator
        self.estimator_ = clone(self.base_estimator)
        self.estimator_.fit(X, Y_encoded)
        return self

    def predict(self, X):
        """
        Predict in encoded space with base_estimator, then decode to original y.

        If self.return_encoded == True, returns encoded predictions instead.
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
            return y_enc_pred

        return self._decode(y_enc_pred)

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
