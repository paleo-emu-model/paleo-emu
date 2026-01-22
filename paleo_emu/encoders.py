import os
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tensorflow.keras import layers, models
import keras

from paleo_emu.config import (
    PaleoEmuConfig,
    _PCAEncoderConfig,
    _LearnedEncoderConfig,
)


@keras.saving.register_keras_serializable()
class _VAE(keras.Model):
    def __init__(self, input_dim, latent_dim):
        super(_VAE, self).__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        # encoder
        self.encoder = models.Sequential([
            layers.InputLayer(input_shape=(input_dim,)),    # input_dim=7008
            layers.Dense(4096, activation="relu"),           # 7008 → 4096
            layers.Dense(2048, activation="relu"),           # 4096 → 2048
            layers.Dense(4096, activation="relu"),           # 2048 → 4096
            layers.Dense(latent_dim * 2)                     # gives mean and logvar
        ])
        # decoder
        self.decoder = models.Sequential([
            layers.InputLayer(input_shape=(latent_dim,)),
            layers.Dense(4096, activation="relu"),
            layers.Dense(2048, activation="relu"),
            layers.Dense(2048, activation="relu"),   
            layers.Dense(1024, activation="relu"),
            layers.Dense(7008)
        ])
    def get_config(self):
        config = super().get_config()
        config.update({
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        # Filter out Keras-specific parameters that aren't VAE constructor arguments
        vae_config = {k: v for k, v in config.items()
                      if k in ['input_dim', 'latent_dim']}  # Add your actual constructor params
        return cls(**vae_config)
    
    def reparameterize(self, mean, logvar):
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        eps = tf.random.normal(shape=(batch, dim))
        return eps * tf.exp(logvar * 0.5) + mean

    def call(self, x):
        x_encoded = self.encoder(x)
        mean, logvar = tf.split(x_encoded, num_or_size_splits=2, axis=1)
        z = self.reparameterize(mean, logvar)
        x_decoded = self.decoder(z)
        return x_decoded, mean, logvar

def _compute_vae_loss(x, x_decoded, mean, logvar):
    reconstruction_loss = tf.reduce_mean(tf.square(x - x_decoded))
    kl_loss = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mean) - tf.exp(logvar))
    return reconstruction_loss + kl_loss

def _save_vae_log(epoch_losses, latent_dim, epochs, learning_rate, batch_size, kl_weight, log_dir="training/logs"):

    os.makedirs(log_dir, exist_ok=True)

    info_str = f"latent{latent_dim}_ep{epochs}_lr{learning_rate}_bs{batch_size}_kl{kl_weight}"

    loss_curve_filename = os.path.join(log_dir, f"loss_curve_{info_str}.png")

    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(epoch_losses)+1), epoch_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"VAE Loss Curve ({info_str})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(loss_curve_filename, dpi=300)
    plt.close()

    print(f"[INFO] Loss curve saved to: {loss_curve_filename}")

    # --- save hyperparameters and final loss to CSV ---
    log_file = os.path.join(log_dir, "vae_hyperparameter_log.csv")

    log_entry = {
        "latent_dim": latent_dim,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "kl_weight": kl_weight,
        "final_loss": epoch_losses[-1]  # the loss of the final epoch
    }

    if not os.path.exists(log_file):
        df = pd.DataFrame([log_entry])
        df.to_csv(log_file, index=False)
    else:
        df = pd.read_csv(log_file)
        df = pd.concat([df, pd.DataFrame([log_entry])], ignore_index=True)
        df.to_csv(log_file, index=False)

    print(f"[INFO] Hyperparameter log updated: {log_file}")



class EncoderGenerator:
    """Utility for building PCA or VAE encoders from data.

    Parameters
    ----------
    Y : array-like, shape (n_samples, n_features)
        Input data to be encoded.
    model_config : PaleoEmuConfig
        Typed configuration object with encoder parameters. Its
        `encoder_config` field is either:
        - _PCAEncoderConfig (for PCA)
        - _LearnedEncoderConfig (for VAE / learned encoder)

    Attributes
    ----------
    mean_val : float
        Mean of flattened input used for normalization.
    std_val : float
        Standard deviation of flattened input used for normalization.
    Y_norm : ndarray
        Normalized input data (same shape as `Y`).
    """

    def __init__(self, Y, model_config: PaleoEmuConfig):
        """Create an EncoderGenerator and compute normalization stats."""
        if not isinstance(model_config, PaleoEmuConfig):
            raise TypeError(
                f"model_config must be a PaleoEmuConfig, got {type(model_config)}"
            )

        self.cfg: PaleoEmuConfig = model_config
        self.encoder_cfg = self.cfg.encoder_config

        # Keep a flat copy for global statistics and compute normalization
        self.Y_flat = np.asarray(Y).ravel()
        self.mean_val = np.mean(self.Y_flat)
        self.std_val = np.std(self.Y_flat)
        self.Y_norm = (np.asarray(Y) - self.mean_val) / self.std_val + 1e-99  # avoid exact zeros
        # self.Y_norm = (np.asarray(Y) - self.mean_val) / self.std_val

        # print(
        #     f"[INFO] Raw Y_flat min={np.min(self.Y_flat)}, "
        #     f"max={np.max(self.Y_flat)}, mean={np.mean(self.Y_flat)}, "
        #     f"std={np.std(self.Y_flat)}"
        # )
        # print("[INFO] Y_flat standardized to mean ~0, std ~1")

    # ------------------------------------------------------------------
    # PCA encoder
    # ------------------------------------------------------------------
    def _generate_pca_encoder(self):
        if not isinstance(self.encoder_cfg, _PCAEncoderConfig):
            raise TypeError(
                f"_generate_pca_encoder called with wrong encoder config type: "
                f"{type(self.encoder_cfg)}"
            )

        # Pydantic guarantees at least one of these is set
        if self.encoder_cfg.pca_variance_ratio is not None:
            # use variance ratio as n_components parameter for PCA
            n_components = self.encoder_cfg.pca_variance_ratio
        else:
            n_components = self.encoder_cfg.n_components

        model = PCA(n_components=n_components)
        Y_encoded = model.fit_transform(self.Y_norm)

        # print(f"[INFO] PCA n_components_: {model.n_components_}")
        # print(
        #     f"[INFO] Sum explained variance: "
        #     f"{np.sum(model.explained_variance_ratio_)}"
        # )

        return Y_encoded, model, self.mean_val, self.std_val

    # ------------------------------------------------------------------
    # VAE encoder
    # ------------------------------------------------------------------
    def _generate_vae_encoder(self):
        if not isinstance(self.encoder_cfg, _LearnedEncoderConfig):
            raise TypeError(
                f"_generate_vae_encoder called with wrong encoder config type: "
                f"{type(self.encoder_cfg)}"
            )

        latent_dim = self.encoder_cfg.latent_dim
        epochs = self.encoder_cfg.epochs
        learning_rate = self.encoder_cfg.learning_rate
        batch_size = self.encoder_cfg.batch_size
        kl_weight = self.encoder_cfg.kl_weight  # for β-VAE if needed

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        input_dim = self.Y_norm.shape[1]
        vae_model = _VAE(input_dim, latent_dim)

        dataset = tf.data.Dataset.from_tensor_slices(
            self.Y_norm.astype("float32")
        )
        dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

        epoch_losses = []

        for epoch in range(epochs):
            total_loss = 0.0
            for step, x_batch in enumerate(dataset):
                with tf.GradientTape() as tape:
                    x_decoded, mean, logvar = vae_model(x_batch)
                    loss = _compute_vae_loss(
                        x_batch, x_decoded, mean, logvar
                    ) * kl_weight

                grads = tape.gradient(loss, vae_model.trainable_variables)
                optimizer.apply_gradients(zip(grads, vae_model.trainable_variables))
                total_loss += float(loss.numpy())

            avg_loss = total_loss / (step + 1)
            epoch_losses.append(avg_loss)

            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"[VAE] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

        _save_vae_log(
            epoch_losses=epoch_losses,
            latent_dim=latent_dim,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            kl_weight=kl_weight,
        )

        # Encode full dataset with trained encoder
        mean_logvar = vae_model.encoder(self.Y_norm)
        mean, logvar = tf.split(mean_logvar, num_or_size_splits=2, axis=1)
        Y_encoded = mean.numpy()
        model = vae_model

        return Y_encoded, model, self.mean_val, self.std_val

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def generate_encoder(self):
        """Generate encoded representation according to encoder_config."""
        if isinstance(self.encoder_cfg, _PCAEncoderConfig):
            return self._generate_pca_encoder()
        elif isinstance(self.encoder_cfg, _LearnedEncoderConfig):
            return self._generate_vae_encoder()
        else:
            raise TypeError(
                f"Unknown encoder_config type: {type(self.encoder_cfg)}"
            )
