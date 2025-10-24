import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from paleo_emu.vae import VAE, compute_vae_loss
from paleo_emu.export import save_training_log

class EncoderGenerator:
    """Utility for building PCA or VAE encoders from data.

    Parameters
    ----------
    Y : array-like, shape (n_samples, n_features)
        Input data to be encoded.
    model_config : object
        Configuration file with encoder parameters.
        Example:

            model_config = type('C', (), {
                'encoder': {'encoder_type': pca, 'n_components': 10},
            })()

            model_config = type('C', (), {
                'encoder': {'encoder_type': vae, 'epochs': 10},
            })()
            
    Attributes
    ----------
    mean_val : float
        Mean of flattened input used for normalization.
    std_val : float
        Standard deviation of flattened input used for normalization.
    Y_norm : ndarray
        Normalized input data (same shape as `Y`).

    Examples
    --------
    Minimal usage showing PCA and VAE:
    >>> import numpy as np
    >>> from paleo_emu.encoder import EncoderGenerator
    >>> Y = np.random.randn(100, 50)
    >>> pca_config = type('C', (), {'pca': {'n_components': 10}})()
    >>> enc = EncoderGenerator(Y, pca_config)
    >>> Y_pca, pca_model, mean, std = enc.generate_encoder()
    >>> vae_config = type('C', (), {'vae': {'latent_dim': 64, 'epochs': 10}})()
    >>> enc = EncoderGenerator(Y, vae_config)
    >>> Y_pca, pca_model, mean, std = enc.generate_encoder()

    
    """

    def __init__(self, Y, model_config):
            """Create an EncoderGenerator and compute normalization stats."""
            self.model_config = model_config
            # Keep a flat copy for global statistics and compute normalization
            self.Y_flat = np.asarray(Y).ravel()
            self.mean_val = np.mean(self.Y_flat)
            self.std_val = np.std(self.Y_flat)
            self.Y_norm = (np.asarray(Y) - self.mean_val) / self.std_val + 1e-99  # avoid exact zeros
            print(f"[INFO] Raw Y_flat min={np.min(self.Y_flat)}, max={np.max(self.Y_flat)}, mean={np.mean(self.Y_flat)}, std={np.std(self.Y_flat)}")
            print(f"[INFO] Y_flat standardized to mean ~0, std ~1")

    def _generate_pca_encoder(self):
        n_components = self.model_config.pca.get("n_components", 20)
        model = PCA(n_components=n_components)
        Y_encoded = model.fit_transform(self.Y_norm)
        print(f"[INFO] PCA n_components_: {model.n_components_}")
        print(f"[INFO] Sum explained variance: {np.sum(model.explained_variance_ratio_)}")
        return Y_encoded, model, self.mean_val, self.std_val

    def _generate_vae_encoder(self):

        latent_dim = self.model_config.vae.get("latent_dim", 256)
        epochs = self.model_config.vae.get("epochs", 150)
        learning_rate = self.model_config.vae.get("learning_rate", 1e-4)
        batch_size = self.model_config.vae.get("batch_size", 64)
        kl_weight = self.model_config.vae.get("kl_weight", 1.0)  # save it for β-VAE if needed

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        input_dim = self.Y_norm.shape[1]
        vae_model = VAE(input_dim, latent_dim)

        dataset = tf.data.Dataset.from_tensor_slices((self.Y_norm.astype('float32')))
        dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

        epoch_losses = []

        for epoch in range(epochs):
            total_loss = 0
            for step, x_batch in enumerate(dataset):
                with tf.GradientTape() as tape:
                    x_decoded, mean, logvar = vae_model(x_batch)
                    loss = compute_vae_loss(x_batch, x_decoded, mean, logvar) * kl_weight  # save it for β-VAE if needed

                grads = tape.gradient(loss, vae_model.trainable_variables)
                optimizer.apply_gradients(zip(grads, vae_model.trainable_variables))
                total_loss += loss

            avg_loss = total_loss / (step + 1)
            epoch_losses.append(avg_loss.numpy())

            if epoch % 10 == 0 or epoch == epochs-1:
                print(f"[VAE] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

        save_training_log(
            epoch_losses=epoch_losses,
            latent_dim=latent_dim,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            kl_weight=kl_weight
        )

        mean_logvar = vae_model.encoder(self.Y_norm)
        mean, logvar = tf.split(mean_logvar, num_or_size_splits=2, axis=1)
        Y_encoded = mean.numpy()
        model = vae_model

        return Y_encoded, model, self.mean_val, self.std_val

    def generate_encoder(self):
        if self.model_config.encoder_type == "pca":
            return self._generate_pca_encoder()
        elif self.model_config.encoder_type == "vae":
            return self._generate_vae_encoder()
        else:
            raise ValueError(f"Unknown encoder type: {self.model_config.encoder_type}")     