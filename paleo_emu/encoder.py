"""
This module is used to encode (i.e., reduce dimensionality) data using PCA or VAE.
"""
import numpy as np

import tensorflow as tf

from sklearn.decomposition import PCA

from paleo_emu.vae import VAE, compute_vae_loss
from paleo_emu.export import save_training_log
import xarray as xr

# old code, keep it for now
def encode(Y_flat, encoder="PCA", pca_variance_ratio=0.999, vae_config=None,fixed_hp=False):
    """
    return:
        Y_pca: encoded data, shape (n_samples, n_components)
        pca_model: trained PCA model or VAE model
        mean_val: mean value of the original data, shape (n_features,)
        std_val: std value of the original data, shape (n_features,)
        rvm_std: std value of the RVM prediction, shape (n_features,)
    """
    mean_val = np.mean(Y_flat)
    std_val = np.std(Y_flat)
    Y_flat_std = (Y_flat - mean_val) / std_val

    residual_variance = None
    
    if encoder == "PCA":
        if fixed_hp:
            print("[INFO] Using fixed prescribed nkeep. ")
            n_samples = Y_flat_std.shape[0]
            pca_model = PCA(n_components=20)
        elif fixed_hp == False:
            print("[INFO] defined variance ratio for PCA instead of nkeep.")
            pca_model = PCA(n_components=pca_variance_ratio)
        else:
            print("[INFO] using user defined hyperparameter")
            #---
            #wait to be added
            #read in the file contains hp and nkeep
            pca_model = PCA(n_components=fixed_hp)

        Y_pca = pca_model.fit_transform(Y_flat_std)
        print(f"[INFO] PCA n_components_: {pca_model.n_components_}")
        
        # calculate the rvm for each PC           
        feat_var = np.var(Y_flat_std, axis=0, ddof=1)  # shape (D,)
        # 已解释方差贡献：Σ_j λ_j * w_j^2
        comps = pca_model.components_                 # (k, D)
        eigvals = pca_model.explained_variance_       # (k,)
        explained_per_feat = (eigvals[:, None] * (comps**2)).sum(axis=0)
        residual = feat_var - explained_per_feat
        residual_variance = np.clip(residual, 0.0, None)        # (D,)
        unexplained_ratio = (np.sum(residual_variance) /
                             (np.sum(explained_per_feat) + np.sum(residual_variance) + 1e-12))
        print(f"[INFO] Residual variance (std space) sum (rvm) ={np.sum(residual_variance):.4e}, "
              f"unexplained ratio ≈ {unexplained_ratio:.6f}")

    elif encoder == "VAE":
        # read in VAE config
        if vae_config is None:
            vae_config = {"latent_dim": 256, "epochs": 150, "learning_rate": 1e-4, "batch_size": 64}

        latent_dim = vae_config.get("latent_dim", 256)
        epochs = vae_config.get("epochs", 150)
        learning_rate = vae_config.get("learning_rate", 1e-4)
        batch_size = vae_config.get("batch_size", 64)
        kl_weight = vae_config.get("kl_weight", 1.0)  # save it for β-VAE if needed

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        input_dim = Y_flat_std.shape[1]
        vae_model = VAE(input_dim, latent_dim)

        dataset = tf.data.Dataset.from_tensor_slices((Y_flat_std.astype('float32')))
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

        mean_logvar = vae_model.encoder(Y_flat_std)
        mean, logvar = tf.split(mean_logvar, num_or_size_splits=2, axis=1)
        eps = tf.random.normal(shape=tf.shape(mean))
        latent = mean + eps * tf.exp(0.5 * logvar)
        Y_pca = mean.numpy()
        pca_model = vae_model
        residual_variance = None  # VAE does not provide a straightforward way to get residual variance

    else:
        raise ValueError("[ERROR] encoder must be either 'PCA' or 'VAE'.")

    return Y_pca, pca_model, mean_val, std_val, residual_variance
