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
def encode(Y_flat, encoder="PCA", model_type="GPR", pca_variance_ratio=0.999, vae_config=None,fixed_hp=False):
    print(f"[INFO] Raw Y_flat min={np.min(Y_flat)}, max={np.max(Y_flat)}, mean={np.mean(Y_flat)}, std={np.std(Y_flat)}")
    
    mean_val = np.mean(Y_flat)
    std_val = np.std(Y_flat)
    Y_flat = (Y_flat - mean_val) / std_val
    print(f"[INFO] Y_flat standardized to mean ~0, std ~1")

    if encoder == "PCA":
        if fixed_hp:
            print("[INFO] Using fixed prescribed hyper parameters. ")
            n_samples = Y_flat.shape[0]
            pca_model = PCA(n_components=20)
        elif fixed_hp == "False":
            print("[INFO] Using PCA for feature extraction.")
            pca_model = PCA(n_components=pca_variance_ratio)
        else:
            print("[INFO] using user defined hyperparameter")
            #---
            #wait to be added
            #read in the file contains hp and nkeep
            pca_model = PCA(n_components=nkeep)
        
        Y_pca = pca_model.fit_transform(Y_flat)
        print(f"[INFO] PCA n_components_: {pca_model.n_components_}")
        print(f"[INFO] Sum explained variance: {np.sum(pca_model.explained_variance_ratio_)}")
        # # --------temporary test---- to be deleted
        # # Save PCA results as NetCDF
        # output_path = "examples/outputs/test.PCA.nc"
        # scaled_amps = pca_model.components_.T * pca_model.singular_values_
        # ds = xr.Dataset(
        #     {
        #         "Y_pca": (["samples", "components"], Y_pca),
        #         "pca_mean": (["features"], pca_model.mean_),
        #         "pca_scaled_amps": (["features", "components"], scaled_amps),
        #     },
        #     coords={
        #         "samples": np.arange(Y_pca.shape[0]),
        #         "components": np.arange(Y_pca.shape[1]),
        #         "features": np.arange(pca_model.mean_.shape[0]),
        #     },
        # )
        # ds.to_netcdf(output_path)
        # print(f"[INFO] PCA results saved to {output_path}")
        # # --------temporary test---- to be deleted

    elif encoder == "VAE":
        print("[INFO] Using VAE for feature extraction.")

        # read in VAE config
        if vae_config is None:
            vae_config = {"latent_dim": 256, "epochs": 150, "learning_rate": 1e-4, "batch_size": 64}

        latent_dim = vae_config.get("latent_dim", 256)
        epochs = vae_config.get("epochs", 150)
        learning_rate = vae_config.get("learning_rate", 1e-4)
        batch_size = vae_config.get("batch_size", 64)
        kl_weight = vae_config.get("kl_weight", 1.0)  # save it for β-VAE if needed

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        input_dim = Y_flat.shape[1]
        vae_model = VAE(input_dim, latent_dim)

        dataset = tf.data.Dataset.from_tensor_slices((Y_flat.astype('float32')))
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

        mean_logvar = vae_model.encoder(Y_flat)
        mean, logvar = tf.split(mean_logvar, num_or_size_splits=2, axis=1)
        eps = tf.random.normal(shape=tf.shape(mean))
        latent = mean + eps * tf.exp(0.5 * logvar)
        # if model_type == "GPR":
        #     Y_pca = mean.numpy()
        # elif model_type == "LGBM":
        #     Y_pca = latent.numpy()
        Y_pca = mean.numpy()

        pca_model = vae_model

    else:
        raise ValueError("[ERROR] encoder must be either 'PCA' or 'VAE'.")

    return Y_pca, pca_model, mean_val, std_val
