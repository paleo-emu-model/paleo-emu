import tensorflow as tf
from tensorflow.keras import layers, models
import keras

@keras.saving.register_keras_serializable()
class VAE(tf.keras.Model):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        # encoder
        self.encoder = models.Sequential([
            layers.InputLayer(input_shape=(input_dim,)),    # input_dim=7008
            layers.Dense(4096, activation="relu"),           # 7008 → 4096
            layers.Dense(2048, activation="relu"),           # 4096 → 2048
            layers.Dense(4096, activation="relu"),           # 2048 → 4096
            layers.Dense(latent_dim * 2)                           # gives mean and logvar
        ])
        # decoder
        self.decoder = models.Sequential([
            layers.InputLayer(input_shape=(latent_dim,)),
            layers.Dense(4096, activation="relu"),
            layers.Dense(2048, activation="relu"),
            layers.Dense(2048, activation="relu"),   # 👈 add another layer here
            layers.Dense(1024, activation="relu"),
            layers.Dense(7008)
        ])


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

def compute_vae_loss(x, x_decoded, mean, logvar):
    reconstruction_loss = tf.reduce_mean(tf.square(x - x_decoded))
    kl_loss = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mean) - tf.exp(logvar))
    return reconstruction_loss + kl_loss
