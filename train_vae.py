import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from vae import VAE
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def open_file():
    # fç apenas para ler o arquivo .h5
    import h5py

    file_path = 'train_images.h5'
    index = 0

    file = h5py.File(file_path, 'r')
    key_names = list(file.keys())
    dset = file[key_names[index]]

    return dset

def encoder():
    latent_dim = 2

    encoder_inputs = keras.Input(shape=(128, 128, 1))
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()

    return encoder

def decoder():
    latent_dim = 2
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(16 * 16 * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((16, 16, 64))(x)
    # x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()

    return decoder

if __name__ == '__main__':
    data_set = open_file()
    msk = np.random.rand(len(data_set)) < 0.7
    msk2 = np.where(msk)  # por algum motivo, não estava funcionando com umas mascara booleana...
    x_train = data_set[msk2]
    msk2 = np.where(~msk)
    x_test = data_set[msk2]
    enc = encoder()
    dec = decoder()

    river = np.concatenate([x_train, x_test], axis=0).astype("float32")
    # river = np.expand_dims(river, -1).astype("float32") / 255

    vae = VAE(encoder=enc, decoder=dec)
    vae.compile(optimizer=keras.optimizers.Adam())

    vae.fit(river, epochs=2, batch_size=250)
    # vae.decoder.save('saved_decoder')
    # vae.encoder.save('saved_encoder')

    vae.get_layer('encoder').save_weights('encoder_weights.h5')  # consultei uma thread no reddit sobre como salvar a demo do keras que me enviou por email
    vae.get_layer('decoder').save_weights('decoder_weights.h5')
    vae.get_layer('encoder').save('encoder_arch')
    vae.get_layer('decoder').save('decoder_arch')

    # for i, x in enumerate(x_train):
    #     _, _, z = vae.encoder.predict(np.newaxis(x))
    #     decoded_data = vae.decoder.predict(z)
    #     plt.imshow(x)
    #     plt.imshow(decoded_data)
    # vae.save('VAE.h5')  # acho que não é necessário (retorna um warning)
