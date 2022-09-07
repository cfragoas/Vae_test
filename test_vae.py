import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from vae import VAE
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from train_vae import Sampling

def open_file():
    # fç apenas para ler o arquivo .h5
    import h5py

    file_path = 'train_images.h5'
    index = 0

    file = h5py.File(file_path, 'r')
    key_names = list(file.keys())
    dset = file[key_names[index]]

    return dset

if __name__ == '__main__':
    encoder = keras.models.load_model('encoder_arch', compile=False)  # carregando o modelo
    decoder = keras.models.load_model('decoder_arch', compile=False)
    vae = VAE(encoder, decoder)
    vae.get_layer('encoder').load_weights('encoder_weights.h5')
    vae.get_layer('decoder').load_weights('decoder_weights.h5')
    vae.compile(optimizer=keras.optimizers.Adam())


    # encoder = keras.models.load_model("saved_encoder")
    # decoder = keras.models.load_model("saved_decoder")
    data_set = open_file()

    z_mean, _, _ = vae.encoder.predict(data_set[:10000])
    plt.scatter(z_mean[:, 0], z_mean[:, 1], s=2)
    plt.colorbar()
    plt.show()

    for x in data_set:
        _, _, z = vae.encoder.predict(x[np.newaxis, :].astype('float32'))
        decoded_data = vae.decoder.predict(z)
        # decoded_data = decoded_data.astype(np.uint8)
        plt.imshow(x)
        plt.show()
        plt.imshow(decoded_data[0], vmax=decoded_data[0].max())
        plt.show()