import numpy as np
import matplotlib.pyplot as plt

from scripts.neuralnetwork.autoencoder import Autoencoder
from scripts.neuralnetwork.rnn import RNN
from scripts.utils.utils import import_tensorflow

tf = import_tensorflow()
tfk = tf.keras
tfkl = tfk.layers


def lpfilter(input_signal, win):
    # Low-pass linear Filter
    # (2*win)+1 is the size of the window that determines the values that influence
    # the filtered result, centred over the current measurement
    from scipy import ndimage

    kernel = np.lib.pad(np.linspace(1, 3, win), (0, win - 1), "reflect")
    kernel = np.divide(kernel, np.sum(kernel))  # normalise
    output_signal = ndimage.convolve(input_signal, kernel)
    return output_signal


class LED:
    def __init__(self, data_path, autoencoder_name, rnn_name, T_macro=100):
        self.load_data(data_path)

        self.autoencoder = Autoencoder(name=autoencoder_name)
        self.latent_dim = self.autoencoder.encoder.output_shape[-1]

        self.rnn = RNN(model_name=rnn_name)

        self.T_micro = self.rnn.rnn.input_shape[1]
        self.T_macro = T_macro

        self.starting_data = self.data[: self.T_micro]
        self.test_data = self.data[self.T_micro : self.T_micro + self.T_macro]

    def load_data(self, data_path):
        # TODO add other formats
        self.data = np.load(data_path)["test_data"][0]

    def run(self, compute_err=False):
        self.encoded_data = self.autoencoder.encode(self.data, save=False)

        self.encoded_smooth = np.zeros_like(self.encoded_data)
        for i in range(8):
            self.encoded_smooth[:, i] = lpfilter(self.encoded_data[:, i], 10)

        future = self.rnn.predict_future(self.encoded_smooth[:200], 801)
        self.forecast = np.concatenate((self.encoded_smooth[:200], future), axis=0)

        self.decoded_future = self.autoencoder.decode(self.forecast)

    def compute_error(self):
        pass


# class LED2:
#     def __init__(self, data_dir, rnn_dir, ae_dir, T_micro, T_macro):
#         self.data = np.load(data_dir)
#         self.rnn = RNN(name=rnn_dir)
#         self.autoencoder = Autoencoder(input_shape=np.shape(self.data))
#         self.T_micro = T_micro
#         self.T_macro = T_macro

#     def predict(self, starting_sequence):
#         encoded_dataset = self.autoencoder.encode(self.data[: self.T_micro])

#         future = self.rnn.predict_future(starting_sequence=encoded_dataset)
#         self.timeline = np.concatenate((starting_sequence, future), axis=1)

#         output = self.autoencoder.decode(self.timeline[-1])

#     def compute_error(self):
#         # err = ||data-timeline||
#         pass
