import tensorflow as tf
import numpy as np

tfk = tf.keras
tfkl = tfk.layers
import matplotlib.pyplot as plt
from autoencoder import Autoencoder
from rnn import RNN

import sys

sys.path.append("..")

from tqdm.auto import tqdm


class LED:
    def __init__(self, data_path, autoencoder_name, rnn_name, T_macro=100):
        self.load_data(data_path)

        self.autoencoder = Autoencoder(name=autoencoder_name)
        self.latent_dim = self.autoencoder.encoder.output_shape[-1]

        self.rnn = RNN(name=rnn_name)

        self.T_micro = self.rnn.input_shape[1]
        self.T_macro = T_macro

        self.starting_data = self.data[:self.T_micro]
        self.test_data = self.data[self.T_micro:self.T_micro+self.T_macro]


    def load_data(self, data_path):
        # TODO add other formats
        self.data = np.load(data_path)

    def run(self, compute_err=False):
        self.encoded_data = self.autoencoder.encode(self.starting_data, save=False)

        self.future = self.rnn.predict_future(length=self.T_macro, starting_sequence=self.encoded_data)

        if compute_err:
            self.compute_error()
    
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
