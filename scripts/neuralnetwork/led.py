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
        self.decoder = self.autoencoder.get_layer('Decoder')
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

        decoded_data = self.get_snapshot(times=np.arange(self.T_micro,self.T_macro))
        diff_data = decoded_data - self.test_data
        err = np.zeros(np.shape(decoded_data)[1:2])

        for x in range(np.shape(decoded_data)[1]):
            for y in range(np.shape(decoded_data)[2]):
                err[x,y] = np.linalg.norm(diff_data,ord=np.inf)

        return err

    def get_snapshot(self,times):

        #TODO: fare un check cosa succede in base a che oggetto è times (array vs intero)

        snapshots = []

        for time in times:
            encoded_snapshot = self.future[time-self.T_micro]
            snapshots.append(self.decoder.predict(encoded_snapshot))

        return snapshots
    
    def get_particle(self,x,y,plot=False):
        
        particle = self.get_snapshot(times=np.arange(self.T_micro,self.T_macro))[:,x,y,:]

        if plot:
            num_variables = np.shape(particle)[-1]

            fig,axs = plt.subplots(num_variables,1,figsize=(8, 6))

            for i in range(num_variables):
                axs[i].plot(np.arange(self.T_micro,self.T_macro), particle[:,i])
                axs[i].set_title('Component ',i)

            plt.tight_layout()
            plt.show()
