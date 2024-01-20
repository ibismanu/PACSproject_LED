# import sys
# sys.path.append('..\..')

import numpy as np
import matplotlib.pyplot as plt

from scripts.utils.utils import import_tensorflow
from scripts.NeuralNetwork.autoencoder import Autoencoder
from scripts.NeuralNetwork.rnn import RNN

tf = import_tensorflow()
tfk = tf.keras
tfkl = tfk.layers


class LED:
    def __init__(
        self, autoencoder_name, rnn_name, length_prediction, smooth=True
    ):
        self.autoencoder = Autoencoder(model_name=autoencoder_name)
        self.latent_dim = self.autoencoder.encoder.output_shape[-1]

        self.rnn = RNN(model_name=rnn_name)
        self.window_size = self.rnn.rnn.input_shape[-2]

        self.length_prediction = length_prediction
        self.smooth = smooth

    def get_data(self, data_path,compressed_name='arr_0'):

        if data_path[-4:] == ".npy":
            self.data = np.load(data_path)[0]
        elif data_path[-4:] == ".npz":
            self.data = np.load(data_path)[compressed_name][0]
        elif data_path[-4:] == ".csv":
            self.data = np.loadtxt(data_path, delimiter=",")[0]
        else:
            raise ValueError("File type not supported")
        
    def run(self):
        self.encoded_data = self.autoencoder.encode(self.data, smooth=self.smooth)

        future = self.rnn.predict_future(
            self.encoded_data[: self.window_size], self.length_prediction
        )
        self.forecast = np.concatenate(
            (self.encoded_data[: self.window_size], future), axis=0
        )

        self.decoded_future = self.autoencoder.decode(self.forecast)


    def compute_error(self):

        decoded_data = self.decoded_future[self.window_size:]
        diff_data = decoded_data - self.data[self.window_size:self.window_size+self.length_prediction]
        err = np.zeros(np.shape(decoded_data)[1:3])

        for x in range(np.shape(decoded_data)[1]):
            for y in range(np.shape(decoded_data)[2]):
                err[x,y] = np.linalg.norm(diff_data[:,x,y,0],ord=np.inf) + \
                           np.linalg.norm(diff_data[:,x,y,1],ord=np.inf)

        return err

    def get_snapshot2(self,times):

        #TODO: fare un check cosa succede in base a che oggetto è times (array vs intero)

        snapshots = []

        for time in times:
            encoded_snapshot = self.forecast[time]
            snapshots.append(self.autoencoder.decoder.predict(encoded_snapshot))

        return snapshots
    
    def get_particle2(self,x,y,plot=False):
        
        particle = self.get_snapshot(times=np.arange(self.T_micro,self.T_macro))[:,x,y,:]

        if plot:
            num_variables = np.shape(particle)[-1]

            fig,axs = plt.subplots(num_variables,1,figsize=(8, 6))

            for i in range(num_variables):
                axs[i].plot(np.arange(self.T_micro,self.T_macro), particle[:,i])
                axs[i].set_title('Component ',i)

            plt.tight_layout()
            plt.show()
