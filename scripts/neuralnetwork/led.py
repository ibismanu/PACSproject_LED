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
        # Import Autoencoder
        self.autoencoder = Autoencoder(latent_dim=None, model_name=autoencoder_name)
        self.latent_dim = self.autoencoder.encoder.output_shape[-1]

        # Import Recurrent Neural Network
        self.rnn = RNN(model_name=rnn_name)
        self.window_size = self.rnn.rnn.input_shape[-2]

        self.length_prediction = length_prediction
        self.smooth = smooth

    # Load data
    def get_data(self, data_path,compressed_name='arr_0'):

        if data_path[-4:] == ".npy":
            self.data = np.load(data_path)[0]
        elif data_path[-4:] == ".npz":
            self.data = np.load(data_path)[compressed_name][0]
        elif data_path[-4:] == ".csv":
            self.data = np.loadtxt(data_path, delimiter=",")[0]
        else:
            raise ValueError("File type not supported")
    
    # Run the LED for "length_prediction" steps
    def run(self,identity=False):
        
        # Encode the sequence generated on the microscopic scale
        if identity:
            self.encoded_data = self.data
        else: 
            self.encoded_data = self.autoencoder.encode(self.data, smooth=self.smooth)

        # Advance in time via the RNN
        future = self.rnn.predict_future(
            self.encoded_data[: self.window_size], self.length_prediction
        )
        self.forecast = np.concatenate(
            (self.encoded_data[: self.window_size], future), axis=0
        )
        
        # Decode the prediction
        if identity: 
            self.decoded_future = self.forecast
        else: 
            self.decoded_future = self.autoencoder.decode(self.forecast)

    # Compute an error estimation (L^order norm in time)
    def compute_error(self, order=2):

        decoded_data = self.decoded_future[self.window_size:]
        
        # Compute difference array
        diff_data = decoded_data - self.data[self.window_size:self.window_size+self.length_prediction]

        # Loop over space dimensions
        if len(np.shape(self.decoded_future))==4:
            err = np.zeros(np.shape(decoded_data)[1:3])
            for x in range(np.shape(decoded_data)[1]):
                for y in range(np.shape(decoded_data)[2]):
                    # Compute error as the sum of the errors for each component
                    err[x,y] = np.linalg.norm(diff_data[:,x,y,0],ord=order) + \
                            np.linalg.norm(diff_data[:,x,y,1],ord=order)
                    
        elif len(np.shape(self.decoded_future))==2:
            err = np.linalg.norm(diff_data[:,0],ord=order) + \
                    np.linalg.norm(diff_data[:,1],ord=order)
        

        return err

    # Extract one or more snapshot of the solution at given times
    def get_snapshot(self,times):

        #TODO: fare un check cosa succede in base a che oggetto è times (array vs intero)

        snapshots = []

        # Loop over desired times
        for time in times:
            snapshots.append(np.asarray(self.decoded_future[time]))
            
        return np.asarray(snapshots)
    
    # Extract the solution profile at any given point
    def get_particle(self,x,y,plot=False):
        
        times=np.arange(0,self.window_size+self.length_prediction)
        particle = self.get_snapshot(times)[:,x,y,:]

        # If desired, plot the profile
        if plot:
            num_variables = np.shape(particle)[-1]

            fig,axs = plt.subplots(num_variables,1,figsize=(8, 6))

            for i in range(num_variables):
                axs[i].plot(times, particle[:,i])
                axs[i].set_title(f"Component {i}")

            plt.tight_layout()
            plt.show()

        return particle
