# import sys
# sys.path.append('..\..')

import numpy as np
import matplotlib.pyplot as plt

from scripts.utils.utils import import_tensorflow
from scripts.neuralnetwork.autoencoder import Autoencoder
from scripts.neuralnetwork.rnn import RNN

tf = import_tensorflow()
tfk = tf.keras
tfkl = tfk.layers


class LED:
    def __init__(self, autoencoder_name, rnn_name, length_prediction, smooth=True):
        # Import Autoencoder
        self.autoencoder = Autoencoder(latent_dim=None, model_name=autoencoder_name)
        self.latent_dim = self.autoencoder.encoder.output_shape[-1]

        # Import Recurrent Neural Network
        self.rnn = RNN(model_name=rnn_name)
        self.window_size = self.rnn.rnn.input_shape[-2]

        self.length_prediction = length_prediction
        self.smooth = smooth

    # Load data
    def get_data(self, data_path, compressed_name="arr_0"):
        if data_path[-4:] == ".npy":
            self.data = np.load(data_path)[0]
        elif data_path[-4:] == ".npz":
            self.data = np.load(data_path)[compressed_name][0]
        elif data_path[-4:] == ".csv":
            self.data = np.loadtxt(data_path, delimiter=",")[0]
        else:
            raise ValueError("File type not supported")

    # Run the LED for "length_prediction" steps
    def run(self, identity=False):
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

    # Compute error estimations
    def compute_error(self):

        decoded_data = self.decoded_future[self.window_size :]

        # Compute difference array
        diff_data = (
            decoded_data - self.data[self.window_size : self.window_size + self.length_prediction]
        )
        
        if len(np.shape(self.decoded_future)) == 4:
            
            # Inizialize error structures
            err_particle = np.zeros(np.shape(decoded_data)[1:3])    # Particle error
            err_snapshot = np.zeros(np.shape(decoded_data)[0])      # Snapshot error
            err_model = 0                                           # Model error
            
            # Loop over particles
            for x in range(np.shape(decoded_data)[1]):
                for y in range(np.shape(decoded_data)[2]):
                    # Compute error as the sum of the errors for each component
                    err_particle[x,y] = np.sqrt(np.linalg.norm(diff_data[:,x,y,0],ord=2)**2 + \
                            np.linalg.norm(diff_data[:,x,y,1],ord=2)**2)
              
            # Loop over snapshots
            for t in range(np.shape(decoded_data)[0]):
                err_snapshot = np.sqrt(np.linalg.norm(diff_data[t,:,:,0],ord='fro')**2 + \
                        np.linalg.norm(diff_data[t,:,:,1],ord='fro')**2)
                    
            # Compute model error
            err_model = np.sqrt(np.linalg.norm(err_particle,ord='fro'))

                                
        elif len(np.shape(self.decoded_future)) == 2:
            
            # Inizialize error structures
            err_particle = 0                                        # Particle error
            err_snapshot = np.zeros(np.shape(decoded_data)[0])      # Snapshot error
            err_model = 0                                           # Model error
            
            # Compute particle error
            err_particle = np.sqrt(np.linalg.norm(diff_data[:,0],ord=2)**2 + \
            np.linalg.norm(diff_data[:,1],ord=2)**2)
                
            # Loop over snapshots
            for t in range(np.shape(decoded_data)[0]):
                err_snapshot[t] = np.linalg.norm(diff_data[t,:],ord=2)
            
            # Compute model error
            err_model = err_snapshot

        return err_particle, err_snapshot, err_model
    

    # Extract one or more snapshot of the solution at given times
    def get_snapshot(self, times, plot=False):
        if np.isscalar(times):
            times = [times]

        snapshots = []

        # Loop over desired times
        for time in times:
            snapshots.append(np.asarray(self.decoded_future[time]))

        snapshots = np.array(snapshots)
        print(times[0])

        if plot:
            min_0 = np.min(snapshots[:, :, :, 0])
            max_0 = np.max(snapshots[:, :, :, 0])
            min_1 = np.min(snapshots[:, :, :, 1])
            max_1 = np.max(snapshots[:, :, :, 1])

            for i in range(len(times)):
                plt.subplot(211)
                plt.title(f"Grid at time {times[i]}")
                plt.imshow(snapshots[i, :, :, 0], vmin=min_0, vmax=max_0)
                plt.colorbar()
                plt.subplot(212)
                plt.imshow(snapshots[i, :, :, 1], vmin=min_1, vmax=max_1)
                plt.colorbar()

                plt.show()

        return snapshots

    # Extract the solution profile at any given point
    def get_particle(self, x, y, plot=False):
        times = np.arange(0, self.window_size + self.length_prediction)
        particle = self.get_snapshot(times)[:, x, y, :]

        # If desired, plot the profile
        if plot:
            num_variables = np.shape(particle)[-1]

            fig, axs = plt.subplots(num_variables, 1, figsize=(8, 6))

            for i in range(num_variables):
                axs[i].plot(times, particle[:, i])
                axs[i].set_title(f"Component {i}")

            plt.tight_layout()
            plt.show()

        return particle
