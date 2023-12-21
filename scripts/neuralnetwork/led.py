import tensorflow as tf
import numpy as np
tfk = tf.keras
tfkl = tfk.layers
import matplotlib.pyplot as plt
from autoencoder import Autoencoder
from rnn import RNN

import sys
sys.path.append('..')

from tqdm.auto import tqdm

class LED:

    def __init__(self,data_dir,rnn_dir,ae_dir,T_micro,T_macro):
        self.data = np.load(data_dir)
        self.rnn = RNN(name=rnn_dir)
        self.autoencoder = Autoencoder(input_shape=np.shape(self.data))
        self.T_micro = T_micro
        self.T_macro = T_macro
        
    
    def predict(self,starting_sequence):

        encoded_dataset = self.autoencoder.encode(self.data[:self.T_micro])

        future = self.rnn.predict_future(starting_sequence = encoded_dataset)
        self.timeline = np.concatenate((starting_sequence,future),axis=1)

        output = self.autoencoder.decode(self.timeline[-1])

    def compute_error(self):
        #err = ||data-timeline||
        pass