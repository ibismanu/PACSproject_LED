# '''eq Pagani'''

# equazioni, ecc

# Datagen();

# LED();


# '''LED class'''
import tensorflow as tf
import numpy as np
tfk = tf.keras
tfkl = tfk.layers
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
from utilities.utils import build_sequences
from utilities.params import NNParams

from tqdm.auto import tqdm

# TODO IMPORTANTE: quando si salvano i dati, fare un trasposto per salvare (time, space)


class LED:

    latent_dim: int
    encoder: tfk.Model
    decoder: tfk.Model
    autoencoder: tfk.Model
    RNN: tfk.Model

    def __init__(self, latent_dim, seed=None):
        self.seed = seed
        self.latent_dim = latent_dim

        pass

    # TODO: prendere dense e conv da file 
    def build_autoencoder(self, input_shape, conv, dense, activation='relu', dropout_rate=0.2, verbose=True, loss=tfk.losses.MeanSquaredError(), optimizer=tfk.optimizers.Adam(), metrics=['mae']):

        # nel README: dense contiene i neuroni, conv contiene le coppie (filters, kernel size)

        # input_shape = (dim_grid,dim_u)

        # conv = [(16,3),(32,5)]
        # dense = [32, 64, 128]

        x = input_shape[0]

        E = []
        D = []

        E.append(tfkl.InputLayer(input_shape=(input_shape)))
        D.insert(0, tfkl.Conv2DTranspose(
            input_shape[-1], kernel_size=(1, 1), strides=(1, 1), padding='same', activation=activation))

        for layer in conv:
            
            print(x)
            
            E.append(tfkl.Conv2D(filters=layer[0], kernel_size=(layer[1], layer[1]), padding='same',
                     activation=activation, kernel_initializer=tfk.initializers.GlorotUniform(self.seed)))
            E.append(tfkl.MaxPool2D(pool_size=(2, 2)))

            if x % 2 == 1:
                D.insert(0, tfkl.Conv2DTranspose(layer[0], kernel_size=(
                    layer[1], layer[1]), strides=(2, 2), padding='valid', activation=activation))
            else:
                D.insert(0, tfkl.Conv2DTranspose(layer[0], kernel_size=(
                    layer[1], layer[1]), strides=(2, 2), padding='same', activation=activation))
            x = x//2

        E.append(tfkl.Flatten())

        x = np.int32(x)
        
        dense_size = dense[0]//(x**2)
        dense_size = dense_size*x*x
        
        
        D.insert(0, tfkl.Reshape((x, x, dense[0]//(x**2))))

        E.append(tfkl.Dense(units=dense_size, activation=activation)) 
        D.insert(0, tfkl.Dense(units=dense_size, activation=activation)) #to adjust dimensions for reshape
        
        for layer in dense:
            E.append(tfkl.Dense(units=layer, activation=activation))
            E.append(tfkl.Dropout(dropout_rate, seed=self.seed))

            D.insert(0, tfkl.Dropout(dropout_rate, seed=self.seed))
            D.insert(0, tfkl.Dense(units=layer, activation=activation))

        E.append(tfkl.Dense(units=self.latent_dim, activation='linear'))

        D.insert(0, tfkl.InputLayer(input_shape=(self.latent_dim)))

        self.encoder = tfk.Sequential(E, name='Encoder')
        self.decoder = tfk.Sequential(D, name='Decoder')

        ae_input = tfkl.Input(shape=(input_shape))
        encoded_input = self.encoder(ae_input)
        ae_output = self.decoder(encoded_input)
        self.autoencoder = tfk.Model(
            inputs=(ae_input), outputs=ae_output, name='autoencoder')
        self.autoencoder.compile(
            loss=loss, optimizer=optimizer, metrics=metrics)

        if verbose:
            self.autoencoder.summary(expand_nested=True)

    def build_RNN(self, input_shape, lstm, dense, activation='relu', dropout_rate=0.4, verbose=True, loss=tfk.losses.MeanSquaredError(), optimizer=tfk.optimizers.Adam(), metrics=['mae']):

        # nel README: dense contiene i neuroni, lstm contiene le coppie (unità, bidirectional True/False)
        # nel README: droput_rate realativo a lstm layers, per i dense è dimezzato

        # input_shape -> LSTM -> dense -> input_shape
        # (units, bidirectional)

        # input_shape: (time,latent_dim)

        input_layer = tfkl.Input(shape=input_shape, name='Input')

        layer = lstm[0]
        if layer[1]:
            bilstm = tfkl.Bidirectional(
                tfkl.LSTM(layer[0], return_sequences=True))(input_layer)
        else:
            bilstm = tfkl.LSTM(layer[0], return_sequences=True)(input_layer)

        dropout = tfkl.Dropout(dropout_rate, seed=self.seed)(bilstm)

        for layer in lstm[1:-1]:
            if layer[1]:
                bilstm = tfkl.Bidirectional(
                    tfkl.LSTM(layer[0], return_sequences=True))(dropout)
            else:
                bilstm = tfkl.LSTM(layer[0], return_sequences=True)(dropout)
            dropout = tfkl.Dropout(dropout_rate, seed=self.seed)(bilstm)

        layer = lstm[-1]
        if layer[1]:
            bilstm = tfkl.Bidirectional(tfkl.LSTM(layer[0]))(dropout)
        else:
            bilstm = tfkl.LSTM(layer[0])(dropout)

        dropout = tfkl.Dropout(dropout_rate, seed=self.seed)(bilstm)

        layer = dense[0]
        dense_layer = tfkl.Dense(units=layer, activation=activation)(dropout)
        dropout = tfkl.Dropout(dropout_rate/2, seed=self.seed)(dense_layer)

        for layer in dense[1:]:
            dense_layer = tfkl.Dense(
                units=layer, activation=activation)(dropout)
            dropout = tfkl.Dropout(dropout_rate/2, seed=self.seed)(dense_layer)

        output_layer = tfkl.Dense(
            units=self.latent_dim, activation=activation)(dropout)

        decoder = self.autoencoder.get_layer('Decoder') 
        decoder.trainable=False
        
        output_layer = decoder(output_layer)
        
        self.RNN = tfk.Model(inputs=input_layer,
                             outputs=output_layer, name='model')
        self.RNN.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        if verbose:
            self.RNN.summary(expand_nested=True)

    def train_autoencoder(self, data_dir, parameters=NNParams(), saving_dir=None, compressed_name = 'arr_0'):

        # saving_dir don't need to specify the extension

        if data_dir[-3:] == "npy":
            X = np.load(data_dir)
        elif data_dir[-3:] == "npz":
            X = np.load(data_dir)[compressed_name]
        elif data_dir[-3:] == "csv":
            X = np.loadtxt(data_dir, delimiter=",")
        else:
            raise ValueError("File type not supported")


        print("---------------PREPARING TEST DATA--------------------")
        X_train = []
        for i in tqdm(range(np.shape(X)[0])):
            #for j in range(np.shape(X)[-1]):
                #X_train.append(X[i, :, :, :, j])
            for j in range(np.shape(X)[1]):
                X_train.append(X[i, j, :, :, :])
        X_train = np.array(X_train)
        print("---------------TEST DATA READY--------------------")
        
        print("---------------STARTING TRAINING--------------------")
        
        history = self.autoencoder.fit(
            X_train,
            X_train,
            batch_size=parameters.batch_size,
            epochs=parameters.epochs,
            validation_split=parameters.validation_split,   
            callbacks=parameters.callbacks
        ).history

        if saving_dir is not None:
            auto_json = self.autoencoder.to_json()
            with open(saving_dir+'.json', 'w') as json_file:
                json_file.write(auto_json)
            self.autoencoder.save_weights(saving_dir+'.h5')

    def load_autoencoder(self, ae_dir):

        # ae_dir don't need to specify the extension

        with open(ae_dir+'.json', 'r') as json_file:
            auto_json = json_file.read()
        self.autoencoder = tfk.models.model_from_json(auto_json)
        self.autoencoder.load_weights(ae_dir+'.h5')

    def encode(self, data_dir, autoencoder_dir=None, compressed_name='arr_0'):

        if data_dir[-3:] == "npy":
            raw_data = np.load(data_dir)
        elif data_dir[-3:] == "npz":
            raw_data = np.load(data_dir)[compressed_name]
        elif data_dir[-3:] == "csv":
            raw_data = np.loadtxt(data_dir, delimiter=",")
        else:
            raise ValueError("File type not supported")

        if autoencoder_dir is not None:
            self.load_autoencoder(autoencoder_dir)

        encoder = self.autoencoder.get_layer('Encoder')

        encoded_data = []
        

        for raw_sequence in raw_data:
            for raw_sample in raw_sequence:
                #print(np.shape(raw_sample))
                #print(np.shape(raw_sequence))
                #print(np.shape(raw_data))
                #encoded_sample = []
                #for i in range(np.shape(raw_sample)[-1]):
                #    x = np.array([raw_sample[:, :, i]])
                #    print(np.shape(x))
                #    encoded_sample.append(encoder.predict(x))
                #    encoded_data.append(encoded_sample)
                
                #TODO: togliere il verbose
                encoded_data.append(encoder.predict(np.expand_dims(raw_sample,0)))

        return np.array(encoded_data)
 

    def train_RNN(self, data_dir, autoencoder_name=None, parameters=NNParams(), saving_name=None):

        X = self.encode(data_name=data_dir, autoencoder_name=autoencoder_name)[0][:-2] #take only first sample
        #Y = X[-1]
        
        #dim_x: (latent_dim, timesteps, nsample)
        
        # decoder = self.autoencoder.get_layer('Decoder')
        
        X_train,Y_train = build_sequences(X[:,:,0])
        
        for i in range(1,np.shape(X)[-1]):
            X_temp,Y_temp = build_sequences(X[:,:,i])
            
            X_train = np.concatenate((X_train,X_temp),0)
            Y_train = np.concatenate((Y_train,Y_temp),0)
        
        # for i in range(Y_train.shape[0]):
        #     for j in range(Y_train.shape[1]):
        #         Y_train[i,j] = decoder(Y_train[i,j])
        
        history = self.RNN.fit(
            X_train,
            Y_train,
            batch_size=parameters.batch_size,
            epochs=parameters.epochs,
            validation_split=parameters.validation_split,
            callbacks=parameters.callbacks
        ).history

        if saving_name is not None:
            saving_file = self.saving_dir+saving_name
            auto_json = self.RNN.to_json()
            with open(saving_file+'.json', 'w') as json_file:
                json_file.write(auto_json)
            self.RNN.save_weights(saving_file+'.h5')
            
        # 0 1 2 3 4 5 6 - -
    
        # 10 11 12 13 14 15 16
        
        
        # 0 1 2 | 3
        # 1 2 3 | 4
        # 2 3 4 | 5
        # 3 4 5 | 6
        # 10 11 12 | 13
        # 11 12 13 | 14
        # 12 13 14 | 15
        # 13 14 15 | 16
        
    def test_autoencoder(self,autoencoder_dir, data_dir_test, compressed_name_test='arr_0',plot=False):

        # compressed_name_test requires the name you gave to the np.array when you saved it as a compressed file. By default is 'arr_0' ad the default name from the function np.savez_compressed 
        # autoencoder_dir don't need to specify the extension of the file

        if data_dir_test[-3:] == "npy":
            X_test = np.load(data_dir_test)
        elif data_dir_test[-3:] == "npz":
            X_test = np.load(data_dir_test)[compressed_name_test]
        elif data_dir_test[-3:] == "csv":
            X_test = np.loadtxt(data_dir_test, delimiter=",")
        else:
            raise ValueError("File type not supported")

        self.load_autoencoder(autoencoder_dir)

        grid_size = np.shape(X_test)[2]
        times = np.shape(X_test)[1]
        n_samples = np.shape(X_test)[0]
        dim_u = np.shape(X_test)[-1]

        X_ae = []
        prediction_loss = []

        # for sample in range(np.shape(X_test)[0]):
        # qui per ora sto facendo il predict solo per il sample 0
        for t in range(np.shape(X_test)[1]):
            prediction = self.autoencoder.predict(np.expand_dims(X_test[0,t],0),verbose=0) 
            X_ae.append(prediction)
            prediction_loss.append(tfk.losses.mae(prediction,X_test[0,t]))

        # error = X_test[0] - X_ae

        # TODO: se vogliamo il plot questo if va messo a posto
        if plot:
            # plot some signals in the diagonal of the grid
            X_ae = np.reshape(X_ae,(n_samples,times,grid_size,grid_size,dim_u))

            for i in range(5):
                fig,(ax1,ax2) = plt.subplots(2,1,figsize=(8, 6))

                ax1.plot(np.arange(0,np.shape(X_test)[1],1), X_ae[0, :, i, i, 0])
                ax1.set_title('Predicted')

                ax2.plot(np.arange(0,np.shape(X_test)[1],1), X_test[0, :, i, i, 0])
                ax2.set_title('Test')

                plt.tight_layout()
                plt.show()

        return X_ae, prediction_loss
