# '''eq Pagani'''

# equazioni, ecc

# Datagen();

# LED();


#'''LED class'''
import tensorflow as tf
import numpy as np
tfk = tf.keras
tfkl = tfk.layers

class LED:

    def __init__(self,seed=None):
        self.seed = seed
        pass
    
    #TODO: prendere dense e conv da file
    def build_autoencoder(self,latent_dim, input_shape, dense, conv, activation='relu',dropout_rate=0.2):
        
        #README: pool not wanted -> set 0
        
        # input_shape = (dim_grid,dim_u)

        # conv = [(16,3),(32,5)]
        # dense = [32, 64, 128] 
         
        x = input_shape[0] 
        loss = tfk.losses.MeanSquaredError()
        optimizer=tfk.optimizers.Adam()
        metrics=['mae']

        E = []
        D = []

        E.append(tfkl.InputLayer(input_shape=(input_shape)))   
        
        for layer in conv:
            E.append(tfkl.conv2D(filters=layer[0],kernel_size=(layer[1],layer[1])),padding='same',activation=activation,kernel_initializer=tfk.initializers.GlorotUniform(self.seed))
            E.append(tfkl.MaxPool2D(pool_size=(2,2)))
            x =  (x + x%2) /2

        E.append(tfkl.flatten())

        for layer in dense:
            E.append(tfkl.Dense(units=layer,activation=activation))
            E.append(tfkl.Dropout(dropout_rate,seed=self.seed))
        
        E.append(tfkl.Dense(units=latent_dim,activation='linear'))


        D.append(tfkl.InputLayer(input_shape=(latent_dim)))     
        # (None,latent_dim)
        # (None, dense_dim)

        for layer in dense:
            D.insert(0,tfkl.Dense(units=layer,activation=activation))
            D.insert(0,tfkl.Dropout(dropout_rate,seed=self.seed))


        D.append(tfkl.Reshape(x,x,dense[0]/(x**2)))
        for layer in np.reverse(conv):
            D.append(tfkl.Conv2DTranspose(layer[0], kernel_size=(layer[1],layer[1]), stride=(2,2), padding='same', activation=activation))

        self.encoder = tfk.Sequential(E)
        self.decoder = tfk.Sequential(D)

        ae_input = tfkl.Input(shape=(input_shape))
        encoded_input = self.encoder(ae_input)
        ae_output = self.decoder(encoded_input)
        self.autoencoder = tfk.Model(inputs=(ae_input), outputs=ae_output, names='autoencoder')
        self.autoencoder.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        self.autoencoder.summary(expand_nested=True)
    
    # def build_RNN():
    #     # RNN = keras.RNN
        
    # def train(what_to_save='all'):
    #     train E,D
        
    #     net = E+RNN+D
        
    #     train net (E,D fixed)
        
    #     #maybe fine tune all
        
    #     save(...)
        
    # def predict():
    #     pass
    
    # def save_network(net, tipo_net):
    # def load_network():
    #     for tipo in tipo_net
    #     if tipo == "E"
    #         E = np.load()
    
# class AdaLED(LED):
#     def train
#     def predict
#     def RNN




autoencoder(tfkl.Dense(units=64,))


