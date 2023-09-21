# '''eq Pagani'''

# equazioni, ecc

# Datagen();

# LED();




#'''LED class'''
import tensorflow as tf
import numpy as np
tfk = tf.keras
tfkl = tfk.layers

#TODO IMPORTANTE: quando si salvano i dati, fare un trasposto per salvare (time, space)
class LED:

    latent_dim: int
    encoder: tfk.Model
    decoder: tfk.Model
    autoencoder: tfk.Model
    RNN: tfk.Model
    
    #data_dir: np.string
    #saving_dir: np.string
    
    def __init__(self, latent_dim, data_dir="../../dataset/", saving_dir="../../models/", seed=None):
        self.seed = seed
        self.latent_dim=latent_dim
        
        
        if data_dir[-1] == "/":
            self.data_dir = data_dir
        else:
            self.data_dir = data_dir+"/"
        
        if saving_dir[-1] == "/":
            self.saving_dir = saving_dir
        else:
            self.saving_dir = saving_dir+"/"
        
        pass
    
    #TODO: prendere dense e conv da file
    def build_autoencoder(self, input_shape, conv, dense, activation='relu',dropout_rate=0.2, verbose=True, loss = tfk.losses.MeanSquaredError(),optimizer=tfk.optimizers.Adam(),metrics=['mae']):
        
        #nel README: dense contiene i neuroni, conv contiene le coppie (filters, kernel size)
        
        # input_shape = (dim_grid,dim_u)

        # conv = [(16,3),(32,5)]
        # dense = [32, 64, 128] 
         
        x = input_shape[0]

        E = []
        D = []

        E.append(tfkl.InputLayer(input_shape=(input_shape)))   
        D.insert(0,tfkl.Conv2DTranspose(input_shape[-1], kernel_size=(1,1), strides=(1,1), padding='same', activation=activation))
        
        for layer in conv:

            E.append(tfkl.Conv2D(filters=layer[0],kernel_size=(layer[1],layer[1]),padding='same',activation=activation,kernel_initializer=tfk.initializers.GlorotUniform(self.seed)))
            E.append(tfkl.MaxPool2D(pool_size=(2,2)))
            
            if x%2==1:
                D.insert(0,tfkl.Conv2DTranspose(layer[0], kernel_size=(layer[1],layer[1]), strides=(2,2), padding='valid', activation=activation))
            else:
                D.insert(0,tfkl.Conv2DTranspose(layer[0], kernel_size=(layer[1],layer[1]), strides=(2,2), padding='same', activation=activation))
            x = x//2

        
        E.append(tfkl.Flatten())

        x = np.int32(x)
        D.insert(0,tfkl.Reshape((x,x,dense[0]//(x**2))))
        
        for layer in dense:
            E.append(tfkl.Dense(units=layer,activation=activation))
            E.append(tfkl.Dropout(dropout_rate,seed=self.seed))
            
            D.insert(0, tfkl.Dropout(dropout_rate,seed=self.seed))
            D.insert(0, tfkl.Dense(units=layer,activation=activation))
            
        
        E.append(tfkl.Dense(units=self.latent_dim,activation='linear'))
    
        D.insert(0,tfkl.InputLayer(input_shape=(self.latent_dim))) 

        self.encoder = tfk.Sequential(E)
        self.decoder = tfk.Sequential(D)

        ae_input = tfkl.Input(shape=(input_shape))
        encoded_input = self.encoder(ae_input)
        ae_output = self.decoder(encoded_input)
        self.autoencoder = tfk.Model(inputs=(ae_input), outputs=ae_output, name='autoencoder')
        self.autoencoder.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        if verbose:
            self.autoencoder.summary(expand_nested=True)
    
    def build_RNN(self, input_shape, lstm, dense, activation='relu',dropout_rate=0.4, verbose=True, loss = tfk.losses.MeanSquaredError(),optimizer=tfk.optimizers.Adam(),metrics=['mae']):
        
        #nel README: dense contiene i neuroni, lstm contiene le coppie (unità, bidirectional True/False)
        #nel README: droput_rate realativo a lstm layers, per i dense è dimezzato
        
        #input_shape -> LSTM -> dense -> input_shape
        #(units, bidirectional)
        
        #input_shape: (time,latent_dim)
        
        
        input_layer = tfkl.Input(shape=input_shape, name='Input')
        
        layer = lstm[0]
        if layer[1]:
            bilstm = tfkl.Bidirectional(tfkl.LSTM(layer[0], return_sequences=True))(input_layer)
        else:
            bilstm = tfkl.LSTM(layer[0], return_sequences=True)(input_layer)
            
        dropout = tfkl.Dropout(dropout_rate, seed=self.seed)(bilstm)
        
        for layer in lstm[1:-1]:
            if layer[1]:
                bilstm = tfkl.Bidirectional(tfkl.LSTM(layer[0], return_sequences=True))(dropout)
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
        dense_layer = tfkl.Dense(units=layer,activation=activation)(dropout)
        dropout = tfkl.Dropout(dropout_rate/2, seed=self.seed)(dense_layer)
        
        for layer in dense[1:]:
            dense_layer = tfkl.Dense(units=layer,activation=activation)(dropout)
            dropout = tfkl.Dropout(dropout_rate/2, seed=self.seed)(dense_layer)
        
        
        output_layer = tfkl.Dense(units=self.latent_dim,activation=activation)(dropout)
        
        self.RNN = tfk.Model(inputs=input_layer, outputs=output_layer, name='model')
        self.RNN.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        
        if verbose:
            self.RNN.summary(expand_nested=True)
            
            
            
            
            
    def train_autoencoder(self, data_name, saving_name=None):
        
        data_file = self.data_dir+data_name
        
        if data_name[-3:] == "npy":
            X_train = np.load(data_file)
        elif data_name[-3:] == "csv":
            X_train = np.loadtxt(data_file,delimiter=",")
        else:
            raise ValueError("File type not supported")
            
        history = self.autoencoder.fit(
             X_train,  
             X_train,
             batch_size=32,
             epochs=1000,
             validation_split=.2,
             callbacks=[
                 tfk.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                 tfk.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr=1e-5),
                 ]
             ).history
        
        if saving_name() is not None:
            saving_file = self.saving_dir+saving_name
            auto_json = self.autoencoder.to_json()
            with open(saving_file+'.json','w') as json_file:
                json_file.write(auto_json)
            self.autoencoder.save_weights(saving_file+'.h5')
        
        
    def load_autoencoder(self, filename):
        with open(filename+'.json', 'r') as json_file:
            auto_json = json_file.read()
        self.autoencoder = tfk.models.model_from_json(auto_json)
        self.autoencoder.load_weights(filename+'.h5')
        
     #    extract E
    #     EncodedData = E(Data)
    
    #     
    #     net = RNN+D
        
    #     train net (EncodedData->Data, D fixed)
        
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




# latent_dim = 10
# NN = LED(latent_dim)
# E = NN.encoder

# input_shape = (31,31,2)


# conv = [(8,3),(16,3),(32,3),(16,3)]
# dense = [64,128]


# NN.build_autoencoder(input_shape, conv, dense)

# input_shape = (4,latent_dim)
# dense = [64,32]
# lstm = [(32,False),(16,True),(32,False)]


# def f(mydense):
#     for layer in mydense[1:]:
#         print("hi")
# f(dense)

# NN.build_RNN(input_shape, lstm, dense)







