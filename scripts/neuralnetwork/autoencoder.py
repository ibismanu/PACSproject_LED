import numpy as np
import os
from tqdm.auto import tqdm

from scripts.utils.utils import import_tensorflow, smooth_filter

tf = import_tensorflow()
tfk = tf.keras
tfkl = tfk.layers


class Autoencoder:
    def __init__(
        self,
        seed=42,
        model_name=None,
        latent_dim=10,
        conv=[(8, 3), (16, 3), (32, 3)],
        dense=[64, 128],
        activation="elu",
        dropout_rate=0.2,
        loss=tfk.losses.MeanSquaredError(),
        optimizer=tfk.optimizers.Adam(),
        metrics=["mae"],
        batch_size=32,
        epochs=500,
        validation_split=0.2,
        callbacks=None,
    ):
        # Seed
        self.seed = seed

        # Building parameters
        self.latent_dim = latent_dim
        self.conv = conv
        self.dense = dense
        self.activation = activation
        self.dropout_rate = dropout_rate

        # Compiling parameters
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics

        # Training parameters
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.callbacks = callbacks

        # For loading an existing model instead of building one
        if model_name is not None:
            self.load_model(model_name)
            self.latent_dim = self.encoder.output_shape[-1]

    def set_seed(self):
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        tfk.utils.set_random_seed(self.seed)

    def load_model(self, model_name):
        path = "../../models/" + model_name + "/" + model_name
        
        with open(path + ".json", "r") as json_file:
            model_json = json_file.read()

        self.autoencoder = tfk.models.model_from_json(model_json)
        self.autoencoder.load_weights(path + ".h5")

        self.encoder = self.autoencoder.get_layer("Encoder")
        self.decoder = self.autoencoder.get_layer("Decoder")

    def get_data(self, file_path, compressed_name="arr_0"):
        
        if file_path[-4:] == ".npy":
            X = np.load(file_path)
        elif file_path[-4:] == ".npz":
            X = np.load(file_path)[compressed_name]
        elif file_path[-4:] == ".csv":
            X = np.loadtxt(file_path, delimiter=",")
        else:
            raise ValueError("File type not supported")
            
        # match file_path[-4:]:
        #     case ".npy":
        #         X = np.load(file_path)
        #     case ".npz":
        #         X = np.load(file_path)[compressed_name]
        #     case ".csv":
        #         X = np.loadtxt(file_path, delimiter=",")
        #     case _:
        #         raise ValueError("File type not supported")

        self.data = []
        for i in tqdm(range(np.shape(X)[0])):
            for j in range(np.shape(X)[1]):
                self.data.append(X[i, j])
        self.data = np.array(self.data)
        
        if len(np.shape(self.data))==2:
            self.data = np.reshape(self.data,(np.shape(self.data)[0],1,1,np.shape(self.data)[-1]))

        self.input_shape = (self.data.shape[1], self.data.shape[2], self.data.shape[3])
        self.output_shape = self.input_shape

    def build_model(self, summary=False):
        
        x = int(self.input_shape[0])

        # Initialize Encoder and Decoder
        E = []
        D = []

        # First layer
        E.append(tfkl.InputLayer(input_shape=(self.input_shape)))
        D.insert(
            0,
            tfkl.Conv2DTranspose(
                self.output_shape[-1],
                kernel_size=(2, 2),
                strides=(1, 1),
                padding="same",
                activation="linear",
            ),
        )
        
        if self.input_shape != self.output_shape:
            D.insert(
                0,
                tfkl.Conv2DTranspose(
                    self.output_shape[-1],
                    kernel_size=(1, 1),
                    strides=(2, 2),
                    padding="same",
                    activation="linear",
                ),
            )
            
        # Convolutional layers  
        for layer in self.conv:
            E.append(
                tfkl.Conv2D(
                    filters=layer[0],
                    kernel_size=(layer[1], layer[1]),
                    padding="same",
                    activation=self.activation,
                    kernel_initializer=tfk.initializers.GlorotUniform(self.seed),
                )
            )
            E.append(tfkl.MaxPool2D(pool_size=(2, 2)))

            if x % 2 == 1:
                padding = "valid"
            else:
                padding = "same"
            D.insert(
                0,
                tfkl.Conv2DTranspose(
                    layer[0],
                    kernel_size=(layer[1], layer[1]),
                    strides=(2, 2),
                    padding=padding,
                    activation=self.activation,
                ),
            )
            x = x // 2

        E.append(tfkl.Flatten())

        dense_size = self.dense[0] - self.dense[0] % (x**2)

        D.insert(0, tfkl.Reshape((x, x, dense_size // (x**2))))

        E.append(tfkl.Dense(units=dense_size, activation=self.activation))

        # Adjust dimensions for reshape
        D.insert(0, tfkl.Dense(units=dense_size, activation=self.activation))

        # Dense layers
        for layer in self.dense:
            E.append(tfkl.Dense(units=layer, activation=self.activation))
            E.append(tfkl.Dropout(self.dropout_rate, seed=self.seed))

            D.insert(0, tfkl.Dropout(self.dropout_rate, seed=self.seed))
            D.insert(0, tfkl.Dense(units=layer, activation=self.activation))

        # Latent layer
        E.append(tfkl.Dense(units=self.latent_dim, activation="linear"))
        D.insert(0, tfkl.InputLayer(input_shape=(self.latent_dim)))

        # Build models
        self.encoder = tfk.Sequential(E, name="Encoder")
        self.decoder = tfk.Sequential(D, name="Decoder")

        # Build the autoencoer
        input = tfkl.Input(shape=self.input_shape)
        encoded = self.encoder(input)
        output = self.decoder(encoded)
        self.autoencoder = tfk.Model(inputs=input, outputs=output, name="Autoencoder")

        # Print summary
        if summary:
            self.autoencoder.summary(expand_nested=True)

        # Compile model
        self.autoencoder.compile(
            loss=self.loss, optimizer=self.optimizer, metrics=self.metrics
        )

    def train_model(self):
        self.set_seed()

        if self.callbacks is None:
            self.callbacks = [
                tfk.callbacks.EarlyStopping(
                    monitor="val_loss", patience=10, restore_best_weights=True
                ),
                # tfk.callbacks.ReduceLROnPlateau(
                #     monitor="val_loss", patience=5, factor=0.5, min_lr=1e-5
                # ),
            ]

        history = self.autoencoder.fit(
            self.data,
            self.data,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.validation_split,
            callbacks=self.callbacks,
        ).history

    def save_model(self, name):
        file_path = "../../models/" + name + "/" + name
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        model_json = self.autoencoder.to_json()
        with open(file_path + ".json", "w") as json_file:
            json_file.write(model_json)
        self.autoencoder.save_weights(file_path + ".h5")

    def encode(self, raw_data, smooth=False, save=False):
        encoded_data = self.encoder.predict(raw_data, verbose=0)

        if smooth:
            for i in range(self.latent_dim):
                encoded_data[:, i] = smooth_filter(encoded_data[:, i], 31, 2)

        if save:
            np.save("../../dataset/encoded_data.npy", encoded_data)
        else:
            return encoded_data

    def decode(self, encoded_data, save=False):
        decoded_data = self.decoder.predict(encoded_data, verbose=0)

        if save:
            np.save("dataset/decoded_data.npy", decoded_data)
        else:
            return decoded_data

class Asymmetrical_Autoencoder(Autoencoder):
    def __init__(
        self,
        seed=42,
        model_name=None,
        latent_dim=10,
        conv=[(8, 3), (16, 3), (32, 3)],
        dense=[64, 128],
        activation="elu",
        dropout_rate=0.2,
        loss=tfk.losses.MeanSquaredError(),
        optimizer=tfk.optimizers.Adam(),
        metrics=["mae"],
        batch_size=32,
        epochs=500,
        validation_split=0.2,
        callbacks=None,
    ):
        super().__init__(seed, model_name, latent_dim, conv, dense, activation, dropout_rate,
                       loss, optimizer, metrics, batch_size, epochs, validation_split, callbacks)
        
    def get_data(self, file_path, compressed_name="arr_0"):
       
       if file_path[-4:] == ".npy":
           X = np.load(file_path)
       elif file_path[-4:] == ".npz":
           X = np.load(file_path)[compressed_name]
       elif file_path[-4:] == ".csv":
           X = np.loadtxt(file_path, delimiter=",")
       else:
           raise ValueError("File type not supported")
           
       self.out_test_data = X[-1]
       self.in_test_data = self.out_test_data[:,::2,::2,:]
       X = X[:900]
       
       # match file_path[-4:]:
       #     case ".npy":
       #         X = np.load(file_path)
       #     case ".npz":
       #         X = np.load(file_path)[compressed_name]
       #     case ".csv":
       #         X = np.loadtxt(file_path, delimiter=",")
       #     case _:
       #         raise ValueError("File type not supported")

       self.output_data = []
       for i in tqdm(range(np.shape(X)[0])):
           for j in range(np.shape(X)[1]):
               self.output_data.append(X[i, j])
       self.output_data = np.array(self.output_data)
       self.input_data = self.output_data[:,::2,::2,:]
       
       self.input_shape = (self.input_data.shape[1], self.input_data.shape[2], self.input_data.shape[3])
       self.output_shape = (self.output_data.shape[1], self.output_data.shape[2], self.output_data.shape[3])    

   
    def train_model(self):
       self.set_seed()

       if self.callbacks is None:
           self.callbacks = [
               tfk.callbacks.EarlyStopping(
                   monitor="val_loss", patience=10, restore_best_weights=True
               ),
               # tfk.callbacks.ReduceLROnPlateau(
               #     monitor="val_loss", patience=5, factor=0.5, min_lr=1e-5
               # ),
           ]

       history = self.autoencoder.fit(
           self.input_data,
           self.output_data,
           batch_size=self.batch_size,
           epochs=self.epochs,
           validation_split=self.validation_split,
           callbacks=self.callbacks,
       ).history

class Autoencoder_identity(Autoencoder):
    def __init__(
        self,
        model_name=None,
        loss=tfk.losses.MeanSquaredError(),
        optimizer=tfk.optimizers.Adam(),
        metrics=["mae"]
    ):
        super().__init__(model_name=model_name, loss=loss, optimizer=optimizer, 
                         metrics=metrics)
       
    def build_model(self,summary=False):

        AE = tfkl.Input(shape=(self.input_shape))

        self.autoencoder = tfk.Model(inputs=AE, outputs=AE, name="Autoencoder")

        # Compile model
        self.autoencoder.compile(
            loss=self.loss, optimizer=self.optimizer, metrics=self.metrics
        )

        self.encoder = self.autoencoder
        self.decoder = self.autoencoder

        if summary:
            self.autoencoder.summary(expand_nested=True)

    def train_model(self):

        raise ValueError("Autoencoder Identity does not support training")



# a = Asymmetrical_Autoencoder(epochs=1000, conv=[(8,3),(16,3),(32,3)], dense=[32,64])
# a.get_data('../../data/dataset/merged_dataset.npz', compressed_name='my_data')

# a.build_model(summary=False)

# a.train_model()

# name = 'Asym_test'
# file_path = "../../models/" + name + "/" + name
# if not os.path.exists(file_path):
#     os.makedirs(file_path)

# model_json = a.autoencoder.to_json()
# with open(file_path + ".json", "w") as json_file:
#     json_file.write(model_json)
# a.autoencoder.save_weights(file_path + ".h5")


# encoded = a.encode(a.in_test_data)
# decoded = a.decode(encoded)

# print(np.shape(a.in_test_data))
# print(np.shape(a.out_test_data))
# print(np.shape(decoded))

# import matplotlib.pyplot as plt

# for i in [10*i for i in range(10)]:
#     fig, ax = plt.subplots(2,1)
#     ax[0].imshow(a.out_test_data[i,:,:,0])
#     im = ax[1].imshow(decoded[i,:,:,0])
#     cbar = fig.colorbar(im,ax=ax.ravel().tolist())
#     cbar.set_ticks(np.arange(0,1,0.5))
#     plt.show()
    
# for i in [5,10]:
#     fig, ax = plt.subplots(2,1)
#     ax[0].plot(decoded[:,i,i,0])
#     ax[0].plot(a.out_test_data[:,i,i,0])
#     ax[0].legend(['predicted','real'])
#     #ax[0].title('u component')
#     ax[1].plot(decoded[:,i,i,1])
#     ax[1].plot(a.out_test_data[:,i,i,1])
#     ax[1].legend(['predicted','real'])
#     #ax[1].title('v component')
#     plt.show()
    
    
    