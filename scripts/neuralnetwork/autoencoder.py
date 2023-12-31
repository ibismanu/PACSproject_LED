import numpy as np
from tqdm.auto import tqdm
import os

import tensorflow as tf

tfk = tf.keras
tfkl = tfk.layers


class Autoencoder:
    def __init__(
        self,
        input_shape=None,  # TODO initialize
        latent_dim=40,
        conv=[(8, 3), (16, 3), (32, 3)],
        dense=[64, 128],
        activation="relu",
        dropout_rate=0.2,
        loss=tfk.losses.MeanSquaredError(),
        optimizer=tfk.optimizers.Adam(),
        metrics=["mae"],
        batch_size=32,
        epochs=1000,
        validation_split=0.2,
        callbacks=[
            tfk.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            ),
            tfk.callbacks.ReduceLROnPlateau(
                monitor="val_loss", patience=5, factor=0.5, min_lr=1e-5
            ),
        ],
        ae_dir=None,
        seed=42,
    ):
        self.seed = seed
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.conv = conv
        self.dense = dense
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics

        self.data = None

        self.epochs = epochs
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.callbacks = callbacks

        if ae_dir is not None:
            self.load_model(ae_dir)

    def load_model(self, ae_dir):

        with open(ae_dir + ".json", "r") as json_file:
            auto_json = json_file.read()

        self.autoencoder = tfk.models.model_from_json(auto_json)
        self.autoencoder.load_weights(ae_dir + ".h5")

        self.encoder = self.autoencoder.get_layer("Encoder")
        self.decoder = self.autoencoder.get_layer("Decoder")

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
                self.input_shape[-1],
                kernel_size=(1, 1),
                strides=(1, 1),
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

    def get_data(self, name, compressed_name="arr_0"):
        match name[-4:]:
            case ".npy":
                X = np.load("dataset/" + name)
            case ".npz":
                X = np.load(name)[compressed_name]
            case ".csv":
                X = np.loadtxt("dataset/" + name, delimiter=",")
            case _:
                raise ValueError("File type not supported")

        self.data = []
        for i in tqdm(range(np.shape(X)[0])):
            for j in range(np.shape(X)[1]):
                self.data.append(X[i, j])
        self.data = np.array(self.data)

    def train_model(self):
        history = self.autoencoder.fit(
            self.data,
            self.data,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.validation_split,
            callbacks=self.callbacks,
        ).history

    def save_model(self, ae_dir):

        if not os.path.exists(ae_dir):
            os.makedirs(ae_dir)

        auto_json = self.autoencoder.to_json()
        with open(ae_dir + ".json", "w") as json_file:
            json_file.write(auto_json)
        self.autoencoder.save_weights(ae_dir + ".h5")

    def encode(self, raw_data, save=True):
        encoded_data = self.encoder.predict(raw_data, verbose=0)

        if save:
            np.save("dataset/encoded_data.npy", encoded_data)
        else:
            return encoded_data
        return
    
    def test_autoencoder(self, data_dir_test, compressed_name_test='arr_0', plot=False):

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

        grid_size = np.shape(X_test)[2]
        times = np.shape(X_test)[1]
        n_samples = np.shape(X_test)[0]
        dim_u = np.shape(X_test)[-1]

        X_ae = []
        prediction_loss = []

        # for s in range(n_samples):
        # qui per ora sto facendo il predict solo per il sample 0
        for t in range(times):
            prediction = self.autoencoder.predict(np.expand_dims(X_test[0,t],0),verbose=0) 
            X_ae.append(prediction)
            prediction_loss.append(tfk.losses.mae(prediction,X_test[0,t]))

        # error = X_test[0] - X_ae

        # # TODO: se vogliamo il plot questo if va messo a posto
        # if plot:
        #     # plot some signals in the diagonal of the grid
        #     X_ae = np.reshape(X_ae,(n_samples,times,grid_size,grid_size,dim_u))

        #     for i in range(5):
        #         fig,(ax1,ax2) = plt.subplots(2,1,figsize=(8, 6))

        #         ax1.plot(np.arange(0,np.shape(X_test)[1],1), X_ae[0, :, i, i, 0])
        #         ax1.set_title('Predicted')

        #         ax2.plot(np.arange(0,np.shape(X_test)[1],1), X_test[0, :, i, i, 0])
        #         ax2.set_title('Test')

        #         plt.tight_layout()
        #         plt.show()

        return X_ae, prediction_loss
