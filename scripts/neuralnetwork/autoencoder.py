import numpy as np
import os
from tqdm.auto import tqdm

from scripts.utils.utils import import_tensorflow, lpfilter

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
        path = "models/" + model_name + "/" + model_name

        with open(path + ".json", "r") as json_file:
            model_json = json_file.read()

        self.autoencoder = tfk.models.model_from_json(model_json)
        self.autoencoder.load_weights(path + ".h5")

        self.encoder = self.autoencoder.get_layer("Encoder")
        self.decoder = self.autoencoder.get_layer("Decoder")

    def get_data(self, file_path, compressed_name="arr_0"):
        match file_path[-4:]:
            case ".npy":
                X = np.load(file_path)
            case ".npz":
                X = np.load(file_path)[compressed_name]
            case ".csv":
                X = np.loadtxt(file_path, delimiter=",")
            case _:
                raise ValueError("File type not supported")

        self.data = []
        for i in tqdm(range(np.shape(X)[0])):
            for j in range(np.shape(X)[1]):
                self.data.append(X[i, j])
        self.data = np.array(self.data)

    def build_model(self, summary=False):
        self.input_shape = (self.data.shape[1], self.data.shape[2], self.data.shape[3])
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
        file_path = "models/" + name + "/" + name
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
                encoded_data[:, i] = lpfilter(encoded_data[:, i], 10)

        if save:
            np.save("dataset/encoded_data.npy", encoded_data)
        else:
            return encoded_data

    def decode(self, encoded_data, save=False):
        decoded_data = self.decoder.predict(encoded_data, verbose=0)

        if save:
            np.save("dataset/decoded_data.npy", decoded_data)
        else:
            return decoded_data
