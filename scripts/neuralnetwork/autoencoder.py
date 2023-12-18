import numpy as np
from tqdm.auto import tqdm
import os

import tensorflow as tf

tfk = tf.keras
tfkl = tfk.layers


class Autoencoder:
    def __init__(
        self,
        input_shape,
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
        name=None,
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

        if name is not None:
            self.load_model("models/" + name)

    def load_model(self, name):
        path = "models/" + name + "/" + name + ".json"

        with open(path, "r") as json_file:
            auto_json = json_file.read()

        self.autoencoder = tfk.models.model_from_json(auto_json)
        self.autoencoder.load_weights(path + ".h5")

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
                X = np.load("dataset/" + name)[compressed_name]
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

    def save_model(self, name):
        file_path = "models/" + name + "/" + name
        if not os.path.exists(file_path):
            os.makedirs(file_path)
            
        auto_json = self.autoencoder.to_json()
        with open(file_path + ".json", "w") as json_file:
            json_file.write(auto_json)
        self.autoencoder.save_weights(file_path + ".h5")

    def encode(self, raw_data):
        encoded_data = self.encoder.predict(raw_data, verbose=0)
        np.save("dataset/encoded_data.npy")
