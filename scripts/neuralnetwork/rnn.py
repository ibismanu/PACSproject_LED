import numpy as np
from scripts.utils.utils import import_tensorflow

tf = import_tensorflow()  # elimina warning inutili

tfk = tf.keras
tfkl = tfk.layers


class RNN:
    def __init__(
        self,
        lstm=[(32, False), (16, True), (32, False)],
        dense=[64, 32],
        seed=42,
        latent_dim=40,
        time_steps=1001,
        activation="relu",
        dropout_rate=0.4,
        loss=tfk.losses.MeanSquaredError(),
        optimizer=tfk.optimizers.Adam(),
        metrics=["mae"],
        window=10,
        stride=1,
        telescope=1,
        batch_size=16,
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
    ):
        # Seed
        self.seed = seed

        # Input shape (time, latent_dim)
        self.input_shape = (time_steps, latent_dim)
        self.output_shape = latent_dim

        # LSTM layers (units: int, bidirectional: bool)
        self.lstm = lstm

        # Dense layers
        self.dense = dense

        # RNN parameters
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics

        # Sequences parameters
        self.window = window - window % stride
        self.stride = stride
        self.telescope = telescope

        # Training parameters
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = (validation_split,)
        self.callbacks = callbacks

        # Model
        self.rnn = None

        # Data
        self.data = None
        self.X_train = None
        self.Y_train = None

    def build_model(self, summary=False):
        # Input layer
        input_layer = tfkl.Input(shape=self.input_shape, name="Input")
        x = input_layer

        # LSTM layer
        for layer in self.lstm[:-1]:
            if layer[1]:
                x = tfkl.Bidirectional(
                    tfkl.LSTM(units=layer[0], return_sequences=True)
                )(x)
            else:
                x = tfkl.LSTM(units=layer[0], return_sequences=True)(x)
            x = tfkl.Dropout(rate=self.dropout_rate, seed=self.seed)(x)

        # Last LSTM layer
        layer = self.lstm[-1]
        if layer[1]:
            x = tfkl.Bidirectional(tfkl.LSTM(units=layer[0]))(x)
        else:
            x = tfkl.LSTM(units=layer[0])(x)
        x = tfkl.Dropout(rate=self.dropout_rate, seed=self.seed)(x)

        # Dense layers
        for layer in self.dense:
            x = tfkl.Dense(units=layer, activation=self.activation)(x)
            x = tfkl.Dropout(rate=0.5 * self.dropout_rate, seed=self.seed)(x)

        output_layer = tfkl.Dense(units=self.output_shape, activation="linear")(x)

        # Build model
        self.rnn = tfk.Model(inputs=input_layer, outputs=output_layer, name="RNN")

        # Print model
        if summary:
            self.rnn.summary(expand_nested=True)

            # Compile model
            self.rnn.compile(
                loss=self.loss, optimizer=self.optimizer, metrics=self.metrics
            )

    def get_data(self, name, compressed_name="arr_0"):
        match name[-4:]:
            case ".npy":
                raw_data = np.load("dataset/" + name)
            case ".npz":
                raw_data = np.load("dataset/" + name)[compressed_name]
            case ".csv":
                raw_data = np.loadtxt("dataset/" + name, delimiter=",")
            case _:
                raise ValueError("File type not supported")

        # CI SERVE UN SOLO SAMPLE

        raw_data = raw_data.T

        self.X_train, self.Y_train = self.build_sequences(raw_data)

    def build_sequences(self, raw_data):
        X = []
        Y = []

        raw_data = np.transpose(raw_data)

        for idx in np.arange(
            0, raw_data.shape[0] - self.window - self.telescope, self.stride
        ):
            X.append(raw_data[idx : idx + self.window])
            Y.append(raw_data[idx + self.window : idx + self.window + self.telescope])

        return np.array(X), np.array(Y)

    def train_model(self):
        history = self.rnn.fit(
            self.X_train,
            self.Y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.validation_split,
            callbacks=self.callbacks,
        )
