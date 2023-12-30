import numpy as np
import os
from scripts.utils.utils import import_tensorflow

tf = import_tensorflow()  # elimina warning inutili

tfk = tf.keras
tfkl = tfk.layers


class RNN:
    def __init__(
        self,
        model_name=None,
        seed=42,
        window_size=150,
        lstm=[64, 128, 256],
        bidirectional=True,
        batch_norm=True,
        dropout_rate=0.0,
        dense=[256, 128, 64],
        activation="relu",
        optimizer="adam",
        loss="mae",
        batch_size=64,
        validation_split=0.2,
        callbacks=None,
        metrics=[],
        epochs=200,
    ):
        if model_name is not None:
            self.load_model(model_name)

        self.seed = seed

        self.window_size = window_size

        self.lstm = lstm
        self.dense = dense
        self.bidirectional = bidirectional
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
        self.activation = activation

        self.optimizer = optimizer
        self.loss = loss

        self.batch_size = batch_size
        self.validation_split = validation_split
        self.callbacks = callbacks
        self.epochs = epochs
        self.metrics = metrics

    def load_model(self, model_name):
        path = "models/" + model_name + "/" + model_name

        with open(path + ".json", "r") as json_file:
            model_json = json_file.read()

        self.rnn = tfk.models.model_from_json(model_json)
        self.rnn.load_weights(path + ".h5")

    def set_seed(self):
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        tfk.utils.set_random_seed(self.seed)

    def get_data(self, file_name, compressed_name="arr_0"):
        match file_name[-4:]:
            case ".npy":
                data = np.load("dataset/" + file_name)
            case ".npz":
                data = np.load("dataset/" + file_name)[compressed_name]
            case ".csv":
                data = np.loadtxt("dataset/" + file_name, delimiter=",")
            case _:
                raise ValueError("File type not supported")

        self.X_train = []
        self.y_train = []

        for ts in range(data.shape[0]):
            time_series = data[ts]
            for i in range(len(time_series) - self.window_size):
                self.X_train.append(time_series[i : i + self.window_size])
                self.y_train.append(time_series[i + self.window_size])

        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)

        self.X_train = self.X_train.reshape(
            self.X_train.shape[0], self.window_size, data.shape[2]
        )

    def build_model(self, summary=False):
        self.rnn = tfk.Sequential()

        # First LSTM layer
        if self.bidirectional:
            self.rnn.add(
                tfkl.Bidirectional(
                    tfkl.LSTM(
                        units=self.lstm[0],
                        return_sequences=(len(self.lstm) > 1),
                        input_shape=(self.X_train.shape[1], self.X_train.shape[2]),
                    )
                )
            )
        else:
            self.rnn.add(
                tfkl.LSTM(
                    units=self.lstm[0],
                    return_sequences=(len(self.lstm) > 1),
                    input_shape=(self.X_train.shape[1], self.X_train.shape[2]),
                )
            )

        if self.batch_norm:
            self.rnn.add(tfkl.BatchNormalization())

        # Middle LSTM layers
        for l in self.lstm[1:-1]:
            if self.bidirectional:
                self.rnn.add(
                    tfkl.Bidirectional(
                        tfkl.LSTM(
                            units=l,
                            return_sequences=True,
                        )
                    )
                )
            else:
                self.rnn.add(
                    tfkl.LSTM(
                        units=l,
                        return_sequences=True,
                    )
                )

            if self.batch_norm:
                self.rnn.add(tfkl.BatchNormalization())

        # Last LSTM layer
        if self.bidirectional:
            self.rnn.add(
                tfkl.Bidirectional(
                    tfkl.LSTM(
                        units=self.lstm[-1],
                    )
                )
            )
        else:
            self.rnn.add(
                tfkl.LSTM(
                    units=self.lstm[-1],
                )
            )

        # Dense layers
        for d in self.dense:
            self.rnn.add(tfkl.Dense(units=d, activation=self.activation))
            if self.batch_norm:
                self.rnn.add(tfkl.BatchNormalization())
            if self.dropout_rate > 0:
                self.rnn.add(tfkl.Dropout(rate=self.dropout_rate))

        # Output layer
        self.rnn.add(tfkl.Dense(units=self.y_train.shape[-1]))

        # Compile model
        self.rnn.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics,
        )

        if summary:
            self.rnn.summary(expand_nested=True)

    def save_model(self, name):
        file_path = "models/" + name + "/" + name
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        model_json = self.rnn.to_json()
        with open(file_path + ".json", "w") as json_file:
            json_file.write(model_json)
        self.rnn.save_weights(file_path + ".h5")

    def train_model(self):
        if self.callbacks is None:
            self.callbacks = [
                tfk.callbacks.EarlyStopping(
                    monitor="val_loss", patience=10, restore_best_weights=True
                )
            ]

        self.rnn.fit(
            self.X_train,
            self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks=self.callbacks,
        )

    def predict_future(self, starting_sequence, length):
        forecast = []
        last_sequence = starting_sequence

        for i in range(length):
            prediction = self.rnn.predict(
                last_sequence.reshape(
                    1, last_sequence.shape[0], last_sequence.shape[1]
                ),
                verbose=0,
            )
            forecast.append(prediction[0])
            last_sequence = np.concatenate((last_sequence[1:], prediction), axis=0)

        return np.array(forecast)
