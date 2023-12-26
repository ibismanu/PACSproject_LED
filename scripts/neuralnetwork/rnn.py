import numpy as np
from scripts.utils.utils import import_tensorflow

tf = import_tensorflow()  # elimina warning inutili

tfk = tf.keras
tfkl = tfk.layers


class RNN:
    def __init__(
        self,
        name=None,
        lstm=[(32, False), (16, True), (32, False)],
        dense=[64, 32],
        seed=42,
        latent_dim=40,
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
        self.validation_split = validation_split
        self.callbacks = callbacks

        # Input shape (window, latent_dim)
        self.input_shape = (self.window, latent_dim)
        self.output_shape = latent_dim

        # Model
        self.rnn = None

        # Data
        self.raw_data = None
        self.X_train = None
        self.Y_train = None

        if name is not None:
            self.load_model(name)

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

    def get_data(self, name, compressed_name="arr_0", to_split=False, split_ratio=0.2):
        match name[-4:]:
            case ".npy":
                self.raw_data = np.load("dataset/" + name)
            case ".npz":
                self.raw_data = np.load("dataset/" + name)[compressed_name]
            case ".csv":
                self.raw_data = np.loadtxt("dataset/" + name, delimiter=",")
            case _:
                raise ValueError("File type not supported")

    # def split():

    def build_sequences(self, raw_data):
        X = []
        Y = []

        # dim(raw_data) = (timesteps,latend_dim)

        for idx in np.arange(
            0, raw_data.shape[0] - self.window - self.telescope, self.stride
        ):
            X.append(raw_data[idx : idx + self.window, :])
            Y.append(
                raw_data[idx + self.window : idx + self.window + self.telescope, :]
            )

        return np.array(X), np.array(Y)

    def train_model(self, raw_data):
        # dim = (n_samples, timesteps, latent_dim)

        # cycle over samples
        for i in range(np.shape(raw_data)[0]):
            X_temp, Y_temp = self.build_sequences(raw_data[i, :, :], self.window)

            if i == 0:
                X_train = X_temp
                Y_train = Y_temp

            else:
                X_train = np.concatenate((X_train, X_temp), 0)
                Y_train = np.concatenate((Y_train, Y_temp), 0)
            # print("round n°",i)

        self.X_train = X_train
        self.Y_train = Y_train

        history = self.rnn.fit(
            self.X_train,
            self.Y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.validation_split,
            callbacks=self.callbacks,
        )

        if self.name is not None:
            saving_dir = "../../models/" + self.name + "/" + self.name
            auto_json = self.RNN.to_json()
            with open(saving_dir + ".json", "w") as json_file:
                json_file.write(auto_json)
            self.RNN.save_weights(saving_dir + ".h5")

    def load_model(self, name):
        path = "models/" + name + "/" + name

        with open(path + ".json", "r") as json_file:
            rnn_json = json_file.read()

        self.rnn = tfk.models.model_from_json(rnn_json)
        self.rnn.load_weights(path + ".h5")

    def test_rnn(self, rnn_dir, data_dir_test, compressed_name="arr_0"):
        self.get_data(data_dir_test, compressed_name=compressed_name)

        self.load_rnn(RNN_dir=rnn_dir)

        X_test, Y_test = self.build_sequences(
            self.raw_data[0, :, :]
        )  # facciamo il test solo sul primo sample

        for seq in range(np.shape(X_test)[0]):
            prediction = self.rnn.predict(np.expand_dims(X_test[seq, :], 0), verbose=0)
            if seq == 0:
                print(np.shape(prediction), np.shape(Y_test[seq, :]))
                X_RNN = prediction
                prediction_loss = np.array(
                    [np.linalg.norm(prediction - Y_test[seq, :], 2)]
                )
            else:
                X_RNN = np.concatenate((X_RNN, prediction), axis=0)
                prediction_loss = np.concatenate(
                    (
                        prediction_loss,
                        np.array([np.linalg.norm(prediction - Y_test[seq, :], 2)]),
                    ),
                    axis=0,
                )
            print("prediction ", seq, " of ", np.shape(X_test)[0])
        return X_RNN, prediction_loss

    def predict_future(self, length, starting_sequence, real_future=None):
        # starting sequence dim: (window, latent_dim)

        future = np.array([])
        X_temp = starting_sequence

        for reg in range(length):
            pred_temp = self.rnn.predict(np.expand_dims(X_temp, 0), verbose=0)
            if len(future) == 0:
                future = pred_temp
                print(np.shape(pred_temp))
            else:
                future = np.concatenate((future, pred_temp), axis=0)
            X_temp = np.concatenate((X_temp[1:, :], pred_temp), axis=0)
            print("prediction ", reg, " of ", length)

        # TODO separate function
        future_err = None
        # if real_future is not None:
        #     future_err = np.array([])
        #     for i in range(length):
        #         future_err.append(tfk.losses.mse(future[:,i],real_future[i,:]))

        return future #, future_err
