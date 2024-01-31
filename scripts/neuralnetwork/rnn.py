import numpy as np
import os
from scripts.utils.utils import import_tensorflow

tf = import_tensorflow()
tfk = tf.keras
tfkl = tfk.layers

# Implement a necurrent neural network, composed by a sequence of  lstm layers (units given by the 
#   user, whom can also set them as bidirectional) and dense layers (neurons given by user)
class RNN:
    def __init__(
        self,
        seed=42,
        window_size=100,
        model_name=None,
        lstm=[64, 128, 256],
        bidirectional=True,
        batch_norm=True,
        dropout_rate=0.0,
        dense=[256, 128, 64],
        activation="relu",
        loss="mae",
        metrics=["mse"],
        optimizer="adam",
        callbacks=None,
        batch_size=32,
        epochs=200,
        validation_split=0.2,
    ):
        # Seed
        self.seed = seed

        # Window size
        self.window_size = window_size

        # Building parameters
        self.lstm = lstm
        self.bidirectional = bidirectional
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
        self.dense = dense
        self.activation = activation

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

    # For reproducibility
    def set_seed(self):
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        tfk.utils.set_random_seed(self.seed)

    # Load existingmodel
    def load_model(self, model_name):
        path = "../../models/" + model_name + "/" + model_name

        with open(path + ".json", "r") as json_file:
            model_json = json_file.read()

        # Load the model
        self.rnn = tfk.models.model_from_json(model_json)
        self.rnn.load_weights(path + ".h5")

        # Extract the input window size
        self.window_size = self.rnn.input_shape[-2]

    # Load data
    def get_data(self, file_path, compressed_name="arr_0"):
        if file_path[-4:] == ".npy":
            data = np.load(file_path)
        elif file_path[-4:] == ".npz":
            data = np.load(file_path)[compressed_name]
        elif file_path[-4:] == ".csv":
            data = np.loadtxt(file_path, delimiter=",")
        else:
            raise ValueError("File type not supported")

        self.X_train = []
        self.y_train = []

        # Build sequences of time steps for training
        for ts in range(data.shape[0]):
            time_series = data[ts]
            for i in range(len(time_series) - self.window_size):
                self.X_train.append(time_series[i : i + self.window_size])      # Input sequence
                self.y_train.append(time_series[i + self.window_size])          # Output value

        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)

        # Place sequences from different samples in the same dimension
        self.X_train = self.X_train.reshape(
            self.X_train.shape[0], self.window_size, data.shape[2]
        )

    # Create the structure of the rnn
    def build_model(self, summary=False):
        self.rnn = tfk.Sequential()

        # Input layer
        self.rnn.add(
            tfk.Input(
                shape=(self.X_train.shape[1], self.X_train.shape[2]),
            )
        )

        # LSTM layers
        for l in self.lstm[:-1]:
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
                
            # Batch Normalization layer
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

    # Train the rnn
    def train_model(self):
        # Set seed for reproducibility
        self.set_seed()

        if self.callbacks is None:
            self.callbacks = [
                tfk.callbacks.EarlyStopping(
                    monitor="val_loss", patience=10, restore_best_weights=True
                )
            ]

        # Train model
        history = self.rnn.fit(
            self.X_train,
            self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks=self.callbacks,
        ).history

    # Save the rnn
    def save_model(self, name):
        file_path = "../../models/" + name
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        model_json = self.rnn.to_json()
        with open(file_path + '/' + name + ".json", "w") as json_file:
            json_file.write(model_json)
        self.rnn.save_weights(file_path + '/' + name + ".h5")

    # Forecast using the rnn
    def predict_future(self, starting_sequence, length):
        
        # Initialize forecasting array
        forecast = []
        
        # Initialize current sequence with the input
        last_sequence = starting_sequence

        # Loop over time
        for i in range(length):
            
            # Predict the next time step
            prediction = self.rnn.predict(
                last_sequence.reshape(
                    1, last_sequence.shape[0], last_sequence.shape[1]
                ),
                verbose=0,
            )
            
            # Save the prediction
            forecast.append(prediction[0])
            
            # Replace oldest element in the current sequence with the new prediction
            last_sequence = np.concatenate((last_sequence[1:], prediction), axis=0)

        return np.array(forecast)
