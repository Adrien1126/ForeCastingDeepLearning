import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Function to create sequences
def create_sequences(data, target_column, timesteps):
    """
    Creates sequences for LSTM training.
    :param data: Feature dataset (NumPy array or DataFrame) of shape (samples, features)
    :param target: Target array or column (e.g., Close prices) of shape (samples,)
    :param timesteps: Number of timesteps to include in each sequence
    :return: Tuple of NumPy arrays (X, y)
    """
    X, y = [], []
    for i in range(len(data) - timesteps):
        # Extract sequences of features
        X.append(data.iloc[i:i + timesteps].values)
        # Extract the target value corresponding to the last timestep in the sequence
        y.append(data.iloc[i + timesteps][target_column])
    return np.array(X), np.array(y)

# Function to create the LSTM model
def build_lstm_model(input_shape):
    """
    Builds and compiles an LSTM model for forecasting the close price.
    :param input_shape: Shape of the input data (timesteps, features)
    :return: Compiled LSTM model
    """
    model = Sequential()

    # Première couche LSTM
    model.add(LSTM(units=32, return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.3))

    # Deuxième couche LSTM
    model.add(LSTM(units=32, return_sequences=False, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.3))

    # Couche dense pour la prédiction
    model.add(Dense(units=1))

    # Compiler le modèle
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model
