import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Function to create sequences (reuse from lstm.py)
def create_sequences(data, target_column, timesteps):
    """
    Creates sequences for GRU training.
    :param data: Feature dataset (NumPy array or DataFrame) of shape (samples, features)
    :param target_column: Target column (e.g., Close prices)
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

# Function to create the GRU model
def build_gru_model(input_shape):
    """
    Builds and compiles a GRU model for forecasting the close price.
    :param input_shape: Shape of the input data (timesteps, features)
    :return: Compiled GRU model
    """
    model = Sequential()

    model.add(Input(shape=input_shape))

    # First GRU layer
    model.add(GRU(units=16, return_sequences=True, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.3))

    # Second GRU layer
    model.add(GRU(units=16, return_sequences=False, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.3))

    # Dense layer for prediction
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model
