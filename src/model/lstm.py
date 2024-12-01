import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Function to create sequences
def create_sequences(data, target, timesteps):
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
        X.append(data[i:i + timesteps])
        # Extract the target value corresponding to the last timestep in the sequence
        y.append(target[i + timesteps])
    return np.array(X), np.array(y)

# Function to create the LSTM model
def build_lstm_model(input_shape):
    """
    Builds and compiles an LSTM model for forecasting the close price.
    :param input_shape: Shape of the input data (timesteps, features)
    :return: Compiled LSTM model
    """
    model = Sequential()
    # LSTM layer with 50 units
    model.add(LSTM(units=50, return_sequences=False, input_shape=input_shape))
    # Dropout for regularization
    model.add(Dropout(0.2))
    # Dense layer to predict the close price
    model.add(Dense(units=1))  
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# Function to reshape the data for LSTM model input
def reshape_data(data, look_back):
    """
    Reshapes the dataset to be suitable for LSTM input (samples, time steps, features).
    :param data: The input dataset (e.g., stock prices, temperature, etc.)
    :param look_back: Number of previous time steps to use as input features
    :return: reshaped data for LSTM (samples, timesteps, features)
    """
    data = np.reshape(data, (data.shape[0], 1, data.shape[1]))  # Reshape for LSTM [samples, timesteps, features]
    return data
