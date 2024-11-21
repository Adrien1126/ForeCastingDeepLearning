import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Function to create dataset matrix
def create_dataset(dataset, look_back=1):
    """
    Converts an array of values into a dataset matrix suitable for time series prediction.
    :param dataset: Array of values (e.g., stock prices, temperature, etc.)
    :param look_back: Number of previous time steps to use for predicting the next time step
    :return: dataset matrix (X, y) where X is the input and y is the output
    """
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        # Creates input-output pairs for time series prediction
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)

# Function to create the LSTM model
def build_lstm_model(input_shape):
    """
    Builds and compiles an LSTM model for time series forecasting.
    :param input_shape: Shape of the input data (samples, timesteps, features)
    :return: Compiled LSTM model
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=False, input_shape=input_shape))
    model.add(Dense(1))  # Output layer with 1 unit (for regression)
    
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
