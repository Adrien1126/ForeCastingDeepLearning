import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

def create_sequences(data, look_back):
    """
    Create sequences for multivariate time series data.
    Args:
        data (numpy array): Multivariate time series data.
        sequence_length (int): Length of each sequence.
    Returns:
        tuple: Arrays of input sequences (X) and targets (y).
    """
    dataX, dataY = [], []
    for i in range(len(data)-look_back-1):
        a = data[i:(i+look_back)]
        dataX.append(a)
        dataY.append(data[i + look_back])
    return np.array(dataX), np.array(dataY)

def build_rnn(sequence_length, num_features):
    """
    Build and compile an RNN model.
    Args:
        sequence_length (int): Length of input sequences.
        num_features (int): Number of features in the input.
    Returns:
        keras.Model: Compiled RNN model.
    """
    model = Sequential([
        Input(shape=(sequence_length, num_features)),  # Explicit input layer
        SimpleRNN(50, activation='relu'),
        Dense(1)  # Predict a single target value
    ])
    model.compile(optimizer='adam', loss='mse')
    return model