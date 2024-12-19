from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Function to create the LSTM model
def build_lstm_model(input_shape):
    """
    Builds and compiles an LSTM model for forecasting the close price.
    :param input_shape: Shape of the input data (timesteps, features)
    :return: Compiled LSTM model
    """
    model = Sequential()

    # Input layer to define the input shape
    model.add(Input(shape=input_shape))

    # Première couche LSTM
    model.add(LSTM(units=16, return_sequences=True, kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.4))

    # Deuxième couche LSTM
    model.add(LSTM(units=16, return_sequences=False, kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.4))

    # Couche dense pour la prédiction
    model.add(Dense(units=1))

    # Compiler le modèle
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

def build_deep_lstm_model(input_shape):
    """
    Build a deeper LSTM model for forecasting with improved capacity.
    :param input_shape: Shape of the input (timesteps, features)
    :return: Compiled Keras model
    """
    model = Sequential()

    # Input layer
    model.add(Input(shape=input_shape))

    # First LSTM layer with return_sequences=True to stack more LSTMs
    model.add(LSTM(units=128, activation='tanh', return_sequences=True))
    model.add(Dropout(0.3))  # Regularization to reduce overfitting

    # Second LSTM layer
    model.add(LSTM(units=128, activation='tanh', return_sequences=True))
    model.add(Dropout(0.3))

    # Third LSTM layer
    model.add(LSTM(units=64, activation='tanh', return_sequences=False))
    model.add(Dropout(0.3))

    # Fully connected layer for learning non-linear relationships
    model.add(Dense(units=50, activation='relu'))

    # Output layer for regression (forecasting)
    model.add(Dense(units=1))  # Single value output for the predicted value

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model
