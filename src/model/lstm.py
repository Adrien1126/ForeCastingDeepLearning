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
