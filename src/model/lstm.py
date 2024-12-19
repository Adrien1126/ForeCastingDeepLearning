from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import keras_tuner as kt

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


def build_deeper_lstm_model(input_shape):
    """
    Build a deeper LSTM model for forecasting the difference between close prices of time t and t-1.
    :param input_shape: Shape of the input (timesteps, features)
    :return: Compiled Keras model
    """
    model = Sequential()

    # Input layer
    model.add(Input(shape=input_shape))

    # First Bidirectional LSTM layer with return_sequences=True to stack more LSTMs
    model.add(Bidirectional(LSTM(units=256, activation='tanh', return_sequences=True)))
    model.add(Dropout(0.4))  # Increased dropout to help regularization

    # Second LSTM layer
    model.add(LSTM(units=256, activation='tanh', return_sequences=True))
    model.add(Dropout(0.4))

    # Third LSTM layer with Layer Normalization
    model.add(LSTM(units=128, activation='tanh', return_sequences=True))
    model.add(LayerNormalization())  # Stabilize the training process
    model.add(Dropout(0.3))

    # Fourth LSTM layer
    model.add(LSTM(units=64, activation='tanh', return_sequences=False))
    model.add(Dropout(0.3))

    # Fully connected layer
    model.add(Dense(units=50, activation='relu'))

    # Output layer for regression (forecasting the price difference)
    model.add(Dense(units=1))  # Single value output for the predicted value (price difference)

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

# Define the model builder
def model_builder(hp):
    model = Sequential()

    # Input layer
    model.add(Input(shape=input_shape))

    # First LSTM layer with tunable units and dropout rate
    model.add(LSTM(units=hp.Int('lstm_units_1', min_value=64, max_value=256, step=32), 
                   activation='tanh', return_sequences=True))
    model.add(Dropout(rate=hp.Float('dropout_1', min_value=0.2, max_value=0.5, step=0.1)))

    # Second LSTM layer
    model.add(LSTM(units=hp.Int('lstm_units_2', min_value=64, max_value=256, step=32), 
                   activation='tanh', return_sequences=True))
    model.add(Dropout(rate=hp.Float('dropout_2', min_value=0.2, max_value=0.5, step=0.1)))

    # Third LSTM layer
    model.add(LSTM(units=hp.Int('lstm_units_3', min_value=32, max_value=128, step=32), 
                   activation='tanh', return_sequences=False))
    model.add(Dropout(rate=hp.Float('dropout_3', min_value=0.2, max_value=0.5, step=0.1)))

    # Fully connected layer
    model.add(Dense(units=hp.Int('dense_units', min_value=32, max_value=128, step=16), activation='relu'))

    # Output layer
    model.add(Dense(units=1))  # Single value output for regression

    # Compile the model with a tunable learning rate
    learning_rate = hp.Choice('learning_rate', values=[0.001, 0.0005, 0.0001])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    
    return model