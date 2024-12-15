from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

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
