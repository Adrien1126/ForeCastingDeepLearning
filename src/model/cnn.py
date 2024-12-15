from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input, LSTM, BatchNormalization, Activation
from tensorflow.keras.regularizers import l2

def build_cnn1d_model(input_shape):
    model = Sequential()

    model.add(Input(shape=input_shape))

    # 1st Convolutional Layer
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))

    # 2nd Convolutional Layer
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))

    # Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(units=50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # Output Layer for Regression

    # Compile the Model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def build_cnn1d_deep_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))

    # 1st Convolutional Block
    model.add(Conv1D(filters=128, kernel_size=5, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))

    # 2nd Convolutional Block
    model.add(Conv1D(filters=128, kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))  # Stop here for pooling
    model.add(Dropout(0.3))

    # 3rd Convolutional Block - NO Pooling
    model.add(Conv1D(filters=64, kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(units=100, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units=1))  # Output Layer

    # Compile
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def build_lstm_cnn(input_shape):
    model = Sequential()

    model.add(Input(shape=input_shape))

    # 1st Convolutional Layer
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))

    # 2nd Convolutional Layer
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(LSTM(64, return_sequences=False))

    # Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='relu'))  # Ensures output >= 0

    # Compile the Model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model