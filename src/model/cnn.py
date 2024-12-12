from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input, LSTM
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

def build_stronger_cnn(input_shape):
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
    model.add(Dense(units=128, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='relu'))  # Ensures output >= 0

    # Compile the Model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model