from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Concatenate, Cropping1D, ZeroPadding1D, Dropout, Flatten, Dense
from tensorflow.keras.regularizers import l2

def build_unet_1d(input_shape):
    inputs = Input(shape=input_shape)
    
    # Encoder Path
    c1 = Conv1D(64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.01))(inputs)    
    c1 = Dropout(0.2)(c1)
    p1 = MaxPooling1D(pool_size=2)(c1)
    
    c2 = Conv1D(128, kernel_size=3, activation='relu', padding='same')(p1)
    c2 = Dropout(0.2)(c2)
    p2 = MaxPooling1D(pool_size=2)(c2)
    
    c3 = Conv1D(256, kernel_size=3, activation='relu', padding='same')(p2)
    c3 = Dropout(0.3)(c3)
    p3 = MaxPooling1D(pool_size=2)(c3)
    
    # Bottleneck
    bn = Conv1D(512, kernel_size=3, activation='relu', padding='same')(p3)
    bn = Dropout(0.4)(bn)
    
    # Decoder Path
    u1 = UpSampling1D(size=2)(bn)
    
    # Match shapes before concatenation
    if u1.shape[1] != c3.shape[1]:
        if u1.shape[1] > c3.shape[1]:
            u1 = Cropping1D(cropping=(0, u1.shape[1] - c3.shape[1]))(u1)
        else:
            u1 = ZeroPadding1D(padding=(0, c3.shape[1] - u1.shape[1]))(u1)
    u1 = Concatenate()([u1, c3])
    c4 = Conv1D(256, kernel_size=3, activation='relu', padding='same')(u1)
    c4 = Dropout(0.3)(c4)
    
    u2 = UpSampling1D(size=2)(c4)
    if u2.shape[1] != c2.shape[1]:
        if u2.shape[1] > c2.shape[1]:
            u2 = Cropping1D(cropping=(0, u2.shape[1] - c2.shape[1]))(u2)
        else:
            u2 = ZeroPadding1D(padding=(0, c2.shape[1] - u2.shape[1]))(u2)
    u2 = Concatenate()([u2, c2])
    c5 = Conv1D(128, kernel_size=3, activation='relu', padding='same')(u2)
    c5 = Dropout(0.2)(c5)
    
    u3 = UpSampling1D(size=2)(c5)
    if u3.shape[1] != c1.shape[1]:
        if u3.shape[1] > c1.shape[1]:
            u3 = Cropping1D(cropping=(0, u3.shape[1] - c1.shape[1]))(u3)
        else:
            u3 = ZeroPadding1D(padding=(0, c1.shape[1] - u3.shape[1]))(u3)
    u3 = Concatenate()([u3, c1])
    c6 = Conv1D(64, kernel_size=3, activation='relu', padding='same')(u3)
    c6 = Dropout(0.2)(c6)
    
    # Output Layer
    outputs = Dense(1, activation='linear')(Flatten()(c6))
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model
