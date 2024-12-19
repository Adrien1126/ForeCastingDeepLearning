import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LayerNormalization, Dropout, MultiHeadAttention, Add
)

# Define Transformer Encoder Layer
class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),  # Feed-forward network
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, inputs, training=False):
        # Multi-head attention
        attn_output = self.attention(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Build Transformer Model for Time Series
def build_transformer(input_shape, embed_dim, num_heads, ff_dim, num_layers, dropout_rate=0.1):
    inputs = Input(shape=input_shape)
    
    # Embedding Layer
    x = Dense(embed_dim)(inputs)  # Project input to embedding dimension
    
    # Add Transformer Encoder Layers
    for _ in range(num_layers):
        x = TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout_rate)(x)
    
    # Flatten and Output Layer
    x = tf.keras.layers.Flatten()(x)
    outputs = Dense(1, activation='linear')(x)  # Output 1 value for forecasting
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Model Configuration
input_shape = (10, 1)  # 10 timesteps, 1 feature (change as needed)
embed_dim = 64         # Embedding dimension
num_heads = 4          # Number of attention heads
ff_dim = 128           # Feed-forward network dimension
num_layers = 2         # Number of transformer encoder layers
dropout_rate = 0.1     # Dropout rate
