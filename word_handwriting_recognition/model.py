from keras import layers
from keras.models import Model
import typing
import tensorflow as tf

def activation_layer(layer, activation="relu", alpha=0.1):
    if activation == "relu":
        layer = layers.ReLU()(layer)
    elif activation == "leaky_relu":
        layer = layers.LeakyReLU(alpha=alpha)(layer)
    return layer


def residual_block(
        x: tf.Tensor,
        filter_num: int,
        strides: typing.Union[int, list] = 2,
        kernel_size: typing.Union[int, list] = 3,
        skip_conv: bool = True,
        padding: str = "same",
        kernel_initializer: str = "he_uniform",
        activation: str = "relu",
        dropout: float = 0.2):
    # Create skip connection tensor
    x_skip = x

    # Perform 1-st convolution
    x = layers.Conv2D(filter_num, kernel_size, padding = padding, strides = strides, kernel_initializer=kernel_initializer)(x)
    x = layers.BatchNormalization()(x)
    x = activation_layer(x, activation=activation)

    # Perform 2-nd convolution
    x = layers.Conv2D(filter_num, kernel_size, padding = padding, kernel_initializer=kernel_initializer)(x)
    x = layers.BatchNormalization()(x)

    # Perform 3-rd convolution if skip_conv is True, matchin the number of filters and the shape of the skip connection tensor
    if skip_conv:
        x_skip = layers.Conv2D(filter_num, 1, padding = padding, strides = strides, kernel_initializer=kernel_initializer)(x_skip)

    # Add x and skip connection and apply activation function
    x = layers.Add()([x, x_skip])     
    x = activation_layer(x, activation=activation)

    # Apply dropout
    if dropout:
        x = layers.Dropout(dropout)(x)

    return x


def train_model(input_dim, output_dim, activation="leaky_relu", dropout=0.2):
    
    inputs = layers.Input(shape=input_dim, name="input")

    # normalize images here instead in preprocessing step
    input = layers.Lambda(lambda x: x / 255)(inputs)

    x1 = residual_block(input, 16, activation=activation, skip_conv=True, strides=1, dropout=dropout)

    x2 = residual_block(x1, 16, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x3 = residual_block(x2, 16, activation=activation, skip_conv=False, strides=1, dropout=dropout)

    x4 = residual_block(x3, 32, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x5 = residual_block(x4, 32, activation=activation, skip_conv=False, strides=1, dropout=dropout)

    x6 = residual_block(x5, 64, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x7 = residual_block(x6, 64, activation=activation, skip_conv=True, strides=1, dropout=dropout)

    x8 = residual_block(x7, 64, activation=activation, skip_conv=False, strides=1, dropout=dropout)
    x9 = residual_block(x8, 64, activation=activation, skip_conv=False, strides=1, dropout=dropout)

    squeezed = layers.Reshape((x9.shape[-3] * x9.shape[-2], x9.shape[-1]))(x9)

    blstm = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(squeezed)
    blstm = layers.Dropout(dropout)(blstm)

    output = layers.Dense(output_dim + 1, activation="softmax", name="output")(blstm)

    model = Model(inputs=inputs, outputs=output)
    return model
