import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers

def conv_block_2(x, n_features, dropout_rate):
    x = layers.Convolution2D(n_features, (3, 3), padding = 'same')(x)
    x = layers.BatchNormalization(axis = 3)(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Convolution2D(n_features, (3, 3), padding='same')(x)
    x = layers.BatchNormalization(axis=3)(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.MaxPooling2D(pool_size = (2, 2))(x)
    return x

def conv_block_3(x, n_features, dropout_rate):
    x = layers.Convolution2D(n_features, (3, 3), padding = 'same')(x)
    x = layers.BatchNormalization(axis = 3)(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Convolution2D(n_features, (3, 3), padding='same')(x)
    x = layers.BatchNormalization(axis=3)(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Convolution2D(n_features, (3, 3), padding='same')(x)
    x = layers.BatchNormalization(axis=3)(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    return x

def fc(x, out_dim, dropout_rate, last_layer_activation):
    x = layers.Dense(1000)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(100)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(out_dim, activation = last_layer_activation)(x)
    return x

def vgg16(im_dim, out_dim, dropout_rate = 0.3, last_layer_activation = 'softmax', input_name = 'im_in'):
    in_ = keras.Input(shape = im_dim, name = input_name)
    x = conv_block_2(in_, 32, dropout_rate)
    x = conv_block_2(x, 64, dropout_rate)
    x = conv_block_3(x, 128, dropout_rate)
    x = conv_block_3(x, 256, dropout_rate)
    x = conv_block_3(x, 256, dropout_rate)
    x = layers.Flatten()(x)
    out_ = fc(x, out_dim, dropout_rate, last_layer_activation)
    mod = keras.Model(inputs = in_, outputs = out_)
    return mod