from tensorflow import keras
from tensorflow.keras import layers

def mod_bl(y_dim):
    nn_bl = keras.Sequential(name = "nn_bl")
    nn_bl.add(keras.Input(shape = (1, ), name = "bl_in"))
    nn_bl.add(layers.Dense(y_dim - 1, activation = "linear", use_bias = False, name = "bl_out"))
    return nn_bl