import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp

def nn_combine(nn_bl, nn_x = None, nn_im = None): # get input shapes
    if((nn_x == None) & (nn_im == None)):
        nn = keras.Model(inputs = nn_bl.input, outputs = nn_bl.output)
    if((nn_x != None) & (nn_im == None)):
        nn = keras.Model(inputs = [nn_bl.input, nn_x.input], outputs = [nn_bl.output, nn_x.output])
    if((nn_x == None) & (nn_im != None)):
        nn = keras.Model(inputs = [nn_bl.input, nn_im.input], outputs = [nn_bl.output, nn_im.output])
    if((nn_x != None) & (nn_im != None)):
        nn = keras.Model(inputs = [nn_bl.input, nn_x.input, nn_im.input], outputs = [nn_bl.output, nn_x.output, nn_im.output])
    return nn

class ontram():
    def __init__(self, nn_bl, nn_x = None, nn_im = None, im_dim = None, response_varying = False,
                 distr = tfp.distributions.Logistic(loc = 0, scale = 1)):
        self.nn_bl = nn_bl
        self.nn_x = nn_x
        self.nn_im = nn_im
        self.distr = distr
        self.response_varying = response_varying
        self.model = nn_combine(nn_bl, nn_x, nn_im)