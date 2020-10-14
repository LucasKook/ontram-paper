import tensorflow as tf
import numpy as np
from functions.utils import gamma_to_theta, tf_diff_axis1
from functions.compute_logLik import compute_logLik

def predict(model, y, bl = None, x = None, x_im = None):
    if(bl is None):
        bl = np.ones((y.shape[0], 1), dtype = "float32")
    if((model.nn_x == None) & (model.nn_im == None)):
        gamma = model.model.predict({"bl_in": bl})
        beta = 0
        eta = 0
        beta_w = 0
    if((model.nn_x != None) & (model.nn_im == None)):
        gamma, beta = model.model.predict({"bl_in": bl, "x_in": x})
        eta = 0
        beta_w = model.model.get_layer('x_out').get_weights()
    if((model.nn_x == None) & (model.nn_im != None)):
        gamma, eta = model.model.predict({"bl_in": bl, "im_in": x_im})
        beta = 0
        beta_w = 0
    if((model.nn_x != None) & (model.nn_im != None)):
        gamma, beta, eta = model.model.predict({"bl_in": bl, "x_in": x, "im_in": x_im})
        beta_w = model.model.get_layer('x_out').get_weights()
    theta = gamma_to_theta(gamma)
    probs = model.distr.cdf(theta - beta - eta)
    dens = tf_diff_axis1(probs)
    cls = tf.math.argmax(dens, axis = 1) # predicted class
    nll = compute_logLik(model, gamma, beta, eta, y)
    return {"cdf": probs.numpy(), "pdf": dens.numpy(), "response": cls.numpy(), "negLogLik": nll.numpy(), 
            "theta": np.delete(theta.numpy(), [0, theta.shape[1]-1], 1), "beta": beta, "eta": eta, "beta_w": beta_w}