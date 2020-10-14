import tensorflow as tf
from functions.compute_logLik import compute_logLik

def apply_gradient(model, bl, x, x_im, y):
    with tf.GradientTape() as tape:
        if((model.nn_x == None) & (model.nn_im == None)):
            gamma = model.model({"bl_in": bl})
            beta = 0
            eta = 0
        if((model.nn_x != None) & (model.nn_im == None)):
            gamma, beta = model.model({"bl_in": bl, "x_in": x})
            eta = 0
        if((model.nn_x == None) & (model.nn_im != None)):
            gamma, eta = model.model({"bl_in": bl, "im_in": x_im})
            beta = 0
        if((model.nn_x != None) & (model.nn_im != None)):
            gamma, beta, eta = model.model({"bl_in": bl, "x_in": x, "im_in": x_im})
        loss_value = compute_logLik(model, gamma, beta, eta, y)
    return loss_value, tape.gradient(loss_value, model.model.trainable_variables)