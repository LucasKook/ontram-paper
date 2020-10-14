import tensorflow as tf

def tf_diff_axis1(x):
    return x[:, 1:] - x[:, :-1]

def gamma_to_theta(gamma):
    rgamma = gamma.shape[0] # number of observations
    cgamma = gamma.shape[1] # number of classes
    theta_0 = tf.constant(float("-inf"), shape = (rgamma, 1))
    theta_1 = tf.reshape(gamma[:, 0], shape = (rgamma, 1))
    theta_K = tf.constant(float("inf"), shape = (rgamma, 1))
    theta_k = tf.math.exp(gamma[:, 1:])
    theta_k = tf.math.cumsum(theta_k, axis = 1)
    thetas = tf.concat([theta_0, theta_1, theta_1 + theta_k, theta_K], axis = 1)
    return thetas