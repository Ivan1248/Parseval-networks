import tensorflow as tf


def l2_regularization(weight_vars):
    return tf.reduce_sum(list(map(tf.nn.l2_loss, weight_vars)))


def orthogonality_penalty(weights):
    """
    A loss that penalizes non-orthogonal matrices. 
    :param weights: a single convolution kernel with shape 
        [ksize, ksize, in_channels, out_channels]
    """
    ortho_penalty = tf.constant(0.0)
    m = tf.reshape(v, (-1, weights.shape[3].value))
    return tf.matmul(m, m, transpose_a=True) - tf.eye(weights.shape[3].value)


def orthogonality_penalty(weight_vars, ord='fro'):
    """
    A loss that penalizes non-orthogonal matrices with an operator norm 
    (defined with ord) of (weight_vars.T @ weight_vars - I). The norm is squared
    in case of Frobenius or 2 norm.
    :param weights: a list of convolutional kernel weights with shape 
        [ksize, ksize, in_channels, out_channels]
    :param ord: operator norm. see ord parameter in 
        https://www.tensorflow.org/api_docs/python/tf/norm
    """
    I = tf.eye(v.shape[3].value)

    def get_loss(w):
        m = tf.reshape(w, (-1, w.shape[3].value))
        d = tf.norm(tf.matmul(m, m, transpose_a=True) - I, ord)
        if ord in ['fro', 2]: d = d**2
        return tf.reduce_sum(d)

    return tf.add_n(list(map(get_loss, weight_vars)))
