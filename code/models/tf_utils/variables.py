import numpy as np
import tensorflow as tf


def conv_weight_variable(ksize, in_channels: int, out_channels: int):
    if type(ksize) is int: ksize = [ksize, ksize]
    shape = list(ksize) + [in_channels, out_channels]
    maxval = (6 / (ksize[0] * ksize[1] * in_channels + out_channels))**0.5
    return tf.get_variable(
        'weights',
        shape=shape,
        initializer=tf.random_uniform_initializer(-maxval, maxval))


def bias_variable(n: int, initial_value=0.05):
    return tf.get_variable(
        'biases',
        shape=[n],
        initializer=tf.constant_initializer(initial_value))
