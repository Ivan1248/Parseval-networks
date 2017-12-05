import numpy as np
import tensorflow as tf


def conv_weight_variable(size: int, in_channels: int, out_channels: int):
    shape = [size, size, in_channels, out_channels]
    return tf.Variable(
        tf.truncated_normal(shape, stddev=0.05, mean=0), name='weights')


def bias_variable(n: int, initial_value=0.05):
    return tf.Variable(tf.constant(initial_value, shape=[n]), name='biases')