import numpy as np
import tensorflow as tf


def conv_weight_variable(size: int, in_channels: int, out_channels: int):
    shape = [size, size, in_channels, out_channels]
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05, mean=0))


def bias_variable(n: int):
    return tf.Variable(tf.constant(0.05, shape=[n]))
