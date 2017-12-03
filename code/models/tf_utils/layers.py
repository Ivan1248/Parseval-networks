import numpy as np
import tensorflow as tf
from variables import conv_weight_variable, bias_variable

# Linear, affine


def add_bias(x, return_params=False):
    b = bias_variable(x.shape[3].value)
    h = x + b
    return (h, b) if return_params else h


# Convolution


def conv2d_par(x, w, b=None, stride=1, dilation=1, padding='SAME'):
    s, d = [stride] * 2, [dilation] * 2
    h = tf.nn.convolution(
        input=x, filter=w, strides=s, dilation_rate=d, padding=padding)
    if b is not None:
        h += b
    return h


def conv2d(x,
           ksize,
           width,
           stride=1,
           dilation=1,
           padding='SAME',
           bias=True,
           return_params=False):
    w = conv_weight_variable(ksize, x.shape[3].value, width)
    params = [w]
    b = None
    if bias:
        b = bias_variable(width)
        params.append(b)
    h = conv2d_par(x, w, b, stride, dilation)
    return (h, params) if return_params else h


# Pooling


def max_pool(x, stride, ksize=None, padding='SAME'):
    if ksize is None:
        ksize = stride
    return tf.nn.max_pool(x, [1, ksize, ksize, 1], [1, stride, stride, 1],
                          padding)


def avg_pool(x, stride, ksize=None, padding='SAME'):
    if ksize is None:
        ksize = stride
    return tf.nn.avg_pool(x, [1, ksize, ksize, 1], [1, stride, stride, 1],
                          padding)


# Rescaling


def _get_rescaled_shape(x, factor):
    return (np.array([d.value
                      for d in x.shape[1:3]]) * factor + 0.5).astype(np.int)


def rescale_nearest_neighbor(x, factor):
    shape = _get_rescaled_shape(x, factor)
    return x if factor == 1 else tf.image.resize_nearest_neighbor(x, shape)


def rescale_bilinear(x, factor):
    shape = _get_rescaled_shape(x, factor)
    return x if factor == 1 else tf.image.resize_bilinear(x, shape)


# Losses

# Special


def batch_normalization(x, is_training=False):
    # TODO: there is something cstrange with the training parameter, check it
    return tf.layers.batch_normalization(
        x, -1, training=is_training, fused=True)


# Blocks


def bn_relu(x, is_training=False, return_params=False):  # TODO
    x = batch_normalization(x, is_training)
    return tf.nn.relu(x)


def residual_block(x,
                   kind=[3, 3],
                   double_dim=False,
                   first_layer=False,
                   is_training=False,
                   return_params=False):  # TODO return_params
    '''
    A ResNet full pre-activation residual block (bn->relu->conv, https://arxiv.org/abs/1603.05027).
    :param x: Tensor
    :param kind: kernel sizes of convolutional layers
    :param double_dim: if True, stride of conv1 is set to 2 and width is doubled
    :param first_layer: if this is the first residual block of the whole network
    :param is_training: Tensor. training indicator for batch normalization
    :param return_params: bool
    :return: Tensor.
    '''
    x_width = x.shape[3].value
    if double_dim is True:  # https://github.com/wenxinxu/resnet-in-tensorflow/blob/master/resnet.py
        x = avg_pool(x, 2, padding='VALID')
        x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [x_width // 2] * 2])

    width = x_width * 2 if double_dim else x_width
    if not first_layer:
        r = bn_relu(x, is_training=is_training)
    r = conv2d(x, kind[0], width, stride=2 if double_dim else 1, bias=False)
    for ksize in kind[1:]:
        r = bn_relu(x, is_training=is_training)
        r = conv2d(x, ksize, width, bias=False)

    return x + r


# Network components


def resnet(x,
           first_layer_width,
           group_lengths,
           block_kind=[3, 3],
           is_training=False):
    '''
    A whole resnet without final global pooling and classification layers.
    :param x: Tensor
    :param first_layer_width: number of channels of the first layer
    :param group_lengths: numbers of blocks per group (with same width)
    :param block_kind: kernel sizes of convolutional layers in a block
    :param is_training: Tensor. training indicator for batch normalization
    :return: Tensor.
    '''
    h = conv2d(x, 3, first_layer_width, bias=False)
    h = bn_relu(h, is_training=is_training)
    h = residual_block(
        h, block_kind, first_layer=True, is_training=is_training)
    for _ in range(group_lengths[0] - 1):  # first group
        h = residual_block(h, block_kind, is_training=is_training)
    for l in group_lengths[1:]:
        h = residual_block(
            h, block_kind, double_dim=True, is_training=is_training)
        for _ in range(l - 1):
            h = residual_block(h, block_kind, is_training=is_training)
    h = add_bias(h)
    h = tf.nn.relu(h)
    return h