import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope
from variables import conv_weight_variable, bias_variable
from layers import *

var_scope = variable_scope.variable_scope

def rbf_conv(x,
             ksize,
             width,
             stride=1,
             dilation=1,
             padding='SAME',
             bias=True,
             reuse=None,
             scope=None):

    in_channels = x.shape[-1].value
    w_const = tf.constant(
        1 / (in_channels * ksize**2), shape=[ksize, ksize, in_channels, 1])

    with var_scope(scope, 'RBFConv', [x], reuse=reuse):
        xx = tf.nn.convolution(
            x**2,
            filter=w_const,
            strides=[stride] * 2,
            dilation_rate=[dilation] * 2,
            padding=padding)
        w = conv_weight_variable(ksize, x.shape[-1].value, width)
        xw = tf.nn.convolution(
            x,
            filter=w,
            strides=[stride] * 2,
            dilation_rate=[dilation] * 2,
            padding=padding)
        ww = tf.reduce_sum(w**2, [0, 1, 2])
        h = xx - 2 * xw + ww
        h = tf.exp(-h)
        if bias:
            h += bias_variable(width)
        return h


def rbf_conv_cw(x,
                ksize,
                width,
                stride=1,
                dilation=1,
                padding='SAME',
                bias=True,
                reuse=None,
                scope=None):

    in_channels = x.shape[-1].value

    w_const = tf.constant(ksize**-2, shape=[ksize, ksize, in_channels, 1])

    with var_scope(scope, 'RBFConvCW', [x], reuse=reuse):
        xx = tf.nn.depthwise_conv2d(
            x**2,
            filter=w_const,
            strides=[1] + [stride] * 2 + [1],
            padding=padding)
        channels = []
        for i in range(width):
            w = conv_weight_variable(ksize, in_channels, 1, 'wspace' + str(i))
            xw = tf.nn.depthwise_conv2d(
                x, filter=w, strides=[1] + [stride] * 2 + [1], padding=padding)
            ww = tf.reshape(tf.reduce_sum(w**2, [0, 1]), [-1])
            h = xx - 2 * xw + ww
            g = tf.get_variable(
                'g' + str(i), shape=[1], initializer=tf.constant_initializer(1))
            h = tf.exp(-(1 + tf.abs(g)) * h)
            c = tf.nn.convolution(
                h,
                filter=conv_weight_variable(1, in_channels, 1,
                                            'wchan' + str(i)),
                padding=padding)
            channels += [c]
        h = tf.concat(channels, axis=3)
        if bias:
            h += bias_variable(width)
        return h


def rbf_residual_block(x,
                       is_training,
                       properties=ResidualBlockProperties([3, 3]),
                       width=16,
                       first_block=False,
                       bn_decay=default_arg(batch_normalization, 'decay'),
                       reuse: bool = None,
                       scope: str = None):
    '''
    A generic ResNet full pre-activation residual block.
    :param x: input tensor
    :param is_training: Tensor. training indicator for batch normalization    
    :param properties: a ResidualBlockProperties instance
    :param width: number of output channels
    :param bn_decay: batch normalization exponential moving average decay
    :param first_block: needs to be set to True if this is the first residual 
        block of the whole ResNet, the first activation is omitted
    '''
    dropout = tf.layers.dropout

    def _bn(x, name):
        return batch_normalization(
            x, is_training, decay=bn_decay, reuse=reuse, scope='bn' + name)

    def _conv(x, ksize, stride, name):
        sc = 'conv' + name
        return rbf_conv_cw(
            x, ksize, width, stride, bias=False, reuse=reuse, scope=sc)

    x_width = x.shape[-1].value
    dim_increase = width > x_width
    with var_scope(scope, 'RBFResBlock', [x], reuse=reuse):
        r = x
        for i, ksize in enumerate(properties.ksizes):
            r = _bn(r, str(i)) if not (first_block and i == 0) else r
            r = _conv(r, ksize, 1 + int(dim_increase and i == 0), str(i))
            if i in properties.dropout_locations:
                r = dropout(r, properties.dropout_rate, training=is_training)
        if dim_increase:
            if properties.dim_increase == 'id':
                x = avg_pool(x, 2, padding='VALID')
                x = tf.pad(x, 3 * [[0, 0]] + [[0, width - x_width]])
            elif properties.dim_increase in ['conv1', 'conv3']:
                x = _bn(x, 'skip')  # TODO: check
                x = _conv(x, int(properties.dim_increase[-1]), 2, 'skip')
        return x + r


def rbf_resnet(x,
               is_training,
               base_width=16,
               widening_factor=1,
               group_lengths=[2, 2, 2],
               block_properties=default_arg(rbf_residual_block, 'properties'),
               bn_decay=default_arg(batch_normalization, 'decay'),
               reuse: bool = None,
               scope: str = None):
    '''
    A pre-activation resnet without final global pooling and 
    classification layers.
    :param x: input tensor
    :param is_training: Tensor. training indicator for batch normalization
    :param base_width: number of output channels of the first layer
    :param widening_factor: (k) block widths are proportional to 
        base_width*widening_factor
    :param group_lengths: (N) numbers of blocks per group (width)
    :param block_properties: kernel sizes of convolutional layers in a block
    :param bn_decay: batch normalization exponential moving average decay
    :param custom_block: a function with the same signature as residual_block 
    '''

    def _bn(x, name):
        return batch_normalization(
            x, is_training, decay=bn_decay, reuse=reuse, scope='bn' + name)

    with var_scope(scope, 'ResNet', [x], reuse=reuse):
        h = rbf_conv_cw(x, 3, base_width)
        h = _bn(h, 'BNReLU0')
        for i, length in enumerate(group_lengths):
            group_width = base_width * widening_factor * (i + 1)
            with tf.variable_scope('group' + str(i), reuse=False):
                for j in range(length):
                    h = rbf_residual_block(
                        h,
                        is_training=is_training,
                        properties=block_properties,
                        width=group_width,
                        first_block=i == 0 and j == 0,
                        bn_decay=bn_decay,
                        reuse=reuse,
                        scope='block' + str(j))
        return _bn(h, 'BNReLU1')


#conv = rbf_conv_cw