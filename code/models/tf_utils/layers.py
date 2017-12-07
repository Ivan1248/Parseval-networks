import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope
from variables import conv_weight_variable, bias_variable

var_scope = variable_scope.variable_scope


def default_arg(func, param):
    import inspect
    return inspect.signature(func).parameters[param].default


# Linear, affine


def add_biases(x, return_params=False, reuse=False, scope='add_biases'):
    with tf.variable_scope(scope, reuse=reuse):
        b = bias_variable(x.shape[-1].value)
        h = x + b
        return (h, b) if return_params else h


# Convolution


def conv(x,
         ksize,
         width,
         stride=1,
         dilation=1,
         padding='SAME',
         bias=True,
         reuse: bool = None,
         scope: str = None):
    '''
    A wrapper for tf.nn.conv2d.
    :param x: 4D input "NHWC" tensor 
    :param ksize: int or tuple of 2 ints representing spatial dimensions 
    :param width: number of output channels
    :param stride: int (or float in case of transposed convolution)
    :param dilation: dilation of the kernel
    :param padding: string. "SAME" or "VALID" 
    :param bias: add biases
    :param reuse: variable scope reuse flag
    :param scope: variable scope name
    '''
    with var_scope(scope, 'Conv', [x], reuse=reuse):
        h = tf.nn.convolution(
            x,
            filter=conv_weight_variable(ksize, x.shape[-1].value, width),
            strides=[stride] * 2,
            dilation_rate=[dilation] * 2,
            padding=padding)
        if bias:
            h += bias_variable(width)
        return h


def conv_transp(x,
                ksize,
                width,
                output_shape,
                stride=1,                
                padding='SAME',
                bias=True,
                reuse: bool = None,
                scope: str = None):
    '''
    A wrapper for tf.nn.tf.nn.conv2d_transpose.
    :param x: 4D input "NHWC" tensor 
    :param ksize: int or tuple of 2 ints representing spatial dimensions 
    :param width: number of output channels
    :param output_shape: tuple of 2 ints
    :param padding: string. "SAME" or "VALID" 
    :param bias: add biases
    :param reuse: variable scope reuse flag
    :param scope: variable scope name
    '''
    with var_scope(scope, 'ConvT', [x], reuse=reuse):
        tf.nn.conv2d_transpose(
            x,
            filter=conv_weight_variable(ksize, x.shape[-1].value, width),
            output_shape=output_shape,
            strides=[stride] * 2,
            padding=padding)
        if bias:
            h += bias_variable(width)
        return h


# Convolution (old)


def conv2d_par(x, w, b=None, stride=1, dilation=1, padding='SAME'):
    # TODO: transposed convolution
    transposed = type(stride) is float
    conv = tf.nn.conv2d
    if transposed:
        stride_inv = 1 / stride
        stride = int(stride_inv + 0.5)
        if abs(stride - stride_inv) > 1e-8:
            raise ValueError(
                "Stride must be an integer or the multiplicative inverse of an integer."
            )
    if transposed:
        x = tf.nn.conv2d_transpose(
            x, filter=w, strides=[stride] * 2, padding=padding)
        if dilation != 1:
            raise ValueError(
                "Dilation not supported for transposed convolution.")
    else:
        x = tf.nn.convolution(
            x,
            filter=w,
            strides=[stride] * 2,
            dilation_rate=[dilation] * 2,
            padding=padding)
    if b is not None:
        x += b
    return x


def conv2d(x,
           ksize,
           width,
           stride=1,
           dilation=1,
           padding='SAME',
           bias=True,
           return_params=False,
           reuse: bool = None,
           scope: str = None):
    with var_scope(scope, 'Conv', [x], reuse=reuse):
        w = conv_weight_variable(ksize, x.shape[-1].value, width)
        params = [w]
        b = None
        if bias:
            b = bias_variable(width)
            params.append(b)
        h = conv2d_par(x, w, b, stride, dilation)
        return (h, params) if return_params else h


# Pooling


def max_pool(x, stride, ksize=None, padding='SAME'):
    if ksize is None: ksize = stride
    return tf.nn.max_pool(x, [1, ksize, ksize, 1], [1, stride, stride, 1],
                          padding)


def avg_pool(x, stride, ksize=None, padding='SAME'):
    if ksize is None: ksize = stride
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


# Special


def batch_normalization(x,
                        is_training,
                        decay=0.99,
                        var_epsilon=1e-4,
                        reuse: bool = None,
                        scope: str = None):
    '''
    Batch normalization that normalizes over all but the last dimension of x.
    :param x: input tensor
    :param is_training: Tensor. training indicator for batch normalization
    :param decay: exponential moving average decay
    :param scope: variable scope name
    '''
    # https://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    with var_scope(scope, 'BN', [x], reuse=reuse):
        shape = [x.shape[-1].value]
        b = tf.get_variable(
            'offset', shape, initializer=tf.constant_initializer(0.0))
        s = tf.get_variable(
            'scale', shape, initializer=tf.constant_initializer(1.0))
        m, v = tf.nn.moments(
            x, axes=[i for i in range(len(x.shape) - 1)], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay)

        def mean_var_with_update():
            with tf.control_dependencies([ema.apply([m, v])]):
                return tf.identity(m), tf.identity(v)

        mean, var = tf.cond(is_training, mean_var_with_update,
                            lambda: (ema.average(m), ema.average(v)))
        return tf.nn.batch_normalization(x, mean, var, b, s, var_epsilon)


# Blocks


def bn_relu(x,
            is_training,
            decay=default_arg(batch_normalization, 'decay'),
            reuse: bool = None,
            scope: str = None):
    ''' Batch normalization followed by ReLU. '''
    with var_scope(scope, 'BNReLU', [x], reuse=reuse):
        x = batch_normalization(x, is_training)
        return tf.nn.relu(x)


def residual_block(x,
                   is_training,
                   kind=[3, 3],
                   double_dim=False,
                   first_block=False,
                   bn_decay=default_arg(bn_relu, 'decay'),
                   reuse: bool = None,
                   scope: str = None):
    '''
    A ResNet full pre-activation residual block.
    :param x: input tensor
    :param is_training: Tensor. training indicator for batch normalization
    :param kind: kernel sizes of convolutional layers
    :param double_dim: if True, stride of conv1 is set to 2 and width is doubled
    :param first_block: needs to be set to True if this is the first residual 
        block of the whole ResNet, the first activation is omitted
    :param bn_decay: batch normalization exponential moving average decay
    :param reuse: bool. variables defined in scope should be reused
    :param scope: variable scope name
    '''
    with var_scope(scope, 'ResBlock', [x], reuse=reuse):
        x_width = x.shape[-1].value
        width = x_width * 2 if double_dim else x_width

        r = x
        if not first_block:
            r = bn_relu(r, is_training, decay=bn_decay, scope='bn_relu0')
        stride = 2 if double_dim else 1
        r = conv(r, kind[0], width, stride=stride, bias=False, scope='conv0')
        for i, ksize in zip(range(1, len(kind)), kind[1:]):
            r = bn_relu(
                r, is_training, decay=bn_decay, scope='bn_relu' + str(i))
            r = conv(r, ksize, width, bias=False, scope='conv' + str(i))

        if double_dim:  # https://github.com/wenxinxu/resnet-in-tensorflow/blob/master/resnet.py
            x = avg_pool(x, 2, padding='VALID')  # TODO: check
            x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [x_width // 2] * 2])

        return x + r  # the last block needs to be followed by bn_relu


# Components


def resnet(x,
           is_training,
           first_layer_width,
           group_lengths,
           block_kind=[3, 3],
           bn_decay=default_arg(bn_relu, 'decay'),
           custom_block=None,
           include_global_pooling=False,
           reuse: bool = None,
           scope: str = None):
    '''
    A pre-activation resnet without final global pooling and 
    classification layers.
    :param x: input tensor
    :param is_training: Tensor. training indicator for batch normalization
    :param first_layer_width: number of output channels of the first layer
    :param group_lengths: numbers of blocks per group (width)
    :param block_kind: kernel sizes of convolutional layers in a block
    :param bn_decay: batch normalization exponential moving average decay
    :param custom_block: a function with the same signature as residual_block 
    :param reuse: bool. variables defined in scope should be reused
    :param scope: variable scope name
    '''
    block = residual_block if custom_block is None else residual_block
    _bn_relu = lambda h, s: bn_relu(h, is_training, bn_decay, reuse=reuse, scope=s)
    with var_scope(scope, 'ResNet', [x], reuse=reuse):
        h = conv(x, 3, first_layer_width, bias=False)
        h = _bn_relu(h, 'BNReLU0')
        for i, length in enumerate(group_lengths):
            with tf.variable_scope('group' + str(i), reuse=False):
                for j in range(length):
                    h = block(
                        h,
                        is_training=is_training,
                        kind=block_kind,
                        double_dim=i != 0 and j == 0,
                        first_block=i == 0 and j == 0,
                        bn_decay=bn_decay,
                        reuse=reuse,
                        scope='block' + str(j))
        return _bn_relu(h, 'BNReLU1')
