import numpy as np
import tensorflow as tf
from variables import conv_weight_variable, bias_variable


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
           return_params=False,
           reuse=False,
           scope='conv'):
    with tf.variable_scope(scope, reuse=reuse):
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

pppp = 0


def batch_normalization(x,
                        is_training,
                        decay=0.99,
                        var_epsilon=1e-3,
                        scope='bn'):
    """ Batch normalization """
    # If there is something strange in training, this is probably the reason.
    # TODO: investigate growing variance
    # https://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    with tf.variable_scope(scope, reuse=False):
        shape = [x.shape[-1].value]
        b = tf.Variable(tf.constant(0.0, shape=shape), name='offset')
        s = tf.Variable(tf.constant(1.0, shape=shape), name='scale')

        m, v = tf.nn.moments(
            x, axes=[i for i in range(len(x.shape) - 1)], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay)

        def mean_var_with_update():
            with tf.control_dependencies([ema.apply([m, v])]):
                return tf.identity(m), tf.identity(v)
                #avg_mean, avg_var = ema.average(m), ema.average(v)
                #print(avg_mean.name) #scope name doubled bug
                #return tf.identity(avg_mean), tf.identity(avg_var) # Ne radi dobro. Zašto?

        mean, var = tf.cond(is_training, mean_var_with_update,
                            lambda: (ema.average(m), ema.average(v)))

        global pppp
        if pppp == 0:
            pppp = 1
            #x = tf.Print(x, [ema.average(v)[0], v[0]])

        return tf.nn.batch_normalization(x, mean, var, b, s, var_epsilon)


# Blocks


def bn_relu(x,
            is_training,
            decay=default_arg(batch_normalization, 'decay'),
            scope='bn_relu'):
    with tf.variable_scope(scope):
        x = batch_normalization(x, is_training)
        return tf.nn.relu(x)


def residual_block(x,
                   is_training,
                   kind=[3, 3],
                   double_dim=False,
                   first_block=False,
                   bn_decay=default_arg(bn_relu, 'decay'),
                   reuse=False,
                   scope='resblock'):  # TODO return_params
    '''
    A ResNet full pre-activation residual block (bn->relu->conv, https://arxiv.org/abs/1603.05027).
    :param x: Tensor
    :param kind: kernel sizes of convolutional layers
    :param double_dim: if True, stride of conv1 is set to 2 and width is doubled
    :param first_block: if this is the first residual block of the whole network
    :param is_training: Tensor. training indicator for batch normalization
    :param return_params: bool
    :return: Tensor.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        x_width = x.shape[-1].value
        width = x_width * 2 if double_dim else x_width

        r = x
        if not first_block:
            r = bn_relu(r, is_training, decay=bn_decay, scope='bn_relu0')
        stride = 2 if double_dim else 1
        r = conv2d(r, kind[0], width, stride=stride, bias=False, scope='conv0')
        for i, ksize in zip(range(1, len(kind)) ,kind[1:]):
            r = bn_relu(
                r, is_training, decay=bn_decay, scope='bn_relu' + str(i))
            r = conv2d(r, ksize, width, bias=False, scope='conv' + str(i))

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
           bn_decay=default_arg(residual_block, 'bn_decay'),
           scope='resnet'):
    '''
    A whole pre-activation resnet without final global pooling and classification layers.
    :param x: Tensor
    :param first_layer_width: number of channels of the first layer
    :param group_lengths: numbers of blocks per group (with equal widths)
    :param block_kind: kernel sizes of convolutional layers in a block
    :param is_training: Tensor. training indicator for batch normalization
    :return: Tensor.
    '''
    with tf.variable_scope(scope, reuse=False):
        h = conv2d(x, 3, first_layer_width, bias=False)
        h = bn_relu(h, is_training, decay=bn_decay)
        for i, length in enumerate(group_lengths):
            with tf.variable_scope('group' + str(i), reuse=False):
                for j in range(length):
                    h = residual_block(
                        h,
                        is_training=is_training,
                        kind=block_kind,
                        double_dim=i != 0 and j == 0,
                        first_block=i == 0 and j == 0,
                        bn_decay=bn_decay,
                        scope='block' + str(j))
        return bn_relu(h, is_training, decay=bn_decay)
