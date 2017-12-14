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
    '''
    # https://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    with var_scope(scope, 'BN', [x], reuse=reuse):
        m, v = tf.nn.moments(
            x, axes=list(range(len(x.shape) - 1)), name='moments')
        ema = tf.train.ExponentialMovingAverage(decay)

        def mean_var_with_update():
            with tf.control_dependencies([ema.apply([m, v])]):
                return tf.identity(m), tf.identity(v)

        mean, var = tf.cond(is_training, mean_var_with_update,
                            lambda: (ema.average(m), ema.average(v)))
        offs, scal = [
            tf.get_variable(
                name,
                shape=[x.shape[-1].value],
                initializer=tf.constant_initializer(val))
            for name, val in [('offset', 0.0), ('scale', 1.0)]
        ]
        return tf.nn.batch_normalization(x, mean, var, offs, scal, var_epsilon)


# Blocks


def bn_relu(x,
            is_training,
            decay=default_arg(batch_normalization, 'decay'),
            reuse: bool = None,
            scope: str = None):
    with var_scope(scope, 'BNReLU', [x], reuse=reuse):
        x = batch_normalization(x, is_training)
        return tf.nn.relu(x)


class ResidualBlockKind:

    def __init__(self,
                 ksizes=[3, 3],
                 dropout_locations=[],
                 dropout_rate=0.3,
                 dim_increase='id'):
        '''
        A ResNet full pre-activation residual block with a (padded) identity 
        shortcut.
        :param ksizes: list of ints or int pairs. kernel sizes of convolutional 
            layers
        :param dropout_locations: indexes of convolutional layers after which
            dropout is to be applied
        :param dropout_rate: int
        :param dim_increase: string. 'id' for identity with zero padding or 
            'conv1' for projection with a 1×1 convolution with stride 2. See 
            section 3.3. in https://arxiv.org/abs/1512.03385. Alternatively, it 
            can also be set to 'conv3' for a 3×3 convolution with stride 2. 
        '''
        self.ksizes = ksizes
        self.dropout_locations = dropout_locations
        self.dropout_rate = dropout_rate
        if dim_increase not in ['id', 'conv1', 'conv3']:
            raise ValueError("dim_increase must be 'id', 'conv1' or 'conv'")
        self.dim_increase = dim_increase


def residual_block(x,
                   is_training,
                   kind=ResidualBlockKind([3, 3]),
                   width=16,
                   first_block=False,
                   bn_decay=default_arg(batch_normalization, 'decay'),
                   reuse: bool = None,
                   scope: str = None):
    '''
    A generic ResNet full pre-activation residual block.
    :param x: input tensor
    :param is_training: Tensor. training indicator for batch normalization    
    :param kind: a ResidualBlockKind instance
    :param width: number of output channels
    :param bn_decay: batch normalization exponential moving average decay
    :param first_block: needs to be set to True if this is the first residual 
        block of the whole ResNet, the first activation is omitted
    '''
    dropout = tf.layers.dropout

    def _bn_relu(x, name):
        sc = 'bn_relu' + name
        return bn_relu(x, is_training, decay=bn_decay, reuse=reuse, scope=sc)

    def _conv(x, ksize, stride, name):
        sc = 'conv' + name
        return conv(x, ksize, width, stride, bias=False, reuse=reuse, scope=sc)

    x_width = x.shape[-1].value
    dim_increase = width > x_width
    with var_scope(scope, 'ResBlock', [x], reuse=reuse):
        r = x
        for i, ksize in enumerate(kind.ksizes):
            r = _bn_relu(r, str(i)) if not (first_block and i == 0) else r
            r = _conv(r, ksize, 1 + int(dim_increase and i == 0), str(i))
            if i in kind.dropout_locations:
                r = dropout(r, kind.dropout_rate, training=is_training)
        if dim_increase:
            if kind.dim_increase == 'id':
                x = avg_pool(x, 2, padding='VALID')
                x = tf.pad(x, 3 * [[0, 0]] + [[0, width - x_width]])
            elif kind.dim_increase in ['conv1', 'conv3']:
                x = _bn_relu(x, 'skip')  # TODO: check
                x = _conv(x, int(kind.dim_increase[-1]), 2, 'skip')
        return x + r


# Components


def resnet(x,
           is_training,
           base_width=16,
           widening_factor=1,
           group_lengths=[2, 2, 2],
           block_kind=default_arg(residual_block, 'kind'),
           bn_decay=default_arg(batch_normalization, 'decay'),
           custom_block=None,
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
    :param block_kind: kernel sizes of convolutional layers in a block
    :param bn_decay: batch normalization exponential moving average decay
    :param custom_block: a function with the same signature as residual_block 
    '''
    block = residual_block if custom_block is None else custom_block
    _bn_relu = lambda h, s: bn_relu(h, is_training, bn_decay, reuse=reuse, scope=s)
    with var_scope(scope, 'ResNet', [x], reuse=reuse):
        h = conv(x, 3, base_width, bias=False)
        h = _bn_relu(h, 'BNReLU0')
        for i, length in enumerate(group_lengths):
            group_width = base_width * widening_factor * (i + 1)
            with tf.variable_scope('group' + str(i), reuse=False):
                for j in range(length):
                    h = block(
                        h,
                        is_training=is_training,
                        kind=block_kind,
                        width=group_width,
                        first_block=i == 0 and j == 0,
                        bn_decay=bn_decay,
                        reuse=reuse,
                        scope='block' + str(j))
        return _bn_relu(h, 'BNReLU1')


# Experimental


def rbf_conv(x,
             ksize,
             width,
             stride=1,
             dilation=1,
             padding='SAME',
             bias=True,
             reuse=None,
             scope=None):

    w_ones = tf.constant(1.0, shape=[ksize, ksize, in_channels, 1])

    with var_scope(scope, 'RBFConv', [x], reuse=reuse):
        xx = tf.nn.convolution(
            x**2,
            filter=w_ones,
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

    w_ones = tf.constant(1.0, shape=[ksize, ksize, in_channels, 1])

    with var_scope(scope, 'RBFConvCW', [x], reuse=reuse):
        xx = tf.nn.depthwise_conv2d(
            x**2,
            filter=w_ones,
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
                       kind=ResidualBlockKind([3, 3]),
                       width=16,
                       first_block=False,
                       bn_decay=default_arg(batch_normalization, 'decay'),
                       reuse: bool = None,
                       scope: str = None):
    '''
    A generic ResNet full pre-activation residual block.
    :param x: input tensor
    :param is_training: Tensor. training indicator for batch normalization    
    :param kind: a ResidualBlockKind instance
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
        for i, ksize in enumerate(kind.ksizes):
            r = _bn(r, str(i)) if not (first_block and i == 0) else r
            r = _conv(r, ksize, 1 + int(dim_increase and i == 0), str(i))
            if i in kind.dropout_locations:
                r = dropout(r, kind.dropout_rate, training=is_training)
        if dim_increase:
            if kind.dim_increase == 'id':
                x = avg_pool(x, 2, padding='VALID')
                x = tf.pad(x, 3 * [[0, 0]] + [[0, width - x_width]])
            elif kind.dim_increase in ['conv1', 'conv3']:
                x = _bn(x, 'skip')  # TODO: check
                x = _conv(x, int(kind.dim_increase[-1]), 2, 'skip')
        return x + r


def rbf_resnet(x,
               is_training,
               base_width=16,
               widening_factor=1,
               group_lengths=[2, 2, 2],
               block_kind=default_arg(rbf_residual_block, 'kind'),
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
    :param block_kind: kernel sizes of convolutional layers in a block
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
                        kind=block_kind,
                        width=group_width,
                        first_block=i == 0 and j == 0,
                        bn_decay=bn_decay,
                        reuse=reuse,
                        scope='block' + str(j))
        return _bn(h, 'BNReLU1')


#conv = rbf_conv_cw