import re
import hickle as hkl

import tensorflow as tf
import numpy as np

def g(inputs, params):
    '''Bottleneck WRN-50-2 model definition
    '''
    def tr(v):
        if v.ndim == 4:
            return v.transpose(2,3,1,0)
        elif v.ndim == 2:
            return v.transpose()
        return v
    params = {k: tf.constant(tr(v)) for k, v in params.iteritems()}
    
    def conv2d(x, params, name, stride=1, padding=0):
        x = tf.pad(x, [[0,0],[padding,padding],[padding,padding],[0,0]])
        z = tf.nn.conv2d(x, params['%s.weight'%name], [1,stride,stride,1],
                         padding='VALID')
        if '%s.bias'%name in params:
            return tf.nn.bias_add(z, params['%s.bias'%name])
        else:
            return z
    
    def group(input, params, base, stride, n):
        o = input
        for i in range(0,n):
            b_base = ('%s.block%d.conv') % (base, i)
            x = o
            o = conv2d(x, params, b_base + '0')
            o = tf.nn.relu(o)
            o = conv2d(o, params, b_base + '1', stride=i==0 and stride or 1, padding=1)
            o = tf.nn.relu(o)
            o = conv2d(o, params, b_base + '2')
            if i == 0:
                o += conv2d(x, params, b_base + '_dim', stride=stride)
            else:
                o += x
            o = tf.nn.relu(o)
        return o
    
    # determine network size by parameters
    blocks = [sum([re.match('group%d.block\d+.conv0.weight'%j, k) is not None
                   for k in params.keys()]) for j in range(4)]

    o = conv2d(inputs, params, 'conv0', 2, 3)
    o = tf.nn.relu(o)
    o = tf.pad(o, [[0,0], [1,1], [1,1], [0,0]])
    o = tf.nn.max_pool(o, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')
    o_g0 = group(o, params, 'group0', 1, blocks[0])
    o_g1 = group(o_g0, params, 'group1', 2, blocks[1])
    o_g2 = group(o_g1, params, 'group2', 2, blocks[2])
    o_g3 = group(o_g2, params, 'group3', 2, blocks[3])
    o = tf.nn.avg_pool(o_g3, ksize=[1,7,7,1], strides=[1,1,1,1], padding='VALID')
    o = tf.reshape(o, [-1, 2048])
    o = tf.matmul(o, params['fc.weight']) + params['fc.bias']
    return o

params = hkl.load('wide-resnet-50-2-export.hkl')
inputs_tf = tf.placeholder(tf.float32, shape=[None,224,224,3])

out = g(inputs_tf, params)

sess = tf.Session()
y_tf = sess.run(out, feed_dict={inputs_tf: inputs.permute(0,2,3,1).numpy()})

# check that difference between PyTorch and Tensorflow is small
assert np.abs(y_tf - y.data.numpy()).max() < 1e-5