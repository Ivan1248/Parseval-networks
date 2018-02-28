import datetime
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.framework import ops

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # /*
from data import Dataset, MiniBatchReader

from .abstract_model import AbstractModel


class QuickNet(AbstractModel):

    def __init__(self,
                 input_shape,
                 class_count,
                 class0_unknown=False,
                 batch_size=128,
                 learning_rate_policy=1e-3,
                 training_log_period=1,
                 name='ClfBaselineA'):
        self.completed_epoch_count = 0
        self.class0_unknown = class0_unknown
        super().__init__(
            input_shape=input_shape,
            class_count=class_count,
            batch_size=batch_size,
            learning_rate_policy=learning_rate_policy,
            training_log_period=training_log_period,
            name=name)

    def _build_graph(self, learning_rate, epoch, is_training):
        from tf_utils.layers import conv, max_pool, rescale_bilinear, avg_pool, bn_relu, separable_conv, var_scope

        input_shape = [None] + list(self.input_shape)
        output_shape = [None, self.class_count]

        # Input image and labels placeholders
        input = tf.placeholder(tf.float32, shape=input_shape)
        target = tf.placeholder(tf.float32, shape=output_shape)

        # Hidden layers
        h = input

        convi = 0

        def sepconv_bn_relu(x, width, scope=None):
            nonlocal convi
            h = separable_conv(
                x, 3, width, bias=False, scope=scope)
            convi += 1
            h = bn_relu(h, is_training)
            return h

        with var_scope(None, 'QuickNet', [h], reuse=False):
            h = sepconv_bn_relu(h, 32, scope="conv1")
            for i in range(3):
                for j in range(2):
                    h = sepconv_bn_relu(h, 128 * 2**i, scope="b"+str(i)+"l"+str(j))
                h = max_pool(h, 2)

            h = tf.layers.dropout(h, 0.5, training=is_training)
            h = conv(h, 1, self.class_count, scope="conv2")

        # Global pooling and softmax classification
        h = tf.reduce_mean(h, axis=[1, 2], keep_dims=True)
        logits = conv(h, 1, self.class_count, scope="conv3")
        logits = tf.reshape(logits, [-1, self.class_count])
        probs = tf.nn.softmax(logits)

        # Loss
        clipped_probs = tf.clip_by_value(probs, 1e-10, 1.0)
        ts = lambda x: x[:, 1:] if self.class0_unknown else x
        loss = -tf.reduce_mean(ts(target) * tf.log(ts(clipped_probs)))

        # Optimization
        optimizer = tf.train.AdamOptimizer(learning_rate)
        training_step = optimizer.minimize(loss)

        # Dense predictions and labels
        preds, dense_labels = tf.argmax(probs, 1), tf.argmax(target, 1)

        # Other evaluation measures
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(preds, dense_labels), tf.float32))

        return AbstractModel.EssentialNodes(
            input=input,
            target=target,
            probs=probs,
            loss=loss,
            training_step=training_step,
            evaluation={'accuracy': accuracy})
