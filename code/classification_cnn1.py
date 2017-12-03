import tensorflow as tf
from abstract_classification_cnn import AbstractClassificationCNN


class ClassificationCNN1(AbstractClassificationCNN):
    def __init__(self, input_shape, num_classes, weight_decay, lr_policy,
                 batch_size, max_epochs, save_dir):
        super().__init__(input_shape, num_classes, weight_decay, lr_policy,
                         batch_size, max_epochs, save_dir)

    def build_model(self, input_shape, num_classes, weight_decay, lr_policy):
        from tensorflow.contrib import layers

        epoch = tf.Variable(1, name='epoch', trainable=False, dtype=tf.int32)
        increment_epoch = tf.assign(epoch, epoch + 1)

        input = tf.placeholder(
            tf.float32, shape=[None] + list(input_shape), name='input')
        labels = tf.placeholder(
            tf.float32, shape=[None, num_classes], name='label')

        weight_decay = tf.constant(weight_decay, dtype=tf.float32)
        learning_rate = tf.train.piecewise_constant(
            epoch,
            boundaries=lr_policy['boundaries'],
            values=lr_policy['values'])

        h = input
        with tf.contrib.framework.arg_scope(
            [layers.conv2d],
                kernel_size=5,
                stride=1,
                data_format='NHWC',
                padding='SAME',
                activation_fn=tf.nn.relu,
                weights_initializer=layers.variance_scaling_initializer(),
                weights_regularizer=layers.l2_regularizer(weight_decay)):
            h = layers.conv2d(h, 16, scope='convrelu1')
            h = layers.max_pool2d(h, 2, 2, scope='pool1')
            h = layers.conv2d(h, 32, scope='convrelu2')
            h = layers.max_pool2d(h, 2, 2, scope='pool2')
        with tf.contrib.framework.arg_scope(
            [layers.fully_connected],
                activation_fn=tf.nn.relu,
                weights_initializer=layers.variance_scaling_initializer(),
                weights_regularizer=layers.l2_regularizer(weight_decay)):
            h = layers.flatten(h, scope='flatten3')
            h = layers.fully_connected(h, 512, scope='fc3')
        logits = layers.fully_connected(
            h, num_classes, activation_fn=None, scope='logits')
        probs = tf.nn.softmax(logits, name='probs')

        ce_loss = tf.losses.softmax_cross_entropy(labels, logits)
        loss = ce_loss + tf.reduce_sum(
            tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_step = optimizer.minimize(loss)

        return AbstractClassificationCNN.Nodes(input, labels, increment_epoch,
                                               logits, probs, loss,
                                               training_step, learning_rate)


if __name__ == "__main__":
    import time
    import os

    import numpy as np
    import tensorflow as tf
    from tensorflow.examples.tutorials.mnist import input_data

    import nn

    import layers

    DATA_DIR = 'D:/datasets/MNIST/'
    SAVE_DIR = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "out/")

    #np.random.seed(100)
    np.random.seed(int(time.time() * 1e6) % 2**31)
    dataset = input_data.read_data_sets(DATA_DIR, one_hot=True)
    train_x = dataset.train.images
    train_x = train_x.reshape([-1, 28, 28, 1])
    train_y = dataset.train.labels
    valid_x = dataset.validation.images
    valid_x = valid_x.reshape([-1, 28, 28, 1])
    valid_y = dataset.validation.labels
    test_x = dataset.test.images
    test_x = test_x.reshape([-1, 28, 28, 1])
    test_y = dataset.test.labels
    train_mean = train_x.mean()
    train_x -= train_mean
    valid_x -= train_mean
    test_x -= train_mean

    nn = ClassificationCNN1(
        train_x[0].shape,
        num_classes=10,
        weight_decay=0.01,
        lr_policy={
            'boundaries': [3, 5, 7],
            'values': [10**-i for i in range(1, 5)]
        },
        batch_size=50,
        max_epochs=8,
        save_dir=SAVE_DIR)
    nn.train(train_x, train_y, valid_x, valid_y)
    nn.evaluate("Test", test_x, test_y)
