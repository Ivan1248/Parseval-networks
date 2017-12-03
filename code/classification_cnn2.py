import tensorflow as tf
from abstract_classification_cnn import AbstractClassificationCNN


class ClassificationCNN2(AbstractClassificationCNN):
    def __init__(self,
                 input_shape,
                 num_classes,
                 weight_decay,
                 lr_policy,
                 batch_size,
                 max_epochs,
                 save_dir,
                 ortho_penalty=0,
                 use_multiclass_hinge_loss=False):
        self.ortho_penalty = ortho_penalty
        self.use_multiclass_hinge_loss = use_multiclass_hinge_loss
        super().__init__(input_shape, num_classes, weight_decay, lr_policy,
                         batch_size, max_epochs, save_dir)

    def build_model(self, input_shape, num_classes, weight_decay, lr_policy):
        from tensorflow.contrib import layers
        from tf_utils import multiclass_hinge_loss

        def get_ortho_penalty():
            vars = tf.contrib.framework.get_variables('')
            filt = lambda x: 'conv' in x.name and 'weights' in x.name
            weight_vars = list(filter(filt, vars))
            loss = tf.constant(0.0)
            for v in weight_vars:
                m = tf.reshape(v, (-1, v.shape[3].value))
                d = tf.matmul(
                    m, m, True) - tf.eye(v.shape[3].value) / v.shape[3].value
                loss += tf.reduce_sum(d**2)
            return loss

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

        mhl = lambda la, lo: 0.1 * multiclass_hinge_loss(la, lo)
        sce = tf.losses.softmax_cross_entropy
        loss = (mhl if self.use_multiclass_hinge_loss else sce)(labels, logits)
        loss = loss + tf.reduce_sum(
            tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        if self.ortho_penalty > 0:
            loss += self.ortho_penalty * get_ortho_penalty()
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_step = optimizer.minimize(loss)

        return AbstractClassificationCNN.Nodes(input, labels, increment_epoch,
                                               logits, probs, loss,
                                               training_step, learning_rate)


if __name__ == "__main__":
    import os
    from data import load_cifar10, shuffle, split, normalize, dense_to_one_hot

    DATA_DIR = 'D:\datasets\cifar-10-batches-py'
    SAVE_DIR = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "out/cifar10")

    class_count = 10
    print("Loading data...")
    train_x, train_y, test_x, test_y = load_cifar10(DATA_DIR)
    print("Preparing data...")
    valid_size = 5000
    train_x, train_y = shuffle(train_x, train_y)
    valid_x, valid_y, train_x, train_y = split(train_x, train_y, valid_size)
    train_x, valid_x, test_x = normalize(train_x, valid_x, test_x)
    ys = [train_y, valid_y, test_y]
    train_y, valid_y, test_y = [dense_to_one_hot(y, class_count) for y in ys]

    print("Preparing CNN...")
    nn = ClassificationCNN2(
        train_x[0].shape,
        num_classes=class_count,
        weight_decay=0.01,
        ortho_penalty=0,
        lr_policy={
            'boundaries': [3, 5, 7],
            'values': [10**-i for i in range(1, 5)]
        },
        batch_size=50,
        max_epochs=8,
        save_dir=SAVE_DIR,
        use_multiclass_hinge_loss=True)

    print("Training...")
    nn.train(train_x, train_y, valid_x, valid_y)
    print("Testing...")
    nn.evaluate("Test", test_x, test_y) 
