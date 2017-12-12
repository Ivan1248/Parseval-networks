import datetime
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.framework import ops

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # /*
sys.path.append(os.path.dirname(__file__))  # /models/
from data import Dataset, MiniBatchReader
from ioutil import path

from abstract_model import AbstractModel
from tf_utils import layers


class ResNet(AbstractModel):

    def __init__(self,
                 input_shape,
                 class_count,
                 batch_size=128,
                 learning_rate_policy=1e-2,
                 block_kind=layers.ResidualBlockKind([3, 3]),
                 group_lengths=[3, 3, 3],
                 base_width=16,
                 widening_factor=1,
                 weight_decay=1e-4,
                 training_log_period=1,
                 name='ResNet'):
        self.completed_epoch_count = 0
        self.block_kind = block_kind
        self.group_lengths = group_lengths
        self.depth = 1 + sum(group_lengths) * len(block_kind.ksizes) + 1
        self.zagoruyko_depth = self.depth - 1 + len(group_lengths)
        self.base_width = base_width
        self.widening_factor = widening_factor
        self.weight_decay = weight_decay
        super().__init__(
            input_shape=input_shape,
            class_count=class_count,
            batch_size=batch_size,
            learning_rate_policy=learning_rate_policy,
            training_log_period=training_log_period,
            name=name)

    def _build_graph(self, learning_rate, epoch, is_training):
        from layers import conv, resnet

        # Input image and labels placeholders
        input_shape = [None] + list(self.input_shape)
        output_shape = [None, self.class_count]
        input = tf.placeholder(tf.float32, shape=input_shape)
        target = tf.placeholder(tf.float32, shape=output_shape)

        # Hidden layers
        h = resnet(
            input,
            base_width=self.base_width,
            widening_factor=self.widening_factor,
            group_lengths=self.group_lengths,
            is_training=is_training)

        # Global pooling and softmax classification
        h = tf.reduce_mean(h, axis=[1, 2], keep_dims=True)
        logits = conv(h, 1, self.class_count)
        logits = tf.reshape(logits, [-1, self.class_count])
        probs = tf.nn.softmax(logits)

        # Loss
        clipped_probs = tf.clip_by_value(probs, 1e-10, 1.0)
        loss = -tf.reduce_mean(target * tf.log(clipped_probs))

        # Regularization
        vars = tf.global_variables()
        weight_vars = filter(lambda x: 'weights' in x.name, vars)
        l2reg = tf.reduce_sum(list(map(tf.nn.l2_loss, weight_vars)))
        loss += self.weight_decay * l2reg

        # Optimization
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
        training_step = optimizer.minimize(loss)

        # Dense predictions and labels
        preds, dense_labels = tf.argmax(probs, 1), tf.argmax(target, 1)

        # Other evaluation measures
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(preds, dense_labels), tf.float32))

        #writer = tf.summary.FileWriter('logs', self._sess.graph)

        return AbstractModel.EssentialNodes(
            input=input,
            target=target,
            probs=probs,
            loss=loss,
            training_step=training_step,
            evaluation={
                'accuracy': accuracy
            })


def main(epoch_count=1):
    from data import Dataset, loaders
    from data.preparers import Iccv09Preparer
    from processing.labels import dense_to_one_hot
    import ioutil
    from ioutil import console

    print("Loading and deterministically shuffling data...")
    data_path = ioutil.path.find_ancestor_sibling(
        __file__, 'projects/datasets/cifar-10-batches-py')
    ds = loaders.load_cifar10(data_path, 'train')
    from processing.preprocessing import get_normalization_statistics, normalize
    mean, std = get_normalization_statistics(ds.images)
    ds = Dataset(normalize(ds.images, mean, std), ds.labels, ds.class_count)
    ds.shuffle()
    ds_train, ds_val = ds.split(0, int(ds.size * 0.8))

    print("Initializing model...")
    n, k = 16, 4  # n-number of weights layers, k-widening factor, (16,4), (28,10)
    block_kind = layers.ResidualBlockKind(
        ksizes=[3, 3],
        dropout_locations=[0],
        dropout_rate=0.3,
        dim_increase='conv1')
    group_count = 3
    blocks_per_group = (n - 4) // (group_count * len(block_kind.ksizes))
    print("group count: {}, blocks per group: {}".format(
        group_count, blocks_per_group))
    model = ResNet(
        input_shape=ds.image_shape,
        class_count=ds.class_count,
        batch_size=128,
        learning_rate_policy={
            'boundaries': [60, 120, 160],
            'values': [1e-1 * 0.2**i for i in range(4)]
        },
        block_kind=block_kind,
        group_lengths=[blocks_per_group] * group_count,
        widening_factor=k,
        weight_decay=5e-4,
        training_log_period=50)
    assert n == model.zagoruyko_depth, "invalid depth (n={}!={})".format(
        n, model.zagoruyko_depth)

    def handle_step(i):
        text = console.read_line(impatient=True, discard_non_last=True)
        if text == 'q':
            return True
        if text == 's':
            writer_path = path.find_ancestor_sibling(__file__, 'data/logs')
            writer = tf.summary.FileWriter(writer_path, graph=model._sess.graph)
        return False

    model.training_step_event_handler = handle_step

    print("Starting training and validation loop...")
    from processing.data_augmentation import augment_cifar

    #model.test(ds_val)
    ds_train_part = ds_train[:ds_val.size]
    for i in range(epoch_count):
        prepr_ds_train = Dataset(
            list(map(augment_cifar, ds_train.images)), ds_train.labels,
            ds_train.class_count)
        model.train(prepr_ds_train, epoch_count=1)
        model.test(ds_val, 'validation data')
        model.test(ds_train_part, 'training data subset')
    save_path = path.find_ancestor_sibling(__file__, 'data/models')
    import datetime
    model.save_state(save_path + '/wrn-%d-%d.' % (n,k) +
                     datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))


if __name__ == '__main__':
    main(epoch_count=200)

# "GTX 970" 43 times faster than "Pentium 2020M @ 2.40GHz × 2"
