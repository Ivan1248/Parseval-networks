import abc
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import os

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # semseg/*
import data
from data import Dataset, MiniBatchReader
from processing.labels import one_hot_to_dense, dense_to_one_hot
from ioutil import file


class AbstractModel(object):
    class EssentialNodes:
        def __init__(self,
                     input,
                     target,
                     probs,
                     loss,
                     training_step,
                     is_training=None):
            self.input = input
            self.target = target
            self.probs = probs
            self.loss = loss
            self.training_step = training_step
            self.is_training = is_training

    def __init__(
            self,
            input_shape,  # [width, height, number of channels], maybe [None, None, number of channels] could be allowed too for variable image size
            class_count,
            batch_size: int,  # mini-batch size
            training_log_period=1,
            name='SS-DCNN'):
        self.name = name

        self.batch_size = batch_size
        self.input_shape, self.class_count = input_shape, class_count

        self._sess = tf.Session()
        self.nodes = self._build_graph()

        self._sess.run(tf.global_variables_initializer())

        self.training_log_period = training_log_period
        self.log = []
        self._saver = tf.train.Saver(
            max_to_keep=10, keep_checkpoint_every_n_hours=2)

        self.training_step_event_handler = lambda step: False

    def __del__(self):  # I am not sure whether this is good
        self._sess.close()
        ops.reset_default_graph()

    def __str__(self):
        return self.name

    def save_state(self, file_path, save_log=True):
        """
            Saves the trained model as `file_path`.
            If `save_log == True`, `self.log` is saved as `file_path`+'.log'.
        """
        file_path = os.path.join(file_path, str(self))
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        self._saver.save(self._sess, file_path)
        with open(path + ".log", mode='w') as fs:
            fs.write("\n".join(self.log))
            fs.flush()
        print("State saved as '" + file_path + "'.")
        return file_path

    def load_state(self, path):
        self._saver.restore(self._sess, path)
        try:
            self.log = file.read_all_lines(path + ".log")
        except:
            self.log = "Log file not found."
        self._log("State loaded (" + str(self.epochs_completed) +
                  " epochs completed).")

    def predict(self, images: list, probs=False):
        """
            Requires the pixelwise-class probabilities TensorFlow graph node
            to be referenced by `self.nodes.probs`.
            It would be good to modify it to do forward propagation in batches
            istead of single images.
        """
        pr_probs = self._run_session([self.nodes.probs], images, None,
                                     False)[0]
        return pr_probs if probs else [one_hot_to_dense(p) for p in pr_probs]

    def train(self,
              train_data: Dataset,
              validation_data: Dataset = None,
              epoch_count: int = 1):
        """ Override if extra fetches are needed """
        self._train(
            train_data=train_data,
            validation_data=validation_data,
            epoch_count=epoch_count,
            extra_fetches=dict())

    def test(self, dataset):
        """ Override if extra fetches (maybe some evaluation measures) are needed """
        self._test(dataset, extra_fetches=dict())

    def _train_minibatch(self, images, labels, extra_fetches: list = []):
        fetches = [self.nodes.training_step, self.nodes.loss
                   ] + list(extra_fetches)
        evals = self._run_session(fetches, images, labels, True)
        cost, extra = evals[1], evals[2:]
        return cost, extra

    def _test_minibatch(self, images, labels, extra_fetches: list = []):
        fetches = [self.nodes.loss] + list(extra_fetches)
        evals = self._run_session(fetches, images, labels, False)
        cost, extra = evals[0], evals[1:]
        return cost, extra

    def _train(self,
               train_data: Dataset,
               validation_data: Dataset = None,
               epoch_count: int = 1,
               extra_fetches: dict = dict()):
        def log_training_start(epoch_count, batch_count, batch_size):
            self._log('Training (epochs: {}; {} batches of size {} per epoch)'
                      .format(epoch_count, batch_count, batch_size))

        def log_training_step(step, cost, extra):
            self._log(' epoch {:d}, step {:d}, cost {:.4f}, {}'.format(
                self.completed_epoch_count, b, cost, extra))

        dr = MiniBatchReader(train_data, self.batch_size)
        log_training_start(epoch_count, dr.number_of_batches, self.batch_size)
        end = False
        for ep in range(epoch_count):
            dr.reset(shuffle=True)
            for b in range(dr.number_of_batches):
                images, labels = dr.get_next_batch()
                cost, extra = self._train_minibatch(images, labels,
                                                    extra_fetches.values())
                if b % self.training_log_period == 0 or b == dr.number_of_batches - 1:
                    log_training_step(b, cost,
                                      dict(zip(extra_fetches.keys(), extra)))
                if self.training_step_event_handler(b) == True:
                    end = True
            self.completed_epoch_count += 1
            if end:
                break

    def _test(self, dataset, extra_fetches: dict = dict()):
        self._log('Testing...')
        cost_sum, extra_sum = 0, np.zeros(len(extra_fetches))
        dr = MiniBatchReader(dataset, self.batch_size)
        for _ in range(dr.number_of_batches):
            images, labels = dr.get_next_batch()
            cost, extra = self._test_minibatch(images, labels,
                                               extra_fetches.values())
            cost_sum += cost
            extra_sum += np.array(extra)
        cost = cost_sum / dr.number_of_batches
        extra = extra_sum / dr.number_of_batches
        ev = dict(zip(extra_fetches.keys(), extra))
        self._log('cost {:.4f}, {}'.format(cost, ev))

    @abc.abstractmethod
    def _build_graph(self):
        """ 
            Builds the TensorFlow graph for the model.
            Override this. It will be automatically called by the constructor
            (assuming super().__init__(...) is called in the constructor of the
            subclass).
            Returns tuple (input node, target labels node, probs node) (nodes 
            are of type tf.Tensor, the first 2 being placeholders)
         """
        return AbstractModel.EssentialNodes(None, None, None, None, None)

    def _run_session(self,
                     fetches: list,
                     images,
                     labels=None,
                     is_training=None):
        feed_dict = {self.nodes.input: images}
        if labels is not None:
            feed_dict[self.nodes.target] = np.array([
                dense_to_one_hot(lab, self.class_count)
                for i, lab in enumerate(labels)
            ])
        if is_training is not None and self.nodes.is_training is not None:
            feed_dict[self.nodes.is_training] = is_training
        return self._sess.run(fetches, feed_dict)

    def _log(self, text: str):
        timestr = datetime.datetime.now().strftime('%H:%M:%S')
        text = "[{}] {}".format(timestr, text)
        self.log.append(text)
        print(text)