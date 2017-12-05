import tensorflow as tf
import numpy as np
from visualization import draw_conv_filters


class AbstractClassificationCNN:
    class Nodes:
        def __init__(self, input, target, increment_epoch, logits, probs, loss,
                     training_step, learning_rate):
            self.input = input
            self.target = target
            self.increment_epoch = increment_epoch
            self.logits = logits
            self.probs = probs
            self.loss = loss
            self.training_step = training_step
            self.learning_rate = learning_rate

    def __init__(self, input_shape, num_classes, weight_decay, lr_policy,
                 batch_size, max_epochs, save_dir):
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.save_dir = save_dir
        self.sess = tf.Session()
        self.n = self.build_model(input_shape, num_classes, weight_decay,
                                  lr_policy)
        self.sess.run(tf.global_variables_initializer())

    def build_model(self, input_shape, num_classes, weight_decay, lr_policy):
        raise NotImplementedError()
        return ConvNet.Nodes(None, None, None, None, None, None, None)

    def _run_sess(self, fetches: list, inputs=None, labels_oh=None):
        feed_dict = dict() if input is None else {self.n.input: inputs}
        if labels_oh is not None:
            feed_dict[self.n.target] = labels_oh
        return self.sess.run(fetches, feed_dict)

    def predict(self, x):
        batch_size = self.batch_size
        num_batches = x.shape[0] // batch_size
        yh = []
        for i in range(num_batches):
            b = x[i * batch_size:(i + 1) * batch_size, :]
            yh += self._run_sess([self.n.probs], b)
        return np.vstack(yh)

    def train(self,
              train_x,
              train_y,
              valid_x,
              valid_y,
              epoch_callback=lambda x: None):
        from visualization import draw_conv_filters
        get_conv1weights = lambda:  tf.contrib.framework.get_variables('convrelu1/weights:0')[0].eval(session=self.sess)
        batch_size = self.batch_size
        save_dir = self.save_dir
        num_examples = train_x.shape[0]
        assert num_examples % batch_size == 0
        num_batches = num_examples // batch_size
        self.evaluate("Validation", valid_x, valid_y)
        draw_conv_filters(0, 0, get_conv1weights(), save_dir)
        for epoch in range(1, self.max_epochs + 1):
            self.sess.run(self.n.increment_epoch)
            # shuffle the data at the beggining of each epoch
            permutation_idx = np.random.permutation(num_examples)
            train_x = train_x[permutation_idx]
            train_y = train_y[permutation_idx]
            #for i in range(100):
            cnt_correct = 0
            for i in range(num_batches):
                # store mini-batch to ndarray
                batch_x = train_x[i * batch_size:(i + 1) * batch_size, :]
                batch_y = train_y[i * batch_size:(i + 1) * batch_size, :]
                fetches = [self.n.logits, self.n.loss, self.n.training_step]
                logits, loss_val, _ = self._run_sess(fetches, batch_x, batch_y)

                # compute classification accuracy
                yp = np.argmax(logits, 1)
                yt = np.argmax(batch_y, 1)
                cnt_correct += (yp == yt).sum()

                if i % 5 == 0:
                    print("epoch %d, step %d/%d, batch loss = %.2f" %
                          (epoch, i * batch_size, num_examples, loss_val))
                if i % 100 == 0:
                    pass
                    draw_conv_filters(epoch, i * batch_size,
                                      get_conv1weights(), save_dir)
                    #draw_conv_filters(epoch, i*batch_size, net[3])
                if i > 0 and i % 50 == 0:
                    print("Train accuracy = %.2f" %
                          (cnt_correct / ((i + 1) * batch_size) * 100))
            print("Train accuracy = %.2f" % (cnt_correct / num_examples * 100))
            epoch_callback(self.evaluate("Validation", valid_x, valid_y))

    def evaluate(self, name, x, y):
        print("\nRunning evaluation: ", name)
        batch_size = self.batch_size
        num_examples = x.shape[0]
        assert num_examples % batch_size == 0
        num_batches = num_examples // batch_size
        cnt_correct = 0
        loss_avg = 0
        yp = np.empty(y.shape[0])
        for i in range(num_batches):
            batch_x = x[i * batch_size:(i + 1) * batch_size, :]
            batch_y = y[i * batch_size:(i + 1) * batch_size, :]
            fetches = [self.n.logits, self.n.loss]
            logits, loss_val = self._run_sess(fetches, batch_x, batch_y)
            yp[i * batch_size:(i + 1) * batch_size] = np.argmax(logits, 1)
            loss_avg += loss_val
            #print("step %d / %d, loss = %.2f" % (i*batch_size, num_examples, loss_val / batch_size))
        from evaluation import evaluate
        cm, accuracy, precisions, recalls = evaluate(yp, np.argmax(y, 1))
        loss_avg /= num_batches
        print(name + " avg loss = %.2f\n" % loss_avg)
        print(name + " accuracy = %.2f" % accuracy)
        print(name + " cm = {}" % cm)
        return loss_avg, cm, accuracy, precisions, recalls
