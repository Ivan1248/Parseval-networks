import tensorflow as tf

from ioutil import path, console
import processing.preprocessing as pp
from data import loaders, Dataset

from models import AbstractModel

SAVED_MODELS_DIR = 'data/models'


class Cifar10Loader(object):
    mean, std = None, None
    data_path = path.find_ancestor_sibling(
        __file__, 'projects/datasets/cifar-10-batches-py')

    @classmethod
    def load_train_val(cls, normalize=True):
        ds = loaders.load_cifar10(cls.data_path, 'train')
        if normalize:
            cls.mean, cls.std = pp.get_normalization_statistics(ds.images)
            ds = Dataset(
                pp.normalize(ds.images, cls.mean, cls.std), ds.labels, ds.class_count)
        ds.shuffle()
        ds_train, ds_val = ds.split(0, int(ds.size * 0.8))
        return ds_train, ds_val

    @classmethod
    def load_test(cls, normalize=True, use_test_set_normalization_statistics=False):
        ds = loaders.load_cifar10(cls.data_path, 'test')
        if not normalize:
            return ds
        mean, std = None, None
        if use_test_set_normalization_statistics:
            mean, std = pp.get_normalization_statistics(ds.images)
        else:
            if cls.mean is None:
                cls.load_train_val()
            mean, std = cls.mean, cls.std
        return Dataset(
            pp.normalize(ds.images, mean, std), ds.labels, ds.class_count)


def train(model: AbstractModel,
          ds_train: Dataset,
          ds_val: Dataset,
          epoch_count=200):

    def handle_step(i):
        text = console.read_line(impatient=True, discard_non_last=True)
        if text == 'q':
            return True
        if text == 's':
            writer_path = path.find_ancestor_sibling(__file__, 'data/logs')
            writer = tf.summary.FileWriter(writer_path, graph=model._sess.graph)
        return False

    model.training_step_event_handler = handle_step

    from processing.data_augmentation import augment_cifar

    model.test(ds_val)
    ds_train_part = ds_train[:ds_val.size]
    for i in range(epoch_count):
        prepr_ds_train = Dataset(
            list(map(augment_cifar, ds_train.images)), ds_train.labels,
            ds_train.class_count)
        model.train(prepr_ds_train, epoch_count=1)
        model.test(ds_val, 'validation data')
        model.test(ds_train_part, 'training data subset')
