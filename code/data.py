import os
import pickle
import numpy as np


def shuffle(data_x, data_y):
    indices = np.arange(data_x.shape[0])
    np.random.shuffle(indices)
    shuffled_data_x = np.ascontiguousarray(data_x[indices])
    shuffled_data_y = np.ascontiguousarray(data_y[indices])
    return shuffled_data_x, shuffled_data_y


def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict


def load_cifar10(data_dir):
    h, w, ch = 32, 32, 3

    train_x = np.ndarray((0, h * w * ch), dtype=np.float32)
    train_y = []
    for i in range(1, 6):
        subset = unpickle(os.path.join(data_dir, 'data_batch_%d' % i))
        train_x = np.vstack((train_x, subset['data']))
        train_y += subset['labels']
    train_x = train_x.reshape((-1, ch, h, w)).transpose(0, 2, 3, 1)
    train_y = np.array(train_y, dtype=np.int32)

    subset = unpickle(os.path.join(data_dir, 'test_batch'))
    test_x = subset['data'].reshape(
        (-1, ch, h, w)).transpose(0, 2, 3, 1).astype(np.float32)
    test_y = np.array(subset['labels'], dtype=np.int32)

    return train_x, train_y, test_x, test_y


def split(x, y, index):
    valid_x, valid_y = x[:index, ...], y[:index, ...]
    train_x, train_y = x[index:, ...], y[index:, ...]
    return valid_x, valid_y, train_x, train_y


def normalize(train_x, valid_x, test_x, returnmeanstd=False):
    train_mean = train_x.mean((0, 1, 2))
    train_std = train_x.std((0, 1, 2))
    train_x = (train_x - train_mean) / train_std
    valid_x = (valid_x - train_mean) / train_std
    test_x = (test_x - train_mean) / train_std
    ret = [train_x, valid_x, test_x]
    return ret + [train_mean, train_std] if returnmeanstd else ret


def dense_to_one_hot(y, class_count):
    return np.eye(class_count)[y]


if __name__ == "__main__":
    DATA_DIR = 'D:\datasets\cifar-10-batches-py'

    h, w, ch = 32, 32, 3

    train_x, train_y, test_x, test_y = load_cifar10(DATA_DIR)

    valid_size = 5000
    train_x, train_y = shuffle(train_x, train_y)
    valid_x, valid_y, train_x, train_y = split(train_x, train_y, valid_size)
    train_x, valid_x, test_x = normalize(train_x, valid_x, test_x)
