import os
import numpy as np
import visualization
import processing.data_augmentation as da
from data.loaders import load_cifar10
from data_utils import Cifar10Loader


import dirs

model = None


def generate(x):
    max, min = np.max(x),np.min(x)
    scale = lambda x: ((x-min)*255/(max-min)).astype(np.ubyte)
    xa = da.augment_cifar(x)
    return visualization.compose([scale(x), scale(xa)], format='0,1')


data_path = os.path.join(os.path.join(dirs.DATASETS, 'cifar-10-batches-py'))
ds, _ = Cifar10Loader.load_train_val()

viz = visualization.Viewer()
viz.display(ds.images, generate)
