import os
import numpy as np
import visualization
import processing.data_augmentation as da
from data.loaders import load_cifar10

import dirs

model = None


def generate(x):
    x = x.astype(np.ubyte)
    return visualization.compose([x, da.augment_cifar(x)], format='0,1')


data_path = os.path.join(os.path.join(dirs.DATASETS, 'cifar-10-batches-py'))
ds = load_cifar10(data_path, 'test')

viz = visualization.Viewer()
viz.display(ds.images, generate)
