import os
import numpy as np
import visualization
import ioutil
import processing.data_augmentation as da
from data import loaders

model = None


def generate(x):
    x = x.astype(np.ubyte)
    return visualization.compose([x, da.augment_cifar(x)], format='0,1')


data_path = ioutil.path.find_ancestor_sibling(
    os.path.dirname(__file__), 'projects/datasets/cifar-10-batches-py')
ds = loaders.load_cifar10(data_path, 'test')

viz = visualization.Viewer()
viz.display(ds.images, generate)
