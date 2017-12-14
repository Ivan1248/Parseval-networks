import os
import numpy as np
import adversarial_examples.generation as advgen
import visualization
from train_test_utils import Cifar10Loader
import dirs

model = None


def generate(x):
    x = x.astype(np.ubyte)
    return visualization.compose([x, x], format='0,1')


data_path = os.path.join(os.path.join(dirs.DATASETS, 'cifar-10-batches-py'))
ds = Cifar10Loader.load_test(normalize=False)

viz = visualization.Viewer()
viz.display(ds.images, generate)
