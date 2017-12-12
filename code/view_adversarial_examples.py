import os
import numpy as np
import adversarial_examples.generation as advgen
import visualization
import ioutil
from data import loaders

from training_utils import Cifar10Loader


model = None
def generate(x):
    x = x.astype(np.ubyte)
    return visualization.compose([x,x], format='0,1')

data_path = os.path.join(
        ioutil.path.find_ancestor(os.path.dirname(__file__), 'projects'),
        'datasets/cifar-10-batches-py')
ds = Cifar10Loader.load_test(normalize=False)

viz = visualization.Viewer()
viz.display(ds.images, generate)


