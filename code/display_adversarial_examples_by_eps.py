import sys

import numpy as np
import matplotlib.pyplot as plt

import cleverhans
from cleverhans.attacks_tf import fgsm
from cleverhans.utils_tf import batch_eval

from models import ResidualBlockProperties, ParsevalResNet, ResNet
from data_utils import Cifar10Loader, Dataset
import visualization
from visualization import compose, Viewer
import dirs
from training import train
import standard_resnets

dimargs = sys.argv[1:]
if len(dimargs) not in [0, 2]:
    print("usage: train-wrn.py [<Zagoruyko-depth> <widening-factor>]")
zaggydepth, k = (28, 10) if len(dimargs) == 0 else map(int, dimargs)

print("Loading and preparing data...")
ds_test = Cifar10Loader.load_test()
print(Cifar10Loader.std)

print("Initializing model...")
from standard_resnets import get_wrn
model = standard_resnets.get_wrn(zaggydepth, k, ds_test.image_shape,
                                 ds_test.class_count)

saved_path = dirs.SAVED_MODELS
saved_path += '/wrn-28-10-t--2018-01-23-19-13/ResNet'  # vanilla
model.load_state(saved_path)

cost, ev = model.test(ds_test)
accuracies = [ev['accuracy']]

top = 10
epss = [0, 0.02, 0.05, 0.2, 0.5, 1]
image_count = 5
batch_size = 64
adv_image_lists = [ds_test.images[:batch_size]]
for eps in epss[1:]:
    print("Creating adversarial examples...")
    clip_max = (255 - np.max(Cifar10Loader.mean)) / np.max(Cifar10Loader.std)
    n_fgsm = fgsm(
        model.nodes.input,
        model.nodes.probs,
        eps=eps,
        clip_min=-clip_max,
        clip_max=clip_max)
    images_adv, = batch_eval(
        model._sess, [model.nodes.input], [n_fgsm], [adv_image_lists[0]],
        args={'batch_size': batch_size},
        feed={model._is_training: False})
    adv_image_lists.append(images_adv)


def generate_visualization(i0):

    def get_row(i):
        ims = adv_image_lists[i][i0:i0+image_count]
        s, m = Cifar10Loader.std, Cifar10Loader.mean
        scale = lambda x: np.clip(x * s + m, 0, 255).astype(np.ubyte)
        return list(map(scale, ims))

    cols = [get_row(i) for i in range(i0, i0 + len(epss))]
    return visualization.compose(cols, format=None)

    images = [im for i in range(i0, i0 + 3) for im in get_row(i)]
    comp_format = "".join([
        str(i) + "," if i % image_count == image_count - 1 else ";"
        for i in range(len(images))
    ])[:-1]
    return visualization.compose(images, format=comp_format)


scaled_eps = eps * np.max(Cifar10Loader.std)
viewer = Viewer("Adversarial examples, scaled eps=" + str(scaled_eps) + ", eps="
                + str(eps))
viewer.display(np.arange(batch_size), generate_visualization)
