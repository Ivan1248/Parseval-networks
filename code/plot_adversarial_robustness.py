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
top = 70
epss = [int(i**2+0.5) / 100 for i in np.arange(0, int(top**0.5+1)+0.5, 0.5)]
print(epss)
accuracieses = []
for parseval in [False, True]:
    aggregation = 'convex' if parseval else 'sum'
    resnet_ctor = ParsevalResNet if parseval else ResNet
    from standard_resnets import get_wrn
    model = standard_resnets.get_wrn(
        zaggydepth,
        k,
        ds_test.image_shape,
        ds_test.class_count,
        aggregation=aggregation,
        resnet_ctor=resnet_ctor)

    saved_path = dirs.SAVED_MODELS
    if parseval:
        saved_path += '/wrn-28-10-p-t--2018-01-24-21-18/ResNet'  # Parseval
    else:
        saved_path += '/wrn-28-10-t--2018-01-23-19-13/ResNet'  # vanilla
    model.load_state(saved_path)

    cost, ev = model.test(ds_test)
    accuracies = [ev['accuracy']]
    for eps in epss[1:]:
        print("Creating adversarial examples...")
        clip_max = (
            255 - np.max(Cifar10Loader.mean)) / np.max(Cifar10Loader.std)
        n_fgsm = fgsm(
            model.nodes.input,
            model.nodes.probs,
            eps=eps,
            clip_min=-clip_max,
            clip_max=clip_max)
        images_adv, = batch_eval(
            model._sess, [model.nodes.input], [n_fgsm],
            [ds_test.images[:model.batch_size*64]],
            args={'batch_size': model.batch_size},
            feed={model._is_training: False})
        adv_ds_test = Dataset(images_adv, ds_test.labels, ds_test.class_count)
        cost, ev = model.test(adv_ds_test)
        accuracies.append(ev['accuracy'])
    accuracieses.append(accuracies)
    print(accuracies)

def plot(epss, curves, names):
    plt.figure()
    plt.rcParams["mathtext.fontset"] = "cm"
    #plt.yticks(np.arange(0, 1, 0.05))
    axes = plt.gca()
    axes.grid(color='0.9', linestyle='-', linewidth=1)
    axes.set_ylim([0, 1])
    axes.set_xlim([0, top/100])
    for c, n in zip(curves, names):
        plt.plot(epss, c, label=n, linewidth=2)
    plt.xlabel("$\epsilon$")
    plt.ylabel("točnost")
    plt.legend()
    plt.show()

plot(epss, accuracieses, ["WRN-28-10-Parseval", "WRN-28-10"])