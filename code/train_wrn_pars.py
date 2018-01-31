import sys
import datetime

from models import ResidualBlockProperties, ParsevalResNet
from data_utils import Cifar10Loader, Dataset
import standard_resnets
from training import train
import dirs

dimargs = sys.argv[1:]
if len(dimargs) not in [0, 2]:
    print("usage: train-wrn.py [<Zagoruyko-depth> <widening-factor>]")
zaggydepth, k = (16, 4) if len(dimargs) == 0 else map(int, dimargs)

print("Loading and preparing data...")
ds_train, ds_val = Cifar10Loader.load_train_val()

print("Initializing model...")
model = standard_resnets.get_wrn(
    zaggydepth, k, ds_train.image_shape, ds_train.class_count, aggregation='convex', resnet_ctor=ParsevalResNet)

print("Starting training and validation loop...")
train(model, ds_train, ds_val, epoch_count=20)

print("Saving model...")
model.save_state(dirs.SAVED_MODELS + '/wrn-%d-%d-p--' % (zaggydepth, k) +
                 datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
