import sys
dimargs = sys.argv[1:]
if len(dimargs) not in [0, 2]:
    print("usage: train-wrn.py [<Zagoruyko-depth> <widening-factor>]")
zaggydepth, k = (16, 4) if len(dimargs) == 0 else map(int, dimargs)

print("Loading and preparing data...")
from data_utils import Cifar10Loader
ds_train, ds_val = Cifar10Loader.load_train_val()

print("Initializing model...")
from models import QuickNet
model = QuickNet(
    input_shape=ds_train.image_shape,
    class_count=ds_train.class_count,
    class0_unknown=True,
    batch_size=128,
    learning_rate_policy={
        'boundaries': [60, 120, 160],
        'values': [1e-4 * 0.2**i for i in range(4)]
    },
    name='QuickNet',
    training_log_period=100)

print("Starting training and validation loop...")
from training import train
train(model, ds_train, ds_val, epoch_count=200)

print("Saving model...")
import datetime
import dirs
model.save_state(dirs.SAVED_MODELS + '/wrn-%d-%d.' % (zaggydepth, k) +
                 datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
