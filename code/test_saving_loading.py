import os, sys

print("Loading and preparing data...")
from data_utils import Cifar10Loader
ds_test = Cifar10Loader.load_test()[:200]

print("Initializing model...")
from models import Dummy

get_model = lambda: Dummy(
    input_shape=ds_test.image_shape,
    class_count=ds_test.class_count,
    batch_size=1,
    training_log_period=100)

import tensorflow as tf
model=get_model()

for i in range(3):

    print("Starting training and validation loop...")
    from training import train
    train(model, ds_test, ds_test, epoch_count=2)

    print("Saving model...")
    import datetime
    import dirs
    save_path = dirs.SAVED_MODELS + '/dummy.' + datetime.datetime.now().strftime(
        "%Y-%m-%d")
    load_path=model.save_state(save_path)

    del(model)
    model = get_model()

    print("Loading model...")
    model.load_state(load_path)
