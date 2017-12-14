import sys
dimargs = sys.argv[1:]
if len(dimargs) not in [0, 2]:
    print("usage: train-wrn.py [<Zagoruyko-depth> <widening-factor>]")
zaggydepth, k = (16, 4) if len(dimargs) == 0 else map(int, dimargs)

print("Loading and preparing data...")
from data_utils import Cifar10Loader
ds_train, ds_val = Cifar10Loader.load_train_val()

print("Initializing model...")
from models import ResidualBlockKind, ResNet


def get_wide_resnet(n, k, input_shape, class_count, dim_increase='conv1'):
    group_count = 3
    ksizes = [3, 3]
    blocks_per_group = (n - 4) // (group_count * len(ksizes))
    print("group count: {}, blocks per group: {}".format(
        group_count, blocks_per_group))
    model = ResNet(
        input_shape=input_shape,
        class_count=class_count,
        batch_size=128,
        learning_rate_policy={
            'boundaries': [60, 120, 160],
            'values': [1e-1 * 0.2**i for i in range(4)]
        },
        block_kind=ResidualBlockKind(
            ksizes=ksizes,
            dropout_locations=[0],
            dropout_rate=0.3,
            dim_increase=dim_increase),
        group_lengths=[blocks_per_group] * group_count,
        widening_factor=k,
        weight_decay=5e-4,
        training_log_period=50)
    assert n == model.zagoruyko_depth, "invalid depth (n={}!={})".format(
        n, model.zagoruyko_depth)
    return model


image_shape, class_count = ds_train.image_shape, ds_train.class_count
model = get_wide_resnet(
    zaggydepth, k, image_shape, class_count, dim_increase='id')

print("Starting training and validation loop...")
from training import train
train(model, ds_train, ds_val, epoch_count=200)

print("Saving model...")
import datetime
import dirs
model.save_state(dirs.SAVED_MODELS + '/wrn-%d-%d.' % (zaggydepth, k) +
                 datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
