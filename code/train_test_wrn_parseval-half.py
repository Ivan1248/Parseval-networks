import sys
dimargs = sys.argv[1:]
if len(dimargs) not in [0, 2]:
    print("usage: train-wrn.py [<Zagoruyko-depth> <widening-factor>]")
zaggydepth, k = (16, 4) if len(dimargs) == 0 else map(int, dimargs)

print("Loading and preparing data...")
from data import Dataset
from data_utils import Cifar10Loader
ds_train, ds_val = Cifar10Loader.load_train_val()
ds_train = Dataset.join(ds_train, ds_val)
ds_test = Cifar10Loader.load_test()

print("Initializing model...")
from models import ResidualBlockProperties, ParsevalResNet


def get_parseval_resnet(n, k, input_shape, class_count, dim_increase):
    group_count = 3
    ksizes = [3, 3]
    blocks_per_group = (n - 4) // (group_count * len(ksizes))
    print("group count: {}, blocks per group: {}".format(
        group_count, blocks_per_group))
    model = ParsevalResNet(
        input_shape=input_shape,
        class_count=class_count,
        batch_size=128,
        learning_rate_policy={
            'boundaries': [30, 60, 80],
            'values': [1e-1 * 0.2**i for i in range(4)]
        },
        block_properties=ResidualBlockProperties(
            ksizes=ksizes,
            dropout_locations=[0],
            dropout_rate=0.3,
            dim_increase=dim_increase,
            aggregation='convex'),
        group_lengths=[blocks_per_group] * group_count,
        widening_factor=k,
        weight_decay=5e-4,
        training_log_period=50)
    assert n == model.zagoruyko_depth, "invalid depth (n={}!={})".format(
        n, model.zagoruyko_depth)
    return model


image_shape, class_count = ds_train.image_shape, ds_train.class_count
model = get_parseval_resnet(zaggydepth, k, image_shape, class_count, 'id')

print("Starting training and validation loop...")
from training import train
train(model, ds_train, ds_test, epoch_count=120)

print("Saving model...")
import datetime
import dirs
model.save_state(dirs.SAVED_MODELS + '/wrnt-pars-h-%d-%d.' % (zaggydepth, k) +
                 datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
