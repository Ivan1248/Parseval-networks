import training_utils, models
from ioutil import path, console
import sys

dimargs = sys.argv[1:]
if len(dimargs) not in [0, 2]:
    print("usage: train-wrn.py [<Zagoruyko-depth> <widening-factor>]")
zaggydepth, k = (16, 4) if len(dimargs) == 0 else map(int, dimargs)

print("Loading and preparing data...")
ds_train, ds_val = training_utils.Cifar10Loader.load_train_val()

print("Initializing model...")
image_shape, class_count = ds_train.image_shape, ds_train.class_count
model = models.get_wide_resnet(zaggydepth, k, image_shape, class_count, dim_increase='id')

print("Starting training and validation loop...")
training_utils.train(model, ds_train, ds_val, epoch_count=200)

print("Saving model...")
import datetime
save_path = path.find_ancestor_sibling(__file__, 'data/models')
model.save_state(save_path + '/wrn-%d-%d.' % (zaggydepth, k) +
                 datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
