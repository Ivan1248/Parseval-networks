#%%
import os
import numpy as np
import tensorflow as tf

from data import load_cifar10, shuffle, split, normalize, dense_to_one_hot
from classification_cnn2 import ClassificationCNN2
from visualization import draw_conv_filters, draw_image

#%%
DATA_DIR = 'D:\datasets\cifar-10-batches-py'
SAVE_DIR = os.path.join(
    os.path.dirname(os.path.realpath('__file__')), "out/cifar10")
os.makedirs(SAVE_DIR, exist_ok=True)

class_count = 10
print("Loading data...")
train_x, train_y, test_x, test_y = load_cifar10(DATA_DIR)
print("Preparing data...")
valid_size = 5000
train_x, train_y = shuffle(train_x, train_y)
valid_x, valid_y, train_x, train_y = split(train_x, train_y, valid_size)
train_x, valid_x, test_x, mean, std = normalize(
    train_x, valid_x, test_x, returnmeanstd=True)
ys = [train_y, valid_y, test_y]
train_y, valid_y, test_y = [dense_to_one_hot(y, class_count) for y in ys]

#%%
tf.reset_default_graph()
print("Preparing CNN...")
nn = ClassificationCNN2(
    train_x[0].shape,
    num_classes=class_count,
    weight_decay=0.01,
    lr_policy={
        'boundaries': [3, 5, 7],
        'values': [10**-i for i in range(1, 5)]
    },
    batch_size=50,
    max_epochs=8,
    save_dir=SAVE_DIR)

conv1_var = tf.contrib.framework.get_variables('convrelu1/weights:0')[0]
conv1_weights = conv1_var.eval(session=nn.sess)
draw_conv_filters(0, 0, conv1_weights, SAVE_DIR)

train_losses, train_accuracies = [], []
valid_losses, valid_accuracies = [], []
learning_rates = []


def epoch_callback(evaluation):
    valid_loss, _, valid_accuracy, _, _ = evaluation
    train_loss, _, train_accuracy, _, _ = nn.evaluate("Training", train_x,
                                                      train_y)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    valid_losses.append(valid_loss)
    valid_accuracies.append(valid_accuracy)
    learning_rates.append(nn.sess.run(nn.n.learning_rate))


#%%
print("Training...")
#train_x = train_x[:1000]
#train_y = train_y[:1000]
nn.train(train_x, train_y, valid_x, valid_y, epoch_callback=epoch_callback)
print("Testing...")
nn.evaluate("Test", test_x, test_y)


#%%
def plot():
    import matplotlib.pyplot as plt
    t = np.arange(1, len(learning_rates) + 1)
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col')
    ax1.set_title("Loss")
    ax1.plot(t, train_losses, label="training")
    ax1.plot(t, valid_losses, label="validation")
    ax2.set_title("Accuracy")
    ax2.plot(t, train_accuracies, label="training")
    ax2.plot(t, valid_accuracies, label="validation")
    ax3.set_title("Learning rate")
    ax3.plot(t, learning_rates)
    plt.show()


plot()


#%%
def show_worst_results():
    print("probs=nn.predict...")
    probs = nn.predict(test_x)
    target_probs = np.sum(probs * test_y, axis=1)
    losses = -np.log(target_probs + 1e-9)
    print("ind = np.argpartition...")
    ind = np.argpartition(losses, -20)[-20:]

    print("draw...")
    images = test_x[ind]
    probs = probs[ind]
    target_probs = target_probs[ind]
    for i, im in enumerate(images):
        draw_image(im, mean, std)
        print(probs[i], np.argmax(test_y, 1)[i])


show_worst_results()
