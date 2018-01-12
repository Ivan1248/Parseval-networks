import numpy as np
import matplotlib.pyplot as plt


def softmax(X):
    exp_X_shifted = np.exp(X - np.max(X)[None].T)
    probs = exp_X_shifted / np.sum(exp_X_shifted)[None].T
    return probs


def ce_loss(logits, target):
    probs = softmax(logits)
    return -np.log(probs[target])


def ce_loss_deriv(logits, target):
    probs = softmax(logits)
    return -probs[target]


def plot(ys):
    plt.figure()
    axes = plt.gca()
    #axes.set_ylim([0, 0.5])
    for i, y in enumerate(ys):
        plt.plot(y)
    plt.xlabel(r"skaliranje logita")
    plt.ylabel(r"$\|L(z)-L(z')\| / \|\| z - z' \|\|_\inf$")
    plt.show()


z = +np.array([1.1, 0, 0]) / 10
z_ = np.array([1.0, 0, 0]) / 10
zs = [z, z_]
target = 0
losses = [np.array([ce_loss(z * i, target) for i in range(100)]) for z in zs]
plot([np.abs(losses[1] - losses[0]) / np.max(np.abs(z_ - z))])
