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
    for i, y in enumerate(ys):
        plt.plot(y)
    plt.xlabel(r"skaliranje logita")
    plt.ylabel(r"$\frac{|L(z)-L(z')|}{\left\Vert  z - z' \right\Vert_\infty}$")
    plt.show()


s = +np.array([1.1, 0, 0]) / 10
s_ = np.array([1.0, 0, 0]) / 10
ss = [s, s_]
target = 0
losses = [np.array([ce_loss(s * i, target) for i in range(100)]) for s in ss]
plot([np.abs(losses[1] - losses[0]) / np.max(np.abs(softmax(s_) - softmax(s)))])
