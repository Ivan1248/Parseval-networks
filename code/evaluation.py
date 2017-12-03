import numpy as np
from sklearn.metrics import confusion_matrix


def evaluate(pred, true):
    """ cm, accuracy, precisions, recalls """
    n = len(pred)
    cm = confusion_matrix(true, pred)
    accuracy = np.trace(cm) / n
    tps = np.diag(cm)
    precisions = tps / np.sum(cm, axis=1)
    recalls = tps / np.sum(cm, axis=0)
    return cm, accuracy, precisions, recalls
