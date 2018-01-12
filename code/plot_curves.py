import os
import glob
import re

import numpy as np
import matplotlib.pyplot as plt

from ioutils import path, file
import dirs


def find_training_log_file(name):
    dir_path = os.path.join(dirs.SAVED_MODELS, name)
    return glob.glob(os.path.join(dir_path, '*.log'))[0]


def parse_log(log_lines):
    read_accuracy = False
    curve = []
    epoch = 0
    for l in log_lines:
        if read_accuracy:
            read_accuracy = False
            acc = re.search("'accuracy': (.*)}", l, re.IGNORECASE).group(1)
            curve.append(float(acc))
        else:
            read_accuracy = re.search("Testing \(validation", l,
                                      re.IGNORECASE) is not None
    return np.array(curve)


def plot(curves, names):
    plt.figure()
    axes = plt.gca()
    axes.set_ylim([0, 0.5])
    for c, n in zip(curves, names):
        plt.plot(c, label=n)
    plt.xlabel("broj epoha")
    plt.ylabel("pogreška")
    plt.legend()
    plt.show()


names = [
    "wrnt-pars-28-10.2018-01-03-18-00/", "wrnt-28-10.2018-01-03-07-17/",
    "wrnt-pars-h-28-10.2018-01-12-01-23/", "wrnt-28-10.2018-01-12-03-31"
]
paths = [find_training_log_file(n) for n in names]
logs = [file.read_all_lines(p) for p in paths]
curves = [1 - parse_log(l) for l in logs]
plot(curves, ["WRN-28-10-Parseval-200", "WRN-28-10-200", "WRN-28-10-Parseval-100", "WRN-28-10-100"])
