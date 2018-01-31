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
    plt.yticks(np.arange(0, 0.51, 0.05))
    axes = plt.gca()
    axes.grid(color='0.9', linestyle='-', linewidth=1)
    axes.set_ylim([0, 0.5])
    axes.set_xlim([0, 200])
    for c, n in zip(curves, names):
        plt.plot(c, label=n, linewidth=2)
    plt.xlabel("broj epoha")
    plt.ylabel("klasifikacijska pogreška")
    plt.legend()
    plt.show()


names_to_labels = {
    #"wrnt-pars-28-10.2018-01-03-18-00/": "WRN-28-10-Parseval",
    #"wrnt-28-10.2018-01-03-07-17/": "WRN-28-10",
    #"wrnt-pars-h-28-10.2018-01-12-01-23/": "WRN-28-10-Parseval-100",
    #"wrnt-28-10.2018-01-12-03-31": "WRN-28-10-100",

    #"wrnt-28-10.2018-01-16-21-54": "WRN-28-10",
    #"wrnt-pars-28-10.2018-01-16-14-56": "WRN-28-10-Parseval",
    #"wrnt-pars-h-28-10.2018-01-16-21-05": "WRN-28-10-Parseval-100",
    #"wrnt-pars-h-2xlr-28-10.2018-01-16-06-42": "WRN-28-10-Parseval2xlr-100", 

    "wrn-28-10-p-t--2018-01-24-21-18": "WRN-28-10-Parseval",
    "wrn-28-10-t--2018-01-23-19-13": "WRN-28-10",
}
paths = map(find_training_log_file, names_to_labels.keys())
logs = map(file.read_all_lines, paths)
curves = [1 - parse_log(l) for l in logs]
plot(curves, [names_to_labels[k] for k in names_to_labels.keys()])
