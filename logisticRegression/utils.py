import codecs
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data_from_file(fname):
    with codecs.open(fname) as f:
        raw_data = f.readlines()

    x_features = []
    y_labels = []

    for row in raw_data:
        x1, x2, y = row.strip().split('\t')
        x_features.append([float(x1), float(x2), 1.])
        y_labels.append([int(y)])

    return np.asarray(x_features), np.asarray(y_labels)


def sigmoid(val):
    return 1 / (1 + np.exp(-val))


def f_sigmoid(val):
    return sigmoid(val) * (1 - sigmoid(val))


def plotRegLine(x_features, y_labels, alphas):
    data_frame = pd.DataFrame({"x1": x_features[:, 0],
                               "x2": x_features[:, 1],
                               "y": y_labels.flatten()})
    sns.lmplot(x = "x1", y = "x2", data = data_frame, hue = 'y', fit_reg = False)

    # fit regression line
    line_pxi = np.linspace(data_frame.x1.min(), data_frame.x1.max(), 100)
    line_y = (-alphas[-1] - alphas[0] * line_pxi) / alphas[1]
    plt.plot(line_pxi, line_y, color = 'lightblue')
    plt.show()