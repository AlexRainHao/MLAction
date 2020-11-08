'''
The weights linear regression has a normal_equation as
    J(`theta`) = 1/2 * (X`theta` - y).T * W * (X`theta` - y)
    `theta` = (X.T * W * X)^-1 * X.T * W * Y

and the weights build as
    weight = exp(-(x - xi)**2 / 2`tau`**2),

where `tau` as a hyper parameter
'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(fname):
    data = pd.read_csv(fname, header = 0)
    return data


def convert_to_features(data_frame: pd.DataFrame):
    # consideration for parameter of beta
    # so the feature should added one columns filled with 1
    x_features = np.expand_dims(data_frame.columns.values.astype(np.float), -1)
    y_value = data_frame.head(1).values.T

    x_features = np.concatenate((x_features, np.ones_like(x_features)), axis = 1)

    return x_features, y_value


def build_weight(x, xi, tau = 5):
    # calculate for `theta` from normal equation
    return np.exp(-(x - xi)[:, 0] ** 2 / (2 * tau ** 2))

def normal_equation(x, y, w):
    """
    @rtype: dimension of 2 * 1
    """
    return np.linalg.inv(x.T.dot(w).dot(x)).dot(x.T).dot(w).dot(y)


def prediction(x, y, tau = 5):
    pred_res = []
    for x_j in x:
        weight = build_weight(x, x_j, tau)
        # convert for matrix calculation
        weight = np.diag(weight)
        theta = normal_equation(x, y, weight)

        pred_res.append(theta.T.dot(x_j)[0])

    return pred_res

    

def plot_for_result(x_features, y_label, tau = 5):
    pred_res = prediction(x_features, y_label, tau)
    sns.regplot(x_features[:, 0], y = y_label, fit_reg = False)
    plt.plot(x[:, 0], pred_res, linewidth = 3)
    plt.show()



if __name__ == '__main__':
    data = load_data("./data/data.csv")
    x, y = convert_to_features(data)
    plot_for_result(x, y)