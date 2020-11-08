import utils as utils
import numpy as np


def gradient_learning(x_features, y_labels, max_iter, learning_rate, error_threshold):
    '''
    full data into gradient calculation
    '''
    epoch = 0

    m, n = x_features.shape
    alphas = np.ones(shape = (n, 1))
    init_error = float('inf')

    while (epoch < max_iter):
        if np.abs(init_error) < error_threshold:
            break

        # m*1
        hx = utils.sigmoid(np.matmul(x_features, alphas))

        # error
        y_error = y_labels - hx

        # n*m * m
        error = np.matmul(x_features.T, y_error)
        alphas += learning_rate * error
        init_error = np.mean(y_error)
        epoch += 1

    return alphas


def sto_gradient_learning(x_features, y_labels, max_iter, learning_rate, error_threshold):
    '''
    stochastic gradient learning
    '''
    epoch = 0

    m, n = x_features.shape
    alphas = np.ones(shape = (n, 1))
    init_error = float('inf')

    while(epoch < max_iter):
        if np.abs(init_error) < error_threshold:
            break

        # random choice
        # x_idx = np.random.choice(range(m))
        x_idxs = list(range(m))
        np.random.shuffle(x_idxs)

        for i, x_idx in enumerate(x_idxs):
            learning_rate = 4 / (1. + epoch + i) + .01
            hx = utils.sigmoid(np.matmul(x_features[x_idx, :], alphas))
            # error
            y_error = y_labels[x_idx] - hx

            error = x_features[x_idx, :] * y_error
            alphas += learning_rate * np.expand_dims(error, -1)

            init_error = y_error
        epoch += 1

    return alphas


# if __name__ == '__main__':
#     x_features, y_labels = utils.load_data_from_file("./data/testSet.txt")
#     alphas = gradient_learning(x_features, y_labels, 200, 1e-3, 1e-6)
#     alphas = sto_gradient_learning(x_features, y_labels, 20, 1e-2, 1e-6)
#     utils.plotRegLine(x_features, y_labels, alphas)
