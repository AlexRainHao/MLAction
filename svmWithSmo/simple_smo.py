import utils
import numpy as np
import time

def simpleSmo(x_features, y_labels, C, toler, maxIter, kernel_type = ("normal", None)):
    '''
    :param x_features:
        ---> n * m dimension list,
        ---> n: numbers of samples
        ---> m: numbers of features
    :param y_labels:
        ---> list with length of n
    :param C:
    :param toler:
    :param maxIter:
    :return:
    '''
    x_features = np.asarray(x_features)
    y_labels = np.expand_dims(np.asarray(y_labels), -1)

    beta = 0
    m, n = x_features.shape
    alphas = np.zeros(shape = (m, 1))
    iter = 0

    if kernel_type:
        kernel_val = utils.calKernelVal(x_features,
                                        kernel_type = kernel_type[0],
                                        kernel_para = kernel_type[1])
    else:
        kernel_val = utils.calKernelVal(x_features, "normal")

    while (iter < maxIter):
        alphaPairChanged = 0

        for i in range(m):
            # gxi = wTx + b, where w = sum(ai * yi * xi)
            gxi = np.matmul((alphas * y_labels).T, kernel_val[i, :]) + beta
            Ei = gxi - y_labels[i]

            if ((y_labels[i] * Ei < -toler).all() and (alphas[i] < C)) \
                    or ((y_labels[i] * Ei > toler).all() and (alphas[i] > 0)):
                j = utils.selectJrand(i, m)
                gxj = np.matmul((alphas * y_labels).T, kernel_val[j, :]) + beta
                Ej = gxj - y_labels[j]

                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()

                if (y_labels[i] != y_labels[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[i] + alphas[j] - C)
                    H = min(C, alphas[i] + alphas[j])

                if L == H:
                    continue

                eta = kernel_val[i, i] + kernel_val[j, j] - 2 * kernel_val[i, j]



                alphas[j] += (y_labels[j] * (Ei - Ej) / eta)
                alphas[j] = utils.clib_boundary(alphas[j], H, L)
                if abs(alphas[j] - alphaJold) < 1e-5: continue

                alphas[i] += y_labels[i] * y_labels[j] * (alphaJold - alphas[j])

                # update beta
                b1 = beta - Ei - y_labels[i] * (alphas[i] - alphaIold) * kernel_val[i, i] - \
                     y_labels[j] * (alphas[j] - alphaJold) * kernel_val[i, j]

                b2 = beta - Ej - y_labels[i] * (alphas[i] - alphaIold) * kernel_val[i, j] - \
                     y_labels[j] * (alphas[j] - alphaJold) * kernel_val[j, j]

                if alphas[i] > 0 and alphas[j] < C:
                    beta = b1
                elif alphas[j] > 0 and alphas[j] < C:
                    beta = b2
                else:
                    beta = .5 * b1 + .5 * b2

                alphaPairChanged += 1

        if (alphaPairChanged == 0): iter += 1
        else:
            iter = 0

    return alphas, beta

if __name__ == '__main__':
    x_features, y_labels = utils.load_data_from_txt('./data/testSet.txt')

    _bt = time.time()
    alphas, beta = simpleSmo(x_features, y_labels, 0.6, 1e-4, 20)
    print('traing time: %f' % (time.time() - _bt))


    w = np.sum(np.expand_dims(alphas.flatten() * y_labels.flatten(), -1) * x_features, axis = 0)

    support_point_index = np.where(alphas.flatten() > 0)[0]

    utils.plotWithboundary(x_features, y_labels, w, beta, support_point_index, "svm_simplified_fit.png")