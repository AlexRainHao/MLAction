import numpy as np
import utils
import time


class optStruct():
    def __init__(self, x_features, y_labels, C, toler):
        self.x_features = x_features
        self.y_labels = y_labels
        self.C = C
        self.toler = toler
        self.m = x_features.shape[0]
        self.alphas = np.zeros(shape = (self.m, 1))
        self.beta = 0
        self.eCache = np.zeros(shape = (self.m, 2))

    def set_kernel(self, kernel_paras = ("rbf", 1.3)):
        self.kernel_val = utils.calKernelVal(self.x_features,
                                             kernel_paras[0],
                                             kernel_paras[1])


def innerL(i, objStruct):
    Ei = utils.calEk(objStruct, i)

    if ((objStruct.y_labels[i] * Ei < -objStruct.toler).all() and objStruct.alphas[i] < objStruct.C) or \
            ((objStruct.y_labels[i] * Ei > objStruct.toler).all() and objStruct.alphas[i] > 0):
        j, Ej = utils.selectJ(i, objStruct, Ei)


        alphaIold = objStruct.alphas[i].copy()
        alphaJold = objStruct.alphas[j].copy()

        if (objStruct.y_labels[i] != objStruct.y_labels[j]):
            L = max(0, objStruct.alphas[j] - objStruct.alphas[i])
            H = min(objStruct.C, objStruct.C + objStruct.alphas[j] - objStruct.alphas[i])
        else:
            L = max(0, objStruct.alphas[i] + objStruct.alphas[j] - objStruct.C)
            H = min(objStruct.C, objStruct.alphas[i] + objStruct.alphas[j])

        if L == H:
            return 0


        eta = objStruct.kernel_val[i, i] + objStruct.kernel_val[j, j] - 2 * objStruct.kernel_val[i, j]

        # if eta >= 0:
        #     return 0

        objStruct.alphas[j] += (objStruct.y_labels[j] * (Ei - Ej) / eta)
        objStruct.alphas[j] = utils.clib_boundary(objStruct.alphas[j], H, L)
        utils.updateEk(objStruct, j)

        if abs(objStruct.alphas[j] - alphaJold) < 1e-5:
            return 0
        objStruct.alphas[i] += objStruct.y_labels[i] * objStruct.y_labels[j] * \
                               (alphaJold - objStruct.alphas[j])
        utils.updateEk(objStruct, i)

        # update beta
        b1 = objStruct.beta - Ei - objStruct.y_labels[i] * (objStruct.alphas[i] - alphaIold) * \
             objStruct.kernel_val[i, i] - \
             objStruct.y_labels[j] * (objStruct.alphas[j] - alphaJold) * \
             objStruct.kernel_val[i, j]

        b2 = objStruct.beta - Ei - objStruct.y_labels[i] * (objStruct.alphas[i] - alphaIold) * \
             objStruct.kernel_val[i, j] - \
             objStruct.y_labels[j] * (objStruct.alphas[j] - alphaJold) * \
             objStruct.kernel_val[j, j]

        if objStruct.alphas[i] > 0 and objStruct.alphas[j] < objStruct.C:
            objStruct.beta = b1
        elif objStruct.alphas[j] > 0 and objStruct.alphas[j] < objStruct.C:
            objStruct.beta = b2
        else:
            objStruct.beta = .5 * b1 + .5 * b2
        return 1

    else:
        return 0

def smoPlat(x_features, y_labels, C, toler, maxIter, kernel_params = ("rbf", 1.3)):
    if len(y_labels.shape) == 1:
        y_labels = np.expand_dims(np.asarray(y_labels), -1)

    objStruct = optStruct(x_features, y_labels, C, toler)
    objStruct.set_kernel(kernel_params)

    iter = 0; alphaPairChanged = 0; entireSet = True

    while (iter < maxIter) and (alphaPairChanged > 0) or entireSet:
        alphaPairChanged = 0
        if entireSet:
            for i in range(objStruct.m):
                alphaPairChanged += innerL(i, objStruct)
            iter += 1

        else:
            nonBoundIs = np.nonzero((objStruct.alphas > 0) * (objStruct.alphas < objStruct.C))[0]
            for i in nonBoundIs:
                alphaPairChanged += innerL(i, objStruct)
            iter += 1


        if entireSet: entireSet = False
        elif alphaPairChanged == 0: entireSet = True

    return objStruct.alphas, objStruct.beta


def calTheta(alphas, x_features, y_labels):
    w = np.sum(np.expand_dims(alphas.flatten() * y_labels.flatten(), -1) * x_features, axis = 0)
    return w



if __name__ == '__main__':
    x_features, y_labels = utils.load_data_from_txt("./data/testSetRBF.txt")
    _bt = time.time()

    alphas, beta = smoPlat(x_features, y_labels, 200, 1e-4, 10000)

    print('traing time: %f' % (time.time() - _bt))
    w = calTheta(alphas, x_features, y_labels)

    support_point_index = np.where(alphas.flatten() > 0)[0]
    utils.plotWithboundary(x_features, y_labels, w, beta, support_point_index, "svm_platt_smo_rbf.png")





