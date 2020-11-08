import codecs
import random
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def load_data_from_txt(fname):
    with codecs.open(fname, encoding = 'utf-8') as f:
        lines = f.read().split('\n')

    x_features = []
    y_labels = []

    for line in lines:
        if not line:
            continue

        x_1, x_2, y = line.strip().split('\t')
        x_features.append([float(x_1), float(x_2)])
        y_labels.append(float(y))

    x_features = np.asarray(x_features)
    y_labels = np.asanyarray(y_labels)

    return x_features, y_labels


def clib_boundary(a_val, H, L):
    if a_val < L:
        return L

    if a_val > H:
        return H

    return a_val


def selectJrand(i, m):
    j = i
    while (j == i):
        j = random.choice(range(m))

    return j


def calEk(objStruct, k):
    gxk = np.matmul((objStruct.alphas * objStruct.y_labels).T, 
                    np.matmul(objStruct.x_features, objStruct.x_features[k, :])) + objStruct.beta
    Ek = gxk - objStruct.y_labels[k]
    return Ek


def selectJ(i, objStruct, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0

    objStruct.eCache[i] = [1, Ei]
    validEcacheList = np.nonzero(objStruct.eCache[:, 0])[0]

    if len(validEcacheList) > 1:
        for k in validEcacheList:
            if k == i: continue

            Ek = calEk(objStruct, k)
            deltaaEk = np.abs(Ei - Ek)
            if (deltaaEk > maxDeltaE):
                maxK = k
                maxDeltaE = deltaaEk
                Ej = Ek
        return maxK, Ej

    else:
        j = selectJrand(i, objStruct.m)
        Ej = calEk(objStruct, j)

    return j, Ej

def updateEk(objStruct, k):
    Ek = calEk(objStruct, k)
    objStruct.eCache[k] = [1, Ek]



def calKernelVal(x_features, kernel_type, kernel_para = None):
    kernel_val = np.zeros(shape = (x_features.shape[0], x_features.shape[0]))
    m = x_features.shape[0]

    if kernel_type == "normal":
        kernel_val = np.matmul(x_features, x_features.T)

    elif kernel_type == "rbf":
        if not kernel_para:
            kernel_para = 1.3

        for i in range(m):
            # for j in range(i, m):
            #     _deltaRow = x_features[i,:]
            _deltaRow = x_features - x_features[i, :]
            _sub_k_val = np.sum(_deltaRow * _deltaRow, axis = 1)
            kernel_val[i,:] = np.exp(_sub_k_val / (-1 * kernel_para ** 2))

    else:
        raise NameError("No support for assigned kernel type")

    return kernel_val




def plotWithboundary(x_features, y_labels, w, beta, support_vecotr_index, figname = "SVM_fit"):
    x_features = np.asarray(x_features)
    y_labels = np.asarray(y_labels).flatten()
    data_frame = pd.DataFrame({"x1": x_features[:, 0], "x2": x_features[:, 1],
                               "y": y_labels})

    sns.lmplot(x = "x1", y = "x2", data = data_frame, fit_reg = False,
               hue = "y")


    sort_index = data_frame["x1"].argsort().values

    y_plot = [x_features[sort_index[0], 0] * w[0] + x_features[sort_index[0], 1] * w[1] + beta,
              x_features[sort_index[-1], 0] * w[0] + x_features[sort_index[-1], 1] * w[1] + beta]
    plt.plot(x_features[[sort_index[0], sort_index[-1]], 0], y_plot, color = "lightblue")

    for _idx in support_vecotr_index:
        plt.scatter(data_frame["x1"][_idx], data_frame["x2"][_idx],
                    marker = "o", color = "gray", s = 300, alpha = .5, )

    plt.xlim(data_frame["x1"].min(),
             data_frame["x1"].max())
    plt.ylim(data_frame["x2"].min(),
             data_frame["x2"].max())

    plt.savefig(figname)
    plt.show()

# if __name__ == '__main__':
#     a, b = load_data_from_txt(fname = './data/testSet.txt')
#     print(a[0])
#     print(b[0])