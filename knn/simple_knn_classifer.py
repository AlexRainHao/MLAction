import utils
import numpy as np
from collections import defaultdict


def classifer(xIn: np.ndarray,
              dataset: np.ndarray,
              labels: np.ndarray,
              K: int = 10) -> int:
    """
    @param xIn: one dimension of np.ndarray
    @param dataset: full dataset with dimension of m * n
    @param labels: with dimension of m * 1
    @param K: the K nearest sample to choose
    """
    labels = labels.flatten()
    distance = (dataset - xIn) ** 2
    sqlDistance = np.sqrt(distance.sum(axis = 1))

    sortedLabel = sqlDistance.argsort()
    classCount = defaultdict(int)

    for i in range(K):
        currLabel = labels[sortedLabel[i]]
        classCount[currLabel] += 1

    maxVotedLabel = sorted(classCount.items(), key = lambda val: val[1], reverse = True)

    return maxVotedLabel[0][0]


def knn_classifer_func(x_features, y_labels, split_ratio = 0.1, K = 10):
    """
    @param x_features: normalized data as format of np.ndarray
    @param y_labels: with dimension of m * 1
    @param split_ratio: train-test dataset split
    """
    m, n = x_features.shape
    errorCount = .0

    tesetIndex = np.random.choice(list(range(m)), size = int(m * split_ratio))

    for idx in tesetIndex:
        currFeature = x_features[idx, :]
        currPred = classifer(currFeature, x_features, y_labels, K = K)

        if currPred != y_labels[idx][0]:
            errorCount += 1

    print(f"Error Rate: {errorCount / m}")


if __name__ == '__main__':
    x_feature, y_label, _ = utils.load_data("./data/datingTestSet.txt")
    x_feature, _, _ = utils.normalize_feature(x_feature)

    knn_classifer_func(x_feature, y_label)