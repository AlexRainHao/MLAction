import numpy as np
from utils import *
from typing import List, Optional, Dict, Tuple, Any


def trainNaiveBayes(x_features: np.ndarray,
                    y_labels: Any) -> Tuple[np.ndarray, np.ndarray, int]:
    m_rows = x_features.shape[0]
    m_vocab = x_features.shape[1]

    p_y_postive = np.sum(y_labels * 1.) / m_rows
    p_x_y1 = np.ones(m_vocab)
    p_x_y0 = np.ones(m_vocab)
    p_x_1, p_x_0 = 2., 2.

    for x, y in zip(x_features, y_labels):
        if y == 1:
            p_x_y1 += x
            p_x_1 += np.sum(x)

        else:
            p_x_y0 += x
            p_x_0 += np.sum(x)

    p1Vec = np.log(p_x_y1 / p_x_1)
    p0Vec = np.log(p_x_y0 / p_x_0)

    return p1Vec, p0Vec, p_y_postive


def NBClassifer(target_vec: np.ndarray,
                p0Vec: np.ndarray,
                p1Vec: np.ndarray, p_1: float) -> int:
    pp = np.sum(target_vec * p1Vec) + np.exp(p_1)
    pn = np.sum(target_vec * p0Vec) + np.exp(1 - p_1)

    if pp > pn:
        return 1
    else:
        return 0






# if __name__ == '__main__':
#     rows, labels = simulated_data()
#     vocab = createVocab(rows)
#
#     x_features = np.asarray([word2bagVec(row, vocab) for row in rows])
#
#     p1Vec, p0Vec, pp = trainNaiveBayes(x_features, np.asarray(labels))
#
#     pre_vec = np.asarray(word2bagVec(["stupid", "garbage"], vocab))
#     print(NBClassifer(pre_vec, p0Vec, p1Vec, pp))