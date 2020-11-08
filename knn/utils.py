import numpy as np
from typing import Tuple, Dict

def load_data(fname):
    x_feature = []
    y_label = []

    with open(fname) as f:
        raw_content = f.readlines()

    for line in raw_content:
        x_1, x_2, x_3, y = line.strip().split('\t')

        x_feature.append([float(x_1), float(x_2), float(x_3)])
        y_label.append(y)

    x_feature = np.asarray(x_feature)

    y_label, label_dict = label_to_feature(y_label)
    y_label = np.expand_dims(np.asarray(y_label), -1)

    return x_feature, y_label, label_dict

def label_to_feature(y_label: list) -> Tuple[np.ndarray, Dict]:
    unique_label = np.unique(y_label)

    label2id = dict(zip(unique_label, range(len(unique_label))))

    return np.asarray([label2id.get(x) for x in y_label]), label2id


def normalize_feature(x_feature):
    min_val = x_feature.min(0)
    max_val = x_feature.max(0)
    range_val = max_val - min_val

    normalData = (x_feature - min_val) / (range_val)

    return normalData, range_val, min_val


# if __name__ == '__main__':
#     x, y, label_dict = load_data(fname = "./data/datingTestSet.txt")