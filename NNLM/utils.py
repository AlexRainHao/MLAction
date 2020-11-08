import pandas as pd
import re
import torch
import random
import os

UNK = "UNK"
PUN = "PUN"
START = "START"

class set_config:
    """a config for train or test"""
    batch_size = 64
    epoch = 10
    save_model = "./output"

    window_size = 3
    hidden_dim = 50
    vector_dim = 60

    whether_direct = True

    learning_rate = 1e-3

    vocab_size = 21530

    use_gpu = False



def load_data(fname):
    """load raw experimental data"""

    data = pd.read_csv(fname, encoding = 'ISO-8859-1')
    return data["text"].tolist()


def remove_urls(data_list):
    """a simplified text cleanup method"""

    pattern = re.compile("http.*")
    at_pattern = re.compile("@\w+\d?[: ]")

    for idx, line in enumerate(data_list):
        _line = pattern.sub('', line) # no url
        _line = at_pattern.sub('', _line) # no @ name
        _line = re.sub('[\'#]', '', _line)
        _line = re.sub('( )+', ' ', _line)
        _line = re.sub('\\x89.* ', '', _line)

        data_list[idx] = _line.strip()

    return data_list



def get_unique_word(data_list):
    """get train dataset vocabulary and corresponding tokens for each sentence"""

    data_set = set()

    for idx, line in enumerate(data_list):
        _word_list = line.split(' ')
        [data_set.add(x) for x in _word_list]
        data_list[idx] = _word_list

    word2idx = {w: (i+3) for i, w in enumerate(data_set)}
    word2idx[UNK] = 0
    word2idx[PUN] = 1
    word2idx[START] = 2

    idx2word = {i: w for w, i in word2idx.items()}

    return word2idx, idx2word, len(data_set), data_list


def get_test_token(data_list):
    """get test dataset each sentence tokens"""

    return [x.split(' ') for x in data_list]


def make_batch(data_list, word2idx, window_size = 3, batch_size = 64, if_gpu = True):
    """get batch data for training or prediction"""

    if if_gpu:
        device = "cuda"
    else:
        device = "cpu"

    batch_idx = 0
    batch = []
    length = len(data_list)

    random.shuffle(data_list)

    for line_idx, line in enumerate(data_list):
        if not line:
            continue

        line = [START] * 2 + line

        window_token = list(zip(*[line[i:] for i in range(window_size)]))

        token_length = len(window_token)

        for idx, token in enumerate(window_token):
            batch.append([word2idx.get(word, '') if word2idx.get(word, '') else word2idx.get(UNK) for word in token])
            batch_idx += 1

            if batch_idx > batch_size - 1 or (line_idx == length - 1 and idx == token_length - 1):
                batch_idx = 0
                if if_gpu:
                    yield torch.LongTensor(batch).cuda(device)
                else:
                    yield torch.LongTensor(batch)
                batch = []


def mkdir_folder(path):
    """create a directory"""
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)