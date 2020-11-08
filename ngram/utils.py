import pandas as pd
import re
from collections import Counter, defaultdict
from math import log

UNK = "UNK"

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


def split_token(data_list):
    return [[UNK] + line.split(' ') + [UNK] for line in data_list]


def get_vocabulary(token_list):
    """get character vocabulary for given dataset"""
    char2idx = {}
    char2idx[UNK] = 0

    for idx, line in enumerate(token_list):
        # _word_list = [UNK] + line.split(' ') + [UNK]

        for char in line:
            if char2idx.get(char): continue
            else: char2idx[char] = len(char2idx)


    idx2char = {idx: chr for chr, idx in char2idx.items()}

    return char2idx, idx2char


def get_ngram_from_sent(tokens, K = 3):
    """pass"""
    tokens = list(zip(*[tokens[i:] for i in range(K)]))

    return tokens


def get_ngram_token(data_list, K = 3):
    """generate n-gram element for dataset

    Returns
    -------
    gram_tokens: (seq_len, K)
    """
    gram_tokens = []

    for idx, sentence in enumerate(data_list):
        tokens = get_ngram_from_sent(tokens = sentence, K = K)
        gram_tokens.append(tokens)

    return gram_tokens

def _ngram_counter_uni(data_list, K = 1, if_laplace = True):
    """get a dictionary of token conditional probability for training data where K = 1

    Returns
    -------
    gram_prob: {'UNK': -2.128231705849268, 'RT': -4.430816798843313}
    """
    assert K == 1, "K must equal to 1 calling ngram counter"

    gram_prob = defaultdict(float)
    for sentence in data_list:
        for token in sentence:
            gram_prob[token] += 1

    all_count = sum(gram_prob.values())

    for key, val in gram_prob.items():
        if if_laplace:
            gram_prob[key] = log(val / (all_count + len(gram_prob)))
        else:
            gram_prob[key] = log(val / all_count)

    return gram_prob


def _ngram_counter_multi(data_list, K = 3, if_laplace = True, exv = 1e-80):
    """get a dictionary of token conditional probability for training data where K > 1

    Returns
    -------
    gram_prob: {'UNK RP': {'Rep.': -3.1918471524802814, '1st': -3.1918471524802814}}
    """
    assert K > 1, "K must greater than 1 calling ngram counter"

    cocurr = defaultdict(int)
    cond = defaultdict(int)

    gram_prob = defaultdict(dict)

    # call token counter
    for sentence in data_list:
        cocurr_gram = get_ngram_from_sent(sentence, K = K)

        cond_gram = get_ngram_from_sent(sentence, K = K -1)

        for gram in cocurr_gram:
            cocurr[' '.join(gram)] += 1
        for gram in cond_gram:
            cond[' '.join(gram)] += 1

    # call condition probability
    for gram, count in cocurr.items():
        tt, ct = gram.split(' ')[-1], ' '.join(gram.split(' ')[:-1])

        if if_laplace:
            prob = log((cocurr.get(gram, 0) + 1) /
                       (cond.get(ct, 0) + len(cond)))

        else:
            prob = log(cocurr.get(gram, 0) / (cond.get(ct) + exv) + exv)

        gram_prob[ct][tt] = prob

    return gram_prob


