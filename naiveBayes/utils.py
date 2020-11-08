import numpy as np
from typing import List, Optional, Dict, Tuple


def simulated_data() -> Tuple[List[List[str]], List[int]]:
    rows = [["my", "dog", "has", "flea", "problems", "help", "please"],
            ["maybe", "not", "take", "him", "to", "dog", "park", "stupid"],
            ["my", "dalmation", "is", "so", "cute", "I", "love", "him"],
            ["stop", "posting", "stupid", "worthless", "garbage"],
            ["mr", "licks", "ate", "my", "steak", "how", "to", "stop", "him"],
            ["quit", "buying", "worthless", "dog", "food", "stupid"]]

    labels = [0, 1, 0, 1, 0, 1]
    return rows, labels


def createVocab(token_rows: Optional[List[List[str]]]) -> List[str]:
    vocab_set = set()
    for row in token_rows:
        vocab_set = vocab_set | set(row)

    return list(vocab_set)


def word2bagVec(tokens: Optional[List[int]],
                 vocab: Optional[List]) -> List[int]:

    tokenVec = [0] * len(vocab)

    for token in tokens:
        tokenVec[vocab.index(token)] += 1

    return tokenVec


def load_data_file(fname: str) -> object:
    pass

# if __name__ == '__main__':
#     rows, labels = simulated_data()
#     vocab = createVocab(rows)
#
#     tokenVec = word2showVec(rows[0], vocab)
#     print(tokenVec)


