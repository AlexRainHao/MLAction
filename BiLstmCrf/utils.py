import codecs
from collections import defaultdict
from copy import deepcopy

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

def load_raw_data(fpath):
    """
    load raw samples dataset and convert to pairs of sentences and tags

    :rtype: List[Tuple[List[Text], List[Text]]], data file path
    E.X.
        (['上', '海', '浦', '东', '开', '发', '与', '法', '制', '建', '设', '同', '步'],
        ['B-LOC', 'E-LOC', 'B-LOC', 'E-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'])
    """
    lines = codecs.open(fpath).read().split('\n')

    examples = []

    unit_tags = set()

    sub_text = []
    sub_tag = []
    for line in lines:
        if line.strip() != "":
            _text, _tag = line.strip().split(' ')
            sub_text.append(_text)
            sub_tag.append(_tag)
            # unit_tags.add(_tag.split('-')[-1])
            unit_tags.add(_tag)

        else:
            examples.append((sub_text, sub_tag))
            sub_text = []
            sub_tag = []

    print(f"total samples {len(examples)}")
    print(f"total tags {len(unit_tags)}")

    return examples, unit_tags

class Vocab():
    """create token vocab"""

    UNK = "UNK"

    def __init__(self):
        self.word2idx = {self.UNK: 0}
        self.word2count = {}
        # self.idx2word = {}

    def add_token(self, token):
        """
        add a certain token to vocab
        :type token: str
        """

        if token not in self.word2idx:
            pidx = len(self.word2idx)
            self.word2idx[token] = pidx
            self.word2count[token] = 1

        else:
            self.word2count[token] += 1

    def add_sentence(self, sentence):
        """
        add each token to vacab for a sentence
        :param sentence: List[str]
        """

        for token in sentence:
            self.add_token(token)

    def filter_vocab(self, threshold = 2):
        """
        filter tokens exists in current vocab and convert it to UNK
        """

        remain_tokens = list(filter(lambda x: self.word2count[x] > threshold, self.word2count))
        self.word2count = {token: self.word2count[token] for token in remain_tokens}

        token2idx = dict(zip([self.UNK] + remain_tokens, range(len(remain_tokens) + 1)))
        self.word2idx = token2idx
        self.get_idx_to_token()

    def get_idx_to_token(self):
        """get idx2token dict"""

        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def get_id_from_token(self, token):
        """obtain corresponding token id"""
        return self.word2idx.get(token, self.word2idx.get(self.UNK))

    def get_token_from_id(self, idx):
        """obtain corresponding token from id"""
        return self.idx2word.get(idx, self.UNK)

    def get_id_from_seq(self, sentence):
        """obtain corresponding sequence token ids
        :type sentence: List[str]
        """
        return [self.get_id_from_token(s) for s in sentence]

    def get_token_from_ids(self, idxs):
        """obtain corresponding tokenss from ids
        :type idxs: List[int]
        """
        return [self.get_token_from_id(id) for id in idxs]

    def size(self):
        """vocab size"""
        return self.__len__()

    def __len__(self):
        return len(self.word2idx)

    def __repr__(self):
        return f"token vocab length {self.__len__()}"

class Tags():
    """create tags vocab"""

    start_tag = "<START>"
    end_tag = "<STOP>"

    def __init__(self):
        self.tag2idx = {self.start_tag: 0, self.end_tag: 1}
        self.idx2tag = {}

    def add_tag(self, tag):
        """add a certain tag to vocab"""
        if tag not in self.tag2idx:
            self.tag2idx[tag] = len(self.tag2idx)

    def add_tag_seq(self, tag_seq):
        """add sequence of tag to vocab
        :type tag_seq: List[str]
        """
        [self.add_tag(tag) for tag in tag_seq]

    def get_idx_to_tag(self):
        self.idx2tag = {v: k for k, v in self.tag2idx.items()}

    def get_id_from_tag(self, tag):
        """obtain corresponding tag id"""
        return self.tag2idx.get(tag, 1)

    def get_tag_from_id(self, idx):
        """obtain corresponding tag from id"""
        return self.idx2tag.get(idx, self.end_tag)

    def get_id_from_seq(self, tags):
        """obtain corresponding tag ids"""
        return [self.get_id_from_tag(s) for s in tags]

    def get_tag_from_ids(self, idxs):
        """obatin corresponding tags from ids"""
        return [self.get_tag_from_id(id) for id in idxs]

    def size(self):
        """vocab size"""
        return self.__len__()

    def __len__(self):
        return len(self.tag2idx)

    def __repr__(self):
        return f"tag vocab length {self.__len__()}"

class trainDataset(Dataset):
    """pass"""

    def __init__(self, vocab_op, tag_op, examples):
        """
        :type vocab_op: Vocab
        :type tag_op: Tags
        :type examples: List[Tuple]
        """
        self.vocab_op = vocab_op
        self.tag_op = tag_op
        self.examples = examples

    def pad_seq(self, x, y):
        """padding for batch"""
        pass

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        sentence = self.examples[idx][0]
        tag = self.examples[idx][1]

        x = self.vocab_op.get_id_from_seq(sentence)
        y = self.tag_op.get_id_from_seq(tag)

        # return {
        #     "x": torch.tensor(x, dtype = torch.long),
        #     "y": torch.tensor(y, dtype = torch.long)
        # }
        return (torch.tensor(x, dtype = torch.long),
                torch.tensor(y, dtype = torch.long))

class trainDataLoader:
    """pass"""

    def __init__(self, dl, batch_size,
                 pad_seq_val = 0,
                 pad_tag_val = 0):
        """
        :type dl: trainDataset
        """
        self.dl = dl
        self.pad_seq_val = pad_seq_val
        self.pad_tag_val = pad_tag_val
        self.batch_size = batch_size

    def refresh(self):
        """generate a epoch data loader generator"""
        dl = deepcopy(self.dl)
        return DataLoader(dl,
                          batch_size = self.batch_size,
                          shuffle = True,
                          collate_fn = self.pad_seq)

    def pad_seq(self, batch):
        """pad for variant length of batch"""

        seqs, labels = zip(*batch)

        seqs = pad_sequence(seqs,
                            batch_first = True,
                            padding_value = self.pad_seq_val)

        labels = pad_sequence(labels,
                              batch_first = True,
                              padding_value = self.pad_tag_val)

        return seqs, labels

if __name__ == '__main__':
    examples, unit_tags = load_raw_data("./data/demo.dev.char")
    vocab_op = Vocab()
    tags_op = Tags()

    print(examples[0])
    print(unit_tags)

    for sen, tag in examples:
        vocab_op.add_sentence(sen)
        tags_op.add_tag_seq(tag)

    vocab_op.get_idx_to_token()
    vocab_op.filter_vocab(0)