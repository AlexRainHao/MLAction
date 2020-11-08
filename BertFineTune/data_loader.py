"""
Convert example to feature,
and convert dataset to torch `DataLoader`
"""


import warnings
warnings.filterwarnings("ignore")


from typing import Any, List, Tuple, Optional, Dict, Union

from collections import defaultdict
from copy import deepcopy
import codecs

import torch
from torch.utils.data import DataLoader, Dataset


class InputExample():
    """wrapper for input example"""

    def __init__(self, text, intent = ''):
        self.text = text
        self.intent = intent


class IntentDataset():
    """pass"""

    def __init__(self, intent):
        """

        Parameters
        ----------
        intent: List[str]
        """
        self.intent = intent
        self.int2idx = defaultdict(int)

    def encoder(self, label):

        if label in self.int2idx:
            pass
        else:
            self.int2idx[label] = len(self.int2idx)


    def reversed(self):
        return {id_v: int_v for int_v, id_v in self.int2idx.items()}


    def __call__(self):
        for lab in self.intent:
            self.encoder(lab)

        return self.int2idx, self.reversed()



class NluClsDataset(Dataset):
    """pass"""

    def __init__(self, message, tokenizer, max_len, label_dict):
        """

        Parameters
        ----------
        message: Any, NLU input data
        tokenizer: Tokenizer
        max_len: int, assign max sequences length
        label_dict: dict, a intent to idx dictionary
        """
        self.message = message
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_dict = label_dict

    def __len__(self):
        return len(self.message)

    def __getitem__(self, idx):
        feature = self.message[idx].text
        label = self.message[idx].intent

        encoding = self.tokenizer.encode_plus(
            feature,
            add_special_tokens = True,
            max_length = self.max_len,
            truncation = True,
            return_token_type_ids = False,
            padding = "max_length",
            return_attention_mask = True,
            return_tensors = "pt",
        )

        return {
            "text": feature,
            "input_ids": encoding["input_ids"].flatten(),
            "input_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(self.label_dict.get(label), dtype = torch.long)
        }


class NluClsDataLoader:
    """pass"""
    def __init__(self, message, tokenizer, max_len, batch_size, label_dict):
        self.dl = self.create_data_loader(message, tokenizer, max_len, label_dict)
        self.batch_size = batch_size


    def create_data_loader(self, message, tokenizer, max_len, label_dict):
        dl = NluClsDataset(
            message = message,
            tokenizer = tokenizer,
            max_len = max_len,
            label_dict = label_dict
        )

        return dl


    def refresh(self):
        dl = deepcopy(self.dl)
        return DataLoader(dl, batch_size = self.batch_size,
                          num_workers = 4, shuffle = True)


