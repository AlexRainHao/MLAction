"""pass"""

from typing import Dict, Text, List, Any, Set, Union, Optional

import os
import numpy as np

class InputExample:
    def __init__(self, intent, feature):
        self.intent = intent
        self.feature = feature

        self.data = {}
        self.set("intent", intent)
        self.set("feature", feature)

    def get(self, prof, default = None):
        return self.data.get(prof, default)

    def set(self, prof, value):
        self.data[prof] = value

def create_folder(model_dir):
    """pass"""
    os.makedirs(model_dir, exist_ok = True)

def create_intent_dict(intents: Union[List, Set]) -> Dict:
    """obtain intent to index dictionary"""

    uni_intents = set(intents)

    return {x: idx for idx, x in enumerate(uni_intents)}


def create_intent_token_dict(intents: Union[List, Set], symbols: Optional[Text]):
    """pass"""

    tokens = set(token for intent in intents for token in intent.split(symbols))

    return {token: idx for idx, token in enumerate(tokens)}


def create_encoded_intent_bag(intent_dict: Dict,
                              symbols: Union[Text, None]) -> np.ndarray:
    """pass"""
    if symbols:
        uni_intents = list(intent_dict.keys())
        token_dict = create_intent_token_dict(uni_intents, symbols)

        encoded_intent_bag = np.zeros((len(uni_intents), len(token_dict)))

        for int, idx in intent_dict.items():
            for tok in int.split(symbols):
                encoded_intent_bag[idx, token_dict[tok]] = 1

        return encoded_intent_bag

    else:
        return np.eye(len(intent_dict))