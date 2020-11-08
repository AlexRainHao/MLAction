"""
Bert Multi-Class Classifier from training to deployment

A chinese fine-tune intent classifier from Bert
Based on `Bert` from paper:
    `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`
    https://arxiv.org/pdf/1810.04805.pdf
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import warnings
warnings.filterwarnings("ignore")

import logging
import os
import os.path as opt
import json
import shutil

from typing import List, Text, Any, Optional, Dict, Tuple

from data_loader import *
from utils import *
from train import *
from model import BertFineTuneModel

try:
    import torch
    import torch.nn as nn
    from transformers import BertTokenizer, BertModel
    from transformers import BertConfig as BC

except ImportError:
    raise ImportError("No torch package found")

try:
    import cPickle as pickle
except ImportError:
    import pickle

logger = logging.getLogger(__name__)
logging.basicConfig(level = logging.DEBUG)


BERTPATH = "bert"
INTENT_RANKING_LENGTH = 2

class BertIntentClassifier():
    """
    Supervised Bert fine-tune model training
    """

    defaults = {
        # global training params
        "batch_size": 32,
        "max_seq_len": 64,
        "epochs": 10,
        "walking_epoch_visual": 1,
        "lr": 2e-5,
        "dropout": 0.2,

        # pre-trained model
        "pre_path": "./Bert/torch_model",
        "bert_config": "bert_config.json",
        "bert_model": "bert_model.bin",
        "vocab_config": "vocab.txt"
    }


    def __init__(self,
                 component_config = None,
                 model = None,
                 int2idx = None,
                 idx2int = None,
                 ):
        """
        Parameters
        ----------
        component_config: dict
        """
        self.component_config = component_config if component_config else {}
        self.device = self._load_device()

        self.model = model
        if self.model:
            self.model.to(self.device)

        self.batch_size = self._load_from_default("batch_size")
        self.max_seq_len = self._load_from_default("max_seq_len")
        self.epochs = self._load_from_default("epochs")
        self.lr = self._load_from_default("lr")
        self.dropout = self._load_from_default("dropout")

        self.walking_epoch_visual = self._load_from_default("walking_epoch_visual")

        self.pre_path = self._load_from_default("pre_path")
        self.bert_config = self._load_from_default("bert_config")
        self.bert_model = self._load_from_default("bert_model")
        self.vocab_config = self._load_from_default("vocab_config")

        self.int2idx = int2idx
        self.idx2int = idx2int

        self._check_encode_status()


    def _load_from_default(self, tag):

        return self.component_config.get(tag, self.defaults.get(tag))


    @staticmethod
    def _load_device():
        """
        default load gpu with cuda tag of 0
        """

        gpus_count = torch.cuda.device_count()

        device = torch.device("cuda:%d" % 0) if gpus_count else torch.device("cpu")

        return device


    def _check_encode_status(self):
        """check the pre-trained or fine-tuning model whether exists"""

        status = True
        if not opt.exists(self.pre_path):
            logger.error(f"Not find bert pretrained path {self.pre_path}")
            status = False

        if not opt.exists(opt.join(self.pre_path, self.bert_config)):
            logger.error(f"Not find bert config file {self.bert_config}")
            status = False

        if not opt.exists(opt.join(self.pre_path, self.bert_model)):
            logger.error(f"Not find bert pretrained model {self.bert_model}")
            status = False

        if not opt.exists(opt.join(self.pre_path, self.vocab_config)):
            logger.error(f"Not find bert pretrained model {self.bert_model}")
            status = False

        if not status:
            raise FileNotFoundError("Not find a legal bert model path")


    def _create_intent_dict(self, training_data) -> Any:
        """create label to integer dictionary"""

        intents = [x.intent for x in training_data]

        int2idx, idx2int = IntentDataset(intents)()
        self.int2idx = int2idx
        self.idx2int = idx2int


    def train(self,
              training_data: List[InputExample],
              **kwargs) -> None:
        """training pipeline start from load dataset and then conduct fine-tune training"""

        encoder, tokenizer = load_pretrained(mpath = self.pre_path,
                                             config = self.bert_config,
                                             model = self.bert_model)

        self._create_intent_dict(training_data)

        f_data_loader = NluClsDataLoader(message = training_data,
                                         tokenizer = tokenizer,
                                         max_len = self.max_seq_len,
                                         batch_size = self.batch_size,
                                         label_dict = self.int2idx)

        train_pipeline = TrainingPipeLine(epochs = self.epochs,
                                          walking_epoch_visual = self.walking_epoch_visual,
                                          lr = self.lr,
                                          dropout = self.dropout,
                                          device = self.device,
                                          int2idx = self.int2idx,
                                          idx2int = self.idx2int)

        self.model = train_pipeline.train(encoder,
                                          data_loader = f_data_loader)


    def process(self, message: InputExample,
                **kwargs) -> Any:
        """inference procedure"""

        logger.info("predicting...")

        intent = {"name": None, "confidence": 0.}
        intent_ranking = []

        tokenizer = load_pretrained_tokenizer(self.pre_path)
        decode_pipeline = TrainingPipeLine(device = self.device,
                                           int2idx = self.int2idx,
                                           idx2int = self.idx2int)

        if message.text.strip():
            score, label = decode_pipeline.decode(model = self.model,
                                                  tokenizer = tokenizer,
                                                  max_len = self.max_seq_len,
                                                  text = message.text,
                                                  ranks = INTENT_RANKING_LENGTH)

            intent = {"name": label[0], "confidence": score[0]}

            intent_ranking = [{"name": x, "confidence": y} for x,y in zip(label[1:],
                                                                          score[1:])]

        res = {"intent": intent, "intent_ranking": intent_ranking}

        return res


    @classmethod
    def load(cls,
             model_dir = None,
             **kwargs) -> Any:
        """load model from config file"""

        with codecs.open(opt.join(model_dir, "train_config.json"), encoding="utf-8") as f:
            meta = json.load(f)


        t_path = meta.get("pre_path")

        if model_dir and t_path:
            t_path = opt.join(model_dir, t_path)
            e_path = opt.join(opt.join(t_path, meta.get("intent_dict")))


            with open(e_path, "rb") as f:
                intent_dict = pickle.load(f)


            # assign weight to fine-tune model
            encoder = load_encoder(t_path, meta.get("bert_config"))

            _state_dict = torch.load(opt.join(t_path, meta.get("bert_model")))

            fine_tune_model = BertFineTuneModel(encoder,
                                                len(intent_dict.get("int2idx")))

            fine_tune_model.load_state_dict(state_dict = _state_dict)

            fine_tune_model.eval()

            meta["pre_path"] = t_path

            return BertIntentClassifier(component_config=meta,
                                        model=fine_tune_model,
                                        int2idx=intent_dict.get("int2idx"),
                                        idx2int=intent_dict.get("idx2int"))



    @staticmethod
    def copy_file_to_dir(input_file, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        shutil.copy2(input_file, output_dir)


    def persist(self, model_dir):
        if self.model is None:
            return {"pre_path": None}

        t_path = opt.join(model_dir, "bert_model.bin")

        try:
            os.makedirs(opt.dirname(t_path))
        except FileExistsError:
            pass

        model_to_save = (self.model.module if hasattr(self.model, "module") else self.model)
        torch.save(model_to_save.state_dict(), t_path)


        self.copy_file_to_dir(input_file = opt.join(self.pre_path,
                                                    self.bert_config),
                              output_dir = opt.join(model_dir))
        self.copy_file_to_dir(input_file = opt.join(self.pre_path,
                                                    self.vocab_config),
                              output_dir = opt.join(model_dir))


        e_path = opt.join(model_dir, "intent_dict.pkl")
        with open(e_path, "wb") as f:
            pickle.dump({"int2idx": self.int2idx,
                         "idx2int": self.idx2int},
                        f)

        config = {"batch_size": self.batch_size,
                  "max_seq_len": self.max_seq_len,
                  "epochs": self.epochs,
                  "walking_epoch_visual": self.walking_epoch_visual,
                  "lr": self.lr,
                  "dropout": self.dropout,
                  "pre_path": opt.abspath(model_dir),
                  "bert_config": "bert_config.json",
                  "bert_model": "bert_model.bin",
                  "vocab_config": "vocab.txt",
                  "intent_dict": "intent_dict.pkl"}

        with codecs.open(opt.join(model_dir, "train_config.json"), 'w', encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii = False, indent = 2)

        logger.info("trained model save to %s" % model_dir)

def toy_exam():
    # load dataset
    import codecs
    a = BertIntentClassifier()
    with codecs.open("./data/moods.txt", encoding = "utf-8") as f:
        lines = f.read().split("\n")

    dataset = []
    for line in lines:
        line = line.split(' ')
        lb = line[-1]
        text = ''.join(line[:-1])
        dataset.append(InputExample(text, lb))

    # train model
    a.train(dataset)

    # save model
    a.persist("./out")

    # load model
    a = BertIntentClassifier()
    a = a.load("./out")

    # do inference
    print(a.process(InputExample(text = "今天天气如何")))

