import os.path as opt
import codecs
from transformers import BertTokenizer, BertModel
from transformers import BertConfig as BC

def load_pretrained(mpath,
                    config = "bert_config.json",
                    model = "bert_model.bin"):

    """load pre-trained bert encoder and tokenizer"""

    encoder = load_pretrained_encoder(mpath, config, model)
    tokenizer = BertTokenizer.from_pretrained(mpath)

    return encoder, tokenizer


def load_encoder(mpath, config = "bert_config.json"):
    """load a relatively static bert encoder architecture"""
    config = load_pretrained_config(mpath, config)
    return BertModel(config)


def load_pretrained_tokenizer(mpath):
    """load pre-trained tokenizer"""

    return BertTokenizer.from_pretrained(mpath)


def load_pretrained_config(mpath, config = "bert_config.json"):
    return BC.from_pretrained(opt.join(mpath, config))


def load_pretrained_encoder(mpath,
                            config = "bert_config.json",
                            model = "bert_model.bin"):
    """load pre-trained bert encoder"""

    b_config = load_pretrained_config(mpath, config)
    encoder = BertModel.from_pretrained(opt.join(mpath, model), config = b_config)

    return encoder

