from typing import Dict, List, Any, Optional, Union, Tuple
from collections import defaultdict
import re
import codecs
from copy import deepcopy

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10

def normalizeString(s):
    """remove redundant punctuations"""
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def filterPairs(pairs):
    """filter sequences length greeter than max length"""
    filter_func = lambda x: len(x[0].split(' ')) < MAX_LENGTH and \
                            len(x[1].split(' ')) < MAX_LENGTH
    
    return [pair for pair in pairs if filter_func(pair)]
    

def lines_parser(lines: List[str]):
    """load data source and obtain pairs of original and target sequences"""
    pairs = list(map(lambda x: x.strip().split('\t'), lines))
    pairs = filterPairs(pairs)
    
    return pairs


class Lang:
    
    def __init__(self, name):
        self.name = name
        self.word2index = {SOS_token: 0, EOS_token: 1}
        self.word2count = defaultdict(int)
        self.index2word = {0: SOS_token, 1: EOS_token}
        self.n_words = 2
        
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)
    
    def addWord(self, word):
        if not word in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] += 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
    
            
def readCorpus(fname):
    """read original """
    with codecs.open(fname, encoding='utf-8') as f:
        lines = f.read().strip().split('\n')
    
    return lines


def readLang(lines) -> Tuple[Lang, Lang, List]:
    """main method load training corpus"""
    pairs = lines_parser(lines)
    ori_lang = Lang("eng")
    tar_lang = Lang("fra")
    
    for ori_pair, tar_pair in pairs:
        ori_lang.addSentence(ori_pair)
        tar_lang.addSentence(tar_pair)
        
    print(f'read {len(pairs)} sentences pairs')
    print(f'[{ori_lang.name}] has [{ori_lang.n_words}] tokens')
    print(f'[{tar_lang.name}] has [{tar_lang.n_words}] tokens')

    return ori_lang, tar_lang, pairs
    

# lines = readCorpus("./data/train.txt")
# ori_lang, tar_lang, pairs = readLang(lines)