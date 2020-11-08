import utils
import random
import math
from copy import deepcopy


class NGram:
    """
    Based NGram model, only support for 1 <= K <= 3

    E.X.
    model = NGram(data_list, K = 2)
    tchr = model.decode(["people"])

    """
    def __init__(self, corpus, K):
        self.K = self._define_checker(K)
        self.corpus = corpus
        self.exv = 1e-80
        self.laplace = True

        self.lambdaPara = self.init_lambdaPara()

        self.full = True
        self.normalized = True

        self.split_ratio = .8

    @staticmethod
    def _define_checker(K):
        assert isinstance(K, int), "NGram model must have a integer of K"

        assert K >= 1, "NGram model must have K of greater than 0"

        assert K <= 3, "NGram model not supported for K of greater than 3"

        return K


    @property
    def vocabs(self):
        """get char2idx, idx2char and tokens for given dataset"""
        if self.full:
            char2idx, idx2char = utils.get_vocabulary(self.corpus)

        else:
            char2idx, idx2char = utils.get_vocabulary(self.train_loader)

        return char2idx, idx2char


    def gram_counter(self, K):

        if K == 1:
            return utils._ngram_counter_uni(self.train_loader, K)

        else:
            return utils._ngram_counter_multi(self.train_loader, K, self.exv, self.laplace)

    @property
    def data_loader(self):
        """a data loader for splitting training set and dev set"""
        data_loader = deepcopy(self.corpus)
        random.shuffle(data_loader)
        return data_loader


    @property
    def train_loader(self):
        """obtain training set from data loader"""
        if self.full:
            return self.data_loader

        return self.data_loader[:int(len(self.data_loader) * self.split_ratio)]

    @property
    def dev_loader(self):
        """obtain validation set from data loader"""
        return self.data_loader[int(len(self.data_loader) * self.split_ratio):]

    @staticmethod
    def get_gram_from_loader(loader, K):
        return utils.get_ngram_token(loader, K)

    def train(self):
        raise NotImplementedError("No training method available")


    def init_lambdaPara(self):
         return [0.] * (self.K + 1)


    def set_lambda(self, *args):
        if len(args) != self.K + 1:
            raise ValueError("lambda used setting must have same dimension with lambdas")

        for id, val in enumerate(args):
            self.lambdaPara[id] = val


    def get_prob(self, counter, *args):
        """conditional probability for a single n-gram tokens"""

        if len(args) == 1:
            return counter.get(args[0], self.exv)

        elif len(args) > 1:
            ct = ' '.join(args[:-1])
            tt = args[-1]

            cond_prob = counter.get(ct)

            if cond_prob:
                prob = cond_prob.get(tt, self.exv)

            else:
                prob = self.exv

            return prob

        else:
            raise ValueError("No token given for prediction")


    def grams_ppl_caler(self, gram_datas, counter):
        """calculation tri-gram ppl for gram datas of a certain sentence
        where ppl = mean(ppl_s), ppl_s = log(prob_s_1) + log(prob_s_2) + ... + log(prob_s_n)
        """
        length = len(gram_datas)

        total = 0.

        for gram in gram_datas:
            ppl = self.get_prob(counter, *gram)

            total -= ppl

        if self.normalized:
            return math.pow(2, total / length)
        else:
            return total / length


    def ppl_caler(self):
        """a method defined for calculating validation dataset ppl"""
        raise NotImplementedError("No ppl calculation handler implemented")

    @property
    def counter(self):
        return self.gram_counter(K = self.K)


    def decode(self, pre_tokens: list):
        """predict next word given K word, a simple decoding version

        Parameters
        ----------
        pre_tokens : list, e.x, ["I", "Have"]
        """

        assert len(pre_tokens) == self.K - 1, f"{self.K - 1} token should given for {self.K}-gram model"

        max_prob = -float("inf")
        t_chr = utils.UNK

        for word in self.vocabs[0].keys():
            grams = pre_tokens + [word]
            prob = self.get_prob(self.counter, *grams)

            if prob > max_prob:
                max_prob = prob
                t_chr = word

        return t_chr
