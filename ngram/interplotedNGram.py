from NGram import NGram
import utils
from copy import deepcopy
import math

class InterplotedSmoothing(NGram):
    """
    Jelinek and Mercer, 1980 interpolated n-gram model
    take the paper `the A Neural Probabilistic Language Model, bengio, et.al` as a reference

    E.X.
    model = InterplotedSmoothing(token_list, K = 2)
    model.train()
    tchr = model.decode(["people"])

    """
    def __init__(self, corpus, K):
        super(InterplotedSmoothing, self).__init__(corpus, K)

    def train(self):
        print("training start...")
        mapping = {1: self._learn_uni_lambda,
                   2: self._learn_bi_labmbda,
                   3: self._learn_thi_lambda}

        alphas = mapping.get(self.K)(*self.lambdaPara[1:])
        a0 = 1 - sum(alphas)
        self.set_lambda(a0, *alphas)

    def decode(self, pre_tokens: list):
        """pass

        Parameters
        ----------
        pre_tokens : list, e.x, ["I", "Have"]
        """

        assert len(pre_tokens) == self.K - 1, f"{self.K - 1} token should given for {self.K}-gram model"

        mapping = {1: [self.uni_counter, self.uni_grams],
                   2: [self.bi_counter, self.bi_grams],
                   3: [self.tri_counter, self.tri_grams]}

        max_prob = -float("inf")
        t_chr = utils.UNK

        for word in self.vocabs[0].keys():
            cp_k = deepcopy(self.K)
            ps = []
            ws = deepcopy(pre_tokens + [word])

            while cp_k > 0:
                p = self.get_prob(mapping.get(cp_k)[0], *ws)
                ps.append(p)
                ws.pop(0)
                cp_k -= 1

            prob = sum(map(lambda x, y: x * y, self.lambdaPara, ps[::-1]))
            if prob > max_prob:
                max_prob = prob
                t_chr = word

        return t_chr


    @property
    def uni_counter(self):
        return self.gram_counter(K = 1)

    @property
    def bi_counter(self):
        return self.gram_counter(K = 2)

    @property
    def tri_counter(self):
        return self.gram_counter(K = 3)

    @property
    def uni_grams(self):
        return self.get_gram_from_loader(self.dev_loader, K = 1)

    @property
    def bi_grams(self):
        return self.get_gram_from_loader(self.dev_loader, K = 2)

    @property
    def tri_grams(self):
        return self.get_gram_from_loader(self.dev_loader, K = 3)


    def ppl_caler(self):
        """obtain validation ppl for lambdaPara training

        where conditional probability for a gram token has formula of:

        `*P(wt|wt-1, wt-2)=a0p0 + a1p1(wt) + a2p2(wt|wt-1) + a3p3(wt|wt-1, wt-2)`

        where tri-gram as example instance.
        then interpolated conditional probability of each gram token could used to calculate a sentence ppl
        """
        p0 = 1 / len(self.vocabs[0])
        ppl = .0

        mapping = {1: [self.uni_counter, self.uni_grams],
                   2: [self.bi_counter, self.bi_grams],
                   3: [self.tri_counter, self.tri_grams]}

        for grams in mapping.get(self.K)[1]:
            sin_ppl = .0
            idx = 0
            cp_k = deepcopy(self.K)

            for gram in grams:
                ps = []
                ws = list(deepcopy(gram))
                while cp_k > 0:
                    p = self.get_prob(mapping.get(cp_k)[0], *ws)
                    ps.append(p)
                    ws.pop(0)
                    cp_k -= 1
                ps.append(p0)


                p_gram = sum(map(lambda x, y: x * y, self.lambdaPara, ps[::-1]))
                sin_ppl -= p_gram
                idx += 1

            if self.normalized:
                ppl += math.pow(2, sin_ppl / idx)
            else:
                ppl += sin_ppl / idx

        return ppl


    def _learn_uni_lambda(self, lower, threshold = 0.001):
        """EM training algorithm based on Jelinek and Mercer, 1980 interpolated uni-gram"""
        upper = 0.999
        lastPPL = None

        bestLambda = (lower + upper) / 2
        learning_step = .2

        while True:
            bestPPL = 1e8
            step = lower

            while step <= upper:
                PPL = self.ppl_caler()

                if PPL < bestPPL:
                    bestPPL = PPL
                    bestLambda = step

                # lambda learning
                step = (upper - lower) * learning_step

            if lastPPL:
                if math.fabs(lastPPL - bestPPL) / bestPPL < threshold:
                    break
            lastPPL = bestPPL

            lower = self.update_lower_bound(bestLambda, lower, upper, learning_step)
            upper = self.update_upper_bound(bestLambda, upper, lower, learning_step)

        return bestLambda


    def _learn_bi_labmbda(self, lower1, lower2, threshold = 0.001):
        """EM training algorithm based on Jelinek and Mercer, 1980 interpolated bi-gram"""
        upper1, upper2 = 0.999, 0.999
        lastPPL = None

        bestLambda1 = (lower1 + upper1) / 2
        bestLambda2 = (lower2 + upper2) / 2
        learning_step = .2

        while True:
            bestPPL = 1e8
            step_1 = lower1

            while step_1 <= upper1:
                step_2 = lower2
                while step_2 <= upper2 and step_1 + step_2 < 1:
                    PPL = self.ppl_caler()

                    if PPL < bestPPL:
                        bestPPL = PPL
                        bestLambda1 = step_1
                        bestLambda2 = step_2

                    # lambda1 learning
                    step_2 += (upper2 - lower2) * learning_step


                # lambda1 learning

                step_1 += (upper1 - lower1) * learning_step

            if lastPPL:
                if math.fabs(lastPPL - bestPPL) / bestPPL < threshold:
                    break

            lastPPL = bestPPL

            # update boundary
            lower1 = self.update_lower_bound(bestLambda1, lower1, upper1, learning_step)
            upper1 = self.update_upper_bound(bestLambda1, lower1, upper1, learning_step)
            lower2 = self.update_lower_bound(bestLambda2, lower2, upper2, learning_step)
            upper2 = self.update_upper_bound(bestLambda2, lower2, upper2, learning_step)

        return bestLambda1, bestLambda2


    def _learn_thi_lambda(self, lower1, lower2, lower3, threshold = .001):
        """EM training algorithm based on Jelinek and Mercer, 1980 interpolated tri-gram"""
        upper1, upper2, upper3 = .999, .999, .999
        lastPPL = None

        bestLambda1 = (lower1 + upper1) / 2
        bestLambda2 = (lower2 + upper2) / 2
        bestLambda3 = (lower3 + upper3) / 2
        learning_step = .2

        while True:
            bestPPL = 1e8
            step_1 = lower1

            while step_1 <= upper1:
                step_2 = lower2
                while step_2 <= upper2:
                    step_3 = lower3
                    while step_3 <= upper3 and sum([step_1, step_2, step_3]) < 1:
                        PPL = self.ppl_caler()

                        if PPL < bestPPL:
                            bestPPL = PPL
                            bestLambda1 = step_1
                            bestLambda2 = step_2
                            bestLambda3 = step_3

                        step_3 += (upper3 - lower3) * learning_step

                    step_2 += (upper2 - lower2) * learning_step

                step_1 += (upper1 - lower1) * learning_step

            if lastPPL:
                if math.fabs(lastPPL - bestPPL) / bestPPL < threshold:
                    break

            lastPPL = bestPPL

            lower1 = self.update_lower_bound(bestLambda1, lower1, upper1, learning_step)
            upper1 = self.update_upper_bound(bestLambda1, lower1, upper1, learning_step)
            lower2 = self.update_lower_bound(bestLambda2, lower2, upper2, learning_step)
            upper2 = self.update_upper_bound(bestLambda2, lower2, upper2, learning_step)
            lower3 = self.update_lower_bound(bestLambda3, lower3, upper3, learning_step)
            upper3 = self.update_upper_bound(bestLambda3, lower3, upper3, learning_step)

        return bestLambda1, bestLambda2, bestLambda3


    def update_lower_bound(self, curr_lambda, lower_bound, upper_bound, lr):
        if curr_lambda != lower_bound:
            return curr_lambda - (upper_bound - lower_bound) * lr

        else:
            return curr_lambda * lr


    def update_upper_bound(self, curr_lambda, lower_bound, upper_bound, lr):
        if curr_lambda != upper_bound:
            return curr_lambda + (upper_bound - lower_bound) * lr

        else:
            return curr_lambda / lr
