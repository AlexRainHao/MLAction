"""
HMM training based on Baum-Welch algorithm, a variant of EM algorithm

reference:
    https://web.stanford.edu/~jurafsky/slp3/A.pdf
    https://github.com/ddbourgin/numpy-ml.git
"""

from typing import List, Optional, Dict, Tuple, Union, Any
import traceback
import numpy as np


class HMM:
    def __init__(self,
                 transition: Union[np.array, None] = None,
                 emission: Union[np.array, None] = None,
                 pi: Union[np.array, None] = None,
                 eps: Union[float, None] = None) -> None:
        """
        Parameters
        ----------
        transition
        emission
        pi
        eps
        """
        self._set_hmm_params(transition, emission, pi, eps)
        self._set_input_params()

    def _set_hmm_params(self, transition, emission, pi, eps):
        """pass"""
        self.H = None
        self.V = None
        self.Transition = None
        self.Emission = None
        self.eps = None

        eps = np.finfo(float).eps.tolist() if not eps else eps

        if transition is None or emission is None or pi is None:
            pass

        else:
            self.H = transition.shape[0]
            transition[transition == 0] = eps

            self.V = emission.shape[1]
            emission[emission == 0] = eps

            pi[pi == 0] = eps

        self.Transition = transition
        self.Emission = emission
        self.Pi = pi
        self.eps = eps


    def _set_input_params(self, seq: Union[np.array, None] = None) -> None:
        """pass"""
        self.O, self.N, self.T = [None] * 3
        if seq is None:
            return
        if seq.ndim == 1:
            seq = seq.reshape(1, -1)

        self.O = seq
        self.N, self.T = seq.shape


    @staticmethod
    def logsumexp(log_probs, axis = None):
        """
        a trick for calling sum of log probability
        for avoiding overflow or underflow
        see: http://bayesjumping.net/log-sum-exp-trick/
        """

        _max = np.max(log_probs)
        ds = log_probs - _max
        exp_sum = np.exp(ds).sum(axis = axis)
        return _max + np.log(exp_sum)


    def fit(self, seq, state_type, obs_type, pi = None, tol = 1e-5, walking_visual = 5):
        """pass"""
        self._set_input_params(seq)
        self.H = len(state_type)
        self.V = len(obs_type)

        # initialization for pi
        if not pi:
            pi = np.ones(self.H)
            self.Pi = pi / pi.sum()

        # initialization for transition probability
        transition = np.ones((self.H, self.H))
        self.Transition = transition / transition.sum(axis = 1)

        # initialization for emission probability
        emission = np.random.rand(self.H, self.V)
        self.Emission = emission / emission.sum(axis = 1)[:, np.newaxis]

        # training procedure
        step = 0
        delta = float("inf")
        last_ll = self.cal_batch_likelihood(self.O)

        while delta > tol:

            phi, gamma, count = self._Estep()
            self.Transition, self.Emission, self.Pi = self._Mstep(phi, gamma, count)
            ll = self.cal_batch_likelihood(self.O)
            delta = ll - last_ll
            last_ll = ll
            step += 1

            if step % walking_visual == 0:
                print(f"Epoch: {step}\tLikelihood: {last_ll: .4f}\tDelta: {delta}")


    def decode(self, sequences: np.array) -> Any:
        if sequences.ndim == 1:
            sequences = sequences.reshape(1, -1)

        N, T = sequences.shape

        viterbi = np.zeros((N, self.H, T))
        backpointer = np.zeros((N, self.H, T)).astype(int)

        # for idx in range(N):
        #     _sta = sequences[idx, 0]
        #
        #     # beginning
        #     for s in range(self.H):
        #         backpointer[idx, s, 0] = 0
        #         viterbi[idx, s, 0] = np.log(self.Pi[s] + self.eps) + np.log(self.Emission[s, _sta] + self.eps)
        #
        #     for t in range(1, T):
        #         pass
        _sta = sequences[:, 0]

        # beginning
        for s in range(self.H):
            backpointer[:, s, 0] = 0
            viterbi[:, s, 0] = np.log(self.Pi[s] + self.eps) + np.log(self.Emission[s, _sta] + self.eps)

        # inference
        for t in range(1, T):
            ot = sequences[:, t]

            for s in range(self.H):
                _vite = lambda x: viterbi[:, x, t - 1]
                _aij = lambda x: np.log(self.Transition[x, s] + self.eps)
                _bj = np.log(self.Emission[s, ot] + self.eps)

                probs = list(map(lambda x: (_vite(x) + _aij(x) + _bj).tolist(), range(self.H)))
                probs = np.reshape(probs, (-1, self.H))
                viterbi[:, s, t] = np.max(probs, axis = -1)
                backpointer[:, s, t] = np.argmax(probs, axis = -1) # N, H, T

        best_path_prob = viterbi[:, :, T - 1].max(axis = -1) # N, 1

        pointer = viterbi[:, :, T - 1].argmax(axis = -1).reshape(-1, 1) # array, N, 1
        best_path = pointer.tolist() # list, N, 1

        for t in reversed(range(1, T)):
            pointer = pointer.reshape(-1)
            for _id, _p in enumerate(pointer):
                _p = backpointer[_id, _p, t]
                best_path[_id].append(_p.tolist())

        best_path = [list(reversed(x)) for x in best_path]

        return best_path, best_path_prob


    def cal_likelihood(self, seq: np.array) -> Union[np.array, float]:
        """call likelihood for a Single sentence"""


        seq = seq.reshape(1, -1)

        _, t = seq.shape

        probs = self._forward(seq[0])
        ll = self.logsumexp(probs[:, t - 1])
        return ll


    def cal_batch_likelihood(self, seq: np.array) -> Union[np.array, float]:
        """call likelihood for a Batch sentence"""

        return np.sum([self.cal_likelihood(row) for row in seq])


    def _forward(self, seq):
        """pass"""

        forward = np.zeros((self.H, self.T))

        ot = seq[0]

        for s in range(1, self.H):
            # the first time stamp
            forward[s, 0] = np.log(self.Pi[s] + self.eps) + np.log(self.Emission[s, ot] + self.eps)

        # recursion
        for t in range(1, self.T):
            ot = seq[t]

            for s in range(self.H):

                _forw = lambda x: forward[x, t - 1]
                _aij = lambda x: np.log(self.Transition[x, s] + self.eps)
                _bj = np.log(self.Emission[s, ot] + self.eps)
                _prob = lambda x: _forw(x) + _aij(x) + _bj

                forward[s, t] = self.logsumexp(list(map(_prob, range(self.H))))

        return forward


    def _backward(self, seq):
        """pass"""

        backward = np.zeros((self.H, self.T))

        for s in range(self.H):
            backward[s, self.T - 1] = np.log(1.)

        for t in reversed(range(self.T - 1)):
            otn = seq[t + 1]

            for s in range(self.H):

                _aij = lambda x: np.log(self.Transition[x, s] + self.eps)
                _bj = np.log(self.Emission[s, otn] + self.eps)
                _back = lambda x: backward[x, t + 1]
                _prob = lambda x: _aij(x) + _bj + _back(x)

                backward[s, t] = self.logsumexp(list(map(_prob, range(self.H))))

        return backward


    def _Estep(self):
        """pass
        """
        phi = np.zeros((self.N, self.H, self.H, self.T))
        gamma = np.zeros((self.N, self.H, self.T))
        count = np.zeros((self.N, self.H))

        for i in range(self.N):
            seq = self.O[i, :]
            fwd = self._forward(seq)
            bwd = self._backward(seq)

            T = self.T - 1
            for s in range(self.H):
                gamma[i, s, T] = fwd[s, T] + bwd[s, T]
                count[i, s] = fwd[s, 0] + bwd[s, 0]


            for t in range(T):
                otn = seq[t + 1]

                for si in range(self.H):
                    gamma[i, si, t] = fwd[si, t] + bwd[si, t]

                    for sj in range(self.H):
                        phi[i, si , si, t] = fwd[si, t] + \
                                             np.log(self.Transition[si, sj] + self.eps) + \
                                             np.log(self.Emission[si, otn] + self.eps) + \
                                             bwd[si, t + 1]

        return phi, gamma, count


    def _Mstep(self, phi, gamma, count):
        """pass"""

        transition = np.zeros((self.H, self.H))
        emission = np.zeros((self.H, self.V))
        pi = np.zeros(self.H)

        count_gamma = np.zeros((self.N, self.H, self.V))
        count_phi = np.zeros((self.N, self.H, self.H))

        for i in range(self.N):
            seq = self.O[i, :]
            for si in range(self.H):
                for vk in range(self.V):
                    if not (seq == vk).any():
                        count_gamma[i, si, vk] = np.log(self.eps)

                    else:
                        count_gamma[i, si, vk] = self.logsumexp(gamma[i, si, seq == vk])

                for sj in range(self.H):
                    count_phi[i, si, sj] = self.logsumexp(phi[i, si, sj, :])

        pi = self.logsumexp(count, axis = 0) - np.log(self.N + self.eps)


        for si in range(self.H):
            for vk in range(self.V):
                emission[si, vk] = self.logsumexp(count_gamma[:, si, vk]) - self.logsumexp(count_gamma[:, si, :])

            for sj in range(self.H):
                transition[si, sj] = self.logsumexp(count_phi[:, si, sj]) - self.logsumexp(count_phi[:, si, :])


        return np.exp(transition), np.exp(emission), np.exp(pi)


if __name__ == "__main__":
    def get_input_data(N, T, V):
        return np.random.randint(0, V, size=(N, T))


    def get_state(H):
        return np.array(range(0, H))


    # simulation
    V = [0, 1, 2, 3, 4]
    X = get_input_data(20, 10, len(V))
    Y = get_state(3)

    # training
    model = HMM()
    model.fit(X, Y, V)

    t = model.Transition
    e = model.Emission
    p = model.Pi

    # decoding
    model = HMM(t, e, p)
    res = model.decode(np.random.randint(0, len(V), size=(2, 10)))
    print(res)