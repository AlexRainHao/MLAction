import torch
import torch.nn as nn
from torch.nn import Parameter
from utils import Vocab, Tags


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab, tags, embed_dim, hidden_dim, batch_size):
        """
        :type vocab: Vocab
        :type tags: Tags
        """
        super(BiLSTM_CRF, self).__init__()
        self.vocab_size = vocab.size()
        # self.tag2idx = tags.tag2idx
        self.tags = tags
        self.tag_size = tags.size()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        # layers
        self.embedding = nn.Embedding(vocab.size(), embed_dim)
        self.lstm = nn.LSTM(input_size = self.embed_dim,
                            hidden_size = self.hidden_dim // 2,
                            bidirectional = True,
                            batch_first = True)

        self.hidden2tag = nn.Linear(hidden_dim, self.tag_size)

        # Parameter
        self.transitions = self.init_transitions()
        self.transitions.data[:, tags.get_id_from_tag(tags.start_tag)] = -10000
        self.transitions.data[tags.get_id_from_tag(tags.end_tag), :] = -10000

    def init_transitions(self):
        """initialize tags transitions matrix"""
        transitions = torch.empty(self.tag_size, self.tag_size)
        nn.init.normal_(transitions, mean = 0.0, std = 0.1)
        return transitions

    def init_hidden(self):
        """initialize lstm hiiden weights"""
        fw_hidden = torch.empty(2, self.batch_size, self.hidden_dim // 2)
        bw_hidden = torch.empty(2, self.batch_size, self.hidden_dim // 2)

        nn.init.normal_(fw_hidden, mean = 0.0, std = 0.1)
        nn.init.normal_(bw_hidden, mean = 0.0, std = 0.1)

        return (fw_hidden, bw_hidden)

    @staticmethod
    def logsumexp(log_probs, dim=None):
        """
        a trick for calling sum of log probability
        for avoiding overflow or underflow
        see: http://bayesjumping.net/log-sum-exp-trick/
        """

        _max = torch.max(log_probs)
        ds = log_probs - _max
        exp_sum = torch.exp(ds).sum(dim=dim)
        return _max + torch.log(exp_sum)

    def get_features(self, sentence):
        """each batch step for lstm forward
        :type sentence: torch.Tensor, [batch, seq_len]
        """
        self.hidden = self.init_hidden() # lstm init hidden

        embeds = self.embedding(sentence) # [batch, seq_len, embed_dim]

        lstm_out, self.hidden = self.lstm(embeds, self.hidden) # [batch, seq_len, hidden_dim]

        lstm_feat = self.hidden2tag(lstm_out)

        return lstm_feat

    def numerator_score(self, feats, tags):
        """obtain sentence score, as where shown numerator
        :param feats: torch.Tensor, [batch, seq_len, tag_size]
        :type tags: torch.Tensor, [batch, seq_len]
        """

        scores = torch.zeros(len(tags))

        _start = torch.tensor(self.tags.get_id_from_tag(self.tags.start_tag)).view(1, -1).repeat(len(tags), 1)
        tags = torch.cat([_start, tags], dim = 1) # [batch, seq_len + 1]

        # for idx, feat in enumerate(feats):
        feats = feats.transpose(1, 0) # [seq_len, batch, tag_size]

        for idx, feat in enumerate(feats):
            scores += self.transitions[tags[:, idx], tags[:, idx+1]] + feat[range(self.batch_size), tags[:, idx+1]]

        scores += self.transitions[tags[:, -1], self.tags.get_id_from_tag(self.tags.end_tag)]

        return scores

    def denominator_score(self, feats):
        """obtain normalized score, as where shown denominator
        :type feats: torch.Tensor, [batch, seq_len, tag_size]
        """

        init_alphas = torch.full((self.batch_size, self.tag_size), -10000.)
        init_alphas[:, self.tags.get_id_from_tag(self.tags.start_tag)] = 0.

        forward_var = init_alphas # [batch, tag_size]

        feats = feats.transpose(1, 0) # [seq_len, batch, tag_size]

        for feat in feats:
            alphas_t = []

            for tag in range(self.tag_size):
                emit_score = feat[:, tag].unsqueeze(-1).repeat(1, self.tag_size) # [batch, tag_size]
                trans_score = self.transitions[tag].repeat(self.batch_size, 1) # [batch, tag_size]

                next_tag_var = forward_var + trans_score + emit_score # [batch, tag_size]

                alphas_t.append(self.logsumexp(next_tag_var, dim = 1)) # [batch]

            forward_var = torch.cat(alphas_t).view(self.batch_size, -1) # [batch, tag_size]

        end_var = forward_var + self.transitions[:, self.tags.get_id_from_tag(self.tags.end_tag)] # [batch, tag_size]

        alpha = self.logsumexp(end_var, dim = 1) # [batch]

        return alpha

    def viterbi_decode(self, feats):
        """decoding from viterbi path
        :type feats: torch.Tensor, [batch, seq_len, tag_size]
        """
        batch_size = feats.shape[0]

        backpointers = [] # should be as [seq_len, tag_size, batch]

        init_alphas = torch.full((self.batch_size, self.tag_size), -10000.)
        init_alphas[:, self.tags.get_id_from_tag(self.tags.start_tag)] = 0.

        forward_var = init_alphas # [batch, tag_size]

        feats = feats.transpose(1, 0)

        for feat in feats:
            step_pointer = []
            step_score = []

            for tag in range(self.tag_size):
                next_tag_var = forward_var + self.transitions[tag] # [batch, tag_size]

                best_tag_id = torch.argmax(next_tag_var, dim = -1)

                step_pointer.append(best_tag_id)
                step_score.append(next_tag_var[range(batch_size), best_tag_id])

            forward_var = torch.cat(step_score).view(batch_size, -1) + feat
            backpointers.append(step_pointer)

        end_var = forward_var + self.transitions[:, self.tags.get_id_from_tag(self.tags.end_tag)] # [batch, tag_size]

        best_tag_id = torch.argmax(end_var, dim = -1) # [batch]
        best_score = end_var[range(batch_size), best_tag_id]

        best_path = [best_tag_id]
        for bidx in reversed(backpointers):
            # bidx: [tag_size, batch]
            best_tag_id = torch.cat(bidx).view(-1, batch_size)[best_tag_id, range(batch_size)]
            best_path.append(best_tag_id)

        return best_score, torch.cat(best_path[::-1]).view(-1, batch_size).T

    def forward(self, sentence, tags):
        """obtain a batch sentence nll
        :type sentence: torch.Tensor, [batch, seq_len]
        :type tags: torch.Tensor, [batch, seq_len]
        """
        feats = self.get_features(sentence)
        sentence_score = self.numerator_score(feats, tags) # [batch, tag_size]
        normalized_score = self.denominator_score(feats)
        return torch.mean(sentence_score - normalized_score)

    def decode(self, sentence):
        """decoding
        :type sentence: torch.Tensor, [batch, seq_len]
        :rtype: score: torch.Tensor, [batch]
        :rtype: tag_seq: torch.Tensor, [batch, seq_len]
        """
        with torch.no_grad():
            feats = self.get_features(sentence)
            score, tag_seq = self.viterbi_decode(feats)

            return score, tag_seq

