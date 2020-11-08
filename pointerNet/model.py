"""LSTM based PointerNet
Only model architecture written, and no data feed for test

https://arxiv.org/pdf/1506.03134.pdf
"""

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F


class PointerNet(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_dim, layers, dropout):
        super(PointerNet, self).__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = Encoder(embed_dim,
                               hidden_dim,
                               layers,
                               dropout)

        self.decoder = Decoder(embed_dim,
                               hidden_dim)

        self.first_decoder_input = Parameter(torch.FloatTensor(embed_dim), requires_grad = False)
        nn.init.uniform(self.first_decoder_input, -1, 1)

    def forward(self, inputs):
        """
        :type inputs: torch.Tensor, [batch, seq_len]
        """

        batch_size = inputs.shape[0]

        first_decoder_input = self.first_decoder_input.expand(batch_size, -1) # [batch, embed_dim]

        enc_inputs = self.embedding(inputs) # [batch, seq_len, embed_dim]

        first_encoder_hidden = self.encoder.init_hidden(inputs) # Tuple(h0, c0)

        # ===============
        # encoder procedure
        # ===============
        enc_out, enc_hidden = self.encoder(enc_inputs, first_encoder_hidden) # [batch, seq_len, layers * hidden_dim]

        # ===============
        # decoder procedure
        # ===============
        first_decoder_hidden = (enc_hidden[0][-1], enc_hidden[1][-1])

        (outputs, pointers), dec_hidden = self.decoder(enc_inputs,
                                                       first_decoder_input,
                                                       first_decoder_hidden,
                                                       enc_out)

        return outputs, pointers


class Encoder(nn.Module):
    """encoder layer, obtain each step encoder outputs"""

    def __init__(self, embed_dim, hidden_dim, layers, dropout):
        super(Encoder, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.dropout = dropout

        self.lstm = nn.LSTM(embed_dim,
                            hidden_dim,
                            layers,
                            dropout = dropout)

    def forward(self, enc_inputs, hidden):
        """
        :type enc_inputs: torch.Tensor, [batch_size, seq_len, embed_dim]

        :rtype: enc_outpus: torch.Tensor, [batch_size, seq_len, layers * hidden_dim]
        :rtype: hidden: Tuple[torch.Tensor, torch.Tensor],
                        (h0, c0), [layers, batch_size, hidden_dim],
                                  [layers, batch_size, hidden_dim]
        """

        enc_inputs = enc_inputs.permute(1, 0, 2) # [seq_len, batch_size, embed_dim]
        enc_outpus, enc_hidden = self.lstm(enc_inputs, hidden)

        return enc_outpus.permute(1, 0, 2), hidden


    def init_hidden(self, inputs):
        batch_size = inputs.shape[0]

        h0 = Parameter(torch.randn((self.layers,
                                    batch_size,
                                    self.hidden_dim)),
                       requires_grad = False)
        c0 = Parameter(torch.randn((self.layers,
                                    batch_size,
                                    self.hidden_dim)),
                       requires_grad = False)
        return h0, c0



class Decoder(nn.Module):
    """decoder layer
        * first decoder hidden
        * first decoder input
        * each step encoder inputs
        * each step encoder outputs
    received for conducting rnn layers and pointer network attention
    then give each decoder step pointer index and corresponding attention alphas
    """

    def __init__(self, embed_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # lstm layer
        self.input2hidden = nn.Linear(embed_dim, hidden_dim * 4)
        self.hidden2hidden = nn.Linear(hidden_dim, hidden_dim * 4)
        self.hidden2out = nn.Linear(2 * hidden_dim, hidden_dim)

        # pointer attention layer
        self.attention = Attention(hidden_dim, hidden_dim)

        self.mask = Parameter(torch.ones(1), requires_grad = False)
        self.runner = Parameter(torch.zeros(1), requires_grad = False) # record each step index


    def forward(self, enc_inputs, dec_inputs, hidden, context):
        """
        :type enc_inputs: torch.Tensor, each encoder step inputs
        :type dec_inputs: torch.Tensor, first decoder inputs fed
        :type hidden: torch.Tensor, first decoder hidden fed
        :type context: torch.Tensor, each encoder step outputs
        """

        batch_size = enc_inputs.shape[0]
        seq_len = enc_inputs.shape[1]

        mask = self.mask.expand(batch_size, seq_len)
        self.attention.init_inf(mask.size())

        runner = self.runner.expand(batch_size, seq_len)
        for i in range(seq_len):
            runner.index_fill_(1, torch.tensor([i]), i)

        outputs = []
        pointers = []

        # ==============
        # decoder for each step
        # ==============
        for _ in range(seq_len):
            g_o, (c_t, h_t) = self.lstm_step(dec_inputs, hidden)
            h_t, alpha = self.attention_step(h_t, context, mask)

            hidden = (h_t, c_t)

            masked_alpha = alpha * mask # [batch_size, seq_len]

            max_prob, max_index = masked_alpha.max(1)

            # update mask
            seen_pointer = (runner == max_index.unsqueeze(1).expand(-1, alpha.shape[1])).float()
            mask = mask * (1 - seen_pointer)

            # update dec_inputs from mask
            embedding_mask = seen_pointer.unsqueeze(2).expand(-1, -1, self.embed_dim)
            dec_inputs = enc_inputs[embedding_mask].view(batch_size, self.embed_dim)

            outputs.append(alpha.unsqueeze(0))
            pointers.append(max_index.unsqueeze(1))

        outputs = torch.cat(outputs, dim = 0).permute(1, 0, 2) # [d_len, batch, seq_len]
        pointers = torch.cat(pointers, dim = 0).permute(1, 0, 2) # [d_len, batch, 1]

        return (outputs, pointers), hidden


    def lstm_step(self, x, hidden):
        """single lstm cell forward"""
        
        h, c = hidden

        gates_cells = self.input2hidden(x) + self.hidden2hidden(hidden)
        g_i, g_f, g_c, g_o = gates_cells.chunk(4, 1)

        g_i = F.sigmoid(g_i)
        g_f = F.sigmoid(g_f)
        g_c = F.tanh(g_c)
        g_o = F.sigmoid(g_o)

        c_t = (g_f * c) + (g_i * g_c)
        h_t = g_o * F.tanh(c_t)

        return g_o, (c_t, h_t)

    def attention_step(self, h_t, context, mask):
        """single pointer network attention forward"""

        alpha, attention_hidden_state = self.attention(h_t, context, torch.eq(mask, 0))
        attention_hidden_state = F.tanh(self.hidden2out(torch.cat((attention_hidden_state, h_t), 1)))
        # seems a fusion layer, but hidden_state from attention layer could used directly

        return attention_hidden_state, alpha


class Attention(nn.Module):
    """pointer network attention mechanism"""

    def __init__(self, input_dim, hidden_dim):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.d_linear = nn.Linear(input_dim, hidden_dim)
        self.e_linear = nn.Linear(input_dim, hidden_dim)
        self.V = Parameter(torch.FloatTensor(hidden_dim), requires_grad = True)

        self.inf = Parameter(torch.FloatTensor([float("-inf")]), requires_grad = False) # aviod grad calculation certain steps

    def init_inf(self, size):
        self.inf = self.inf.expand(*size)

    def forward(self, hidden, context, mask):
        """
        :type hidden: torch.Tensor, [batch_size, hidden_dim]
        :type context: torch.Tensor, [batch_size, seq_len, hidden_dim]
        :type mask: torch.Tensor, [batch_size, seq_len]
        """

        batch_size = context.shape[0]
        seq_len = context.shape[1]
        di = self.d_linear(hidden).expand(-1, -1, seq_len)

        ei = self.e_linear(context) # [batch, seq_len, hidden_dim]

        ui = torch.bmm(self.V.expand(batch_size, 1, -1),
                       F.tanh(di + ei).permute(0, 2, 1)).squeeze(1) # [batch, seq_len]

        # mask poster unit for softmax
        ui[mask] = self.inf[mask]

        alpha = F.softmax(ui) # [batch, seq_len]

        attention_hidden_state = torch.bmm(ei.permute(0, 2, 1), alpha.unsqueeze(2)).squeeze(2) # [batch, hidden_dim]

        return alpha, attention_hidden_state

