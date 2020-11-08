import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers = 1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.dummy_param = nn.Parameter(torch.empty(0))
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        
    def initHidden(self):
        # w = torch.empty(self.n_layers, self.batch_size, self.hidden_size)
        # nn.init.normal_(w, mean = 0.0, std = 0.1)
        w = torch.zeros(self.n_layers, 1, self.hidden_size)
        return w
        
    def forward(self, input, hidden):
        """single encoder unit flow

        Parameters
        ----------
        input : Tensor, [batch_size]
        hidden: Tensor, [n_layers, batch_size, hidden_size]

        Returns
        -------
        output: Tensor, [1, batch_size, hidden_size]
        hidden: Tensor, [n_layers, batch_size, hidden_size]
        """
        embeded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embeded, hidden)
        return output, hidden
    

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers = 1):
        super(DecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, output_size)
        
    def initHidden(self):
        # w = torch.empty(self.n_layers, self.batch_size, self.hidden_size)
        # nn.init.normal_(w, mean=0.0, std=0.1)
        w = torch.zeros(self.n_layers, 1, self.hidden_size)
        return w
    
    def forward(self, input, hidden):
        """
        single decoder unit flow
        
        Parameters
        ----------
        input : Tensor, (batch_size)
        hidden: Tensor, (n_layers, batch_size, hidden_size)

        Returns
        -------
        output: Tensor, (batch_size, output_size)
        hidden: Tensor, (n_layers, batch_size, hidden_size)
        """
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.classifier(output[0]), dim = 1)
        
        return output, hidden, None
    
    
class BahdanauDecoderRNN(nn.Module):
    """Bahdanau Attention Mechanism
    https://arxiv.org/pdf/1409.0473.pdf
    """
    def __init__(self, hidden_size, output_size, n_layers = 1):
        super(BahdanauDecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_size, hidden_size)
        
        # define attention layers
        self.fc_hidden = nn.Linear(self.hidden_size, self.hidden_size, bias = False)
        self.fc_encoder = nn.Linear(self.hidden_size, self.hidden_size, bias = False)
        # self.weight = nn.Parameters(torch.FloatTensor(1, hidden_size))
        self.weight = torch.empty(1, hidden_size).to(device)
        nn.init.normal_(self.weight, mean=0.0, std=0.1)
        
        self.gru = nn.GRU(hidden_size * 2, hidden_size)
        self.classifier = nn.Linear(hidden_size, output_size)
        
    def initHidden(self):
        w = torch.zeros(self.n_layers, 1, self.hidden_size)
        return w
    
    def forward(self, input, hidden, encoder_outputs: torch.Tensor):
        """
        single decoder unit flow

        Parameters
        ----------
        input : Tensor, (batch_size)
        hidden: Tensor, (n_layers, batch_size, hidden_size)
        encoder_outputs: Tensor, (e_seq_len, hidden_size)

        Returns
        -------
        output: Tensor, (batch_size, output_size)
        hidden: Tensor, (n_layers, batch_size, hidden_size)
        """
        encoder_outputs = encoder_outputs.squeeze()
        
        embeded = self.embedding(input).view(1, -1)
        
        # attention alignment scores
        unali_scores = torch.tanh(self.fc_hidden(hidden[-1]) + self.fc_encoder(encoder_outputs)).unsqueeze(0)
        ali_socres = unali_scores.bmm(self.weight.unsqueeze(2)) # 1, e_seq_len, 1
        atten_weights = F.softmax(ali_socres.view(1, -1), dim = 1) # 1, e_seq_len
        
        # context vector
        context_vector = torch.bmm(atten_weights.unsqueeze(0),
                                   encoder_outputs.unsqueeze(0))
        
        output = torch.cat((embeded, context_vector[0]), 1).unsqueeze(0)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.classifier(output[0]), dim = 1)

        return output, hidden, atten_weights
        
        
class LuongDecoderRNN(nn.Module):
    """Luong Attention Mechanism
    https://arxiv.org/pdf/1409.0473.pdf
    """
    def __init__(self, hidden_size, output_size, attention_method = "general", n_layers = 1):
        super(LuongDecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        
        self.attention = LuongAttention(hidden_size, attention_method)
        
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.classifier = nn.Linear(self.hidden_size * 2, self.output_size)
        
    def forward(self, input, hidden, encoder_outputs):
        """
        single decoder unit flow

        Parameters
        ----------
        input : Tensor, (batch_size)
        hidden: Tensor, (n_layers, batch_size, hidden_size)
        encoder_outputs: Tensor, (e_seq_len, hidden_size)

        Returns
        -------
        output: Tensor, (batch_size, output_size)
        hidden: Tensor, (n_layers, batch_size, hidden_size)
        """
        embeded = self.embedding(input).view(1, 1, -1)
        gru_output, hidden = self.gru(embeded, hidden)
        
        alignment_scores = self.attention(hidden, encoder_outputs)
        atten_weights = F.softmax(alignment_scores.view(1, -1), dim = 1)

        context_vector = torch.bmm(atten_weights.unsqueeze(0),
                                   encoder_outputs.unsqueeze(0))
        
        output = torch.cat((gru_output, context_vector), -1)
        output = F.log_softmax(self.classifier(output[0]), dim =1)
        return output, hidden, atten_weights
        

class LuongAttention(nn.Module):
    def __init__(self, hidden_size, method = "dot"):
        super(LuongAttention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.init_method()
        
    def init_method(self):
        if self.method == "general":
            self.fc = nn.Linear(self.hidden_size, self.hidden_size, bias = False)
            
        elif self.method == "concat":
            self.fc = nn.Linear(self.hidden_size, self.hidden_size, bias = False)
            self.weight = torch.empty(1, self.hidden_size).to(device)
            nn.init.normal_(self.weight, mean=0.0, std=0.1)
            
    def forward(self, decoder_hidden, encoder_outputs):
        if self.method == "dot":
            encoder_outputs = encoder_outputs.unsqueeze(0)
            
            return encoder_outputs.bmm(decoder_hidden[-1].view(1, -1, 1)).squeeze(0)
        
        elif self.method == "general":
            out = self.fc(decoder_hidden[-1])
            return encoder_outputs.unsqueeze(0).bmm(out.view(1, -1, 1)).squeeze(0)
        
        elif self.method == "concat":
            out = torch.tanh(self.fc(decoder_hidden[-1] + encoder_outputs)) # seq, hidden
            return out.unsqueeze(0).bmm(self.weight.unsqueeze(-1)).squeeze(0)


            
