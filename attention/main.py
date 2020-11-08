import utils
import model

import random

import torch
import torch.nn as nn
import torch.optim as optim

class Params:
    pass


def set_index_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensor_from_sentence(lang, sentence):
    index = set_index_from_sentence(lang, sentence)
    index.append(utils.EOS_token)
    return torch.tensor(index, dtype = torch.long, device = model.device).view(-1, 1)


def tensor_from_pairs(src_lang, tar_lang, pair):
    src_tensor = tensor_from_sentence(src_lang, pair[0])
    tar_tensor = tensor_from_sentence(tar_lang, pair[1])
    return src_tensor, tar_tensor


def bch_train(input_tensor, target_tensor, encoder, decoder,
             e_optim, d_optim, criterion, max_len = utils.MAX_LENGTH):
    """
    training for single pair of sentence
    
    Parameters
    ----------
    input_tensor: Tensor, (length)
    target_tensor: Tensor, (length)
    """
    e_hidden = encoder.initHidden().to(model.device)
    
    e_optim.zero_grad()
    d_optim.zero_grad()
    
    input_len = input_tensor.size(0)
    target_len = target_tensor.size(0)
    
    e_outs = torch.zeros(max_len, encoder.hidden_size, device = model.device)
    
    loss = 0
    
    for ei in range(input_len):
        e_out, e_hidden = encoder(input_tensor[ei], e_hidden)
        e_outs[ei] = e_out[0, 0]
        
    d_input = torch.tensor([[utils.SOS_token]], device = model.device)
    d_hidden = e_hidden
    
    for di in range(target_len):
        d_out, d_hidden, atten_w = decoder(d_input, d_hidden, e_outs)
        topv, topi = d_out.topk(1)
        d_input = topi.squeeze().detach()
        
        loss += criterion(d_out, target_tensor[di])
        
        if d_input.item() == utils.EOS_token:
            break
            
    loss.backward()
    
    e_optim.step()
    d_optim.step()
    
    return loss.item() / target_len
    
    
def train(src_lang, tar_lang, pairs, encoder, decoder, iters = 500, lr = 1e-2):
    step_loss = []
    
    e_optim = optim.SGD(encoder.parameters(), lr = lr)
    d_optim = optim.SGD(decoder.parameters(), lr = lr)
    
    training_pairs = [tensor_from_pairs(src_lang, tar_lang, random.choice(pairs)) for _ in range(iters)]
    criterion = nn.NLLLoss()
    
    for iter in range(iters):
        train_pair = training_pairs[iter]
        input_tensor = train_pair[0]
        target_tensor = train_pair[1]
        
        loss = bch_train(input_tensor, target_tensor, encoder, decoder,
                         e_optim, d_optim, criterion)
        step_loss.append(loss)
        
        if iter % 20 == 0:
            print(f'Loss:\t{loss}')
    
    
        
if __name__ == '__main__':
    lines = utils.readCorpus("./data/train.txt")
    ori_lang, tar_lang, pairs = utils.readLang(lines)
    hidden_size = 256
    encoder = model.EncoderRNN(ori_lang.n_words, hidden_size).to(model.device)
    # decoder = model.DecoderRNN(hidden_size, tar_lang.n_words)
    # decoder = model.BahdanauDecoderRNN(hidden_size, tar_lang.n_words).to(model.device)
    decoder = model.LuongDecoderRNN(hidden_size, tar_lang.n_words, attention_method="concat").to(model.device)
    train(ori_lang, tar_lang, pairs, encoder, decoder)

