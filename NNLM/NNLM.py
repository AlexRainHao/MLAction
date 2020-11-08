'''
pass
'''

from copy import deepcopy
from tqdm import tqdm
import os.path as opt
import utils
import torch
import torch.nn as nn
import torch.optim as optim

class NNLM_Model(nn.Module):
    """NNLM model structure"""

    def __init__(self, config):
        super(NNLM_Model, self).__init__()

        self.config = config
        self.load_config()

        self.C = nn.Embedding(self.V, self.m)
        self.H = nn.Linear((self.n - 1) * self.m, self.h)

        self.U = nn.Linear(self.h, self.V)

        if self.direct:
            self.W = nn.Linear((self.n - 1) * self.m, self.V)

    def load_config(self):
        self.n = self.config.window_size
        self.m = self.config.vector_dim
        self.h = self.config.hidden_dim
        self.V = self.config.vocab_size

        self.lr = self.config.learning_rate
        self.direct = self.config.whether_direct

        self.batch_size = self.config.batch_size
        self.use_gpu = self.config.use_gpu


    def forward(self, input_vec):
        """
        :type input_vec: [batch_size, n]
        """
        input_vec = input_vec[:, :-1] # batch, n-1
        target_vec = input_vec[:, 0] # batch, 1
        word_feature = self.C(input_vec) # batch, n - 1, m
        word_feature = word_feature.view(word_feature.shape[0], -1) # batch, m(n-1)
        hidden_features = torch.tanh(self.H(word_feature))

        output_logits = self.U(hidden_features)
        if self.direct:
            output_logits += self.W(word_feature)

        return output_logits, target_vec



def train_epoch(data_generator):
    """train model for each epoch"""
    pbar = tqdm(total = len(train_data_list) % config.batch_size + 1)

    try:
        while data_generator:
            batch = next(data_generator)
            batch_logits, batch_label = model(batch)
            batch_loss = loss(batch_logits, batch_label)
            pbar.update(1)

            batch_loss.backward()
            optimizer.step()
    except StopIteration:
        return batch_loss


def predict(dataloader):
    """a simplified version of prediction"""
    model.eval()
    predLoss = 0.
    idx = 0
    for batch in dataloader:
        pred_logit, tar_label = model(batch)
        pred_label = pred_logit.max(1, keepdim=True)[1]
        predLoss += loss(pred_logit, tar_label)
        idx += 1

    return predLoss


if __name__ == '__main__':
    config = utils.set_config
    epoch = config.epoch

    model = NNLM_Model(config = config)
    model.train()
    if model.use_gpu:
        model.to("cuda")
    else:
        model.to("cpu")

    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = model.lr)

    # load data
    data = utils.load_data("./data/Political-media-DFE.csv")
    data = utils.remove_urls(data)
    train_data = data[:100]
    test_data = data[100:120]
    test_data_list = utils.get_test_token(test_data)

    word2idx, idx2word, length, train_data_list = utils.get_unique_word(train_data)

    utils.mkdir_folder(config.save_model)

    # train epoch
    for ep in range(epoch):
        data_generator = utils.make_batch(train_data_list, word2idx,
                                          window_size=config.window_size,
                                          batch_size=config.batch_size,
                                          if_gpu=config.use_gpu)
        test_data_generator = utils.make_batch(test_data_list, word2idx,
                                               window_size = config.window_size,
                                               batch_size = 8,
                                               if_gpu = config.use_gpu)

        batch_loss = train_epoch(data_generator)
        test_loss = predict(test_data_generator)

        if ep % 100 == 0:
            print(f'Epoch: {ep}, Train Loss: {batch_loss: .5f}')
            print(f'Epoch: {ep}, Test Loss: {test_loss: .5f}')
            model.train()


            # save model
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(),
                       opt.join(config.save_model, f"ckpt_{ep}.pth"))

