import torch.nn as nn
from torch.nn import CrossEntropyLoss

class BertFineTuneModel(nn.Module):

    def __init__(self, encoder, num_labels, dropout = .2):
        super(BertFineTuneModel, self).__init__()

        self.bert = encoder
        self.dropout = nn.Dropout(dropout)
        self.num_labels = num_labels
        self.cls_layer = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.loss_func = CrossEntropyLoss(reduction = "mean")



    def forward(self, input_ids, input_mask, labels = None):
        _, pool_output = self.bert(input_ids = input_ids,
                                         attention_mask = input_mask)

        pool_output = self.dropout(pool_output)
        logits = self.cls_layer(pool_output)

        loss, accuray = 0., 0.

        if self.training:
            loss = self.call_loss(logits, labels)
            accuray = self.call_acc(logits, labels)

        return logits, loss, accuray


    def call_loss(self, logits, labels):
        loss = self.loss_func(logits.view(-1, self.num_labels),
                              labels.view(-1))

        return loss

    def call_acc(self, logits, labels):
        acc = (logits.argmax(dim = -1) == labels).float().mean().item()
        return acc