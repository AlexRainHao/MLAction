from tqdm import tqdm
import logging

import torch
from transformers import AdamW, get_linear_schedule_with_warmup

from model import BertFineTuneModel

logger = logging.getLogger(__name__)


class TrainingPipeLine():
    """pass"""
    def __init__(self,
                 epochs = 10,
                 walking_epoch_visual = 1,
                 lr = 2e-5,
                 dropout = 0.2,
                 device = torch.device("cpu"),
                 int2idx = None,
                 idx2int = None):

        self.epochs = epochs
        self.walking_epoch_visual = walking_epoch_visual
        self.lr = lr
        self.dropout = dropout
        self.device = device
        self.int2idx = int2idx
        self.idx2int = idx2int

    def train(self, encoder, data_loader):
        """
        Parameters
        ----------
        data_loader: NluClsDataLoader
        """

        model = BertFineTuneModel(encoder, len(self.int2idx), self.dropout)
        model.to(self.device)

        last_loss = 0.
        acc = 0.
        pbar = tqdm(range(self.epochs), desc = "Epochs")

        optimizer = AdamW(model.parameters(), lr = self.lr)
        total_steps = len(data_loader.dl) * self.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps = 0,
                                                    num_training_steps = total_steps)


        for ep in pbar:
            ep_loss = 0.
            model.train()
            b_data_loader = data_loader.refresh()

            for batch in b_data_loader:
                optimizer.zero_grad()
                b_input_ids = batch["input_ids"].to(self.device)
                b_input_mask = batch["input_mask"].to(self.device)
                b_label = batch["label"].to(self.device)

                logits, loss, acc = model(b_input_ids,
                                          b_input_mask,
                                          b_label)

                ep_loss += loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()

            if ep % self.walking_epoch_visual == 0:
                pbar.set_postfix({"epoch": f"{ep}",
                                  "loss": f"{ep_loss:.3f}",
                                  "acc": f"{acc:.3f}"})
                last_loss += ep_loss

        if self.walking_epoch_visual:
            logger.info("Finished training albert finetune policy, "
                        "loss={:.3f}, train accuracy={:.3f}"
                        "".format(last_loss, acc))

        return model


    def decode(self, model, tokenizer, max_len, text, ranks):
        model.eval()

        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens = True,
            max_length = max_len,
            truncation = True,
            return_token_type_ids = False,
            padding = "max_length",
            return_attention_mask = True,
            return_tensors = "pt",
        )

        input_ids = encoding["input_ids"].view(1, -1).to(self.device)
        input_mask = encoding["attention_mask"].view(1, -1).to(self.device)

        logits, _, _ = model(input_ids, input_mask, None)

        pointer = logits.flatten().argsort(descending = True).tolist()[:ranks]
        score = torch.exp(logits.flatten()) / torch.exp(logits.flatten()).sum()
        score = score[pointer].tolist()

        return score, [self.idx2int[x] for x in pointer]

