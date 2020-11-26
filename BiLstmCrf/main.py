from utils import load_raw_data, Vocab, Tags, trainDataset, trainDataLoader
from biLSTM_crf import BiLSTM_CRF

from tqdm import tqdm

import torch
import torch.optim as optim


EPOCH = 10
LR = 1e-2
BATCH_SIZE = 4
EMBED_DIM = 8
HIDDEN_DIM = 16

def train(vocab, tags, dataloader):
    """
    :type dataloader: trainDataLoader
    """

    model = BiLSTM_CRF(vocab = vocab, tags = tags,
                       embed_dim = EMBED_DIM, hidden_dim = HIDDEN_DIM,
                       batch_size = BATCH_SIZE)

    optimizer = optim.Adam(model.parameters(), lr = LR)

    total_step = len(dataloader.dl) * EPOCH
    pbar =tqdm(total = total_step, desc = "batch")

    for ep in range(EPOCH):
        dl = dataloader.refresh()

        for bch in dl:
            optimizer.zero_grad()
            x = bch[0]
            y = bch[1]

            loss = model(x, y)

            loss.backward()
            optimizer.step()

            pbar.set_postfix({"ep": f"{ep}",
                              "loss": f"{loss}"})
            pbar.update(1)

    model_to_save = model.module if hasattr(model, "module") else model
    torch.save(model_to_save.state_dict(), "./ner.pth")

def pipeline():
    # ============
    # get examples
    # ============

    examples, unit_tags = load_raw_data("./data/demo.dev.char")
    vocab_op = Vocab()
    tags_op = Tags()

    for sen, tag in examples:
        vocab_op.add_sentence(sen)
    vocab_op.get_idx_to_token()

    tags_op.add_tag_seq(unit_tags)

    # ============
    # train model
    # ============
    dataloader = trainDataLoader(dl = trainDataset(vocab_op, tags_op, examples),
                                 batch_size = BATCH_SIZE,
                                 pad_seq_val = vocab_op.get_id_from_token(vocab_op.UNK),
                                 pad_tag_val = tags_op.get_id_from_tag("O"))

    train(vocab_op, tags_op, dataloader)


if __name__ == "__main__":
    pipeline()


