# -*- coding: utf-8 -*-
"""Data Code."""

import torch
import spacy
import numpy as np
from random import randint
from spacy.symbols import ORTH

# special tokens
UNK_TOKEN = "<unk>"
SOT_TOKEN = "<sot>"
EOT_TOKEN = "<eot>"
SP_TOKENS = [UNK_TOKEN, SOT_TOKEN, EOT_TOKEN]


class CustomDataset(torch.utils.data.Dataset):
    """Custom Dataset for flatten ids."""

    def __init__(self, token_id_arr, n_iter=1024):
        """Init.

        - n_iter: number of iterations per one epoch.
        """
        super().__init__()
        self.token_id_arr = token_id_arr
        self.length = token_id_arr.shape[0] - 1
        if n_iter != 0:
            self.n_iter = n_iter  # number of iterations per a epoch.
        else:
            self.n_iter = self.length

    def __len__(self):
        """Length."""
        return self.n_iter

    def __getitem__(self, i):
        """Get a item."""
        # select a sentence
        if self.n_iter != 0:
            i = randint(0, self.length)
        return self.token_id_arr[i]

    def collate_fn(self, batch):
        """Collate fn for batch."""
        tensors = [torch.from_numpy(arr) for arr in batch]
        tensor = torch.stack(tensors)
        tensor = tensor.type(torch.long)
        return tensor


def get_text_data(args):
    """Get text data for training."""
    vocab = torch.load(args.vocab_path)
    train_ids = np.load(args.train_id_path)
    valid_ids = np.load(args.valid_id_path)

    train_dataset = CustomDataset(train_ids, n_iter=1024 * args.batch_size)
    valid_dataset = CustomDataset(valid_ids, n_iter=0)

    spacy_en = spacy.load('en_core_web_sm')
    for token in SP_TOKENS:
        spacy_en.tokenizer.add_special_case(token, [{ORTH: token}])
    tokenizer = spacy_en.tokenizer

    datasets = {'train': train_dataset,
                'valid': valid_dataset}

    return vocab, tokenizer, datasets
