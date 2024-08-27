# -*- coding: utf-8 -*-
"""Evaluation Code."""

import torch
from utils import AverageMeter


def evaluate_step(model, dataloader, criterion, device):
    """Evaluate step."""
    model.eval()
    loss_meter = AverageMeter()
    with torch.no_grad():
        for x in dataloader:
            x = x.to(device)
            y, _ = model(x[:, :-1])
            loss = criterion(y.permute(0, 2, 1), x[:, 1:])

            loss_meter.update(loss.item(), n=x.shape[0])

    return loss_meter.avg


def generate(model, sentence, vocab, tokenizer, device, max_len=256, end_id=2):
    """Generate a text."""
    model.eval()

    tokens = tokenizer(sentence.lower())
    indices = vocab.lookup_indices([token.text for token in tokens])
    with torch.no_grad():
        for _ in range(max_len - len(indices)):
            x = torch.LongTensor(indices).unsqueeze(0).to(device)
            y, attention = model(x)
            pred = y.argmax(2)[:, -1].item()

            indices.append(pred)
            if pred == end_id:
                break
    g = ' '.join(vocab.lookup_tokens(indices))

    return g, attention
