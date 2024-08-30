# -*- coding: utf-8 -*-
"""Configurations."""

import torch
import logging
from models import get_model
from data import get_text_data
from utils import AverageMeter
from evaluate import evaluate_step, generate


def train_step(model, dataloader, criterion, optimizer, device, print_interval=0.05):
    """Train a one epoch."""
    global global_step
    model.train()
    loss_meter = AverageMeter()

    if print_interval < 1:
        print_interval = int(len(dataloader) * print_interval)

    for x in dataloader:
        x = x.to(device)
        y, _ = model(x[:, :-1])
        loss = criterion(y.permute(0, 2, 1), x[:, 1:])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), n=x.shape[0])

        global_step += 1
        if global_step % print_interval == 0:
            logging.info((f"Step: {global_step}, "
                          f"Loss: {loss_meter.avg:1.4f}"))

    return loss_meter.avg


def train_model(args):
    """Train a model."""
    if args.wb_flag:
        import wandb

    global global_step
    global_step = 0

    device = torch.device('cuda:0')

    model = get_model(args).to(device)

    vocab, tokenizer, datasets = get_text_data(args)

    train_dataloader = torch.utils.data.DataLoader(datasets['train'],
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   collate_fn=datasets['train'].collate_fn)

    valid_dataloader = torch.utils.data.DataLoader(datasets['valid'],
                                                   batch_size=args.batch_size,
                                                   shuffle=False,
                                                   collate_fn=datasets['valid'].collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr,
                                  betas=args.lr_betas,
                                  weight_decay=args.weight_decay,
                                  eps=1e-9)
    class_weights = None
    if args.sp_weight is not None:
        class_weights = torch.ones(args.vocab_size, dtype=torch.float32).to(device)
        class_weights[1:3] = args.sp_weight
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights,
                                          label_smoothing=args.label_smoothing)

    best_valid_loss = float('inf')
    for epoch in range(args.epoch):
        train_loss = train_step(model,
                                train_dataloader,
                                criterion,
                                optimizer, device, args.print_interval)

        valid_loss = evaluate_step(model, valid_dataloader, criterion, device)

        logging.info((f"Epoch: {epoch}/{args.epoch}, Step: {global_step}, "
                      f"Train Loss: {train_loss:1.4f}, Valid Loss: {valid_loss:1.4f}"))
        if args.wb_flag:
            wandb.log({'train loss': train_loss,
                       'valid loss': valid_loss},
                      step=global_step)

        # generate sample texts
        token_ids = datasets['train'][0]
        tokens = vocab.lookup_tokens(token_ids)
        text = ' '.join(tokens)
        logging.info(f"Full\t: {text}")
        # use first half tokens as input
        inp = ' '.join(tokens[:args.context_size // 2])
        out, _ = generate(model, inp, vocab, tokenizer, device, args.max_len)
        logging.info(f"Input\t: {inp}\n")
        logging.info(f"Output\t: {out}\n")

        torch.save({'step': global_step, 'state_dict': model.state_dict()},
                   args.save_root / 'trained_model.pt')

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save({'step': global_step, 'state_dict': model.state_dict()},
                       args.save_root / 'best_val.pt')
