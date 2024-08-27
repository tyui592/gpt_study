# -*- coding: utf-8 -*-
"""Main code."""

from train import train_model
from config import get_arguments

if __name__ == '__main__':
    args = get_arguments()

    if args.wb_flag:
        import wandb

        run = wandb.init(project='gpt',
                         job_type='train',
                         name=args.wb_name,
                         notes=args.wb_notes,
                         tags=args.wb_tags,
                         config=args)
    train_model(args)
