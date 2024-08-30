# -*- coding: utf-8 -*-
"""Configurations."""

import argparse
import logging
from pathlib import Path
from utils import save_dict, set_logger


def build_parser():
    """Get arguments from cmd."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_root',
                        type=Path,
                        default='./data-store/TinyStories')

    parser.add_argument('--min_freq',
                        type=int,
                        default=2)

    parser.add_argument('--vocab_path',
                        type=str,
                        default=None)

    parser.add_argument('--train_id_path',
                        type=str,
                        default=None)

    parser.add_argument('--valid_id_path',
                        type=str,
                        default=None)

    parser.add_argument('--save_root',
                        type=Path,
                        default='./model-store/')

    parser.add_argument('--label_smoothing',
                        type=float,
                        default=0.1)

    parser.add_argument('--sp_weight',
                        type=float,
                        default=None)

    parser.add_argument('--weight_decay',
                        type=float,
                        default=5e-4)

    parser.add_argument('--lr',
                        type=float,
                        default=2.5e-4)

    parser.add_argument('--lr_betas',
                        type=float,
                        nargs='+',
                        default=[0.9, 0.98])

    parser.add_argument('--context_size',
                        type=int,
                        default=128)

    parser.add_argument('--max_len',
                        type=int,
                        default=256)

    parser.add_argument('--vocab_size',
                        type=int,
                        default=30000)

    parser.add_argument('--batch_size',
                        type=int,
                        default=64)

    parser.add_argument('--d_model',
                        type=int,
                        default=256)

    parser.add_argument('--d_ff',
                        type=int,
                        default=512)

    parser.add_argument('--n_layers',
                        type=int,
                        default=8)

    parser.add_argument('--n_heads',
                        type=int,
                        default=16)

    parser.add_argument('--attention_drop',
                        type=float,
                        default=0.1)

    parser.add_argument('--residual_drop',
                        type=float,
                        default=0.1)

    parser.add_argument('--embedding_drop',
                        type=float,
                        default=0.1)

    parser.add_argument('--print_interval',
                        type=float,
                        default=0.2)

    parser.add_argument('--epoch',
                        type=int,
                        default=512)

    parser.add_argument('--wb_flag',
                        action='store_true',
                        default=False,
                        help="Use wandb")

    parser.add_argument('--wb_project',
                        type=str,
                        default='gpt')

    parser.add_argument('--wb_name',
                        type=str,
                        default=None)

    parser.add_argument('--wb_notes',
                        type=str,
                        default=None)

    parser.add_argument('--wb_tags',
                        type=str,
                        nargs='+',
                        default=None)

    return parser.parse_args()


def get_arguments():
    """Get arguments."""
    args = build_parser()
    args.save_root.mkdir(exist_ok=True, parents=True)

    # set logger
    set_logger(args.save_root)

    # make data paths and dir
    dir_path = args.dataset_root / f"vocab_size-{args.vocab_size}"

    args.vocab_path = dir_path / 'vocab.pth'
    args.train_id_path = dir_path / 'train_context_arr.npy'
    args.valid_id_path = dir_path / 'valid_context_arr.npy'

    # check arguments
    logging.info("Arguments...")
    for k, v in vars(args).items():
        logging.info(f"{k}: {v}")

    # save arguments
    save_dict(args.save_root / 'arg.pkl', args)

    return args
