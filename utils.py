# -*- coding: utf-8 -*-
"""Utility Code."""

import pickle
import logging


class AverageMeter:
    """AverageMeter."""

    def __init__(self):
        """Set zero value."""
        self.reset()

    def reset(self):
        """Reset."""
        self.avg = 0
        self.sum = 0
        self.count = 0

        return None

    def update(self, val, n=1):
        """Update value."""
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        return None


def save_dict(path, dic):
    """Save a dict."""
    with open(path, 'wb') as f:
        pickle.dump(dic, f)
    return


def load_dict(path):
    """Load a dict."""
    with open(path, 'rb') as f:
        dic = pickle.load(f)
    return dic


def set_logger(root_path):
    """Set logger."""
    formatter = logging.Formatter("[%(asctime)s] %(message)s")

    # logging.info() --> write log in the console and the file
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # logging.debug() --> write log to the file
    file_handler = logging.FileHandler(root_path / 'info.log')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.DEBUG)

    return None
