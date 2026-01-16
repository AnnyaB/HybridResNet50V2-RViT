# scripts/utils.py
import os
import time
import random
import numpy as np
import torch


def set_seed(seed=22089065):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AvgMeter:
    def __init__(self):
        self.sum = 0.0
        self.n = 0

    def update(self, v, k=1):
        self.sum += float(v) * k
        self.n += int(k)

    @property
    def avg(self):
        return self.sum / max(1, self.n)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def now():
    return time.time()
