import random
import numpy as np
import torch


def seed_all(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)


def get_mean(lis):
    return sum(lis) / len(lis)
