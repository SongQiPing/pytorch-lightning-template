import torch
import torch.nn.functional as F
from torch import autograd
import random
import numpy as np


def set_random_seed(seed):
    """ set random seeds for all possible random libraries"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False