from collections.abc import Iterable
from numpy import exp

import random
import numpy as np
import torch

def sigmoid(X):
   return 1/(1+exp(-X))


def flatten(l):
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


def fix_random_seed(seed_value=1763):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
