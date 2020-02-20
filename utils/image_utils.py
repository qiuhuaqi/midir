import torch
import torch.nn.functional as F
import numpy as np

# -- image normalisation to 0 - 255
def normalise_numpy(x, norm_min=0.0, norm_max=255.0):
    return float(norm_max - norm_min) * (x - np.min(x)) / (np.max(x) - np.min(x))


def normalise_torch(x, nmin=0.0, nmax=255.0):
    return (nmax - nmin) * (x - x.min()) / (x.max() - x.min())

