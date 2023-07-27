# Reference: https://github.com/qinzheng93/GeoTransformer

import torch.nn as nn
import torch.nn.functional as F
import torch
from lib.utils import square_distance
import numpy as np


def get_activation(activation, **kwargs):
    if activation == 'relu':
        return nn.ReLU(inplace=True)
    elif activation == 'lrelu':
        if 'negative_slope' in kwargs:
            negative_slope = kwargs['negative_slope']
        else:
            negative_slope = 0.01
        return nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
    elif activation == 'elu':
        return nn.ELU(inplace=True)
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'gelu':
        return nn.GLU()
    else:
        raise RuntimeError('Activation function {} is not supported.'.format(activation))


class MonteCarloDropout(nn.Module):
    def __init__(self, p=0.1):
        super(MonteCarloDropout, self).__init__()
        self.p = p

    def forward(self, x):
        out = nn.functional.dropout(x, p=self.p, training=True)
        return out


def get_dropout(p, monte_carlo_dropout=False):
    if p is not None and p > 0:
        if monte_carlo_dropout:
            return MonteCarloDropout(p)
        else:
            return nn.Dropout(p)
    else:
        return None
