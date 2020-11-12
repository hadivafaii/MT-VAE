import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .configuration import ReadoutConfig
from .model_utils import get_null_adj_nll, print_num_params
import sys; sys.path.append('..')
from utils.generic_utils import to_np


class Readout(nn.Module):
    def __init__(self,
                 config: ReadoutConfig,
                 verbose=False,):
        super(Readout, self).__init__()

        self.config = config

        self.temporal_fcs = nn.ModuleList([
            nn.Linear(
                in_features=config.time_lags // 2**i,
                out_features=config.nb_tk[i],
                bias=False,)
            for i in config.include_lvls
        ])
        _spat_dims = [15, 8, 4]
        self.spatial_fcs = nn.ModuleList([
            nn.Linear(
                in_features=_spat_dims[i]**2,
                out_features=config.nb_sk[i],
                bias=False,)
            for i in config.include_lvls
        ])

        nb_filters = 0
        for i in config.include_lvls:
            nb_filters += config.nb_tk[i] * config.nb_sk[i] * config.core_dim * 2**i
        self.layer = nn.Linear(nb_filters, len(config.useful_cells[config.expt]), bias=True)
        # self.norm = nn.LayerNorm(nb_filters)
        self.relu = nn.ReLU(inplace=True)
        self.softplus = nn.Softplus()
        self.dropout = nn.Dropout(config.dropout)
        self.criterion = nn.PoissonNLLLoss(log_input=False, reduction="sum")

        if verbose:
            print_num_params(self)

    def forward(self, *args):
        assert len(args) == len(self.config.include_lvls)
        x = (self.temporal_fcs[i](item).permute(0, 1, -1, 2) for i, item in enumerate(args))
        z = (self.spatial_fcs[i](item).flatten(start_dim=1) for i, item in enumerate(x))

        z = torch.cat(list(z), dim=-1)
        z = self.relu(z)
        z = self.dropout(z)
        # z = self.norm(z)

        y = self.layer(z)
        y = self.softplus(y)

        return y
