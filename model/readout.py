import torch
from torch import nn
from torch.nn.utils import weight_norm

from .common import get_init_fn, Permute, LearnedSoftPlus, LearnedSwish
from .configuration import ReadoutConfig
from .model_utils import print_num_params


class Readout(nn.Module):
    def __init__(self, config: ReadoutConfig, verbose=False):
        super(Readout, self).__init__()

        self.config = config

        _temp_dims = [11, 6, 3]
        _spat_dims = [15, 8, 4]
        spatiotemporal = [
            nn.Sequential(
                nn.Dropout3d(p=config.dropout, inplace=True),
                nn.Linear(in_features=_temp_dims[i],    # config.time_lags // 2**i,
                          out_features=config.nb_tk[i], bias=True,),
                Permute(dims=(0, 1, -1, 2, 3)),
                nn.Flatten(start_dim=3),
                weight_norm(nn.Linear(
                    in_features=_spat_dims[i] ** 2,
                    out_features=config.nb_sk[i], bias=True,)),
                nn.Flatten(),)
            for i in config.include_lvls
        ]
        self.spatiotemporal = nn.ModuleList(spatiotemporal)
        self.activation = LearnedSwish(slope=1.0)

        # total filters to pool from
        self.nb_filters = {i: config.nb_tk[i] * config.nb_sk[i] * config.core_dim * 2**i for i in config.include_lvls}
        # self.register_buffer('mask', self._compute_mask())
        nf = sum(list(self.nb_filters.values()))
        nc = len(config.useful_cells[config.expt])
        layers = []
        for cc in range(nc):
            layers += [nn.Sequential(nn.Linear(nf, 1, bias=True), LearnedSoftPlus(beta=1.0))]
        self.layers = nn.ModuleList(layers)
        # self.layer = nn.Linear(nb_filters, nb_cells, bias=True)
        # self.activations = nn.Softplus()
        self.criterion = nn.PoissonNLLLoss(log_input=False, reduction="sum")

        self.apply(get_init_fn(config.init_range))
        if verbose:
            print_num_params(self)

    def forward(self, *args):
        x = (self.spatiotemporal[i](args[lvl]) for i, lvl in enumerate(self.config.include_lvls))
        x = torch.cat(list(x), dim=-1)
        x = self.activation(x)
        # x.mul_(self.mask)

        y = (layer(x) for layer in self.layers)
        y = torch.cat(list(y), dim=-1)
        # x = self.layer(x)
        # x = self.softplus(x)
        return y

    def _compute_mask(self):
        mask = []
        for i, nf in self.nb_filters.items():
            nb_exc = nf // 2
            nb_inh = nf - nb_exc
            mask.extend([1.] * nb_exc + [-1.] * nb_inh)
        return torch.tensor(mask, dtype=torch.float)


class ConvReadout(nn.Module):
    def __init__(self, config: ReadoutConfig, verbose=False):
        super(ConvReadout, self).__init__()

        self.config = config

        _temp_dims = [11, 6, 3]
        _spat_dims = [15, 8, 4]
        spatiotemporal = [
            nn.Sequential(
                weight_norm(nn.Conv3d(
                    in_channels=config.core_dim * 2**i,
                    out_channels=config.nb_units[i],
                    kernel_size=config.kernel_sizes[i],
                    groups=config.groups[i],)),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(),
            )
            for i in config.include_lvls
        ]
        self.spatiotemporal = nn.ModuleList(spatiotemporal)
        self.dropout = nn.Dropout(config.dropout)

        self.layer = nn.Linear(sum(config.nb_units), len(config.useful_cells[config.expt]), bias=True)
        self.softplus = nn.Softplus()
        self.criterion = nn.PoissonNLLLoss(log_input=False, reduction="sum")

        self.apply(get_init_fn(config.init_range))
        if verbose:
            print_num_params(self)

    def forward(self, *args):
        x = (self.spatiotemporal[i](args[lvl]) for i, lvl in enumerate(self.config.include_lvls))
        x = torch.cat(list(x), dim=-1)
        x = self.dropout(x)
        # x = self.relu(x)
        x = self.layer(x)
        x = self.softplus(x)
        return x



class SingleCellReadout(nn.Module):
    def __init__(self, config: ReadoutConfig, verbose=False):
        super(SingleCellReadout, self).__init__()

        self.config = config
        self.nc = len(config.useful_cells[config.expt])

        # spatio-temporal filters
        self.temporal = nn.ModuleList([
            nn.Linear(
                in_features=config.time_lags // 2**i,
                out_features=config.nb_tk[i],
                bias=False,)
            for i in config.include_lvls
        ])
        _spat_dims = [15, 8, 4]
        self.spatial = nn.ModuleList([
            nn.Linear(
                in_features=_spat_dims[i]**2,
                out_features=config.nb_sk[i] * self.nc,
                bias=False,)
            for i in config.include_lvls
        ])

        # last layer
        nb_filters = sum(config.nb_tk[i] * config.nb_sk[i] * config.core_dim * 2**i for i in config.include_lvls)
        self.layers = nn.ModuleDict(
            {"{:d}".format(c): nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Dropout(config.dropout),
                nn.Linear(in_features=nb_filters,
                          out_features=1, bias=True,),
                nn.Softplus(),)
                for c in range(self.nc)}
        )
        self.criterion = nn.PoissonNLLLoss(log_input=False, reduction="sum")

        if verbose:
            print_num_params(self)

    def forward(self, *args):
        outputs = []
        for nb_sk, tk, sk, x in zip(self.config.nb_sk, self.temporal, self.spatial, args):
            output = tk(x).permute(0, 1, 3, 2)
            output = sk(output)
            output = output.split(nb_sk, dim=-1)
            output = tuple(item.flatten(start_dim=1) for item in output)
            outputs.append(output)
        outputs = tuple(torch.cat([item[c] for item in outputs], dim=-1) for c in range(self.nc))
        outputs = tuple(layer(features) for features, layer in zip(outputs, self.layers.values()))
        y = torch.cat(outputs, dim=-1)
        return y

