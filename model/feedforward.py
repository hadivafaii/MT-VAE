import os
from datetime import datetime
from os.path import join as pjoin
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Union

import torch
from torch import nn
from torch.nn.utils import weight_norm
from torch.optim import Adam

from .configuration import FFConfig
from .common import get_init_fn
from .model_utils import print_num_params


class GLM(nn.Module):
    def __init__(self, config: FFConfig, verbose=False):
        super(GLM, self).__init__()

        self.config = config

        spat_dim = 15
        self.kernel = weight_norm(nn.Linear(
            in_features=config.time_lags * 2 * spat_dim ** 2,
            out_features=1,
            bias=False,
        ))
        self.flatten = nn.Flatten()
        self.softplus = nn.Softplus()
        self.criterion = nn.PoissonNLLLoss(log_input=False, reduction="sum")

        self.apply(get_init_fn(self.config.init_range))
        if verbose:
            print_num_params(self)

    def forward(self, x):
        x = self.flatten(x)
        x = self.kernel(x)
        y = self.softplus(x)
        return y


class DSGLM(nn.Module):
    def __init__(self, config: FFConfig, verbose: bool = False):
        super(DSGLM, self).__init__()
        assert not config.multicell, "For single cell modeling only"
        self.config = config

        num_units = [1] + config.nb_vel_tuning_units + [1]
        layers = []
        for i in range(len(config.nb_vel_tuning_units) + 1):
            layers += [nn.Conv2d(num_units[i], num_units[i + 1], 1), nn.LeakyReLU()]

        self.vel_tuning = nn.Sequential(*layers)
        self.dir_tuning = nn.Linear(2, 1, bias=False)

        self.temporal_kernel = nn.Linear(config.time_lags, 1, bias=False)
        self.spatial_kernel = nn.Linear(config.grid_size ** 2, 1, bias=True)

        self.criterion = nn.PoissonNLLLoss(log_input=False, reduction="sum")
        self.softplus = nn.Softplus()

        self.apply(get_init_fn(config.init_range))
        self._load_vel_tuning()

        if verbose:
            print_num_params(self)

    def forward(self, x):
        x = x.permute(0, 4, 2, 3, 1)  # N x tau x grd x grd x 2
        rho = torch.norm(x, dim=-1)

        # angular component
        f_theta = torch.exp(self.dir_tuning(x).squeeze(-1) / rho.masked_fill(rho == 0., 1e-8))

        # radial component
        original_shape = rho.size()  # N x tau x grd x grd
        rho = rho.flatten(end_dim=1).unsqueeze(1)  # N*tau x 1 x grd x grd
        f_r = self.vel_tuning(rho)
        f_r = f_r.squeeze(1).view(original_shape)

        # full subunit
        subunit = f_theta * f_r
        subunit = subunit.flatten(start_dim=2)  # N x tau x H*W

        # apply spatial and temporal kernels
        y = self.spatial_kernel(subunit).squeeze()  # N x tau
        y = self.temporal_kernel(y)  # N x 1
        y = self.softplus(y)

        return y

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def _load_vel_tuning(self):
        _dir = pjoin(os.environ['HOME'], 'Documents/PROJECTS/MT_LFP/vel_dir_weights')
        try:
            print('[INFO] loading vel tuning identity weights')
            self.vel_tuning.load_state_dict(torch.load(pjoin(_dir, 'id_weights.bin')))
        except (FileNotFoundError, RuntimeError):
            print('[INFO] file does not exist, training from scratch')
            self._train_vel_tuning()
            os.makedirs(_dir, exist_ok=True)
            torch.save(self.vel_tuning.state_dict(), pjoin(_dir, 'id_weights.bin'))

    def _train_vel_tuning(self):
        if torch.cuda.is_available():
            self.cuda()

        tmp_optim = Adam(self.vel_tuning.parameters())
        loss_fn = nn.MSELoss()

        ratio = 10
        nb_epochs = 2000
        batch_size = 8192
        pbar = tqdm(range(nb_epochs))
        for epoch in pbar:
            tmp_data = torch.rand((batch_size, 1, self.config.grid_size, self.config.grid_size))
            tmp_data = tmp_data * ratio - 0.02
            tmp_data = tmp_data.cuda()

            pred = self.vel_tuning(tmp_data)
            loss = loss_fn(pred, tmp_data)

            tmp_optim.zero_grad()
            loss.backward()
            tmp_optim.step()

            pbar.set_description("epoch # {:d}, loss: {:.5f}".format(epoch, loss.item()))

        print('[INFO] training vel tuning identity weights done')

    def extras_to_device(self, device):
        for reg_type, reg_mat in self.reg_mats_dict.items():
            self.reg_mats_dict[reg_type] = reg_mat.to(device)

    def visualize(self, xv_nnll, xv_r2, save=False):
        dir_tuning = self.dir_tuning.weight.data.flatten().cpu().numpy()
        b_abs = np.linalg.norm(dir_tuning)
        theta = np.arccos(dir_tuning[1] / b_abs)

        tker = self.temporal_kernel.weight.data.flatten().cpu().numpy()
        sker = self.spatial_kernel.weight.data.view(self.config.grid_size, self.config.grid_size).cpu().numpy()

        if max(tker, key=abs) < 0:
            tker *= -1
            sker *= -1

        sns.set_style('dark')
        plt.figure(figsize=(16, 4))
        plt.subplot(121)
        t_rng = np.array([39, 36, 32, 27, 22, 15, 7, 0])
        plt.xticks(t_rng, (self.config.time_lags - t_rng - 1) * -self.config.temporal_res)
        plt.xlabel('Time (ms)', fontsize=25)
        plt.plot(tker)
        plt.grid()
        plt.subplot(122)
        plt.imshow(sker, cmap='bwr')
        plt.colorbar()

        plt.suptitle(
            '$\\theta_p = $ %.2f deg,     b_abs = %.4f     . . .     xv_nnll:  %.4f,       xv_r2:  %.2f %s'
            % (np.rad2deg(theta), b_abs, xv_nnll, xv_r2, '%'), fontsize=15)

        if save:
            result_save_dir = os.path.join(self.config.base_dir, 'results/PyTorch')
            os.makedirs(result_save_dir, exist_ok=True)
            save_name = os.path.join(
                result_save_dir,
                'DS_GLM_{:s}_{:s}.png'.format(self.config.experiment_names, datetime.now().strftime("[%Y_%m_%d_%H:%M]"))
            )
            plt.savefig(save_name, facecolor='white')

        plt.show()