import os
import h5py
import numpy as np
import pandas as pd
from os.path import join as pjoin
from typing import List, Union, Dict

NORM_CHOICES = ['batch', 'layer', 'group']
CONV_NORM_CHOICES = ['weight', 'spectral']


class Config:
    def __init__(
        self,
            predictive_model=False,      # TODO: temporary
            time_gals=None,      # TODO: temporary
            data_file=None,  # TODO: temporary

            useful_cells: Dict[str, list] = None,
            temporal_res: int = 25,
            grid_size: int = 15,
            time_lags: int = 12,

            nb_lvls: int = 2,
            nb_blocks: int = 1,
            z_dim: int = 8,
            nb_rot_kernels: int = 4,
            nb_rotations: int = 8,
            rot_kernel_size: Union[int, List[int]] = 3,
            decoder_init_grid_size: List[int] = None,

            base_dir: str = 'Documents/PROJECTS/MT_LFP',
            hyperflow_h_file: str = None,
            readout_h_file: str = None,
    ):

        # generic configs
        self.temporal_res = temporal_res
        self.grid_size = grid_size
        self.time_lags = time_lags

        # VAE config
        self.nb_lvls = nb_lvls
        self.nb_blocks = nb_blocks
        self.z_dim = z_dim

        # encoder
        self.nb_rot_kernels = nb_rot_kernels
        self.nb_rotations = nb_rotations
        if isinstance(rot_kernel_size, int):
            self.rot_kernel_size = [rot_kernel_size] * 3    # Y x X x time
        else:
            self.rot_kernel_size = rot_kernel_size

        # decoder
        if decoder_init_grid_size is None:
            self.decoder_init_grid_size = [4, 4, 3]
        else:
            self.decoder_init_grid_size = decoder_init_grid_size

        # dir configs
        self.base_dir = pjoin(os.environ['HOME'], base_dir)
        if hyperflow_h_file is None:
            _file = 'hyperflow.h'
            self.hyperflow_h_file = pjoin(self.base_dir, 'synth_hyperflow', _file)
        else:
            self.hyperflow_h_file = hyperflow_h_file
        readout_h_file = data_file       # TODO: temporary
        if readout_h_file is None:
            _file = 'mt_lfp_grd{:d}tres{:d}.h5'
            _file = _file.format(grid_size, temporal_res)
            self.readout_h_file = pjoin(self.base_dir, 'python_processed', _file)
        else:
            self.readout_h_file = readout_h_file

        if useful_cells is None:
            self.useful_cells = load_cellinfo(self.base_dir)
        else:
            self.useful_cells = useful_cells


class TrainConfig:
    def __init__(self,
                 lr: float = 1e-2,
                 batch_size: int = 64,
                 weight_decay: float = 1e-1,

                 scheduler_type: str = 'cosine',
                 scheduler_gamma: float = 0.9,
                 scheduler_period: int = 10,
                 lr_min: float = 1e-8,

                 beta_warmup_steps: int = None,

                 log_freq: int = 100,
                 chkpt_freq: int = 1,
                 eval_freq: int = 5,
                 random_state: int = 42,
                 xv_folds: int = 5,
                 use_cuda: bool = True,
                 runs_dir: str = 'Documents/MT/runs',):

        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay

        _allowed_schedulers = ['cosine', 'exponential', 'step', 'cyclic', None]
        assert scheduler_type in _allowed_schedulers,\
            "allowed scheduler types: {}".format(_allowed_schedulers)
        self.scheduler_type = scheduler_type
        self.scheduler_gamma = scheduler_gamma
        self.scheduler_period = scheduler_period
        self.lr_min = lr_min

        self.beta_warmup_steps = beta_warmup_steps

        self.log_freq = log_freq
        self.chkpt_freq = chkpt_freq
        self.eval_freq = eval_freq
        self.random_state = random_state
        self.xv_folds = xv_folds
        self.use_cuda = use_cuda
        self.runs_dir = pjoin(os.environ['HOME'], runs_dir)


class VAEConfig:
    def __init__(
        self,
            z_dim: int = 8,
            time_lags: int = 12,
            nb_rot_kers: int = 8,
            nb_rotations: int = 8,
            nb_latent_scales: int = 2,
            layers_per_scale: int = 2,
            norm: str = 'batch',
            conv_norm: str = 'spectral',
            use_se: bool = True,
            rot_ker_sizes: Union[int, List[int]] = 3,
            dec_init_sizes: Union[int, List[int]] = None,

            temporal_res: int = 25,
            grid_size: int = 15,
            base_dir: str = 'Documents/MT',
            h_file: str = None,
    ):
        # VAE
        self.z_dim = z_dim
        self.time_lags = time_lags
        self.nb_latent_scales = nb_latent_scales
        self.layers_per_scale = layers_per_scale
        self.norm = norm if norm in NORM_CHOICES else None
        self.conv_norm = conv_norm if conv_norm in CONV_NORM_CHOICES else None
        self.use_se = use_se

        # encoder/decoder
        self.nb_rot_kers = nb_rot_kers
        self.nb_rotations = nb_rotations
        self.rot_ker_sizes = [rot_ker_sizes] * 3 if isinstance(rot_ker_sizes, int) else rot_ker_sizes
        self.dec_init_sizes = [4, 4, 3] if dec_init_sizes is None else dec_init_sizes

        # dir configs
        self.temporal_res = temporal_res
        self.grid_size = grid_size
        self.base_dir = pjoin(os.environ['HOME'], base_dir)
        if h_file is None:
            _file = 'hyperflow.h'   # TODO: this also has sres and tres info in file name
            # TODO: hfile has both train and test in it
            self.h_file = pjoin(self.base_dir, 'synth_hyperflow', _file)
        else:
            self.h_file = h_file


class BaseConfig(object):
    def __init__(self,
                 time_lags: int = 12,
                 init_range: float = 0.05,
                 base_dir: str = 'Documents/MT',
                 temporal_res: int = 25,
                 grid_size: int = 15,
                 h_file: str = None,
                 useful_cells: Dict[str, list] = None,
                 ):

        self.time_lags = time_lags
        self.init_range = init_range

        # dir configs
        self.base_dir = pjoin(os.environ['HOME'], base_dir)
        if h_file is None:
            _file = 'mt_lfp_grd{:d}tres{:d}.h5'
            _file = _file.format(grid_size, temporal_res)
            self.h_file = pjoin(self.base_dir, 'python_processed', _file)
        else:
            self.h_file = h_file

        # useful cells
        self.useful_cells = self.load_cellinfo() if useful_cells is None else useful_cells

    def load_cellinfo(self):
        clu = pd.read_csv(pjoin(self.base_dir, "extra_info", "cellinfo.csv"))
        ytu = pd.read_csv(pjoin(self.base_dir, "extra_info", "cellinfo_ytu.csv"))

        clu = clu[np.logical_and(1 - clu.SingleElectrode, clu.HyperFlow)]
        ytu = ytu[np.logical_and(1 - ytu.SingleElectrode, ytu.HyperFlow)]

        useful_cells = {}

        for name in clu.CellName:
            useful_channels = []
            for i in range(1, 16 + 1):
                if clu[clu.CellName == name]["chan{:d}".format(i)].item():
                    useful_channels.append(i - 1)

            if len(useful_channels) > 1:
                useful_cells.update({name: useful_channels})

        for name in ytu.CellName:
            useful_channels = []
            for i in range(1, 24 + 1):
                if ytu[ytu.CellName == name]["chan{:d}".format(i)].item():
                    useful_channels.append(i - 1)

            if len(useful_channels) > 1:
                useful_cells.update({name: useful_channels})

        return useful_cells


class ReadoutConfig(BaseConfig):
    def __init__(self,
                 expt: str,
                 core_dim: int = 64,
                 dropout: float = 0.0,
                 include_lvls: List[int] = None,
                 nb_sk: List[int] = None,
                 nb_tk: List[int] = None,
                 **kwargs,
                 ):
        super(ReadoutConfig, self).__init__(**kwargs)

        self.expt = expt
        self.core_dim = core_dim
        self.dropout = dropout
        self.include_lvls = [0, 1, 2] if include_lvls is None else include_lvls
        self.nb_sk = [16, 8, 4] if nb_sk is None else nb_sk
        self.nb_tk = [2, 2, 1] if nb_tk is None else nb_tk
        assert len(self.include_lvls) == len(self.nb_sk) == len(self.nb_tk)


class FFConfig(BaseConfig):
    def __init__(self,
                 expt: str,
                 time_lags: int = 12,
                 **kwargs,
                 ):
        super(FFConfig, self).__init__(**kwargs)

        # readout config
        self.expt = expt
        self.time_lags = time_lags
