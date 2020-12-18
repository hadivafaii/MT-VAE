import os
import yaml
import numpy as np
from typing import Tuple
from copy import deepcopy as dc
from prettytable import PrettyTable
from os.path import join as pjoin

import torch
from torch import nn

from .configuration import Config, VAEConfig, ReadoutConfig, TrainConfig
from utils.generic_utils import now


def save_model(
        model: nn.Module,
        comment: str,
        chkpt: int = -1,
):
    config_dict = vars(model.config)
    to_hash_dict_ = dc(config_dict)
    hash_str = str(hash(frozenset(sorted(to_hash_dict_))))

    save_dir = pjoin(
        model.config.base_dir,
        'saved_models',
        type(model).__name__,
        '{}_{}'.format(comment, hash_str),
        '{0:04d}'.format(chkpt),
    )
    os.makedirs(save_dir, exist_ok=True)
    bin_file = pjoin(save_dir, '{:s}.bin'.format(type(model).__name__))
    torch.save(model.state_dict(), bin_file)

    config_file = pjoin(save_dir, '{:s}.yaml'.format(type(model.config).__name__))
    with open(config_file, 'w') as f:
        yaml.dump(config_dict, f)

    with open(pjoin(save_dir, '{}.txt'.format(now(exclude_hour_min=False))), 'w') as f:
        f.write("chkpt {:d} saved".format(chkpt))


def load_model(
        keyword: str,
        chkpt_id: int = -1,
        verbose: bool = False,
        base_dir: str = 'Documents/MT',
):
    match = False
    model_dir = pjoin(os.environ['HOME'], base_dir, 'saved_models')
    for root, dirs, files in os.walk(model_dir):
        match = next(filter(lambda x: keyword in x, dirs), None)
        if match:
            model_dir = pjoin(root, match)
            if verbose:
                print('models found:\nroot: {:s}\nmatch: {:s}'.format(root, match))
            break

    if not match:
        raise RuntimeError('no match found for keyword: {:s}'.format(keyword))

    available_chkpts = sorted(os.listdir(model_dir), key=lambda x: int(x))
    if verbose:
        print('there are {:d} chkpts to load'.format(len(available_chkpts)))
    load_dir = pjoin(model_dir, available_chkpts[chkpt_id])

    if verbose:
        print('\nLoading from:\n{}\n'.format(load_dir))

    config_name = next(filter(lambda s: 'yaml' in s, os.listdir(load_dir)), None)
    with open(pjoin(load_dir, config_name), 'r') as stream:
        try:
            config_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    if 'ReadoutConfig' in config_name:
        config = ReadoutConfig(**config_dict)
    elif 'VAEConfig' in config_name:
        raise NotImplementedError
        # config = VAEConfig(**config_dict)
    else:
        raise RuntimeError('unknown config: {}'.format(config_name))

    if type(config).__name__ == 'ReadoutConfig':
        from .readout import Readout
        loaded_model = Readout(config, verbose=verbose)
    elif type(config).__name__ == 'VAEConfig':
        raise NotImplementedError
        # from .vae import VAE
        # loaded_model = VAE(config, verbose=verbose)
    else:
        raise RuntimeError("invalid config type encountered")

    bin_file = pjoin(load_dir, '{:s}.bin'.format(type(loaded_model).__name__))
    loaded_model.load_state_dict(torch.load(bin_file))
    loaded_model.eval()

    chkpt = available_chkpts[chkpt_id]
    metadata = {"model_name": str(match), "chkpt": int(chkpt)}

    return loaded_model, metadata


def save_model2(model, comment, chkpt=-1):
    config_dict = vars(model.config)
    to_hash_dict_ = dc(config_dict)
    hash_str = str(hash(frozenset(sorted(to_hash_dict_))))
    '{0:05d}'.format(25)
    save_dir = pjoin(
        model.config.base_dir,
        'saved_models',
        type(model).__name__,
        '{}_{}'.format(comment, hash_str),
        '{0:03d}'.format(chkpt),
    )
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), pjoin(save_dir, 'model.bin'))

    if type(model.config).__name__ == 'VAECondig':
        config_file = pjoin(save_dir, 'vae_config.yaml')
    elif type(model.config).__name__ == 'ReadoutConfig':
        config_file = pjoin(save_dir, 'readout_config.yaml')
    else:
        raise RuntimeError("invalid config type encountered")

    with open(config_file, 'w') as f:
        yaml.dump(config_dict, f)

    with open(pjoin(save_dir, '{}.txt'.format(now(exclude_hour_min=False))), 'w') as f:
        f.write("chkpt {:d} saved".format(chkpt))


def load_model2(keyword, chkpt_id=-1, config=None, verbose=False, base_dir='Documents/MT'):
    _dir = pjoin(os.environ['HOME'], base_dir, 'saved_models')
    available_models = os.listdir(_dir)
    if verbose:
        print('Available models to load:\n{:s}'.format(available_models))

    match_found = False
    model_id = -1
    for i, model_name in enumerate(available_models):
        if keyword in model_name:
            model_id = i
            match_found = True
            break

    if not match_found:
        raise RuntimeError("no match found for keyword: {:s}".format(keyword))

    model_dir = pjoin(_dir, available_models[model_id])
    available_chkpts = sorted(os.listdir(model_dir))
    if verbose:
        print('\nAvailable chkpts to load:\n{}'.format(available_chkpts))
    load_dir = pjoin(model_dir, available_chkpts[chkpt_id])

    if verbose:
        print('\nLoading from:\n{}\n'.format(load_dir))

    if config is None:
        config_name = next(filter(lambda s: 'yaml' in s, os.listdir(load_dir)), None)
        with open(pjoin(load_dir, config_name), 'r') as stream:
            try:
                config_dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        if 'vae' in config_name:
            config = VAEConfig(**config_dict)
        elif 'readout' in config_name:
            config = ReadoutConfig(**config_dict)
        elif config_name == 'config.yaml':  # TODO: depriciated, delete later
            config = Config(**config_dict)

    if type(config).__name__ == 'VAECondig':
        from .vae import VAE
        loaded_model = VAE(config, verbose=verbose)
    elif type(config).__name__ == 'ReadoutConfig':
        from .readout import Readout, SingleCellReadout  # TODO: is single cell readout worth it?
        loaded_model = Readout(config, verbose=verbose)
    elif type(config).__name__ == 'Config':  # TODO: depriciated, delete later
        from .vae import VAE
        loaded_model = VAE(config, verbose=verbose)
    else:
        raise RuntimeError("invalid config type encountered")

    loaded_model.load_state_dict(torch.load(pjoin(load_dir, 'model.bin')))
    loaded_model.eval()

    model_name = available_models[model_id]
    chkpt = available_chkpts[chkpt_id].split('_')[0]
    chkpt = chkpt.split(':')[-1]
    metadata = {"model_name": str(model_name), "chkpt": int(chkpt)}

    return loaded_model, metadata


def print_num_params(module: nn.Module):
    t = PrettyTable(['Module Name', 'Num Params'])

    for name, m in module.named_modules():
        total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        if '.' not in name:
            if isinstance(m, type(module)):
                t.add_row(["{}".format(m.__class__.__name__), "{}".format(total_params)])
                t.add_row(['---', '---'])
            else:
                t.add_row([name, "{}".format(total_params)])
    print(t, '\n\n')


def add_weight_decay(model, weight_decay: float = 1e-1, skip_keywords: Tuple[str, ...] = ('bias',)):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) <= 1 or any(k in name for k in skip_keywords):
            no_decay.append(param)
        else:
            decay.append(param)

    param_groups = [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay},
    ]
    return param_groups


def r2_score(pred: np.ndarray, true: np.ndarray, axis: int = 0, clean: bool = True):
    r2 = 1 - np.var(true, axis=axis) / np.linalg.norm(pred-true, axis=axis)
    if clean:
        r2 = np.maximum(0.0, r2 * 100)
    return r2


def get_null_adj_nll(pred: np.ndarray, true: np.ndarray, axis: int = 0):

    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)
    if not isinstance(true, np.ndarray):
        true = np.array(true)

    assert not (pred < 0).sum(), "predicted firing rate must be non-negative"

    nll = _get_nll(pred, true, axis)

    r_0 = true.mean(axis)
    null_nll = _get_nll(r_0, true, axis)

    return -nll + null_nll


def _get_nll(pred, true, axis):
    _eps = np.finfo(np.float32).eps
    return np.sum(pred - true * np.log(pred + _eps), axis=axis) / np.maximum(_eps, np.sum(true, axis=axis))
