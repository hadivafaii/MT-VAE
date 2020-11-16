import os
import h5py
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy.linalg import norm
from os.path import join as pjoin

import torch
from torch.utils.data import Dataset, DataLoader

from .configuration import ReadoutConfig, TrainConfig


def normalize_fn(x, dim=None):
    if isinstance(x, dict):
        return {k: (v - v.mean(axis=dim, keepdims=True)) / v.std(axis=dim, keepdims=True) for (k, v) in x.items()}
    else:
        return (x - x.mean(axis=dim, keepdims=True)) / x.std(axis=dim, keepdims=True)


class ReadoutDataset(Dataset):
    def __init__(self,
                 stim: np.ndarray,
                 spks: np.ndarray,
                 idxs: np.ndarray,
                 extras: dict = None,
                 transform=None,):
        assert len(stim) == len(spks) == len(idxs), "input/output num samples must be equal"
        self.stim = stim
        self.spks = spks
        self._idxs = idxs
        self.extras = {} if extras is None else extras
        self.transform = transform

    def __len__(self):
        return len(self._idxs)

    def __getitem__(self, i):
        src = self.stim[i]
        tgt = self.spks[i]

        if self.transform is not None:
            src = self.transform(src)

        return src, tgt


# TODO: capture errors whenever output_r is empty (no repeat data)
# Then dl_test should be an empty dataset
def create_readout_dataset(config: ReadoutConfig, train_config: TrainConfig):
    output, output_r = load_readout_data(config)
    stim, spks, good_indxs, extras = output
    stim_r, spks_r, extras_r = output_r
    good_indxs_tst = np.arange(config.time_lags, len(spks_r))

    nt = len(good_indxs)
    train_inds, valid_inds = generate_xv_folds(nt, num_folds=train_config.xv_folds)

    rng = np.random.RandomState(train_config.random_state)
    rng.shuffle(train_inds)

    good_indxs_trn = good_indxs[train_inds]
    good_indxs_vld = good_indxs[valid_inds]

    stim_trn, spks_trn = process_readout_data(good_indxs_trn, stim, spks, config.time_lags)
    stim_vld, spks_vld = process_readout_data(good_indxs_vld, stim, spks, config.time_lags)
    stim_tst, spks_tst = process_readout_data(good_indxs_tst, stim_r, spks_r, config.time_lags)

    ds_train = ReadoutDataset(
        stim=stim_trn,
        spks=spks_trn,
        idxs=good_indxs_trn,
        extras=extras,
        transform=normalize_fn,)
    ds_valid = ReadoutDataset(
        stim=stim_vld,
        spks=spks_vld,
        idxs=good_indxs_vld,
        extras=extras,
        transform=normalize_fn,)
    ds_test = ReadoutDataset(
        stim=stim_tst,
        spks=spks_tst,
        idxs=good_indxs_tst,
        extras=extras_r,
        transform=normalize_fn,)

    dl_train = DataLoader(
        dataset=ds_train,
        batch_size=train_config.batch_size,
        shuffle=True,
        drop_last=True,)
    dl_valid = DataLoader(
        dataset=ds_valid,
        batch_size=train_config.batch_size,
        shuffle=False,
        drop_last=False,)
    dl_test = DataLoader(
        dataset=ds_test,
        batch_size=train_config.batch_size,
        shuffle=False,
        drop_last=False,)

    return dl_train, dl_valid, dl_test


def process_readout_data(idxs, stim, spks, time_lags):
    src = []
    for i in idxs:
        src.append(np.expand_dims(stim[..., i - time_lags: i], axis=0))
    src = np.concatenate(src).astype(float)
    tgt = spks[idxs].astype(float)

    assert len(src) == len(tgt), "input and output must have same length"
    return src, tgt


def load_readout_data(config: ReadoutConfig):
    f = h5py.File(config.h_file, 'r')
    grp = f[config.expt]

    spks = np.array(grp['spks'])
    spks = spks[:, config.useful_cells[config.expt]]
    stim = np.array(grp['stim'], dtype=float)
    stim = np.transpose(stim, (3, 1, 2, 0))  # 2 x grd x grd x nt

    badspks = np.array(grp['badspks'])
    goodspks = 1 - badspks
    good_indxs = np.where(goodspks == 1)[0]
    good_indxs = good_indxs[good_indxs > config.time_lags]
    true_good_indxs = refine_good_indices(stim, good_indxs, config.time_lags)

    extras = {}     # TODO: later might wanna add lfp
    output = (
        stim.astype(float),
        spks.astype(float),
        true_good_indxs.astype(int),
        extras,
    )

    if 'repeats' in grp:
        repeats_grp = grp['repeats']
        spks_r = np.array(repeats_grp['spksR'], dtype=float)
        spks_r = spks_r[:, config.useful_cells[config.expt]]
        psth = np.array(repeats_grp['psth_raw_all'], dtype=float)
        psth = psth[config.useful_cells[config.expt]]
        start_inds = np.array(repeats_grp['tind_start_all'], dtype=int)
        start_inds = start_inds[config.useful_cells[config.expt]]
        stim_r = np.array(repeats_grp['stimR'], dtype=float)
        stim_r = np.transpose(stim_r, (3, 1, 2, 0))  # 2 x grd x grd x nt

        badspks_r = np.array(repeats_grp['badspksR'], dtype=int)
        goodspks_r = 1 - badspks_r
        good_indxs_r = np.where(goodspks_r == 1)[0]
        true_good_indxs_r = refine_good_indices(stim_r, good_indxs_r, config.time_lags)
        true_good_indxs_r = true_good_indxs_r - config.time_lags  # the whole thing is shifted

        extras_r = {
            'psth': psth.astype(int),
            'start_inds': start_inds.astype(int),
            'good_indxs_r': true_good_indxs_r.astype(int),
        }
        output_r = (
            stim_r.astype(float),
            spks_r.astype(float),
            extras_r,
        )
    else:
        output_r = ()

    f.close()

    return output, output_r


def refine_good_indices(stim, good_indxs, time_lags):
    _eps = 0.1
    nt = stim.shape[-1]
    stim_norm = norm(stim.reshape(-1, nt), axis=0)
    bad_indices = np.where(stim_norm < _eps)[0]

    true_good_indxs = []
    for i in good_indxs:
        if not set(range(i - time_lags, i + 1)).intersection(bad_indices):
            true_good_indxs.append(i)

    return np.array(true_good_indxs)


"""
class UnSupervisedDataset(Dataset):
    def __init__(self, data_dict, time_lags, time_gals, transform=None):

        self.stim = data_dict['stim']
        self.good_indxs = data_dict['good_indxs']
        self.train_indxs = data_dict['train_indxs']
        self.valid_indxs = data_dict['valid_indxs']

        assert not set(self.train_indxs).intersection(set(self.valid_indxs)), "train/valid indices must be disjoint"
        assert len(self.valid_indxs) + len(self.train_indxs) == len(self.good_indxs)

        self.time_lags = time_lags
        self.time_gals = time_gals
        self.transform = transform

    def __len__(self):
        return len(self.train_indxs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        i = self.good_indxs[self.train_indxs[idx]]

        source = self.stim[..., i - self.time_lags: i]
        target = self.stim[..., i: i + self.time_gals]

        if self.transform is not None:
            source = self.transform(source)
            target = self.transform(target)

        return source, target

def create_datasets(config, xv_folds, rng, load_unsupervised=False, load_processed=True):
    _dir = pjoin(config.base_dir, "pytorch_processed")
    files = os.listdir(_dir)

    if load_processed and len(files) == 3:
        print('processed data found: {}. loading . . .'.format(files))

        supervised_final = joblib.load(pjoin(_dir, "supervised.sav"))
        # supervised_dataset = SupervisedDataset(supervised_final, config.time_lags, normalize_fn)

        nardin_final = joblib.load(pjoin(_dir, "nardin.sav"))
        # nardin_dataset_train = NardinDataset(nardin_final, config.time_lags, True, normalize_fn)
        # nardin_dataset_valid = NardinDataset(nardin_final, config.time_lags, False, normalize_fn)

        if load_unsupervised:
            unsupervised_final = joblib.load(pjoin(_dir, "unsupervised.sav"))
            unsupervised_dataset = UnSupervisedDataset(unsupervised_final, config.time_lags, config.time_gals, normalize_fn)
            return supervised_dataset, nardin_dataset_train, nardin_dataset_valid, unsupervised_dataset
        else:
            return supervised_dataset, nardin_dataset_train, nardin_dataset_valid, None

    supervised_data_dict, unsupervised_data_dict = _load_data(config)

    # supervised part
    supervised_final = {}
    for expt, data_dict in supervised_data_dict.items():
        nt = len(data_dict['good_indxs'])
        train_inds, valid_inds = generate_xv_folds(nt, num_folds=xv_folds)
        rng.shuffle(train_inds)
        data = {
            'stim': data_dict['stim'],
            'spks': data_dict['spks'],
            'train_indxs': data_dict['good_indxs'][train_inds],
            'valid_indxs': data_dict['good_indxs'][valid_inds],
        }
        supervised_final.update({expt: data})
    joblib.dump(supervised_final, pjoin(_dir, "supervised.sav"))
    supervised_dataset = SupervisedDataset(supervised_final, config.time_lags, normalize_fn)

    # nardin part
    nardin_final = _load_nardin_data(config)[0]
    joblib.dump(nardin_final, pjoin(_dir, "nardin.sav"))
    nardin_dataset_train = NardinDataset(nardin_final, config.time_lags, True, normalize_fn)
    nardin_dataset_valid = NardinDataset(nardin_final, config.time_lags, False, normalize_fn)

    # unsupervised part
    stim_all = []
    bad_indxs = []
    _eps = 1
    total_nt = 0
    for k, v in unsupervised_data_dict.items():
        bad_indxs.extend(range(total_nt, total_nt + config.time_lags + 1))

        nt = v.shape[-1]
        stim_norm = norm(v.reshape(-1, nt), axis=0)
        zero_norm_indices = np.where(stim_norm < _eps)[0]

        if len(zero_norm_indices) != 0:
            diff_mat = np.eye(len(zero_norm_indices)) - np.eye(len(zero_norm_indices), k=-1)
            boundary_indxs = np.where(diff_mat @ zero_norm_indices != 1)[0]

            for i in range(len(boundary_indxs) - 1):
                start = total_nt + zero_norm_indices[boundary_indxs[i]] - config.time_gals
                end = total_nt + zero_norm_indices[boundary_indxs[i + 1] - 1] + config.time_lags
                bad_indxs.extend(range(start, end))
            start = total_nt + zero_norm_indices[boundary_indxs[-1]] - config.time_gals
            end = total_nt + zero_norm_indices[-1] + config.time_lags
            bad_indxs.extend(range(start, end))

        total_nt += nt
        bad_indxs.extend(range(total_nt - config.time_gals - 1, total_nt))
        stim_all.append(v)

    good_indxs = set(range(total_nt)).difference(set(bad_indxs))
    stim_all = np.concatenate(stim_all, axis=-1)

    zero_norm_indices = np.where(norm(stim_all.reshape(-1, total_nt), axis=0) < _eps)[0]
    assert not set(zero_norm_indices).intersection(good_indxs), "good_indxs must exclude indices where ||stim|| = 0"
    assert stim_all.shape[-1] == total_nt

    train_inds, valid_inds = generate_xv_folds(len(good_indxs), num_folds=xv_folds, num_blocks=1)
    rng.shuffle(train_inds)

    unsupervised_final = {
        'stim': stim_all.astype(float),
        'good_indxs': list(good_indxs),
        'train_indxs': list(train_inds),
        'valid_indxs': list(valid_inds),
    }
    joblib.dump(unsupervised_final, pjoin(_dir, "unsupervised.sav"))
    unsupervised_dataset = UnSupervisedDataset(unsupervised_final, config.time_lags, config.time_gals, normalize_fn)

    return supervised_dataset, nardin_dataset_train, nardin_dataset_valid, unsupervised_dataset
"""


def generate_xv_folds(nt, num_folds=5, num_blocks=3, which_fold=None):
    """Will generate unique and cross-validation indices, but subsample in each block
        NT = number of time steps
        num_folds = fraction of data (1/fold) to set aside for cross-validation
        which_fold = which fraction of data to set aside for cross-validation (default: middle of each block)
        num_blocks = how many blocks to sample fold validation from"""

    valid_inds = []
    nt_blocks = np.floor(nt / num_blocks).astype(int)
    block_sizes = np.zeros(num_blocks, dtype=int)
    block_sizes[range(num_blocks - 1)] = nt_blocks
    block_sizes[num_blocks - 1] = nt - (num_blocks - 1) * nt_blocks

    if which_fold is None:
        which_fold = num_folds // 2
    else:
        assert which_fold < num_folds, 'Must choose XV fold within num_folds = {}'.format(num_folds)

    # Pick XV indices for each block
    cnt = 0
    for bb in range(num_blocks):
        start = np.floor(block_sizes[bb] * (which_fold / num_folds))
        if which_fold < num_folds - 1:
            stop = np.floor(block_sizes[bb] * ((which_fold + 1) / num_folds))
        else:
            stop = block_sizes[bb]

        valid_inds = valid_inds + list(range(int(cnt + start), int(cnt + stop)))
        cnt = cnt + block_sizes[bb]

    valid_inds = np.array(valid_inds, dtype='int')
    train_inds = np.setdiff1d(np.arange(0, nt, 1), valid_inds)

    return list(train_inds), list(valid_inds)


def _load_nardin_data(config):
    path = pjoin(config.base_dir, "nardin", "python_processed")
    info = pd.read_pickle(pjoin(path, "info_w_ctrs_w_fr.pd"))
    info['2_name'] = np.array(info['2_name']).astype('str')

    threshold_fr = 0.1
    good_cells = list(info['4_fr'] > threshold_fr)
    info = info[good_cells]
    info = info.reset_index()

    unique_expt_names = list(np.unique(info["2_name"]))
    cell_ids = {}
    for expt in unique_expt_names:
        key = "nardin-{:s}".format(expt)
        indxs = list(info[info["2_name"] == expt].index)
        cell_ids.update({key: indxs})

    stim = np.transpose(np.load(os.path.join(path, "stim1.npy")), (3, 1, 2, 0))  # 2 x grd x grd x nt
    spks = np.load(os.path.join(path, "spks.npy"))[:, good_cells]
    filters = np.load(os.path.join(path, "data_filters.npy"))[:, good_cells]

    nt = stim.shape[-1]
    train_indxs, valid_indxs = generate_xv_folds(nt, num_folds=5, num_blocks=3)
    first_third = range(nt // 3)

    stim = stim[..., first_third]
    spks = spks[first_third]
    filters = filters[first_third]
    train_indxs = train_indxs[:len(train_indxs) // 3]
    valid_indxs = valid_indxs[:len(valid_indxs) // 3]

    train_indxs = [x for x in train_indxs if x > config.time_lags]
    valid_indxs = [x for x in valid_indxs if x > config.time_lags]

    data = {
        "stim": stim,
        "spks": spks,
        "filters": filters,
        "train_indxs": train_indxs,
        "valid_indxs": valid_indxs,
        "cell_ids": cell_ids,
    }

    return data, info


def _load_data(config):
    supervised_data_dict = {}
    unsupervised_data_dict = {}

    ff = h5py.File(config.data_file, 'r')
    for expt in tqdm(ff.keys()):
        grp = ff[expt]
        stim = np.transpose(np.array(grp['stim']), (3, 1, 2, 0))  # 2 x grd x grd x nt
        unsupervised_data_dict.update({expt: stim.astype(float)})

        if expt in config.useful_cells.keys():
            badspks = np.array(grp['badspks'])
            goodspks = 1 - badspks
            good_indxs = np.where(goodspks == 1)[0]
            good_indxs = good_indxs[good_indxs > config.time_lags]

            nt = stim.shape[-1]
            good_channels = config.useful_cells[expt]
            spks = np.zeros((nt, len(good_channels)))

            for i, cc in enumerate(good_channels):
                spks[:, i] = np.array(grp['ch_%d' % cc]['spks_%d' % cc]).squeeze()

            true_good_indxs = refine_good_indices(stim, good_indxs, config.time_lags)

            _data = {
                'stim': stim.astype(float),
                'spks': spks.astype(float),
                'good_indxs': true_good_indxs.astype(int),
            }
            supervised_data_dict.update({expt: _data})

    ff.close()

    return supervised_data_dict, unsupervised_data_dict
