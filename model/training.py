import os
import numpy as np
from tqdm.notebook import tqdm
from typing import Union, Tuple
from os.path import join as pjoin
from sklearn.metrics import r2_score

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, StepLR

from .vae import VAE, Encoder
from .readout import Readout
from .configuration import TrainConfig
from .dataset import create_readout_dataset
from .model_utils import save_model, get_null_adj_nll
from utils.generic_utils import to_np, now


class BaseTrainer(object):
    def __init__(self,
                 model: nn.Module,
                 train_config: TrainConfig,
                 **kwargs,):
        super(BaseTrainer, self).__init__()
        kwargs_defaults = {
            'verbose': False,
        }
        for k, v in kwargs_defaults.items():
            if k not in kwargs:
                kwargs[k] = v

        os.environ["SEED"] = str(train_config.random_state)
        torch.manual_seed(train_config.random_state)
        np.random.seed(train_config.random_state)

        cuda_condition = torch.cuda.is_available() and train_config.use_cuda
        self.device = torch.device("cuda" if cuda_condition else "cpu")

        self.model = model.to(self.device).eval()
        self.config = model.config
        self.train_config = train_config
        self.writer = None

        self.dl_train = None
        self.dl_valid = None
        self.dl_test = None
        self.setup_data()

        self.optim = None
        self.optim_schedule = None
        self.setup_optim()

        if kwargs['verbose']:
            print("\nTotal Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, nb_epochs: Union[int, range], comment: str):
        assert isinstance(nb_epochs, (int, range)), "Please provide either range or int"

        writer_dir = pjoin(
            self.train_config.runs_dir,
            type(self.model).__name__,
            "{}".format(now(exclude_hour_min=True)),
            "{}".format(comment),
        )
        self.writer = SummaryWriter(writer_dir)

        epochs_range = range(nb_epochs) if isinstance(nb_epochs, int) else nb_epochs
        pbar = tqdm(epochs_range)
        for epoch in pbar:
            avg_loss = self.iteration(epoch=epoch)
            pbar.set_description('epoch # {:d}, avg loss: {:3f}'.format(epoch + 1, avg_loss))
            if self.optim_schedule is not None:
                self.optim_schedule.step()

            if (epoch + 1) % self.train_config.chkpt_freq == 0:
                save_model(self.model, comment=comment, chkpt=epoch + 1)

            if (epoch + 1) % self.train_config.eval_freq == 0:
                nb_iters = len(self.dl_train)
                global_step = (epoch + 1) * nb_iters
                _ = self.validate(global_step, verbose=False)
                if self.dl_test is not None:
                    _ = self.test(global_step, verbose=False)

    def iteration(self, epoch: int = 0):
        raise NotImplementedError

    def validate(self, global_step: int = None, verbose: int = True):
        raise NotImplementedError

    def test(self, global_step: int = None, verbose: int = True):
        raise NotImplementedError

    def setup_data(self):
        raise NotImplementedError

    def swap_model(self, new_model, full_setup: bool = True):
        self.model = new_model.to(self.device).eval()
        self.config = new_model.config
        if full_setup:
            self.setup_data()
            self.setup_optim()

    def setup_optim(self):
        self.optim = AdamW(
            self.model.parameters(),
            lr=self.train_config.lr,
            weight_decay=self.train_config.weight_decay,
        )
        if self.train_config.scheduler_type == 'cosine':
            self.optim_schedule = CosineAnnealingLR(
                self.optim,
                T_max=self.train_config.scheduler_period,
                eta_min=self.train_config.eta_min,
            )
        elif self.train_config.scheduler_type == 'exponential':
            self.optim_schedule = ExponentialLR(
                self.optim,
                gamma=self.train_config.scheduler_gamma,
            )
        elif self.train_config.scheduler_type == 'step':
            self.optim_schedule = StepLR(
                self.optim,
                step_size=self.train_config.scheduler_period,
                gamma=self.train_config.scheduler_gamma,
            )

    def to_cuda(self, x, dtype=torch.float32) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if isinstance(x, (tuple, list)):
            if any(torch.is_tensor(item) for item in x):
                return tuple(map(lambda z: z.to(device=self.device, dtype=dtype), x))
            else:
                return tuple(map(lambda z: torch.tensor(z, device=self.device, dtype=dtype), x))
        else:
            if torch.is_tensor(x):
                return x.to(device=self.device, dtype=dtype)
            else:
                return torch.tensor(x, device=self.device, dtype=dtype)


class ReadoutTrainer(BaseTrainer):
    def __init__(self,
                 model: Readout,
                 train_config: TrainConfig,
                 core: Encoder,
                 **kwargs,):
        super(ReadoutTrainer, self).__init__(model, train_config, **kwargs)
        self.core = core.to(self.device).eval()

    def iteration(self, epoch: int = 0):
        self.model.train()

        cuml_loss = 0.0
        nb_iters = len(self.dl_train)
        pbar = tqdm(enumerate(self.dl_train), total=nb_iters, leave=False)
        for i, (x, y) in pbar:
            global_step = epoch * nb_iters + i
            x, y = self.to_cuda((x, y))

            with torch.no_grad():
                x1, x2, x3, _ = self.core(x)[0]
            x = tuple(map(lambda z: z.flatten(start_dim=2, end_dim=3), (x1, x2, x3)))

            pred = self.model(*x)
            loss = self.model.criterion(pred, y) / self.dl_train.batch_size
            _check_for_nans(loss, global_step)
            cuml_loss += loss.item()

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            if (global_step + 1) % self.train_config.log_freq == 0:
                self.writer.add_scalar("loss/train", loss.item(), global_step)
            self.writer.add_scalar('extras/lr', self.optim.param_groups[0]['lr'], global_step)
        return cuml_loss / nb_iters

    def validate(self, global_step: int = None, verbose: bool = True):
        loss, pred, true = self.generate_prediction('valid')

        nnll = get_null_adj_nll(pred, true)
        r2 = r2_score(true, pred, multioutput='raw_values') * 100

        if global_step is not None:
            self.writer.add_scalar("loss/valid", loss, global_step)
            self.writer.add_scalar("mean/nnll/valid", np.mean(nnll), global_step)
            self.writer.add_scalar("median/nnll/valid", np.median(nnll), global_step)
            self.writer.add_scalar("mean/r2/valid", np.mean(r2), global_step)
            self.writer.add_scalar("median/r2/valid", np.median(r2), global_step)
        if verbose:
            msg = "valid,   loss: {:.3f},   mean nnll:  {:.3f},   mean r2:  {:.2f} {:s}"
            msg = msg.format(loss, np.mean(nnll), np.mean(r2), '%')
            print(msg)

        output = {
            'true': true,
            'pred': pred,
            'loss': loss,
            'nnll': nnll,
            'r2': r2,
        }
        return output

    def test(self, global_step: int = None, verbose: bool = True):
        loss, pred, true = self.generate_prediction('test')

        psth = self.dl_test.dataset.extras['psth']
        start_inds = self.dl_test.dataset.extras['start_inds']
        good_idxs = self.dl_test.dataset.extras['good_indxs_r']

        true_stacked = np.zeros(psth.shape)
        pred_stacked = np.zeros(psth.shape)
        nnll_stacked = np.zeros(psth.shape[:-1])

        nc, ntrials, trial_length = psth.shape
        for cell in range(nc):
            for trial_id, start_ind in enumerate(start_inds[cell]):
                slice_ = range(start_ind - self.config.time_lags,
                               start_ind - self.config.time_lags + trial_length)

                true_stacked[cell][trial_id] = true[slice_, cell]
                pred_stacked[cell][trial_id] = pred[slice_, cell]

                nnll_indxs = set(slice_).intersection(set(good_idxs))
                nnll_indxs = list(nnll_indxs)

                nnll_stacked[cell, trial_id] = get_null_adj_nll(
                    pred=pred[nnll_indxs, cell],
                    true=true[nnll_indxs, cell],
                )

        r2 = r2_score(psth.mean(1).T, pred_stacked.mean(1).T, multioutput='raw_values') * 100

        if global_step is not None:
            self.writer.add_scalar("loss/test", loss, global_step)
            self.writer.add_scalar("mean/nnll/test", np.mean(nnll_stacked), global_step)
            self.writer.add_scalar("median/nnll/test", np.median(nnll_stacked), global_step)
            self.writer.add_scalar("mean/r2/test", np.mean(r2), global_step)
            self.writer.add_scalar("median/r2/test", np.median(r2), global_step)
        if verbose:
            msg = "test,  num trials {:d},   loss: {:.3f},\n\n"
            msg += "nnll over trials:\nmean:\n{},\nvar:\n{}\n\n"
            msg += "\nnnll mean: {:.4f},   nnll median: {:.4f}\n\n"
            msg += "r2:\n{},\n\nr2 mean: {:.2f} {:s},   r2 median: {:.2f} {:s}\n\n"
            msg = msg.format(ntrials, loss,
                             nnll_stacked.mean(-1), nnll_stacked.var(-1),
                             np.mean(nnll_stacked), np.median(nnll_stacked),
                             r2, np.mean(r2), '%', np.median(r2), '%')
            print(msg)

        output = {
            'true': true_stacked,
            'pred': pred_stacked,
            'nnll': nnll_stacked,
            'psth': psth,
            'loss': loss,
            'r2': r2,
        }
        return output

    def generate_prediction(self, mode):
        if mode == 'train':
            loader = self.dl_train
        elif mode == 'valid':
            loader = self.dl_valid
        elif mode == 'test':
            loader = self.dl_test
        else:
            raise ValueError("invalid mode: {}".format(mode))

        self.model.eval()

        loss_list = []
        pred_list = []
        true_list = []
        for (x, y) in loader:
            x, y = self.to_cuda((x, y))

            with torch.no_grad():
                x1, x2, x3, _ = self.core(x)[0]
                x = tuple(map(lambda z: z.flatten(start_dim=2, end_dim=3), (x1, x2, x3)))
                pred = self.model(*x)

            loss = self.model.criterion(pred, y)
            loss_list.append(loss.item())
            pred_list.append(pred)
            true_list.append(y)

        loss = np.mean(loss_list) / loader.batch_size
        pred = torch.cat(pred_list)
        true = torch.cat(true_list)
        pred, true = tuple(map(to_np, (pred, true)))

        return loss, pred, true

    def setup_data(self):
        dl_train, dl_valid, dl_test = create_readout_dataset(self.config, self.train_config)
        self.dl_train = dl_train
        self.dl_valid = dl_valid
        self.dl_test = dl_test


# TODO: write a VAETrainer

def _check_for_nans(loss, global_step: int):
    if torch.isnan(loss).sum().item():
        msg = "nan encountered in loss. optimizer will detect this and skip. global step = {}"
        msg = msg.format(global_step)
        raise RuntimeWarning(msg)
