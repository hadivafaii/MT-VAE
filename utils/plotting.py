import os
import torch
import numpy as np
from typing import Dict
from copy import deepcopy as dc
from tqdm.notebook import tqdm
from .generic_utils import to_np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import FigureCanvasPdf, PdfPages
from matplotlib.patches import FancyBboxPatch, BoxStyle
from matplotlib.collections import PatchCollection
import seaborn as sns


def mk_2d_tuning_curve(
        tuning_curve: np.ndarray,
        lim: float = 5.0,
        save_file: str = None,
        display: bool = True,
        figsize=(32, 27),
        dpi=100,):

    num_latents = tuning_curve.shape[0]
    num = int(np.sqrt(tuning_curve.shape[-1]))
    xx = np.linspace(-lim, lim, num)
    yy = np.linspace(-lim, lim, num)

    sns.set_style('white')
    fig, axes = plt.subplots(16, 16, figsize=figsize, dpi=dpi, sharex='all', sharey='all')

    ticks = [0, num // 2 - 1, num - 1]
    xtick_labels = np.round(xx[[0, num // 2 - 1, num - 1]], decimals=1)
    ytick_labels = np.round(yy[[0, num // 2 - 1, num - 1]], decimals=1)

    fr_min = np.min(tuning_curve)
    fr_max = np.max(tuning_curve)

    for dim1 in tqdm(range(num_latents)):
        if dim1 == 0:
            for i in range(num_latents):
                axes[dim1, i].set_title('z idx = {:d}'.format(i))
        remaining = list(range(num_latents))
        remaining.remove(dim1)
        for dim2 in tqdm(remaining, leave=False):
            im = axes[dim1, dim2].imshow(
                X=tuning_curve[dim1, dim2].reshape(num, num).T,
                cmap='gist_ncar',
                vmin=fr_min,
                vmax=fr_max,
                interpolation='bilinear',
            )

    axes[-1, 0].set_xticks(ticks)
    axes[-1, 0].set_xticklabels(xtick_labels)
    axes[-1, 0].set_yticks(ticks)
    axes[-1, 0].set_yticklabels(ytick_labels)

    _ = fig.colorbar(im, ax=axes.ravel().tolist(), aspect=80, pad=0.02)

    msg = "2D tuning curves. {:s}\nmin firing rate:  {:.2f}  /  max firing rate:  {:.2f}"
    msg = msg.format(save_file, fr_min, fr_max)
    sup = fig.suptitle(msg, fontsize=30, y=0.95)

    save_fig(fig, sup, save_file, display)
    return fig, axes


def mk_psth_plots(
        output_test: Dict[str, np.ndarray],
        output_valid: Dict[str, np.ndarray],
        true_color: str = 'black',
        pred_color: str = 'dodgerblue',
        save_file: str = None,
        display: bool = True,
        figsize=(20, 16),
        dpi=100,):
    sns.set_style('dark')

    psth, pred = output_test['psth'], output_test['pred']
    nc, ntrials, nt = psth.shape

    axes = []
    figs = []
    sups = []
    for cell in range(nc):
        raster = psth[cell].copy()
        raster[raster > 0] = 1
        fr = psth[cell].mean(0)

        f, ax_arr = plt.subplots(2, 1, sharex='col', figsize=figsize, dpi=dpi)

        width = 0.01
        height = 0.5

        facecolor = true_color
        alpha = 0.8
        edgecolor = true_color

        r = [FancyBboxPatch((y, x), width, height, boxstyle=BoxStyle("Square", pad=0.3))
             for (x, y) in zip(*np.where(raster != 0))]
        pc = PatchCollection(r, facecolor=facecolor, alpha=alpha, edgecolor=edgecolor)

        xticks = range(0, nt + 1, 40)
        xtick_labels = [int(t * 25 / 1000) for t in range(0, nt + 1, 40)]

        ax_arr[0].set_xlim(0, nt + 1)
        ax_arr[0].set_ylim(0, ntrials + 1)
        ax_arr[0].add_collection(pc)
        ax_arr[0].set_aspect('1')
        ax_arr[0].invert_yaxis()
        ax_arr[0].set_ylabel("Trials", fontsize=20)

        ax_arr[1].plot(fr, color=true_color, lw=2)
        ax_arr[1].set_xticks(xticks)
        ax_arr[1].set_xticklabels(xtick_labels)
        ax_arr[1].set_xlabel("Time (s)", fontsize=20)
        ax_arr[1].set_ylabel("Firing rate (Hz)", fontsize=20)
        ax_arr[1].grid()

        ax_arr[1].plot(pred[cell].mean(0), color=pred_color, lw=5, label='VAE')
        ax_arr[1].legend(loc='upper right', prop={'size': 25})

        msg = "{:s} - ch # {:d}\n\n"
        msg += "test nnll = {:.3f},    ===>  test $R^2$ =  {:.1f} {:s}  <=== \n\n"
        msg += "valid nnll = {:.3f},  valid $R^2$ =  {:.1f} {:s}"

        msg = msg.format(
            output_test['expt'], cell,
            np.mean(output_test['nnll'][cell]), np.mean(output_test['r2'][cell]), '%',
            output_valid['nnll'][cell], output_valid['r2'][cell], '%',
        )
        sup = f.suptitle(msg, fontsize=25, y=1.02)
        f.tight_layout()

        figs.append(f)
        sups.append(sup)
        axes.append(np.expand_dims(ax_arr, axis=0))
    axes = np.concatenate(axes)

    # TODO: fix this so the cases are automatically handled
    save_fig(figs, sups, save_file, display, multi=True)
    return figs, axes, sups


def animate_vel_field():
    pass


def plot_vel_field(data, fig_size=None, scale=None, save_file=None, dpi=100, estimate_center=False):
    # TODO: fix this using ax_arr stuff
    if torch.is_tensor(data):
        data = to_np(dc(data))

    grd = data.shape[1]
    xx, yy = np.mgrid[0:grd, 0:grd]

    if len(data.shape) != 4:
        data_to_plot = np.expand_dims(data, axis=-1)
    else:
        data_to_plot = np.copy(data)

    num = data_to_plot.shape[-1]
    if fig_size is None:
        fig_size = (12, 2.5 * num)

    ctrs_all = []
    vminmax = np.max(np.abs(data_to_plot))
    for i in range(num):
        uu, vv = data_to_plot[0, ..., i], data_to_plot[1, ..., i]
        cc = np.sqrt(np.square(uu) + np.square(vv))

        plt.figure(figsize=fig_size, dpi=dpi)
        plt.subplot(num, 3, 3 * i + 1)
        plt.imshow(uu, vmin=-vminmax, vmax=vminmax, cmap='bwr')
        plt.colorbar()
        plt.xlim(-1, grd)
        plt.ylim(-1, grd)

        plt.subplot(num, 3, 3 * i + 2)
        plt.imshow(vv, vmin=-vminmax, vmax=vminmax, cmap='bwr')
        plt.colorbar()
        plt.xlim(-1, grd)
        plt.ylim(-1, grd)

        plt.subplot(num, 3, 3 * i + 3)
        plt.quiver(xx, yy, uu, vv, cc, alpha=1, cmap='PuBu', scale=scale)
        plt.colorbar()
        plt.scatter(xx, yy, s=0.001, color='bisque')
        plt.xlim(-1, grd)
        plt.ylim(-1, grd)
        plt.axis('image')

        if estimate_center:
            ctr_y, ctr_x = np.unravel_index(np.argmax(cc), (grd, grd))
            plt.plot(ctr_y, ctr_x, 'r.', markersize=10)
            ctrs_all.append((ctr_x, ctr_y))

    if save_file is not None:
        plt.savefig(save_file, facecolor='white')
        plt.show()
    else:
        plt.show()

    if estimate_center:
        return ctrs_all


def save_fig(fig, sup, save_file, display, multi=False):
    if save_file is not None:
        save_dir = os.path.dirname(save_file)
        try:
            os.makedirs(save_dir, exist_ok=True)
        except FileNotFoundError:
            pass
        if not multi:
            fig.savefig(save_file, dpi=fig.dpi, bbox_inches='tight', bbox_extra_artists=[sup])
        else:
            assert len(fig) == len(sup)
            with PdfPages(save_file) as pages:
                for f, s in zip(fig, sup):
                    if f is None:
                        continue
                    canvas = FigureCanvasPdf(f)
                    if s is not None:
                        canvas.print_figure(pages, dpi=f.dpi, bbox_inches='tight', bbox_extra_artists=[s])
                    else:
                        canvas.print_figure(pages, dpi=f.dpi, bbox_inches='tight')

    if display:
        if isinstance(fig, list):
            for f in fig:
                plt.show(f)
        else:
            plt.show(fig)
    else:
        if isinstance(fig, list):
            for f in fig:
                plt.close(f)
        else:
            plt.close(fig)
