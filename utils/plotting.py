import torch
import numpy as np
from copy import deepcopy as dc
from .generic_utils import to_np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')


def plot_vel_field(data, fig_size=None, scale=None, save_file=None, estimate_center=False):
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

        plt.figure(figsize=fig_size)
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
        plt.scatter(xx, yy, s=0.005)
        plt.xlim(-1, grd)
        plt.ylim(-1, grd)
        plt.axis('image')

        if estimate_center:
            ctr_y, ctr_x = np.unravel_index(np.argmax(cc), (grd, grd))
            plt.plot(ctr_y, ctr_x, 'r.', markersize=10)
            ctrs_all.append((ctr_x, ctr_y))

    if save_file is not None:
        plt.savefig(save_file, facecolor='white')
        plt.close()
    else:
        plt.show()

    if estimate_center:
        return ctrs_all
