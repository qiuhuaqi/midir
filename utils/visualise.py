import numpy as np
import torch
import os
import random
from matplotlib import pyplot as plt


def plot_warped_grid(ax, disp, bg_img=None, interval=3, title="$\mathcal{T}_\phi$", fontsize=30, color='c'):
    """disp shape (2, H, W)"""
    if bg_img is not None:
        background = bg_img
    else:
        background = np.zeros(disp.shape[1:])

    id_grid_H, id_grid_W = np.meshgrid(range(0, background.shape[0] - 1, interval),
                                       range(0, background.shape[1] - 1, interval),
                                       indexing='ij')

    new_grid_H = id_grid_H + disp[0, id_grid_H, id_grid_W]
    new_grid_W = id_grid_W + disp[1, id_grid_H, id_grid_W]

    kwargs = {"linewidth": 1.5, "color": color}
    # matplotlib.plot() uses CV x-y indexing
    for i in range(new_grid_H.shape[0]):
        ax.plot(new_grid_W[i, :], new_grid_H[i, :], **kwargs)  # each draws a horizontal line
    for i in range(new_grid_H.shape[1]):
        ax.plot(new_grid_W[:, i], new_grid_H[:, i], **kwargs)  # each draws a vertical line

    ax.set_title(title, fontsize=fontsize)
    ax.imshow(background, cmap='gray')
    # ax.axis('off')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)


def plot_result_fig(vis_data_dict, save_path=None, title_font_size=20, dpi=100, show=False, close=False):
    """Plot visual results in a single figure/subplots.
    Images should be shaped (*sizes)
    Disp should be shaped (ndim, *sizes)

    vis_data_dict.keys() = ['target', 'source', 'target_original',
                            'target_pred', 'warped_source',
                            'disp_gt', 'disp_pred']
    """
    fig = plt.figure(figsize=(30, 18))
    title_pad = 10

    ax = plt.subplot(2, 4, 1)
    plt.imshow(vis_data_dict["target"], cmap='gray')
    plt.axis('off')
    ax.set_title('Target', fontsize=title_font_size, pad=title_pad)

    ax = plt.subplot(2, 4, 2)
    plt.imshow(vis_data_dict["target_original"], cmap='gray')
    plt.axis('off')
    ax.set_title('Target original', fontsize=title_font_size, pad=title_pad)

    # calculate the error before and after reg
    error_before = vis_data_dict["target"] - vis_data_dict["target_original"]
    error_after = vis_data_dict["target"] - vis_data_dict["target_pred"]

    # error before
    ax = plt.subplot(2, 4, 3)
    plt.imshow(error_before, vmin=-2, vmax=2, cmap='seismic')  # assuming images were normalised to [0, 1]
    plt.axis('off')
    ax.set_title('Error before', fontsize=title_font_size, pad=title_pad)

    # error after
    ax = plt.subplot(2, 4, 4)
    plt.imshow(error_after, vmin=-2, vmax=2, cmap='seismic')  # assuming images were normalised to [0, 1]
    plt.axis('off')
    ax.set_title('Error after', fontsize=title_font_size, pad=title_pad)

    # predicted target image
    ax = plt.subplot(2, 4, 5)
    plt.imshow(vis_data_dict["target_pred"], cmap='gray')
    plt.axis('off')
    ax.set_title('Target predict', fontsize=title_font_size, pad=title_pad)

    # warped source image
    ax = plt.subplot(2, 4, 6)
    plt.imshow(vis_data_dict["warped_source"], cmap='gray')
    plt.axis('off')
    ax.set_title('Warped source', fontsize=title_font_size, pad=title_pad)

    # warped grid: ground truth
    ax = plt.subplot(2, 4, 7)
    bg_img = np.zeros_like(vis_data_dict["target"])
    plot_warped_grid(ax, vis_data_dict["disp_gt"], bg_img, interval=3, title="$\phi_{GT}$", fontsize=title_font_size)

    # warped grid: prediction
    ax = plt.subplot(2, 4, 8)
    plot_warped_grid(ax, vis_data_dict["disp_pred"], bg_img, interval=3, title="$\phi_{pred}$", fontsize=title_font_size)

    # adjust subplot placements and spacing
    plt.subplots_adjust(left=0.0001, right=0.99, top=0.9, bottom=0.1, wspace=0.001, hspace=0.1)

    # saving
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight', dpi=dpi)

    if show:
        plt.show()

    if close:
        plt.close()
    return fig


def visualise_result(data_dict, axis=0, save_result_dir=None, epoch=None, dpi=50):
    """
    Save one validation visualisation figure for each epoch.
    - 2D: 1 random slice from N-slice stack (not a sequence)
    - 3D: the middle slice on the chosen axis

    Args:
        data_dict: (dict) images shape (N, 1, *sizes), disp shape (N, ndim, *sizes)
        save_result_dir: (string) Path to visualisation result directory
        epoch: (int) Epoch number (for naming when saving)
        axis: (int) For 3D only, choose the 2D plane orthogonal to this axis in 3D volume
        dpi: (int) Image resolution of saved figure
    """
    # check cast to Numpy array
    for n, d in data_dict.items():
        if isinstance(d, torch.Tensor):
            data_dict[n] = d.cpu().numpy()

    ndim = data_dict["target"].ndim - 2
    sizes = data_dict["target"].shape[2:]

    # put 2D slices into visualisation data dict
    vis_data_dict = {}
    if ndim == 2:
        # randomly choose a slice for 2D
        z = random.randint(0, data_dict["target"].shape[0]-1)
        for name, d in data_dict.items():
            vis_data_dict[name] = data_dict[name].squeeze()[z, ...]  # (H, W) or (2, H, W)

    else:  # 3D
        # visualise the middle slice of the chosen axis
        z = int(sizes[axis] // 2)
        for name, d in data_dict.items():
            if name in ["disp_pred", "disp_gt"]:
                # dvf.yaml: choose the two axes/directions to visualise
                axes = [0, 1, 2]
                axes.remove(axis)
                vis_data_dict[name] = d[0, axes, ...].take(z, axis=axis+1)  # (2, X, X)
            else:
                # images
                vis_data_dict[name] = d[0, 0, ...].take(z, axis=axis)  # (X, X)

    # housekeeping: dummy dvf_gt for inter-subject case
    if not "disp_gt" in data_dict.keys():
        vis_data_dict["disp_gt"] = np.zeros_like(vis_data_dict["disp_pred"])

    # set up figure saving path
    if save_result_dir is not None:
        fig_save_path = os.path.join(save_result_dir, f'epoch{epoch}_axis_{axis}_slice_{z}.png')
    else:
        fig_save_path = None

    fig = plot_result_fig(vis_data_dict, save_path=fig_save_path, dpi=dpi)
    return fig
