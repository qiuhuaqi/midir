"""Miscellaneous Utility functions such as parameter parsing, model saving/loading, visualising results"""
import os
import json
import logging
import shutil

import torch
from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt
import matplotlib
import imageio
import numpy as np

from utils.metrics import computeJacobianDeterminant2D
from utils.vis import flow_to_hsv, show_warped_grid

import random

class Params(object):
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """
    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        if os.path.exists(log_path):
            print("Logger already exists. Overwritting.")
            os.system("mv -f {} {}.backup".format(log_path, log_path))

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(stream_handler)


def set_summary_writer(model_dir, run_name):
    """
    Returns a Tensorboard summary writer
    which writes to [model_dir]/tb_summary/[run_name]/

    Args:
        model_dir: directory of the model
        run_name: sub name of the summary writer (usually 'train' or 'val')

    Returns:
        summary writer

    """
    summary_dir = os.path.join(model_dir, 'tb_summary', run_name)
    if not os.path.exists(summary_dir):
        print("TensorboardX summary directory does not exist...\n Making directory {}".format(summary_dir))
        os.makedirs(summary_dir)
    else:
        print("TensorboardX summary directory already exist at {}...\nOverwritting!".format(summary_dir))
        os.system("rm -rf {}".format(summary_dir))
        os.makedirs(summary_dir)
    return SummaryWriter(summary_dir)


def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        if state['epoch'] == 1:
            print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def plot_results(target, source, warped_source, dvf, save_path=None, title_font_size=20, show_fig=False, dpi=100):
    """Plot all motion related results in a single figure
    dvf is expected to be in number of pixels"""

    # convert flow into HSV flow with white background
    hsv_flow = flow_to_hsv(dvf, max_mag=0.15, white_bg=True)

    ## set up the figure
    fig = plt.figure(figsize=(30, 18))
    title_pad = 10

    # source
    ax = plt.subplot(2, 4, 1)
    plt.imshow(source, cmap='gray')
    plt.axis('off')
    ax.set_title('Source', fontsize=title_font_size, pad=title_pad)

    # warped source
    ax = plt.subplot(2, 4, 2)
    plt.imshow(warped_source, cmap='gray')
    plt.axis('off')
    ax.set_title('Warped Source', fontsize=title_font_size, pad=title_pad)

    # calculate the error before and after reg
    error_before = target - source
    error_after = target - warped_source

    # error before
    ax = plt.subplot(2, 4, 3)
    plt.imshow(error_before, vmin=-255, vmax=255, cmap='gray')
    plt.axis('off')
    ax.set_title('Error before', fontsize=title_font_size, pad=title_pad)

    # error after
    ax = plt.subplot(2, 4, 4)
    plt.imshow(error_after, vmin=-255, vmax=255, cmap='gray')
    plt.axis('off')
    ax.set_title('Error after', fontsize=title_font_size, pad=title_pad)

    # target image
    ax = plt.subplot(2, 4, 5)
    plt.imshow(target, cmap='gray')
    plt.axis('off')
    ax.set_title('Target', fontsize=title_font_size, pad=title_pad)

    # hsv flow
    ax = plt.subplot(2, 4, 7)
    plt.imshow(hsv_flow)
    plt.axis('off')
    ax.set_title('HSV', fontsize=title_font_size, pad=title_pad)

    # quiver, or "Displacement Vector Field" (DVF)
    ax = plt.subplot(2, 4, 6)
    interval = 3  # interval between points on the grid
    background = source
    quiver_flow = np.zeros_like(dvf)
    quiver_flow[:, :, 0] = dvf[:, :, 0]
    quiver_flow[:, :, 1] = dvf[:, :, 1]
    mesh_x, mesh_y = np.meshgrid(range(0, dvf.shape[1] - 1, interval),
                                 range(0, dvf.shape[0] - 1, interval))
    plt.imshow(background[:, :], cmap='gray')
    plt.quiver(mesh_x, mesh_y,
               quiver_flow[mesh_y, mesh_x, 1], quiver_flow[mesh_y, mesh_x, 0],
               angles='xy', scale_units='xy', scale=1, color='g')
    plt.axis('off')
    ax.set_title('DVF', fontsize=title_font_size, pad=title_pad)

    # det Jac
    ax = plt.subplot(2, 4, 8)
    jac_det, mean_grad_detJ, negative_detJ = computeJacobianDeterminant2D(dvf)
    spec = [(0, (0.0, 0.0, 0.0)), (0.000000001, (0.0, 0.2, 0.2)),
            (0.12499999999, (0.0, 1.0, 1.0)), (0.125, (0.0, 0.0, 1.0)),
            (0.25, (1.0, 1.0, 1.0)), (0.375, (1.0, 0.0, 0.0)),
            (1, (0.94509803921568625, 0.41176470588235292, 0.07450980392156863))]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('detjac', spec)
    plt.imshow(jac_det, vmin=-1, vmax=7, cmap=cmap)
    plt.axis('off')
    ax.set_title('Jacobian (Grad: {0:.2f}, Neg: {1:.2f}%)'.format(mean_grad_detJ, negative_detJ * 100),
                 fontsize=int(title_font_size*0.9), pad=title_pad)
    # split and extend this axe for the colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(cax=cax1)
    cb.ax.tick_params(labelsize=20)

    # adjust subplot placements and spacing
    plt.subplots_adjust(left=0.0001, right=0.99, top=0.9, bottom=0.1, wspace=0.001, hspace=0.1)

    # saving
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight', dpi=dpi)

    if show_fig:
        plt.show()
    plt.close()


def save_train_result(target, source, warped_source, dvf, save_result_dir, epoch, fps=20, dpi=40):
    """
    Args:
        target: (N, H, W)
        source: (N, H, W)
        warped_source: (N, H, W)
        dvf: (N, H, W, 2)
        save_result_dir:
        epoch:
        fps:

    Returns:

    """
    # loop over time frames
    png_buffer = []
    for fr in range(dvf.shape[0]):
        dvf_fr = dvf[fr, :, :, :]  # (H, W, 2)
        target_fr = target[fr, :, :]  # (H, W)
        source_fr = source[fr, :, :]  # (H, W)
        warped_source_fr = warped_source[fr, :, :]  # (H, W)

        fig_save_path = os.path.join(save_result_dir, f'frame_{fr}.png')
        plot_results(target_fr, source_fr, warped_source_fr, dvf_fr, save_path=fig_save_path, dpi=dpi)

        # read back the PNG to save a GIF animation
        png_buffer += [imageio.imread(fig_save_path)]
        os.remove(fig_save_path)
    imageio.mimwrite(os.path.join(save_result_dir, f'epoch_{epoch}.gif'), png_buffer, fps=fps)


def plot_results_t1t2(target,
                      target_original,
                      source,
                      warped_source,
                      target_pred,
                      dvf,
                      dvf_gt,
                      save_path=None, title_font_size=20, show_fig=False, dpi=100):
    """Plot all motion related results in a single figure
    dvf is expected to be in number of pixels"""


    ## set up the figure
    fig = plt.figure(figsize=(30, 18))
    title_pad = 10

    ax = plt.subplot(2, 4, 1)
    plt.imshow(target, cmap='gray')
    plt.axis('off')
    ax.set_title('Target (T1-syn)', fontsize=title_font_size, pad=title_pad)

    ax = plt.subplot(2, 4, 2)
    plt.imshow(target_original, cmap='gray')
    plt.axis('off')
    ax.set_title('Target original (T1)', fontsize=title_font_size, pad=title_pad)

    # calculate the error before and after reg
    error_before = target - target_original
    error_after = target - target_pred

    # error before
    ax = plt.subplot(2, 4, 3)
    plt.imshow(error_before, vmin=-2, vmax=2, cmap='gray')  # assuming images were normalised to [0, 1]
    # plt.imshow(error_before, cmap='gray')
    plt.axis('off')
    ax.set_title('Error before', fontsize=title_font_size, pad=title_pad)

    # error after
    ax = plt.subplot(2, 4, 4)
    plt.imshow(error_after, vmin=-2, vmax=2, cmap='gray')  # assuming images were normalised to [0, 1]
    # plt.imshow(error_after, cmap='gray')
    plt.axis('off')
    ax.set_title('Error after', fontsize=title_font_size, pad=title_pad)

    ax = plt.subplot(2, 4, 5)
    plt.imshow(target_pred, cmap='gray')
    plt.axis('off')
    ax.set_title('Target predict', fontsize=title_font_size, pad=title_pad)

    ax = plt.subplot(2, 4, 6)
    plt.imshow(warped_source, cmap='gray')
    plt.axis('off')
    ax.set_title('Warped source', fontsize=title_font_size, pad=title_pad)

    # deformed grid ground truth
    ax = plt.subplot(2, 4, 7)
    bg_img = np.zeros_like(target)
    show_warped_grid(ax, dvf_gt, bg_img, interval=3, title="$\phi_{GT}$", fontsize=title_font_size)

    ax = plt.subplot(2, 4, 8)
    show_warped_grid(ax, dvf, bg_img, interval=3, title="$\phi_{pred}$", fontsize=title_font_size)

    # # hsv flow
    # convert flow into HSV flow with white background
    hsv_flow = flow_to_hsv(dvf, max_mag=0.15, white_bg=True)
    # todo: DVF shape change to be applied
    # ax = plt.subplot(2, 4, 7)
    # plt.imshow(hsv_flow)
    # plt.axis('off')
    # ax.set_title('HSV', fontsize=title_font_size, pad=title_pad)

    # # quiver, or "Displacement Vector Field" (DVF)
    # todo: DVF shape change to (2, H, W) to be applied
    # ax = plt.subplot(2, 4, 6)
    # interval = 3  # interval between points on the grid
    # background = source
    # quiver_flow = np.zeros_like(dvf)
    # quiver_flow[:, :, 0] = dvf[:, :, 0]
    # quiver_flow[:, :, 1] = dvf[:, :, 1]
    # mesh_x, mesh_y = np.meshgrid(range(0, dvf.shape[1] - 1, interval),
    #                              range(0, dvf.shape[0] - 1, interval))
    # plt.imshow(background[:, :], cmap='gray')
    # plt.quiver(mesh_x, mesh_y,
    #            quiver_flow[mesh_y, mesh_x, 1], quiver_flow[mesh_y, mesh_x, 0],
    #            angles='xy', scale_units='xy', scale=1, color='g')
    # plt.axis('off')
    # ax.set_title('DVF', fontsize=title_font_size, pad=title_pad)

    # # det Jac
    # ax = plt.subplot(2, 4, 8)
    # todo: DVF shape change to (2, H, W) to be applied
    # jac_det, mean_grad_detJ, negative_detJ = computeJacobianDeterminant2D(dvf)
    # spec = [(0, (0.0, 0.0, 0.0)), (0.000000001, (0.0, 0.2, 0.2)),
    #         (0.12499999999, (0.0, 1.0, 1.0)), (0.125, (0.0, 0.0, 1.0)),
    #         (0.25, (1.0, 1.0, 1.0)), (0.375, (1.0, 0.0, 0.0)),
    #         (1, (0.94509803921568625, 0.41176470588235292, 0.07450980392156863))]
    # cmap = matplotlib.colors.LinearSegmentedColormap.from_list('detjac', spec)
    # plt.imshow(jac_det, vmin=-1, vmax=7, cmap=cmap)
    # plt.axis('off')
    # ax.set_title('Jacobian (Grad: {0:.2f}, Neg: {1:.2f}%)'.format(mean_grad_detJ, negative_detJ * 100),
    #              fontsize=int(title_font_size*0.9), pad=title_pad)
    # # split and extend this axe for the colorbar
    # from mpl_toolkits.axes_grid1 import make_axes_locatable
    # divider = make_axes_locatable(ax)
    # cax1 = divider.append_axes("right", size="5%", pad=0.05)
    # cb = plt.colorbar(cax=cax1)
    # cb.ax.tick_params(labelsize=20)

    # adjust subplot placements and spacing
    plt.subplots_adjust(left=0.0001, right=0.99, top=0.9, bottom=0.1, wspace=0.001, hspace=0.1)

    # saving
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight', dpi=dpi)

    if show_fig:
        plt.show()
    plt.close()



def save_val_visual_results(target,
                            target_original,
                            source,
                            warped_source,
                            target_pred,
                            dvf,
                            dvf_gt,
                            save_result_dir, epoch, dpi=50):
    """
    Randomly save 1 slice from N-slice stack (not a sequence)
    Args:
        target: (N, 1, H, W)
        source: (N, 1, H, W)
        warped_source: (N, 1, H, W)
        dvf: (N, 2, H, W)
        save_result_dir:
        epoch:
        dpi: image resolution
    """
    z = random.randint(0, target.shape[0]-1)

    target = target[z, 0, ...]  # (H, W)
    source = source[z, 0, ...]  # (H, W)
    warped_source = warped_source[z, 0, ...]  # (H, W)
    target_original = target_original[z, 0, ...]  # (H, W)
    target_pred = target_pred[z, 0, ...]  # (H, W)

    dvf = dvf[z, ...]  # (2, H, W)
    dvf_gt = dvf_gt[z, ...] # (2, H, W)

    fig_save_path = os.path.join(save_result_dir, f'epoch{epoch}_slice_{z}.png')
    plot_results_t1t2(target,
                      target_original,
                      source,
                      warped_source,
                      target_pred,
                      dvf,
                      dvf_gt,
                      save_path=fig_save_path, dpi=dpi)

