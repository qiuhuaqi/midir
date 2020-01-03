import torch
import numpy as np
import torch.nn as nn

from model.mutual_info.window_func import cubic_bspline_numpy, rect_window_numpy
from model.mutual_info.window_func import cubic_bspline_torch, rect_window_torch
from utils.image_utils import normalise_numpy, normalise_torch

def joint_hist_vanilla_numpy(img_ref, img_tar, num_bins_ref=64, num_bins_tar=64):

    # flatten the images
    img_ref_vec = img_ref.ravel().astype(float)
    img_tar_vec = img_tar.ravel().astype(float)

    # normalise intensity range of both images to [0, 1]
    img_ref_vec = normalise_numpy(img_ref_vec, 0.0, 1.0)
    img_tar_vec = normalise_numpy(img_tar_vec, 0.0, 1.0)

    # set bin edges
    bin_width_ref = 1.0 / num_bins_ref
    bin_width_tar = 1.0 / num_bins_tar
    bins_ref = np.arange(num_bins_ref) * bin_width_ref
    bins_tar = np.arange(num_bins_tar) * bin_width_tar

    # histogram 2D
    H, bins_ref, bins_tar = np.histogram2d(img_ref_vec, img_tar_vec, bins=(bins_ref, bins_tar))

    return H, bins_ref, bins_tar





def joint_hist_parzen_numpy(img_ref, img_tar, window="cubic_bspline", num_bins_ref=64, num_bins_tar=64,
                            c_ref=1.0, c_tar=1.0):
    """Joint histogram written in numpy, noramlise image intensity range to [0,1]"""

    # flatten the images and cast to float32
    img_ref_vec = img_ref.ravel().astype(float)
    img_tar_vec = img_tar.ravel().astype(float)

    # normalise intensity range of both images to [0, 1]
    img_ref_vec = normalise_numpy(img_ref_vec, 0.0, 1.0)
    img_tar_vec = normalise_numpy(img_tar_vec, 0.0, 1.0)

    # set bin edges
    bin_width_ref = 1.0 / num_bins_ref
    bin_width_tar = 1.0 / num_bins_tar
    bins_ref = np.arange(num_bins_ref) * bin_width_ref
    bins_tar = np.arange(num_bins_tar) * bin_width_tar

    # window width parameter (eps)
    # to satisfy the partition of unity constraint, eps can be no larger than bin_width
    # and can only be reduced by integer factor to increase support,
    # cubic B-spline function has support of 4*eps, hence maximum number of bins affected is 4/c
    # maximum number of bins affected is 4/c)
    eps_ref = c_ref * bin_width_ref
    eps_tar = c_tar * bin_width_tar

    # parse window function from string argument
    if window == "cubic_bspline":
        window_func = cubic_bspline_numpy
    elif window == "rectangle":
        window_func = rect_window_numpy
    else:
        raise Exception("Window function not recognised.")

    # calculate Parzen window function response
    W_ref = window_func((bins_ref[:, np.newaxis] - img_ref_vec) / eps_ref)
    W_tar = window_func((bins_tar[:, np.newaxis] - img_tar_vec) / eps_tar)

    # calculate joint histogram
    H = W_ref.dot(W_tar.transpose()) / (eps_ref * eps_tar)
    return H, bins_ref, bins_tar



class JointHistParzenTorch(nn.Module):
    # todo: this abstraction seems redundant
    #  (only used to make sure the bin edges are sent to the correct device)
    def __init__(self, num_bins_ref=64, num_bins_tar=64, c_ref=1.0, c_tar=1.0):
        super().__init__()

        self.c_ref = c_ref
        self.c_tar = c_tar

        # set bin edges
        self.bin_width_ref = 1.0 / num_bins_ref
        self.bin_width_tar = 1.0 / num_bins_tar
        self.bins_ref = torch.arange(num_bins_ref, dtype=torch.float) * self.bin_width_ref
        self.bins_tar = torch.arange(num_bins_tar, dtype=torch.float) * self.bin_width_tar
        self.bins_ref = nn.Parameter(self.bins_ref, requires_grad=False)
        self.bins_tar = nn.Parameter(self.bins_tar, requires_grad=False)


    def forward(self, img_ref, img_tar, window="cubic_bspline"):
        """
        Joint histogram written in Pytorch, image intensity will be normalised to range [0,1]

        Args:
            img_ref: (Tensor) reference image (fixed image), shape (T-1, 1, H, W)
            img_tar: (Tensor) target image (moved image), shape (T-1, 1, H, W)
            window: (String) name of window function

        Returns:
            H: (Tensor) joint histogram shape (N, #bins_ref, #bins_tar)
        """

        # cast images to float32
        img_ref = img_ref.float()
        img_tar = img_tar.float()

        # normalise intensity range of both images to [0, 1] and cast to float 32
        img_ref = normalise_torch(img_ref, 0.0, 1.0)
        img_tar = normalise_torch(img_tar, 0.0, 1.0)

        # flatten the images to shape (T-1, 1, H*W) and cast to float32
        size_ref = img_ref.size()
        size_tar = img_tar.size()
        img_ref = img_ref.view(size_ref[0], size_ref[1], -1)
        img_tar = img_tar.view(size_tar[0], size_tar[1], -1)

        # window width parameter (eps)
        # to satisfy the partition of unity constraint, eps can be no larger than bin_width
        # and can only be reduced by integer factor to increase support,
        # cubic B-spline function has support of 4*eps, hence maximum number of bins affected is 4/c
        eps_ref = self.c_ref * self.bin_width_ref
        eps_tar = self.c_tar * self.bin_width_tar

        # parse window function from string argument
        if window == "cubic_bspline":
            window_func = cubic_bspline_torch
        elif window == "rectangle":
            window_func = rect_window_torch
        else:
            raise Exception("Window function not recognised.")

        # calculate Parzen window function response
        # bin edges of size (#bins, 1) and images of size (T-1, 1, H*W) should be broadcastable
        W_ref = window_func((self.bins_ref.unsqueeze(1) - img_ref) / eps_ref) # (N, #bins, H*W)
        W_tar = window_func((self.bins_tar.unsqueeze(1) - img_tar) / eps_tar) # (N, #bins, H*W)

        # calculate joint histogram (using batch matrix multiplication)
        H = W_ref.bmm(W_tar.transpose(1, 2)) / (eps_ref * eps_tar)  # (N, #bins, #bins)
        return H
