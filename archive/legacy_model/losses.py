"""
Loss functions
"""

import torch.nn as nn

from model.loss.reg_loss import l2reg, bending_energy
from model.loss.sim_loss import MILossGaussian
from utils.image import bbox_from_mask


def loss_fn(data_dict, params):
    """
    Construct loss function

    Args:
        data_dict: (dict) dictionary containing data
            {
                "target": (Tensor, shape (N, 1, *sizes)) target image
                "warped_source": (Tensor, shape (N, 1, *sizes)) deformed source image
                "dvf_pred": (Tensor, shape (N, dim, *sizes)) DVF predicted
                ...
            }
        params: (object) parameters from params_mirtk.json

    Returns:
        loss: (scalar) loss value
        losses: (dict) dictionary of individual losses (weighted)
    """
    sim_losses = {"MSE": nn.MSELoss(),
                  "NMI": MILossGaussian(num_bins_tar=params.mi_num_bins,
                                        num_bins_src=params.mi_num_bins,
                                        sigma_tar=params.mi_sigma,
                                        sigma_src=params.mi_sigma)}

    reg_losses = {"diffusion": l2reg,
                  "be": bending_energy}

    tar = data_dict["target"]
    warped_src = data_dict["warped_source"]

    # (optional) only evaluate similarity loss within ROI bounding box
    if params.loss_roi:
        bbox, _ = bbox_from_mask(data_dict["roi_mask"].squeeze(1).numpy())
        for i in range(params.dim):
            tar = tar.narrow(i+2, int(bbox[i][0]), int(bbox[i][1] - bbox[i][0]))
            warped_src = warped_src.narrow(i+2, int(bbox[i][0]), int(bbox[i][1] - bbox[i][0]))

    sim_loss = sim_losses[params.sim_loss](tar, warped_src) * params.sim_weight
    reg_loss = reg_losses[params.reg_loss](data_dict["dvf_pred"]) * params.reg_weight

    return {"loss": sim_loss + reg_loss,
            params.sim_loss: sim_loss,
            params.reg_loss: reg_loss}


