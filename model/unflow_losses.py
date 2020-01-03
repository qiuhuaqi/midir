"""
UnFlow losses with
- forward and backward consistency
- occlusion
- Census transform
- 2nd order smoothness
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.submodules import resample_transform

from utils import image_utils
# Values for occlusion decision threshold
ALPHA = 0.01
BETA = 0.5

# ALPHA = params.ALPHA
# BETA = params.BETA

##############################################################################################
# --- UnFlow loss function --- #
##############################################################################################
def combine_unflow_loss(img1, img2, flows_fw, flows_bw, params):
    """Compute UnFlow loss"""

    # parse loss configs
    # loss_list = ['ternary', 'occ', 'smooth_2nd', 'fb']
    loss_list = params.loss_list

    # layer_weights = params.layer_weights
    # layer_weights = [12.7, 4.35, 3.9, 3.4, 1.1]
    layer_weights = params.layer_weights

    # initialise combined loss dict
    combined_losses = dict()
    for loss_name in loss_list:
        combined_losses[loss_name] = 0.0

    combined_loss = 0.0

    for layer_num in range(len(layer_weights)):
        # downsample images to the resolution of the layer
        img1_layer = F.interpolate(img1, scale_factor=(1/2)**layer_num, mode='bilinear', align_corners=True)
        img2_layer = F.interpolate(img2, scale_factor=(1/2)**layer_num, mode='bilinear', align_corners=True)

        flow_fw, flow_bw = flows_fw[layer_num], flows_bw[layer_num]
        losses = compute_unflow_losses(img1_layer, img2_layer, flow_fw, flow_bw, occ=params.occ)

        layer_weight = layer_weights[layer_num]

        # weighted sum of different losses of this layer
        layer_loss = 0.0
        for loss_name in loss_list:
            weight_name = loss_name + '_weight'
            if params.dict.get(weight_name):
                layer_loss += params.dict[weight_name] * losses[loss_name]
                # keeping track of each loss summed across layers
                combined_losses[loss_name] += layer_weight * params.dict[weight_name] * losses[loss_name]

        combined_loss += layer_weight * layer_loss

    return combined_loss, combined_losses


def compute_unflow_losses(img1, img2, flow_fw, flow_bw, box_distance=1, occ=True):
    """Compute the variety of losses of UnFlow for one layer"""

    # -- occlusion masks
    flow_bw_resampled = resample_transform(flow_bw, flow_fw)
    flow_diff_fw = flow_fw + flow_bw_resampled
    flow_diff_sq_fw = l2_norm_sq(flow_diff_fw)

    flow_fw_resampled = resample_transform(flow_fw, flow_bw)
    flow_diff_bw = flow_bw + flow_fw_resampled
    flow_diff_sq_bw = l2_norm_sq(flow_diff_bw)

    if occ:
        # mask logic: [Not occluded NOR out] = 1, [occluded or out] = 0
        # occlusion mask forward
        occ_thresh_fw = ALPHA * (l2_norm_sq(flow_fw) + l2_norm_sq(flow_bw_resampled)) + BETA
        fb_occ_mask_fw = flow_diff_sq_fw < occ_thresh_fw
        # out_mask_fw = create_outgoing_mask(flow_fw)
        # occ_mask_fw = fb_occ_mask_fw * out_mask_fw
        occ_mask_fw = fb_occ_mask_fw
        occ_mask_fw = occ_mask_fw.float().cuda()

        # occlusion mask backward
        occ_thresh_bw = ALPHA * (l2_norm_sq(flow_bw) + l2_norm_sq(flow_fw_resampled)) + BETA
        fb_occ_mask_bw = flow_diff_sq_bw < occ_thresh_bw
        # out_mask_bw = create_outgoing_mask(flow_bw)
        # occ_mask_bw = fb_occ_mask_bw * out_mask_bw
        occ_mask_bw = fb_occ_mask_bw
        occ_mask_bw = occ_mask_bw.float().cuda()
    else:
        occ_mask_fw = torch.ones(img1.size()).cuda()
        occ_mask_bw = torch.ones(img1.size()).cuda()
        assert occ_mask_fw.size() == occ_mask_bw.size()



    # -- warping images
    img2_warped = resample_transform(img2, flow_fw)
    img1_warped = resample_transform(img1, flow_bw)

    im_diff_fw = img1 - img2_warped
    im_diff_bw = img2 - img1_warped


    # --- calculate the losses
    losses = {}

    # data loss
    # photometric loss as data loss (vanilla data loss)
    losses['photo'] = (charbonnier_loss(im_diff_fw, occ_mask_fw) +
                       charbonnier_loss(im_diff_bw, occ_mask_bw))

    # ternary loss as data loss
    losses['ternary'] = (ternary_loss(img1, img2_warped, occ_mask_fw, box_distance=box_distance) +
                         ternary_loss(img2, img1_warped, occ_mask_bw, box_distance=box_distance))

    losses['occ'] = (charbonnier_loss(occ_mask_fw) + charbonnier_loss(occ_mask_bw))

    # -- smoothness loss
    losses['smooth_2nd'] = (second_order_loss(flow_fw) +
                            second_order_loss(flow_bw))

    # -- consistency loss
    losses['fb'] = (charbonnier_loss(flow_diff_fw, occ_mask_fw) +
                    charbonnier_loss(flow_diff_bw, occ_mask_bw))

    # -- unused losses
    # losses['sym'] = (charbonnier_loss(occ_fw - disocc_bw) +
    #                  charbonnier_loss(occ_bw - disocc_fw))

    # losses['grad'] = (gradient_loss(im1, im2_warped, mask_fw) +
    #                   gradient_loss(im2, im1_warped, mask_bw))

    # losses['smooth_1st'] = (smoothness_loss(flow_fw) +
    #                         smoothness_loss(flow_bw))


    return losses


# --- Building blocks of the UnFlow loss functions --- #

def ternary_loss(img1, img2_warped, occ_mask, box_distance=1):
    """Compute the ternary loss in place for the vanilla photometric distance"""
    box_size = 2 * box_distance + 1

    def _ternary_transform(img):
        """Perform ternary Census transform of an image
        Input image shape (N, Ch, H, W)"""

        out_channels = box_size * box_size
        weights = torch.from_numpy(np.eye(out_channels).reshape((out_channels, 1, box_size, box_size)))
        weights = weights.float().cuda()
        box_filter_output = F.conv2d(img, weights, stride=1, padding=box_distance)

        census_img = box_filter_output - img
        census_img_norm = census_img / torch.sqrt(0.81 + census_img ** 2)
        # output size should be (N, out_channels, H, W)
        return census_img_norm

    def _hamming_distance(t1, t2):
        dist = (t1 - t2) ** 2
        dist_norm  = dist / (0.1 + dist)
        dist_sum = torch.sum(dist_norm, dim=1, keepdim=True)
        # size should be (N, 1, H, W)
        return dist_sum

    t1 = _ternary_transform(img1)
    t2 = _ternary_transform(img2_warped)
    census_dist = _hamming_distance(t1, t2)  # (N, 1, H, W)

    # handles the border by filtering out the results calculated using padded zeros
    paddings = [[box_distance, box_distance], [box_distance, box_distance]]
    inner_mask = create_inner_mask(census_dist, paddings).cuda()
    assert occ_mask.size() == inner_mask.size(), 'Occlusion mask has different size as the inner mask'
    mask = occ_mask * inner_mask  # occ_mask AND inner_mask

    return charbonnier_loss(census_dist, mask)


def _second_order_deltas(flow):
    """
    Compute discrete 2nd order derivatives

    Args:
        flow: (tensor) flow shape (N, 2, H, W)

    Returns:
        delta_u: u direction
        delta_v:
        mask:

    """
    # use h-w coordinates
    filter_h = [[0, 1, 0],
                [0, -2, 0],
                [0, 1, 0]]
    filter_w = [[0, 0, 0],
                [1, -2, 1],
                [0, 0, 0]]
    filter_diag1 = [[1, 0, 0],
                    [0, -2, 0],
                    [0, 0, 1]]
    filter_diag2 = [[0, 0, 1],
                    [0, -2, 0],
                    [1, 0, 0]]

    # formulate the derivatives as weights in a filter bank
    weights = np.array([filter_h, filter_w, filter_diag1, filter_diag2])
    weights = torch.from_numpy(weights).unsqueeze(1).float().cuda()

    # apply the derivatives using conv2d
    flow_h = flow[:, 0, :, :].unsqueeze(1)
    flow_w = flow[:, 1, :, :].unsqueeze(1)
    delta_h = F.conv2d(flow_h,  weights, stride=1, padding=1)
    delta_w = F.conv2d(flow_w,  weights, stride=1, padding=1)
    # shape of deltas (N, 4, H, W)
    assert delta_h.size()[1] == 4 and delta_w.size()[1] == 4, '2nd order derivatives shape incorrect'

    # create mask to filter out the padded area
    # all mask shape (N, 1, H, W) same as the flow, final mask is (N, 4, H, W)
    inner_mask_h = create_inner_mask(flow, paddings=[[1, 1], [0, 0]])
    inner_mask_w = create_inner_mask(flow, paddings=[[0, 0], [1, 1]])
    inner_mask_diag = create_inner_mask(flow, paddings=[[1, 1], [1, 1]])
    inner_masks = torch.cat([inner_mask_h, inner_mask_w, inner_mask_diag, inner_mask_diag],  dim=1)

    return delta_h, delta_w, inner_masks


def second_order_loss(flow):
    """Compute total variation loss of 2nd order derivatives of flow"""
    delta_h, delta_w, inner_masks = _second_order_deltas(flow)
    inner_masks = inner_masks.cuda()
    loss_h = charbonnier_loss(delta_h, inner_masks)
    loss_w = charbonnier_loss(delta_w, inner_masks)
    return loss_h + loss_w


def charbonnier_loss(x, mask=None, truncate=None, alpha=0.45, beta=1.0, epsilon=0.001):
    """Compute Charbonnier loss of an input tensor x,
    normalised by total pixel numbers
    Positions on the mask with value 0 is ignored
    Final loss sum averaged over total number of pixels
    Ip shape (N, 1, H, W) for both x and mask, Op scalar tensor"""

    loss = ( (x * beta) ** 2 + epsilon ** 2 ) ** alpha
    if mask is not None:
        loss = loss * mask
    # todo: truncate?
    return torch.sum(loss) / np.prod(x.size())


def create_outgoing_mask(flow):
    """Computes a mask that is ones at all positions where the flow
    would carry a pixel over the image boundary.
    Mask logic: inside = 1, outside = 0
    """
    N, _, H, W = flow.size()
    grid_h, grid_w = torch.meshgrid([torch.linspace(-1, 1, H), torch.linspace(-1, 1, W)])

    grid_h = grid_h.cuda()
    grid_w = grid_w.cuda()


    # add flow by broadcasting (H, W) to (N, H, W)
    grid_h = grid_h + flow[:, 0, :, :]
    grid_w = grid_w + flow[:, 1, :, :]


    # check whether overflow
    inside_h = (grid_h >= -1) * (grid_h <= 1)
    inside_w = (grid_w >= -1) * (grid_w <= 1)

    out_mask = inside_h * inside_w   # (N, H, W)
    return out_mask.unsqueeze(1)  # (N, 1, H, W)


def create_inner_mask(x, paddings):
    """
    Create a mask to filter out a padded area
    Args:
        x: the tensor to apply the mask
        paddings: size padded to x,
                    nested list [[h_pad_before, h_pad_after], [w_pad_before, w_pad_after]]

    Returns:
        mask: shape (N, 1, H, W)
    """
    N, _, H, W = x.size()
    inner_height = H - paddings[0][0] - paddings[0][1]
    inner_width = W - paddings[1][0] - paddings[1][1]

    inner_mask = np.ones((inner_height, inner_width))
    inner_mask = np.pad(inner_mask, paddings, mode='constant')
    inner_mask = torch.from_numpy(inner_mask)

    inner_mask_4d = inner_mask.expand([N, 1, H, W])
    return inner_mask_4d.requires_grad_(requires_grad=False).float()


def l2_norm_sq(x):
    """Calculate the square of L2 norm of an array along dimension 1
    Ip array (N, 2, H, W), Op array (N, 1, H, W)"""
    return torch.sum(x**2, dim=1, keepdim=True)



