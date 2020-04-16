import numpy as np
import cv2
from scipy.spatial.distance import directed_hausdorff
import SimpleITK as sitk
import matplotlib
import matplotlib.pyplot as plt
import os

from utils.image import bbox_from_mask, bbox_crop


def calculate_metrics(metric_data, metric_groups):
    """
    Wrapper function for calculating all metrics.
    Args:
        metric_data: (dict) data used for calculation of metrics
        metric_groups: (list of strings) name of metric groups

    Returns:
        metrics_results: (dict) {metric_name: metric_value}
    """
    metric_results = {}

    # keys must match those in metric_groups and params.metric_groups
    metric_group_fns = {'dvf_metrics': calculate_dvf_metrics,
                        'image_metrics': calculate_image_metrics}

    for metric_group in metric_groups:
        metric_results.update(metric_group_fns[metric_group](metric_data))

    return metric_results


"""
Transformation metrics
"""
def calculate_dvf_metrics(metric_data):
    """
    Calculate DVF-related metrics.
    If roi_mask is given, the DVF is masked and only evaluate in the bounding box of the mask.

    Args:
        metric_data: (dict) dvf shape (N, dim, *dims), roi mask shape (N, 1, *(dims))

    Returns:
        metric_results: (dict)
    """

    # unpack metric data, keys must match metric_data input
    dvf_pred = metric_data['dvf_pred']
    dvf_gt = metric_data['dvf_gt']

    if 'roi_mask' in metric_data.keys():
        roi_mask = metric_data['roi_mask']  # (N, 1, *(dims))

        # find brian mask bbox mask
        mask_bbox, mask_bbox_mask = bbox_from_mask(roi_mask[:, 0, ...])

        # mask by roi mask(N, dim, *dims) * (N, 1, *dims) = (N, dim, *dims)
        dvf_pred *= roi_mask
        dvf_gt *= roi_mask

        # crop out DVF within the roi mask bounding box (N, dim, *dims_cropped)
        dvf_pred = bbox_crop(dvf_pred, mask_bbox)
        dvf_gt = bbox_crop(dvf_gt, mask_bbox)

    ## Jacobian metrics
    mean_grad_detJ, mean_negative_detJ = detJac_stack(dvf_pred.transpose(0, 2, 3, 1),  # (N, H, W, 2)
                                                      rescaleFlow=False)

    # metric keys must match params specification
    return {'aee': calculate_aee(dvf_pred, dvf_gt),
            'rmse_dvf': calculate_rmse_dvf(dvf_pred, dvf_gt),
            'mean_grad_detJ': mean_grad_detJ,
            'mean_negative_detJ': mean_negative_detJ}


def calculate_aee(x, y):
    """
    Average End point error (mean over point-wise L2 norm)
    Input DVF shape: (N, dim, *(sizes))
    """
    return np.sqrt(((x - y) ** 2).sum(axis=1)).mean()


def calculate_rmse_dvf(x, y):
    """
    RMSE of DVF (square root over mean of sum squared)
    Input DVF shape: (N, dim, *(sizes))
    """
    return np.sqrt(((x - y) ** 2).sum(axis=1).mean())


def computeJacobianDeterminant2D(dvf, rescaleFlow=False, save_path=None):
    """
    Calculate determinant of Jacobian of the transformation

    Args:
        dvf: (ndarry, shape HxHx2)
        rescaleFlow: scale the deformation field by image size/2,
                        if True [-1, 1] coordinate system is assumed for flow
        save_path:

    Returns:
        jac_det: the determinant of Jacobian for each point, same dimension as input
        mean_grad_jac_det: mean of the value of det(J)
        below_zero_jac_det: ration (0~1) of points that have negative det(J)

    """
    if rescaleFlow:
        # scale the deformation field to convert coordinate system from [-1, 1] range to pixel number
        dvf = dvf * np.asarray((dvf.shape[0] / 2., dvf.shape[1] / 2.))

    # calculate det Jac using SimpleITK
    flow_img = sitk.GetImageFromArray(dvf, isVector=True)
    jac_det_filt = sitk.DisplacementFieldJacobianDeterminant(flow_img)
    jac_det = sitk.GetArrayFromImage(jac_det_filt)

    mean_grad_detJ = np.mean(np.abs(np.gradient(jac_det)))
    negative_detJ = np.sum((jac_det < 0)) / (jac_det.shape[0] * jac_det.shape[1])  # ratio of negative det(Jac)

    # render and save det(Jac) image
    if save_path is not None:
        spec = [(0, (0.0, 0.0, 0.0)), (0.000000001, (0.0, 0.2, 0.2)),
                (0.12499999999, (0.0, 1.0, 1.0)), (0.125, (0.0, 0.0, 1.0)),
                (0.25, (1.0, 1.0, 1.0)), (0.375, (1.0, 0.0, 0.0)),
                (1, (0.94509803921568625, 0.41176470588235292, 0.07450980392156863))]
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('detjac', spec)
        save_path = os.path.join(save_path, 'detJ.png')
        plt.imsave(save_path, jac_det, vmin=-1, vmax=7,
                   cmap=cmap)  # vmin=-2., vmax=2., cmap='RdBu_r') # cmap=plt.cm.gray)
        # plt.imshow(jac_det, vmin=-1, vmax=7, cmap=cmap)
        # plt.show()
    return jac_det, mean_grad_detJ, negative_detJ


def detJac_stack(flow_stack, rescaleFlow=False):
    """
    Calculate determinant of Jacobian for a stack of 2D displacement fields.

    Args:
        flow_stack: (ndarray shape N, H, W, 2) 2D stack of disp/flow fields
        rescaleFlow: rescale flow to reverse coordinate normalisation,
                        i.e. if True DVF input is assumed in [-1,1] coordinate

    Returns:
        mean_grad_jac_det, mean_negative_detJ: averaged over slices in the stack

    """
    mean_grad_detJ_buffer = []
    mean_negatvie_detJ_buffer = []

    for slice_idx in range(flow_stack.shape[0]):
        flow = flow_stack[slice_idx, :, :, :]
        _, mean_grad_detJ, negative_detJ = computeJacobianDeterminant2D(flow, rescaleFlow=rescaleFlow)
        mean_grad_detJ_buffer += [mean_grad_detJ]
        mean_negatvie_detJ_buffer += [negative_detJ]

    mean_grad_detJ_mean = np.mean(mean_grad_detJ_buffer)
    negative_detJ_mean = np.mean(mean_negatvie_detJ_buffer)

    return mean_grad_detJ_mean, negative_detJ_mean


""""""

"""
Image intensity  metrics
"""
def calculate_image_metrics(metric_data):
    # unpack metric data, keys must match metric_data input
    img = metric_data['target']
    img_pred = metric_data['target_pred']  # (N, 1, *(dims))

    if 'roi_mask' in metric_data.keys():
        roi_mask = metric_data['roi_mask']

        # find brian mask bbox mask
        mask_bbox, mask_bbox_mask = bbox_from_mask(roi_mask[:, 0, ...])

        # crop out image within the roi mask bounding box
        img = bbox_crop(img, mask_bbox)
        img_pred = bbox_crop(img_pred, mask_bbox)
    return {'rmse': calculate_rmse(img, img_pred)}


def calculate_rmse(x, y):
    """Standard RMSE formula, square root over mean
    (https://wikimedia.org/api/rest_v1/media/math/render/svg/6d689379d70cd119e3a9ed3c8ae306cafa5d516d)
    """
    return np.sqrt(((x - y) ** 2).mean())


""""""

"""
Segmentation metrics
"""


def contour_distances_2d(image1, image2, dx=1):
    """
    Calculate contour distances between binary masks.
    The region of interest must be encoded by 1

    Args:
        image1: 2D binary mask 1
        image2: 2D binary mask 2
        dx: physical size of a pixel (e.g. 1.8 (mm) for UKBB)

    Returns:
        mean_hausdorff_dist: Hausdorff distance (mean if input are 2D stacks) in pixels
    """

    # Retrieve contours as list of the coordinates of the points for each contour
    # convert to contiguous array and data type uint8 as required by the cv2 function
    image1 = np.ascontiguousarray(image1, dtype=np.uint8)
    image2 = np.ascontiguousarray(image2, dtype=np.uint8)

    # extract contour points and stack the contour points into (N, 2)
    contours1, _ = cv2.findContours(image1.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour1_pts = np.array(contours1[0])[:, 0, :]
    for i in range(1, len(contours1)):
        cont1_arr = np.array(contours1[i])[:, 0, :]
        contour1_pts = np.vstack([contour1_pts, cont1_arr])

    contours2, _ = cv2.findContours(image2.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour2_pts = np.array(contours2[0])[:, 0, :]
    for i in range(1, len(contours2)):
        cont2_arr = np.array(contours2[i])[:, 0, :]
        contour2_pts = np.vstack([contour2_pts, cont2_arr])

    # distance matrix between two point sets
    dist_matrix = np.zeros((contour1_pts.shape[0], contour2_pts.shape[0]))
    for i in range(contour1_pts.shape[0]):
        for j in range(contour2_pts.shape[0]):
            dist_matrix[i, j] = np.linalg.norm(contour1_pts[i, :] - contour2_pts[j, :])

    # symmetrical mean contour distance
    mean_contour_dist = 0.5 * (np.mean(np.min(dist_matrix, axis=0)) + np.mean(np.min(dist_matrix, axis=1)))

    # calculate Hausdorff distance using the accelerated method
    # (doesn't really save computation since pair-wise distance matrix has to be computed for MCD anyways)
    hausdorff_dist = directed_hausdorff(contour1_pts, contour2_pts)[0]

    return mean_contour_dist * dx, hausdorff_dist * dx


def contour_distances_stack(stack1, stack2, label_class, dx=1):
    """
    Measure mean contour distance metrics between two 2D stacks

    Args:
        stack1: stack of binary 2D images, shape format (W, H, N)
        stack2: stack of binary 2D images, shape format (W, H, N)
        label_class: class of which to calculate distance
        dx: physical size of a pixel (e.g. 1.8 (mm) for UKBB)

    Return:
        mean_mcd: mean contour distance averaged over non-empty slices
        mean_hd: Hausdorff distance averaged over non-empty slices
    """

    # assert the two stacks has the same number of slices
    assert stack1.shape[-1] == stack2.shape[-1], 'Contour dist error: two stacks has different number of slices'

    # mask by class
    stack1 = (stack1 == label_class).astype('uint8')
    stack2 = (stack2 == label_class).astype('uint8')

    mcd_buffer = []
    hd_buffer = []
    for slice_idx in range(stack1.shape[-1]):
        # ignore empty masks
        if np.sum(stack1[:, :, slice_idx]) > 0 and np.sum(stack2[:, :, slice_idx]) > 0:
            slice1 = stack1[:, :, slice_idx]
            slice2 = stack2[:, :, slice_idx]
            mcd, hd = contour_distances_2d(slice1, slice2, dx=dx)

            mcd_buffer += [mcd]
            hd_buffer += [hd]

    return np.mean(mcd_buffer), np.mean(hd_buffer)


def categorical_dice_stack(mask1, mask2, label_class=0):
    """
    todo: this evaluation function should ignore slices that has empty masks at either ED or ES frame
    Dice scores of a specified class between two masks or two 2D "stacks" of masks
    If the inputs are stacks of multiple 2D slices, dice scores are averaged
    (classes are encoded but by label class number not one-hot )

    Args:
        mask1: N label masks, numpy array shaped (H, W, N)
        mask2: N label masks, numpy array shaped (H, W, N)
        label_class: the class over which to calculate dice scores

    Returns:
        mean_dice: the mean dice score, scalar

    """
    mask1_pos = (mask1 == label_class).astype(np.float32)
    mask2_pos = (mask2 == label_class).astype(np.float32)

    pos1and2 = np.sum(mask1_pos * mask2_pos, axis=(0, 1))
    pos1or2 = np.sum(mask1_pos + mask2_pos, axis=(0, 1))

    # numerical stability is needed because of possible empty masks
    dice = np.mean(2 * pos1and2 / (pos1or2 + 1e-3))

    return dice


def categorical_dice_volume(mask1, mask2, label_class=0):
    """
    Dice score of a specified class between two volumes of label masks.
    (classes are encoded but by label class number not one-hot )
    Note: stacks of 2D slices are considered volumes.

    Args:
        mask1: N label masks, numpy array shaped (H, W, N)
        mask2: N label masks, numpy array shaped (H, W, N)
        label_class: the class over which to calculate dice scores

    Returns:
        volume_dice
    """
    mask1_pos = (mask1 == label_class).astype(np.float32)
    mask2_pos = (mask2 == label_class).astype(np.float32)
    dice = 2 * np.sum(mask1_pos * mask2_pos) / (np.sum(mask1_pos) + np.sum(mask2_pos))

    return dice


""""""
