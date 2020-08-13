import numpy as np
import torch
import cv2
from scipy.spatial.distance import directed_hausdorff
import SimpleITK as sitk
from utils.image import bbox_from_mask, bbox_crop
from model.transformations import spatial_transform


# Calculate metrics #

def calculate_metrics(metric_data, metric_groups, return_tensor=False):
    """
    Wrapper function for calculating all metrics in Numpy
    Args:
        metric_data: (dict) data used for calculation of metrics, could be Tensor or Numpy Array
        metric_groups: (list of strings) name of metric groups
        return_tensor: (bool) return Torch Tensor if True

    Returns:
        metrics_results: (dict) {metric_name: metric_value}
    """

    # cast Tensor to Numpy Array if needed
    for k, x in metric_data.items():
        if isinstance(x, torch.Tensor):
            metric_data[k] = x.cpu().numpy()

    # keys must match metric_groups and params.metric_groups
    # (using groups to share pre-scripts)
    metric_group_fns = {'dvf_metrics': calculate_dvf_metrics,
                        'image_metrics': calculate_image_metrics,
                        'seg_metrics': calculate_seg_metrics}
    metric_results = dict()
    for group in metric_groups:
        metric_results.update(metric_group_fns[group](metric_data))

    # cast results back to Tensor if needed
    if return_tensor:
        for k, x in metric_results.items():
            metric_results[k] = torch.tensor(x)

    return metric_results


def calculate_dvf_metrics(metric_data):
    """
    Calculate DVF-related metrics.
    If roi_mask is given, the DVF is masked and only evaluate in the bounding box of the mask.

    Args:
        metric_data: (dict) DVF shapes (N, dim, *sizes), roi mask shape (N, 1, *sizes)

    Returns:
        metric_results: (dict)
    """
    # new object to avoid changing data in metric_data
    dvf_pred = metric_data['dvf_pred']
    if 'dvf_gt' in metric_data.keys():
        dvf_gt = metric_data['dvf_gt']

    # mask the DVF with roi mask if given
    if 'roi_mask' in metric_data.keys():
        roi_mask = metric_data['roi_mask']  # (N, 1, *(dims))

        # find roi mask bbox mask
        mask_bbox, mask_bbox_mask = bbox_from_mask(roi_mask[:, 0, ...])

        # mask and bbox crop dvf gt and pred by roi_mask
        dvf_pred = dvf_pred * roi_mask
        dvf_pred = bbox_crop(dvf_pred, mask_bbox)

        if 'dvf_gt' in metric_data.keys():
            dvf_gt = dvf_gt * roi_mask
            dvf_gt = bbox_crop(dvf_gt, mask_bbox)

    # Jacobian metrics
    folding_ratio, mag_det_jac_det = calculate_jacobian_metrics(dvf_pred)

    dvf_metric_results = dict()
    dvf_metric_results.update({'folding_ratio': folding_ratio,
                               'mean_negative_detJ': mag_det_jac_det})

    if 'dvf_gt' in metric_data.keys():
        dvf_metric_results.update({'aee': calculate_aee(dvf_pred, dvf_gt),
                                   'rmse_dvf': calculate_rmse_dvf(dvf_pred, dvf_gt)})
    return dvf_metric_results


def calculate_image_metrics(metric_data):
    # unpack metric data, keys must match metric_data input
    img = metric_data['target']
    img_pred = metric_data['target_pred']  # (N, 1, *sizes)

    # crop out image by the roi mask bounding box if given
    if 'roi_mask' in metric_data.keys():
        roi_mask = metric_data['roi_mask']
        mask_bbox, mask_bbox_mask = bbox_from_mask(roi_mask[:, 0, ...])
        img = bbox_crop(img, mask_bbox)
        img_pred = bbox_crop(img_pred, mask_bbox)

    return {'rmse': calculate_rmse(img, img_pred)}


def calculate_seg_metrics(metric_data):
    """TODO: 2D (could use a dummy loop)"""
    seg_metric_results = {}

    # dice for cortical segmentation
    cor_seg_gt = metric_data['target_cor_seg']
    cor_seg_pred = metric_data['target_cor_seg_pred']

    for label_cls in np.unique(cor_seg_gt):
        if label_cls == 0:
            continue  # exclude background
        seg_metric_results[f'cor_dice_class_{label_cls}'] = calculate_dice_volume(cor_seg_gt, cor_seg_pred,
                                                                                  label_class=label_cls)

    # dice for sub-cortical segmentation
    subcor_seg_gt = metric_data['target_subcor_seg']
    subcor_seg_pred = metric_data['target_subcor_seg_pred']

    for label_cls in np.unique(subcor_seg_gt):
        if label_cls == 0:
            continue  # exclude background
        seg_metric_results[f'subcor_dice_class_{label_cls}'] = calculate_dice_volume(subcor_seg_gt, subcor_seg_pred,
                                                                                     label_class=label_cls)
    # mean dice
    seg_metric_results['mean_dice'] = np.mean([dice for k, dice in seg_metric_results.items()])
    return seg_metric_results


def calculate_aee(x, y):
    """
    Average End point Error (AEE, mean over point-wise L2 norm)
    Input DVF shape: (N, dim, *(sizes))
    """
    return np.sqrt(((x - y) ** 2).sum(axis=1)).mean()


def calculate_rmse_dvf(x, y):
    """
    RMSE of DVF (square root over mean of sum squared)
    Input DVF shape: (N, dim, *(sizes))
    """
    return np.sqrt(((x - y) ** 2).sum(axis=1).mean())


def calculate_rmse(x, y):
    """Standard RMSE formula, square root over mean
    (https://wikimedia.org/api/rest_v1/media/math/render/svg/6d689379d70cd119e3a9ed3c8ae306cafa5d516d)
    """
    return np.sqrt(((x - y) ** 2).mean())


def calculate_jacobian_metrics(dvf):
    """
    Calculate Jacobian related regularity metrics.

    Args:
        dvf: (numpy.ndarray, shape (N, dim, *sizes) Displacement field

    Returns:
        folding_ratio: (scalar) Folding ratio (ratio of Jacobian determinant < 0 points)
        mag_grad_jac_det: (scalar) Mean magnitude of the spatial gradient of Jacobian determinant
    """
    folding_ratio = []
    mag_grad_jac_det = []
    for n in range(dvf.shape[0]):
        dvf_n = np.moveaxis(dvf[n, ...], 0, -1)  # (*sizes, dim)
        jac_det_n = calculate_jacobian_det(dvf_n)

        folding_ratio += [(jac_det_n < 0).sum() / np.prod(jac_det_n.shape)]
        mag_grad_jac_det += [np.abs(np.gradient(jac_det_n)).mean()]
    return np.mean(folding_ratio), np.mean(mag_grad_jac_det)


def calculate_jacobian_det(dvf):
    """
    Calculate Jacobian determinant of displacement field of one image/volume (2D/3D)

    Args:
        dvf: (numpy.ndarray, shape (*sizes, dim) Displacement field

    Returns:
        jac_det: (numpy.adarray, shape (*sizes) Point-wise Jacobian determinant
    """
    dvf_img = sitk.GetImageFromArray(dvf, isVector=True)
    jac_det_img = sitk.DisplacementFieldJacobianDeterminant(dvf_img)
    jac_det = sitk.GetArrayFromImage(jac_det_img)
    return jac_det


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


def calculate_dice_stack(mask1, mask2, label_class=0):
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


def calculate_dice_volume(mask1, mask2, label_class=0):
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
