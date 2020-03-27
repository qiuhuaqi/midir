"""Image/Array utils"""
import numpy as np
from PIL import Image


# -- image normalisation to 0 - 255
def normalise_numpy(x, norm_min=0.0, norm_max=255.0):
    return float(norm_max - norm_min) * (x - np.min(x)) / (np.max(x) - np.min(x))


def normalise_torch(x, nmin=0.0, nmax=255.0):
    return (nmax - nmin) * (x - x.min()) / (x.max() - x.min())

def upsample_image(image, size):
    return np.array(Image.fromarray(image).resize((size, size)))

def bbox_from_mask(mask, pad_ratio=0.2):
    """
    Find a bounding box indices of a mask (with positive > 0)
    (largest bounding box of N masks)
    The indices follows Python indexing rule and can be directly used for slicing

    Args:
        mask: (ndarray NxHxW)
        pad_ratio: ratio of distance between the edge of mask bounding box to image boundary to pad

    Return:
        None: if structure in mask is too small

        bbox: list [[dim_i_min, dim_i_max], [dim_j_min, dim_j_max], ...] otherwise
        bbox_mask: (ndarray NxHxW) binary mask which is 1 inside the bbox, 0 outside

    """
    mask_indices = np.nonzero(mask > 0)
    bbox_i = (mask_indices[1].min(), mask_indices[1].max()+1)
    bbox_j = (mask_indices[2].min(), mask_indices[2].max()+1)
    # bbox_k = (mask_indices[3].min(), mask_indices[3].max()+1)


    # pad 20% of minimum distance to the image boundaries (10% each side)
    if pad_ratio > 1:
        print("Invalid padding value (>1), set to 1")
        pad_ratio = 1

    bbox_pad_i = pad_ratio * min(bbox_i[0], mask.shape[1] - bbox_i[1])
    bbox_pad_j = pad_ratio * min(bbox_j[0], mask.shape[2] - bbox_j[1])
    # bbox_pad_k = 0.2 * min(bbox_k[0], image.shape[3] - bbox_k[1])

    bbox_i = (bbox_i[0] - int(bbox_pad_i/2), bbox_i[1] + int(bbox_pad_i/2))
    bbox_j = (bbox_j[0] - int(bbox_pad_j/2), bbox_j[1] + int(bbox_pad_j/2))
    # bbox_k = (bbox_k[0] - int(bbox_pad_k/2), bbox_k[1] + int(bbox_pad_k/2))
    bbox = [bbox_i, bbox_j]

    # bbox mask
    bbox_mask = np.zeros(mask.shape, dtype=np.float32)
    bbox_mask[:, bbox_i[0]:bbox_i[1], bbox_j[0]:bbox_j[1]] = 1.0

    return bbox, bbox_mask


