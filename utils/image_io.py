"""
Utility functions to handle image io.
Huaqi Qiu, Jan 2019.
"""

import nibabel as nib
import imageio
import os
import numpy as np
from utils.image import upsample_image

def load_nifti(path, data_type="float32", nim=False):
    xnim = nib.load(path)
    x = xnim.get_data().astype(data_type)
    if nim:
        return x, xnim
    else:
        return x


def save_nifti(x, path, nim=None, verbose=False):
    """
    Save a numpy array to a nifti file

    Args:
        x: (numpy.ndarray) data
        path: destination path
        nim: Nibabel nim object, to provide the nifti header
        verbose: (boolean)

    Returns:
        N/A
    """
    if nim is not None:
        nim_save = nib.Nifti1Image(x, nim.affine, nim.header)
    else:
        nim_save = nib.Nifti1Image(x, np.eye(4))
    nib.save(nim_save, path)

    if verbose:
        print("Nifti saved to: {}".format(path))


def save_gif(images, path, fps=20):
    """
    Save numpy array to gif

    Args:
        images: numpy array of shape (H, W, Frames) for grayscale images
                or (H, W, ch, Frames) for colored images
        path: save destination file path
        fps: frame rate of the gif
    """
    images = images.astype(np.uint8)
    frame_list = [upsample_image(images[..., fr], 300) for fr in range(images.shape[-1])]
    imageio.mimwrite(path, frame_list, fps=fps)


def save_png(images, path_dir):
    """
    Save numpy array to a series of PNG files

    Args:
        images: numpy array of shape (H, W, Frames) for grayscale images
                or (H, W, ch, Frames) for colored images
        path_dir: save destination directory path
    """
    images = images.astype(np.uint8)
    for fr in range(images.shape[-1]):
        image = upsample_image(images[..., fr], 300)
        imageio.imwrite(os.path.join(path_dir, 'frame_{}.png'.format(fr)), image)

