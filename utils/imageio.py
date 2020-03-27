"""
Utility functions to handle image IO.
Huaqi Qiu, Jan 2019.
"""

import nibabel as nib
import imageio
import os
import numpy as np

from utils.image import upsample_image


def save_nifti(ndarray, path, nim, verbose=False):
    """
    Save a numpy array to a nifti file

    Args:
        ndarray: numpy array
        path: destination path
        nim: nibabel's nim object, to provide the nifti header

    Returns:
        N/A
    """
    nim_save = nib.Nifti1Image(ndarray, nim.affine, nim.header)
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

