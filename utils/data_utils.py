"""Utility functions for general data processing"""
import numpy as np
import nibabel as nib
import os


def split_volume(image_name, output_name):
    """ Split an image volume into a number of slices. """
    nim = nib.load(image_name)
    Z = nim.header['dim'][3]
    affine = nim.affine
    image = nim.get_data()

    for z in range(Z):
        image_slice = image[:, :, z]
        image_slice = np.expand_dims(image_slice, axis=2)
        affine2 = np.copy(affine)
        # modify Image-to-World affine transformation matrix to compensate lost of z-dimension
        affine2[:3, 3] += z * affine2[:3, 2]
        nim2 = nib.Nifti1Image(image_slice, affine2)
        nib.save(nim2, '{0}{1:02d}.nii.gz'.format(output_name, z))


def split_volume_idmat(image_name, output_name):
    """ Split an image volume into slices with identity matrix as I2W transformation
    This is to by-pass the issue of not accumulating displacement in z-direction
     in MIRTK's `convert-dof` function
     """
    nim = nib.load(image_name)
    Z = nim.header['dim'][3]
    image = nim.get_data()

    for z in range(Z):
        image_slice = image[:, :, z]
        image_slice = np.expand_dims(image_slice, axis=2)
        nim2 = nib.Nifti1Image(image_slice, np.eye(4))
        nib.save(nim2, '{0}{1:02d}.nii.gz'.format(output_name, z))


def split_sequence(image_name, output_name):
    """ Split an image sequence into a number of time frames. """
    nim = nib.load(image_name)
    T = nim.header['dim'][4]
    affine = nim.affine
    image = nim.get_data()

    for t in range(T):
        image_fr = image[:, :, :, t]
        nim2 = nib.Nifti1Image(image_fr, affine)
        nib.save(nim2, '{0}{1:02d}.nii.gz'.format(output_name, t))

