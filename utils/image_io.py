import nibabel as nib
import imageio
import os
import numpy as np
from PIL import Image


def load_nifti(path, data_type=np.float32, nim=False):
    xnim = nib.load(path)
    x = np.asanyarray(xnim.dataobj).astype(data_type)
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


def upsample_image(image, size):
    return np.array(Image.fromarray(image).resize((size, size)))


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


def split_volume(image_name, output_name):
    """ Split an image volume into a number of slices. """
    # image saved in shape (H, W, N)
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


def split_volume_idmat(image_path, output_prefix, data_type=np.float32):
    """ Split an image volume into slices with identity matrix as I2W transformation
    This is to by-pass the issue of not accumulating displacement in z-direction
     in MIRTK's `convert-dof` function
     """
    # image data saved in shape (H, W, N)
    nim = nib.load(image_path)
    Z = nim.header['dim'][3]
    image = np.asanyarray(nim.dataobj).astype(data_type)

    for z in range(Z):
        image_slice = image[:, :, z]
        image_slice = np.expand_dims(image_slice, axis=2)
        nim2 = nib.Nifti1Image(image_slice, np.eye(4))
        nib.save(nim2, '{0}{1:02d}.nii.gz'.format(output_prefix, z))


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