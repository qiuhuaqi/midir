"""Utility functions for Pytorch dataloaders"""
import numpy as np
import torch


class CenterCrop(object):
    """
    Central crop typical numpy array loaded from nifti files
    Expected input shape: (N, H, W)
    Expected output shape: (N, H', W')
    """
    def __init__(self, output_size=192):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2, "output_size is not an input of 1 or 2 elements"
            self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[-2:]

        # pad to output size with zeros if image is smaller than crop size
        if h < self.output_size[0]:
            h_before = (self.output_size[0] - h) // 2
            h_after = self.output_size[0] - h - h_before
            image = np.pad(image, ((0 ,0), (h_before, h_after), (0, 0)), mode='constant')

        if w < self.output_size[1]:
            w_before = (self.output_size[1] - w) // 2
            w_after = self.output_size[1] - w - w_before
            image = np.pad(image, ((0, 0), (0, 0), (w_before, w_after)), mode='constant')

        # then continue with normal cropping
        h, w = image.shape[-2:]  # update shape numbers after padding
        h_start = h//2 - self.output_size[0]//2
        w_start = w//2 - self.output_size[1]//2

        h_end = h_start + self.output_size[0]
        w_end = w_start + self.output_size[1]

        cropped_image = image[..., h_start:h_end, w_start:w_end]

        assert cropped_image.shape[-2:] == self.output_size
        return cropped_image


class Normalise(object):
    """
    Normalise image of any shape.
    (image - mean) / std
    If mode = 'max, use mean = 0, std = max(image) to normalise to [0~1]
    """
    def __init__(self, mode='max'):
        self.mode = mode

    def __call__(self, image):
        if self.mode == 'max':
            mean = 0.0
            std = np.max(np.abs(image))
        else:
            mean = np.mean(np.abs(image))
            std = np.std(np.abs(image))
        return (image - mean) / std


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, image):
        return torch.from_numpy(image)


class AffineTransform(object):
    pass
