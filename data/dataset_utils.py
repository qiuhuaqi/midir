"""Utility functions for Pytorch dataloaders"""
import numpy as np
import torch


class CenterCrop(object):
    """
    Central crop numpy array
    Input shape: (N, H, W)
    Output shape: (N, H', W')
    """
    def __init__(self, output_size=192):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2, "'output_size' can only be a single integer or a pair of integers"
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
    Normalise image of any shape to range
    (image - mean) / std
    mode:
        'minmax': normalise the image using its min and max to range [0, 1]
        'fixed': normalise the image by a fixed ration determined by the input arguments (preferred for image registration)
        'meanstd': normalise to mean=0, std=1
    """

    def __init__(self, mode='minmax',
                 min_in=0.0, max_in=255.0,
                 min_out=0.0, max_out=1.0):
        self.mode = mode
        self.min_in = min_in,
        self.max_in = max_in
        self.min_out = min_out
        self.max_out = max_out

        if self.mode == 'fixed':
            self.norm_ratio = (max_out - min_out) * (max_in - min_in)

    def __call__(self, image):
        if self.mode == 'minmax':
            min_in = image.min()
            max_in = image.max()
            image_norm = (image - min_in) * (self.max_out - self.min_out) / (max_in - min_in)

        elif self.mode == 'fixed':
            image_norm = image * self.norm_ratio

        elif self.mode == 'meanstd':
            image_norm = (image - image.mean()) / image.std()

        else:
            raise ValueError("Normalisation mode not recogonised.")
        return image_norm


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, image):
        return torch.from_numpy(image)


class AffineTransform(object):
    pass
