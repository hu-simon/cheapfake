"""
Python file containing commonly used transformations, written as classes to follow with good PyTorch style.
"""

import os
import time
import random

import cv2
import torch
import numpy as np

"""
Can probabliy just combine the two functions and do the flipping on the for loop.
"""


class BatchRescale(object):
    """Rescales a collection of images in a given sample, to a specified size.

    """

    def __init__(
        self,
        output_size,
        return_tensor=True,
        channel_first=True,
        interpolation=cv2.INTER_LANCZOS4,
    ):
        """Instantiates a new BatchResize object.
        
        Parameters
        ----------
        output_size : int or tuple
            The output size of the image (height and width). If an integer is passed as input, then the output size of the image is determined by scaling the height and width of the original image. 
        return_tensor : {True, False}, bool, optional
            If True then the output is returned as a torch.Tensor instance, by default True. Otherwise the output is returned as a numpy.ndarray instance.
        channel_first : {True, False}, bool, optional
            If True then the input and output are assumed to be of the shape (sample, channel, height, width). Otherwise, the input and output are assumed to be of the shape (sample, height, width, channel). 
        interpolation : int, optional
            The interpolation scheme to be used, by default cv2.INTER_LANCZOS4. This option is passed onto ``cv2.resize()`` as a parameter.

        """
        assert isinstance(output_size, (int, tuple))
        assert isinstance(return_tensor, bool)
        assert isinstance(channel_first, bool)
        assert isinstance(interpolation, int)

        self.output_size = output_size
        self.return_tensor = return_tensor
        self.channel_first = channel_first
        self.interpolation = interpolation

    def __call__(self, frames):
        """Resizes a collection of images to the desired output size.
        
        Parameters
        ----------
        frames : numpy.ndarray or torch.Tensor instance
            Numpy array or Torch tensor containing the collection of frames to be resized.
        
        Returns
        -------
        resized_frames : numpy.ndarray or torch.Tensor instance
            Numpy array or Torch tensor containing the resized frames.

        """
        if self.channel_first:
            frames = np.einsum("ijkl->iklj", frames)

        if isinstance(self.output_size, int):
            (new_height, new_width) = frames[0].shape[:2]
            new_height = int(new_height / self.output_size)
            new_width = int(new_width / self.output_size)
        else:
            (new_height, new_width) = self.output_size[:2]

        resized_frames = np.empty((frames.shape[0], new_height, new_width, 3))
        for k, frame in enumerate(frames):
            resized_frames[k] = cv2.resize(
                frame, dsize=(new_width, new_height), interpolation=self.interpolation
            )

        if self.channel_first:
            resized_frames = np.einsum("ijkl->iljk", resized_frames)
        if self.return_tensor:
            resized_frames = torch.from_numpy(resized_frames)

        return resized_frames


class BatchRescaleFlip(object):
    """Resizes and stochastically flips (left-right) a collection of images in a given sample, to a specified size.
    
    """

    def __init__(
        self,
        output_size,
        return_tensor=True,
        channel_first=True,
        interpolation=cv2.INTER_LANCZOS4,
        p_flip=0.3,
    ):
        """Instantiates a new BatchRescaleFlip object.
        
        Parameters
        ----------
        output_size : int or tuple
            The output size of the image (height and width). If an integer is passed as input, then the output size of the image is determined by scaling the height and width of the original image.
        return_tensor : {True, False}, bool, optional
            If True then the output is returned as a torch.Tensor instance, by default True. Otherwise the output is returned as a numpy.ndarray instance.
        channel_first : {True, False}, bool, optional
            If True then the input and output are assumed to be of the shape (sample, channel, height, width). Otherwise the input and output are assumed to be of the shape (sample, height, width, channel).
        interpolation : int
            The interpolation scheme to be used, by default cv2.INTER_LANCZOS4. This option is passed onto ``cv2.resize()`` as a parameter.    
        p_flip : float, optional
            The probability of flipping the image along the vertical axis (i.e. left-right flipping), by default 0.3.

        Returns
        -------
        resized_frames : numpy.ndarray or torch.Tensor instance
            Numpy array or Torch tensor containing the resized images, which are potentially flipped along the vertical axis (i.e. left-right flipping).

        """
        assert isinstance(output_size, (int, tuple))
        assert isinstance(return_tensor, bool)
        assert isinstance(channel_first, bool)
        assert isinstance(interpolation, int)
        assert isinstance(p_flip, float)
        assert p_flip <= 1 and p_flip > 0, "Probability must be in (0, 1]."

        self.output_size = output_size
        self.return_tensor = return_tensor
        self.channel_first = channel_first
        self.interpolation = interpolation
        self.p_flip = p_flip

    def __call__(self, frames):
        """Resizes a collection of images in a sample to the desired output size, with a probability of flipping the images along the vertical axis (i.e. flip the image left-right).
        
        Parameters
        ----------
        frames : numpy.ndarray or torch.Tensor instance
            Numpy array or Torch tensor containing the collection of frames to be resized.

        Returns
        -------
        resized_frames : numpy.ndarray or torch.Tensor instance
            Numpy array or Torch tensor containing the resized frames.

        """
        resized_frames = BatchReshape(
            output_size=self.output_size,
            return_tensor=self.return_tensor,
            channel_first=self.channel_first,
            interpolation=self.interpolation,
        )

        # Flip the resized_frames based on the probability. TODO

        return resized_frames


class BatchNormalizeRGB(object):
    """Normalizes the color dimension, on a collection of images, to have zero mean and unit standard deviation.

    """

    def __init__(self, return_tensor=True, channel_first=True):
        """Instantiates a new NormalizeRGB object.

        Parameters
        ----------
        return_tensor : {True, False}, bool, optional
            If True then the output is returned as a torch.Tensor instance, by default True. Otherwise the output is returned as a numpy.ndarray instance.
        channel_first : {True, False}, bool, optional
            If True then the input and output are assumed to be of the shape (sample, channel, height, width). Otherwise the input and output are assumed to be of the shape (sample, height, width, channel).
    
        """
        assert isinstance(return_tensor, bool)
        assert isinstance(channel_first, bool)

        self.return_tensor = return_tensor
        self.channel_first = channel_first

    def __call__(self, frames):
        """Normalizes the collection of frames to have zero mean and unit standard deviation.

        The normalization is done for each channel.

        Parameters
        ----------
        frames : numpy.ndarray or torch.Tensor instance
            Numpy array or Torch tensor containing the frames.
        
        Returns
        -------
        normalized_frames : numpy.ndarray or torch.Tensor instance
            Numpy array or Torch tensor containing the frames with the channel dimensions normalized to have zero mean and unit standard deviation.
        
        """
        assert isinstance(
            frames, (np.ndarray, torch.Tensor)
        ), "Array must be a numpy.ndarray or torch.Tensor instance."
        assert len(frames) > 0, "Array must contain at least one frame."

        if self.channel_first:
            axis_order = (0, 2, 3)
        else:
            axis_order = (0, 1, 2)

        normalized_frames = (frames - frames.mean(axis=axis_order, keepdims=True)) / (
            frames.std(axis=axis_order, keepdims=True)
        )

        return normalized_frames


class BatchResizeNormalize(object):
    """Resizes and normalizes the color channels, for a collection of images, to have zero mean and unit standard deviation.

    """

    def __init__(self, output_size, return_tensor=True, channel_first=True):
        """Instantiates a new BatchResizeNormalize object.

        Parameters
        ----------
        output_size : int or tuple
            The output size of the image (height and width). If an integer is passed as input, then the output size of the image is determined by scaling the height and width of the original image.
        return_tensor : {True, False}, bool, optional
            If True then the output is returned as a torch.Tensor instance, by default True. Otherwise the output is returned as a numpy.ndarray instance.
        channel_first : {True, False}, bool, optional
            If True then the input and output are assumed to be of the shape (sample, channel, height, width). Otherwise the input and output are assumed to be of the shape (sample, height, width, channel).
    
        """
        assert isinstance(output_size, (int, tuple))
        assert isinstance(return_tensor, bool)
        assert isinstance(channel_first, bool)

        self.output_size = output_size
        self.return_tensor = return_tensor
        self.channel_first = channel_first

        self.rescale_transform = BatchResize(
            output_size=self.output_size,
            return_tensor=self.return_tensor,
            channel_first=self.channel_first,
        )
        self.normalize_transform = BatchNormalizeRGB(
            return_tensor=self.return_tensor, channel_first=self.channel_first
        )

    def __call__(self, frames):
        """Resizes and normalizes the color channels, for a collection of images, to have zero mean and unit standard deviation.

        Parameters
        ----------
        frames : numpy.ndarray or torch.Tensor instance
            Numpy array or Torch tensor containing the frames to be resized and normalized.
        
        Returns
        -------
        transformed_frames : numpy.ndarray or torch.Tensor instance
            Numpy array or Torch tensor containing the resized and color normalized frames, normalized to have zero mean and unit standard deviation.

        """
        resized_frames = self.rescale_transform(frames)
        transformed_frames = self.normalize_transform(resized_frames)

        return transformed_frames

