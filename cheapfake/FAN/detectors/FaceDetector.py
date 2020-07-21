"""
Implements the FaceDetector class that represents an abstract Face Detector.
"""

import os
import sys
import time
import logging

import cv2
import glob
import torch
import numpy as np
from tqdm import tqdm
from skimage import io


class FaceDetector(object):
    """
    Abstract class representing a Face Detector.
    """

    def __init__(self, device, verbose=True):
        """
        Instantiates a FaceDetector object.

        Parameters
        ----------
        device :
        verbose : {True, False}, bool, optional

        Returns
        -------
        None
        """
        self.device = device
        self.verbose = verbose

        if verbose:
            if "cpu" in device:
                logger = logging.getLogger(__name__)
                logger.warning(
                    "Face Detection is running on CPU. This may cause slowdown."
                )

        if "cpu" not in device and "cuda" not in device:
            if verbose:
                logger.error(
                    "The only acceptable values are: {cpu, cuda} but got {} instead.".format(
                        device
                    )
                )
            raise ValueError

    def detect_face_from_image(self, input):
        """
        Main function for detecting faces from images. 

        The input can be either a torch.Tensor instance, representing the image, or a path to the image, depending on the implementation from the sub-class that inherits this class.

        This function must be implemented by all sub-classes that inherit this class.

        Parameters
        ----------
        input : torch.Tensor instance OR str
            The input used to perform the face detection.
        """
        raise NotImplementedError

    def detect_face_from_directory(
        self, path, extensions=["jpg, png"], show_progress=True
    ):
        """
        Main function for detecting faces from a directory containing images.

        The input to this is a path to a directory containing a directory of images, with the extensions provided in ``extensions``. Note that this function searchs for ALL image files with the extensions.

        Parameters
        ----------
        path : str
        extensions : list (of str), optional
        
        Notes
        -----
        This function calls the ``detect_face_from_image`` function, which is implemented by the sub-class.
        
        See Also
        --------
        detect_face_from_image : related function
        """
        if self.verbose:
            logger = logging.getLogger(__name__)

        assert (
            len(extensions) > 0
        ), "This function expects at least one extension but none were provided."

        if self.verbose:
            logger.info("Constructing list of images...")

        files = list()
        for extension in extensions:
            files.extend(glob.glob(path, "*.{}".format(extension)))

        if self.verbose:
            logger.info(
                "Finished constructing list of images. {} images found...".format(
                    len(files)
                )
            )
        predictions = {}
        for image_path in tqdm(files, disable=(not show_progress_bar)):
            if self.verbose:
                logger.info("Running face detection on image {}...".format(image_path))
            predictions[image_path] = self.detect_from_image(image_path)

        if self.verbose:
            logger.info(
                "Finished detecting images. Successfully found faces in {}/{} images...".format(
                    len(predictions), len(files)
                )
            )

        return predictions

    @staticmethod
    def to_ndarray(input, rgb_flag=True):
        """
        Converts the input to a numpy.array instance.

        Note that ``input`` can either be a torch.Tensor instance or a string representing the image path. 

        Parameters
        ----------
        input : torch.Tensor instance or str
        rgb_flag : {True, False}, bool, optional

        Returns
        -------
        ndarray : numpy.array instance
        """
        if isinstance(input, str)):
            if not rgb_flag:
                return cv2.imread(input)
            else:
                return io.imread(input)
        elif torch.is_tensor(input):
            if not rgb_flag:
                return input.cpu().numpy()[..., ::-1].copy()
            else:
                return input.cpu().numpy()
        elif isinstance(input, np.ndarray):
            if not rgb_flag:
                return input[..., ::-1].copy()
            else:
                return input
        else:
            raise TypeError