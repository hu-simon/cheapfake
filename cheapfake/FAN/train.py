"""
Python script to train the FAN models (2D and 3D).
"""

import os
import time

import cv2
import enum
import torch
import skimage
import numpy as np

import cheapfake.utils.fanutils as fanutils
from cheapfake.FAN.models import FaceAlignmentNetwork
from cheapfake.FAN.resnetdepth import ResNetWithDepth

options = __import__("config")
os.environ["CUDA_VISIBLE_DEVICES"] = options.gpu


class LandmarkType(enum.Enum):
    """
    Child of enumeration class that represents the type of landmarks to be detected.
    """

    _2D = 2
    _2D_HALF = 2.5
    _3D = 3

    def __init__(self):
        pass


class NetworkSize(enum.Enum):
    """
    Child of enumeration class that represents the size of the network.
    """

    SMALL = 2
    MEDIUM = 3
    LARGE = 4

    def __new__(cls, value):
        member = object.__new__(cls)
        member._value_ = value

        return member

    def __int__(self):
        return self.value


class FaceAlignment:
    """
    Implements the FaceAlignment network with pre-existing weights.
    """

    def __init__(
        self,
        landmarks_type,
        network_size=NetworkSize.LARGE,
        device=torch.device("cpu"),
        flip_input=False,
        face_detector="dlib",
        verbose=False,
    ):
    """
    Instantiates a FaceAlignment object.

    Parameters
    ----------
    landmarks_type : 
    network_size : 
    device : 
    flip_input : {False, True}, bool, optional
    face_detector : {"dlib", "sfd"}, str, optional
    verbose : {False, True}, bool, optional

    Returns
    -------
    None
    """
    self.landmarks_type = landmarks_type
    self.network_size = int(network_size)
    self.device = device
    self.flip_input = flip_input
    self.face_detector = face_detector
    self.verbose = verbose

    
