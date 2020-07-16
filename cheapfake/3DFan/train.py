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
from cheapfake.3DFan.models import FaceAlignmentNetwork
from cheapfake.3DFan.resnetdepth import ResNetWithDepth

class LandmarkType(enum.Enum):
    """
    Child of enumeration class that represents the type of landmarks to be detected.
    """

    _2D = 2
    _2D_HALF = 2.5
    _3D = 3

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

    def __int__(self)
        return self.value

