"""
Python file that creates the necessary objects for training cheapfake.
"""

import os
import time

import torch
import librosa
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset


class DeepFakeDataset(Dataset):
    """Python implementation of the DeepFakeDataset class.

    """

    def __init__(self, root_prefix, frame_transform=None, audio_transform=None):
        """Instantiates a new DeepFakeDataset object.

        Parameters
        ----------
        root_prefix : str
        frame_transform : callable, optional
            A callable function that is used to transform extracted frames from videos, by default None. If None, then no transform is applied to the extracted frames. If a non-callable function is passed as input, then the default transform (i.e. no transform) is used.
        audio_transform : callable, optional
            A callable function that is used to transform extracted audio from videos, by default None. If None, then no transform is applied to the extracted audio. If a non-callable function is passed as input, then the default transform (i.e. no transform) is used.
        
        """
        self.root_prefix = root_prefix
        self.frame_transform = frame_transform
        self.audio_transform = audio_transform

        # Compute the list of absolute paths here.

    def __getitem__(self, index):
        """Extracts the frames and audio from the 
        """
