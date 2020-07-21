import os
import time

import cv2
from cheapfake.FAN.core.FaceDetector import FaceDetector


class DLibDetector(FaceDetector):
    """
    Implements a face detection class that detects faces using DLib.
    """

    def __init__(self, device, path_to_detector, verbose=False):
        """
        Instantiates a face detector that uses DLib.

        Parameters
        ----------
        device :
        path_to_detector : str
        verbose : {False, True}, bool, optional

        Returns
        -------
        None
        """
        print("WARNING: DLib is not a very good face detector")
