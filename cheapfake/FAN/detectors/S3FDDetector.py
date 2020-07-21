import os
import time

import cv2
import torch
from torch.utils.model_zoo import load_url
from cheapfake.FAN.core.S3FDNet import S3FDNet
from cheapfake.FAN.core.FaceDetector import FaceDetector

# Import the utilities file here.


class S3FDDetector(FaceDetector):
    """
    Implements a face detection class using the S3FD model.
    """

    def __init__(self, device, path_to_detector=None, verbose=False):
        """
        Instantiates a face detection object that uses the S3FD model for face detection.
        
        Parameters
        ----------
        device : 
        path_to_detector : str, optional
        verbose : {False, True}, bool, optional

        Returns
        -------
        None
        """
        super(S3FDDetector, self).__init__(device, verbose)

        if model_weights is None:
            model_weights = load_url(1)
        else:
            model_weights = torch.load(path_to_detector)

        self.detector = S3FDNet().to(device)
        self.detector.load_state_dict(model_weights)
        self.detector.eval()

    def detect_face_from_image(self, input):
        """
        Main function for detecting faces from images. 

        The input can be either a torch.Tensor instance, representing the image, or a path to the image, depending on the implementation from the sub-class that inherits this class.

        This function must be implemented by all sub-classes that inherit this class.

        Parameters
        ----------
        input : torch.Tensor instance OR str
            The input used to perform the face detection.
        
        Returns
        -------
        bbox_list : list (of tuples)
            List containing the coordinates of the bounding box, surrounding a face.
        """
        image = self.to_ndarray(input)

        bbox_list = s3fdutils.detect(self.detector, image, device=self.device)
        idx_keep = s3fdutils.non_max_suppression(bbox_list, 0.3)
        bbox_list = bbox_list[idx_keep, :]
        bbox_list = [bbox for bbox in bbox_list if bbox[-1] > 0.5]

        return bbox_list

    @property
    def reference_scale(self):
        return 195

    @property
    def reference_shift_x(self):
        return 0

    @property
    def reference_shift_y(self):
        return 0
