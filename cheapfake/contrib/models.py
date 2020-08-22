"""
Need to rethink the entire forward pass. The idea is as follows.

The resized frames should be sent into FAN, with the expected shape. This returns facial landmarks (if a face is detected). Using the resizing information, we can rescale the predicted facial landmarks to get an overall estimate of where the lips are located in the full-size image. Then, the lip region is extracted from the full-size image (with some buffer room) and gets resized to 64 x 128 which is the size expected by LipNet. The predicted facial features and the embeddings learning by the LipNet are sent into the MLP, along with the audio embeddings learned by the residual network.
"""

import os
import time
import enum

import cv2
import torch
import numpy as np
import face_alignment
import torch.nn as nn
import torch.nn.functional as F

import cheapfake.lipnet.models as lipnet
import cheapfake.contrib.dataset as dataset
import cheapfake.contrib.video_processor as video_processor

lipnet_options = __import__("lipnet_config")
fan_options = __import__("face_alignment_config")


class NetworkType(enum.Enum):
    FAN = 1
    LIPNET = 2
    RESNETAUDIO = 3


class FeaturesEncoder(nn.Module):
    """Implements a feature encoder to create embeddings that can fit into a multi-layer perceptron.

    """

    def __init__(self, network_type=NetworkType.FAN):
        """Instantiates a new FeatureEncoder object.

        Parameters
        ----------
        network_type : cheapfake.contrib.models.NetworkType instance, optional
            The type of network to instantiate, by default NetworkType.FAN. The supported options are NetworkType.FAN, NetworkType.LIPNET, and NetworkType.RESNETAUDIO.

        """
        super(FeaturesEncoder, self).__init__()

        assert network_type.value in (1, 2, 3), "Network architecture not supported."
        
        self.network_type = network_type
        
        # FAN.
        # For this model we are going to assume the coordinates are an image with shape (75, 68, 2) i.e. there are two channels.
        if network_type.value == 1:
            self.conv1 = nn.Conv2d(2, 4, kernel_size=3, stride=1, padding=1)
            self.batchnorm1 = nn.BatchNorm2d(4)
            self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1)
            self.batchnorm2 = nn.BatchNorm2d(4)
            self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(4 * 18 * 17, 256)
            self.relu = nn.ReLU(inplace=True)
        elif network_type.value == 2:
            self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
            self.batchnorm1 = torch.nn.BatchNorm2d(16)
            self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = torch.nn.Conv2d(16, 25, kernel_size=3, stride=1, padding=1)
            self.batchnorm2 = torch.nn.BatchNorm2d(25)
            self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
            self.flatten = torch.nn.Flatten()
            self.fc1 = torch.nn.Linear(25 * 18 * 128, 256)
            self.relu = torch.nn.ReLU(inplace=True)
        
    def forward(self, x):
        """Performs a forward pass of the input ``x`` through the network.

        Parameters
        ----------
        x : torch.Tensor instance
            Torch tensor containing the input to the network.

        """
        if self.network_type.value == 1:
            x = self.conv1(x)
            x = self.batchnorm1(x)
            x = self.maxpool1(x)
            x = self.conv2(x)
            x = self.batchnorm2(x)
            x = self.maxpool2(x)
            x = self.flatten(x)
            x = self.fc1(x)
            x = self.relu(x)
        elif self.network_type.value == 2:
            x = self.conv1(x)
            x = self.batchnorm1(x)
            x = self.maxpool1(x)
            x = self.conv2(x)
            x = self.batchnorm2(x)
            x = self.maxpool2(x)
            x = self.flatten(x)
            x = self.fc1(x)
            x = self.relu(x)
        return x


class MLP(nn.Module):
    """Implements the Multi-Layer Perceptron network used at the end of the CheapFake network.

    """

    # For this network, we are going to havae an input of shape (3 x 256) where we have concatenated the outputs from the three networks. Then this can just be a few network layers if we want.
    def __init__(self, verbose=False):
        """Instantiates a new MLP object.

        Parameters
        ----------
        verbose : {False, True}, bool, optional
            If True then verbose output about the progress is sent to the system console, by default False.        

        """
        super(MLP, self).__init__()

        assert isinstance(verbose, bool)


class CheapFake(nn.Module):
    """Implements the CheapFake network architecture.

    """

    def __init__(
        self,
        dropout_rate=0.5,
        num_modules=1,
        verbose=False,
        device=torch.device("cpu"),
    ):
        """Instantiates a new CheapFake object.

        Parameters
        ----------
        input_size : tuple
            The input size of the images that are being fed into the network, by default (64, 128).
        dropout_rate : float, optional
            The dropoutout rate applied after each convolutional layer, by default 0.5.
        num_modules : int, optional
            The number of modules, by default 1.
        verbose : {False, True}, bool, optional
            If verbose then progress of the forward pass is printed, by default True.
        device : torch.device instance
            The device where all computations are carried out, by default torch.device("cpu").

        """
        super(CheapFake, self).__init__()

        assert isinstance(dropout_rate, float)
        assert isinstance(num_modules, int)
        assert isinstance(verbose, bool)
        assert isinstance(device, torch.device)
        assert dropout_rate <= 1 and dropout_rate > 0, "Probability must be in (0, 1]."

        self.dropout_rate = dropout_rate
        self.num_modules = num_modules
        self.verbose = verbose
        self.device = device

        self._load_lipnet()
        self._load_fan()
        self._load_encoder()

    def _load_lipnet(self, load_weights=True):
        """Creates a new instance of the LipNet model and also loads pre-trained weights if they are available.
        
        Parameters
        ----------
        load_weights : {True, False}, bool, optional
            If True then the pre-trained LipNet weights are loaded.
        
        """
        assert isinstance(load_weights, bool)

        self.lipnet_model = lipnet.LipNet(verbose=self.verbose)
        self.lipnet_model.to(self.device)
        self.lipnet_model.eval()

        if load_weights:
            if hasattr(lipnet_options, "weights"):
                pretrained_dict = torch.load(
                    lipnet_options.weights, map_location=self.device
                )
                pretrained_dict["fully_connected.weight"] = pretrained_dict.pop(
                    "FC.weight"
                )
                pretrained_dict["fully_connected.bias"] = pretrained_dict.pop("FC.bias")
                model_dict = self.lipnet_model.state_dict()
                pretrained_dict = {
                    k: v
                    for k, v in pretrained_dict.items()
                    if k in model_dict.keys() and v.size() == model_dict[k].size()
                }
                missed_params = [
                    k for k, v in model_dict.items() if not k in pretrained_dict.keys()
                ]
                if self.verbose:
                    print(
                        "Loaded parameters / Total parameters: {}/{}".format(
                            len(pretrained_dict), len(model_dict)
                        )
                    )
                model_dict.update(pretrained_dict)
                self.lipnet_model.load_state_dict(model_dict)
            else:
                print(
                    "[WARNING] Invalid path to pre-trained weights. No weights will be loaded."
                )
        else:
            if self.verbose:
                print("[INFO] No pre-trained weights were loaded.")

    def _load_fan(self):
        """Creates a new instance of the FAN model and loads pre-trained weights.

        The pre-trained weights are automatically loaded.

        See Also
        --------
        cheapfake.face_alignment.api : Handles initialization of the Face Alignment Network.

        """

        if hasattr(fan_options, "landmarks_type"):
            assert fan_options.landmarks_type in [
                "2D",
                "3D",
            ], "Landmarks must be either 2D or 3D."
            self.landmarks_type = fan_options.landmarks_type
        else:
            self.landmarks_type = "2D"

        if self.landmarks_type == "2D":
            landmarks_type = face_alignment.LandmarksType._2D
        else:
            landmarks_type = face_alignment.LandmarksType._3D

        if hasattr(fan_options, "face_detector"):
            assert fan_options.face_detector in [
                "sfd",
                "dlib",
            ], "Only sfd and dlib are supported for face detection."
            self.face_detector = fan_options.face_detector
        else:
            self.face_detector = "sfd"

        device = "cuda" if "cuda" in self.device.type else "cpu"
        self.face_alignment_model = face_alignment.FaceAlignment(
            landmarks_type=self.landmarks_type,
            device=device,
            face_detector=self.face_detector,
            verbose=self.verbose,
        )
    
    def _load_encoder(self):
        """Creates a new instance of the encoder model and loads any pretrained weights.
        
        """
        self.fan_encoder = FeaturesEncoder(NetworkType.FAN).to(self.device)
        self.lipnet_encoder = FeaturesEncoder(NetworkType.LIPNET).to(self.device)
        # Load any weights here.

    def _find_bounding_box(self, points, tol=(2, 2, 2, 2)):
        """Finds the minimum bounding box containing the points, with tolerance in all directions.

        Parameters
        ----------
        points : numpy.ndarray or torch.Tensor instance
            Numpy array or Torch tensor containing the predicted xy-coordinates of the detected facial landmarks.
        tol : tuple or int, optional
            The tolerance (in pixels) in each direction (left, top, right, bottom), by default (2, 2, 2, 2). If an integer is passed then all directions take on this value.
        
        Returns
        -------
        bbox : tuple
            Tuple (min_x, min_y, max_x, max_y) containing the coordinates of the bounding box, with tolerance in each direction.

        """
        assert isinstance(tol, (int, tuple))
        if isinstance(tol, tuple):
            assert len(tol) == 4, "Need at least four tolerances."
        if isinstance(tol, int):
            tol = (tol, tol, tol, tol)

        x_coords, y_coords = zip(*points)
        bbox = (
            min(x_coords) - tol[0],
            min(y_coords) - tol[1],
            max(x_coords) + tol[2],
            max(y_coords) + tol[3],
        )
        bbox = tuple([int(item) for item in bbox])

        return bbox

    def _find_bounding_boxes(self, landmarks, tol=(2, 2, 2, 2)):
        """Find the minimum bounding boxes for a batch of facial landmarks.

        Parameters
        ----------
        landmarks : numpy.ndarray or torch.Tensor instance
            Numpy array or Torch tensor containing the xy-coordinates of the detected facial landmarks.
        tol : tuple or int, optional
            The tolerance (in pixels) in each direction (left, top, right, bottom), by default (2, 2, 2, 2). If an integer is passed then all directions take on this value.
        
        Returns
        -------
        bboxes : list (of tuples)
            List containing tuples containing the coordinates of the bounding box.

        """
        bboxes = list()
        landmarks = landmarks[:, 48:68]
        for landmark in landmarks:
            bboxes.append(self._find_bounding_box(points=landmark, tol=tol))

        return bboxes

    def _crop_lips(
        self, frames, landmarks, tol=(2, 2, 2, 2), channels_first=True,
    ):
        """Crops the lip area from a batch of frames. 

        Parameters
        ----------
        frames : torch.Tensor instance
            Torch tensor instance containing the frames to crop the lip areas from. 
        landmarks : numpy.ndarray or torch.Tensor instance
            Numpy array or Torch tensor containing the xy-coordinates of the detected facial landmarks.
        tol : int or tuple, optional
            The tolerance (in pixels) in each direction (left, top, right, bottom), by default (2, 2, 2, 2). If an integer is passed then all directions take on this value.
        channels_first : {True, False}, bool, optional
            If True then the input and output are assumed to have shape (sample, channel, height, width), by default True. Otherwise, the input and output are assumed to have shape (sample, height, width, channel).
        
        Returns
        -------
        cropped_frames : numpy.ndarray or torch.Tensor instance
            Numpy array or Torch tensor containing the cropped lips.

        """
        assert isinstance(tol, (int, tuple))
        assert isinstance(frames, torch.Tensor)
        assert isinstance(channels_first, bool)

        if isinstance(tol, tuple):
            assert len(tol) == 4, "Need at least four tolerances."
        if isinstance(tol, int):
            tol = (tol, tol, tol, tol)

        if channels_first:
            frames = np.einsum("ijkl->iklj", frames.cpu().numpy())

        bboxes = self._find_bounding_boxes(landmarks=landmarks, tol=tol)

        cropped_frames = torch.empty(frames.shape[0], 64, 128, frames.shape[-1])
        for k, (bbox, frame) in enumerate(zip(bboxes, frames)):
            cropped_frame = frame[bbox[1] : bbox[3], bbox[0] : bbox[2], :]
            cropped_frames[k] = torch.from_numpy(
                cv2.resize(
                    cropped_frame, dsize=(128, 64), interpolation=cv2.INTER_CUBIC
                )
            )

        return cropped_frames

    def forward(self, x):
        """Performs a forward pass of the input ``x`` through the network.

        Parameters
        ----------
        x : torch.Tensor instance
            Torch tensor containing the input to the network.
        
        Returns
        -------
        Figure this out.

        """
        fan_output = self.face_alignment_model.get_landmarks_from_batch(x)
        fan_output = np.asarray(fan_output).squeeze(axis=1)
        cropped_lips = self._crop_lips(x, fan_output)
        permuted_fan_output = np.einsum("ijk->kij", fan_output)
        permuted_fan_output = permuted_fan_output[None, :, :, :]
        fan_embedding = self.fan_encoder(torch.from_numpy(permuted_fan_output).float().cuda())
        cropped_lips = np.einsum("ijkl->lijk", cropped_lips)
        cropped_lips = cropped_lips[None, :, :, :]
        cropped_lips = torch.from_numpy(cropped_lips)
        lipnet_embedding = self.lipnet_model(cropped_lips.float().cuda())
        lipnet_embedding = torch.squeeze(lipnet_embedding, axis=1)
        lipnet_embedding = self.lipnet_encoder(lipnet_embedding[None, :, :, None].permute(0, -1, 1, 2).float().to(self.device))
        
        
        return fan_output, fan_embedding, lipnet_embedding
