"""
Python file containing the models.

Need to add support for adding in weights to the encoder model.
"""

import os
import time
import enum

import torch
import face_alignment
import torch.nn as nn
import torch.nn.functional as F

import cheapfake.lipnet.models as lipnet

lipnet_options = __import__("lipnet_config")
face_alignment_options = __import__("face_alignment_config")


class NetworkType(enum.Enum):
    face_alignment = 1
    lipnet = 2
    resnet_audio = 3


class FeaturesEncoder(nn.Module):
    """Implementation of a feature encoder that creates embeddings that fit into a multi-layer perceptron.

    """

    def __init__(self, network_type=NetworkType.face_alignment):
        """Instantiates a new FeaturesEncoder network.

        Parameters
        ----------
        network_type : cheapfake.contrib.models.NetworkType instance, optional
            The type of network to instantiate, by default NetworkType.face_alignment. The supported options are NetworkType.face_alignment, NetworkType.lipnet, and NetworkType.resnet_audio.

        """
        super(FeaturesEncoder, self).__init__()

        assert network_type.value in (1, 2, 3), "Network architecture not supported."

        self.network_type = network_type

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
        """Performs a forward pass of the input ``x`` through the encoder network.

        Parameters
        ----------
        x : torch.Tensor instance
            Torch tensor containing the input to the network.

        Returns
        -------
        x : torch.Tensor instance
            Torch tensor containing the embedded features.

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


class AugmentedFAN(nn.Module):
    """Augmented Face Alignment Network (AFAN) that includes a feature embedding network that acts on the output of FAN.

    """

    def __init__(self, device=torch.device("cpu"), num_modules=1, verbose=True):
        """Instantiates a new AugmentedFAN object.

        Pre-trained weights are automatically loaded.

        Parameters
        ----------
        device : torch.device instance
            The device on which all procedures are carried out, by default torch.device("cpu").
        num_modules : int, optional
            The number of modules, by default 1.    
        verbose : bool, optional
            If True then verbose output will be printed onto the system console.
        
        """
        super(AugmentedFAN, self).__init__()

        assert isinstance(device, torch.device)
        assert isinstance(num_modules, int)
        assert isinstance(verbose, bool)

        if hasattr(face_alignment_options, "landmarks_type"):
            assert face_alignment_options.landmarks_type in [
                "2D",
                "3D",
            ], "Landmarks must either be in 2D or 3D."
            self.landmarks_type = face_alignment_options.landmarks_type
        else:
            self.landmarks_type = "2D"

        if self.landmarks.type == "2D":
            landmarks_type = face_alignment.LandmarksType._2D
        else:
            landmarks_type = face_alignment.LandmarksType._3D

        if hasattr(face_alignment_options, "face_detector"):
            assert face_alignment_options.face_detector in [
                "sfd",
                "dlib",
            ], "Only S3FD and DLib are supported as face detection schemes."
            self.face_detector = face_alignment_options.face_detector
        else:
            self.face_detector = "sfd"

        self.device = device
        device = "cuda" if "cuda" in self.device.type else "cpu"
        self.num_modules = num_modules
        self.verbose = verbose

        self.face_alignment_model = face_alignment.FaceAlignment(
            landmarks_type=self.landmarks_type,
            device=device,
            face_detector=self.face_detector,
            verbose=self.verbose,
        )
        self.encoder_model = FeaturesEncoder(network_type=NetworkType.face_alignment)

    def _load_encoder_weights(self):
        pass

    def forward(self, x):
        """Performs a forward pass of ``x`` through the network.

        Parameters
        ----------
        x : torch.Tensor instance
            Torch tensor containing the input to the network.

        Returns
        -------
        landmarks : numpy.ndarray or torch.Tensor instance
            Numpy array or Torch tensor containing the xy-coorddinates of the detected facial landmarks.
        face_embeddings : numpy.ndarray or torch.Tensor instance
            Numpy array or Torch tensor containing the embeddings of the facial features returned by the FAN.

        """
        landmarks = self.face_alignment_model.get_landmarks_from_batch(x)
        landmarks = torch.Tensor(landmarks)
        landmarks = torch.squeeze(landmarks, axis=1).to(self.device)

        landmarks_permutted = landmarks.permute(2, 0, 1).float().to(self.device)
        face_embedding = self.encoder_model(landmarks_permutted)

        return landmarks, face_embedding


class AugmentedLipNet(nn.Module):
    """Augmented LipNet that includes a feature embedding network that acts on the output of LipNet.

    """

    def __init__(
        self, device=torch.device("cpu"), dropout_rate=0.5, num_modules=1, verbose=True
    ):
        """Instantiates a new AugmentedLipNet object.

        Parameters
        ----------
        device : torch.device instance
            The device on which all procedures are carried out, by default torch.device("cpu").
        dropout_rate : float, optional
            The dropout rate applied after each convolutional layer, by default 0.5.
        verbose : bool, optional
            If True then progress of the forward pass is printed to the system console, by default True.
        
        """
        super(AugmentedLipNet, self).__init__()

        assert isinstance(device, torch.device)
        assert isinstance(dropout_rate, float)
        assert isinstance(verbose, bool)
        assert (
            dropout_rate > 0 and dropout_rate <= 1
        ), "Dropout rate must be a proper probability."

        self.device = device
        self.dropout_rate = dropout_rate
        self.verbose = verbose

        self.lipnet_model = lipnet.LipNet(
            dropout=self.dropout_rate, verbose=self.verbose
        ).to(self.device)
        self.encoder_model = FeaturesEncoder(network_type=NetworkType.lipnet).to(
            self.device
        )

        if hasattr(lipnet_options, "weights"):
            self._load_lipnet_weights()
        else:
            print("[INFO] Pre-trained weights were not loaded.")

    def _load_lipnet_weights(self):
        pretrained_dict = torch.load(lipnet_options.weights, map_location=self.device)
        pretrained_dict["fully_connected.weight"] = pretrained_dict.pop("FC.weight")
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
        print("[WARNING] Invalid path to pre-trained weights, and thus no weights will be loaded.")

    def _load_encoder_weights(self):
        pass
    
    def forward(self, x):
        """Performs a forward pass of the input ``x`` through the network.

        Parameters
        ----------
        x : torch.Tensor instance
            Torch tensor containing the input to the network.

        Returns
        -------
        lipnet_embedding : torch.Tensor instance
            Torch tensor containing the learned embedding from LipNet.

        """
        x = self.lipnet_model(x)
        x = torch.squeeze(x, axis=1)
        x = x.permute(0, -1, 1, 2).float().to(self.device)
        
        lipnet_embedding = self.encoder_model(x)

        return lipnet_embedding
