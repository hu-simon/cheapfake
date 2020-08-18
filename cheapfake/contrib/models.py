"""
Need to rethink the entire forward pass. The idea is as follows.

The resized frames should be sent into FAN, with the expected shape. This returns facial landmarks (if a face is detected). Using the resizing information, we can rescale the predicted facial landmarks to get an overall estimate of where the lips are located in the full-size image. Then, the lip region is extracted from the full-size image (with some buffer room) and gets resized to 64 x 128 which is the size expected by LipNet. The predicted facial features and the embeddings learning by the LipNet are sent into the MLP, along with the audio embeddings learned by the residual network.
"""

import os
import time

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


class MLP(nn.Module):
    """Implements the Multi-Layer Perceptron network used at the end of the CheapFake network.

    """

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

    def _load_lipnet(self, load_weights=True):
        """Creates a new instance of the LipNet model and also loads pre-trained weights if they are available.
        
        Parameters
        ----------
        load_weights : {True, False}, bool, optional
            If True then the pre-trained LipNet weights are loaded.
        
        """
        assert isinstance(load_weights, bool)

        self.lipnet_model = lipnet.LipNet()
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

        if self.landmarks_type is "2D":
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

    def _permute_fan(self, x):
        """Permutes the input ``x`` so that it is in the shape expected by the Face Alignment Network (FAN). 

        The input ``x`` is assumed to be of the shape (batch, channel, sample, height, width) which is expected by LipNet. However, the input to the FAN is assumed to be of the shape (batch, sample, channel, height, width). 

        Parameters
        ----------
        x : torch.Tensor or numpy.ndarray instance
            Torch tensor or Numpy array containing the input to be permutted, assumed to have shape (batch, channel, sample, height, width).
        
        Returns
        -------
        x : torch.Tensor or numpy.ndarray instance
            Torch tensor or Numpy array containing the permutted input, assumed to have shape (batch, sample, channel, height, width).

        """
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        x = np.einsum("ijklm->ikjlm", x)
        x = torch.from_numpy(x)

        return x

    def _forward_lipnet(self, x):
        """Performs a forward pass of the input ``x`` through LipNet.

        Parameters
        ----------
        x : torch.Tensor instance 
            Torch tensor containing the input to the network.

        Returns
        -------
        lipnet_features : torch.Tensor instance
            Torch tensor containing the latent features coming from removing the classification layer from LipNet.

        """
        pass

    def _forward_fan(self, x):
        """Performs a forward pass of the input ``x`` through the Face Alignment Network.

        Parameters
        ----------
        x : torch.Tensor instance
            Torch tensor containing the input to the network.

        Returns
        -------

        """
        pass

    def forward(self, x):
        """Performs a forward pass of the input ``x``.

        Parameters
        ----------
        x : torch.Tensor instance
            Torch tensor containing the input to the network. The input to the network should be a list of tensors, [frames, audio, audio_stft]. 

        Returns
        -------
        x : torch.Tensor instance
            Torch tensor containing the output of the network.
        
        """
        pass
