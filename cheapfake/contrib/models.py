import os
import time

import torch
import face_alignment
import torch.nn as nn
import torch.nn.functional as F

import cheapfake.lipnet.models as lipnet
import cheapfake.contrib.dataset as dataset
import cheapfake.contrib.video_processor as video_processor


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
        self, input_size=(64, 128), dropout_rate=0.5, num_modules=1, verbose=False
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

        """
        super(CheapFake, self).__init__()

        assert isinstance(dropout_rate, float)
        assert isinstance(num_modules, int)
        assert isinstance(verbose, bool)
        assert dropout_rate <= 1 and dropout_rate > 0, "Probability must be in (0, 1]."

        self.dropout_rate = dropout_rate
        self.num_modules = num_modules
        self.verbose = verbose

        self._load_lipnet()

    def _load_lipnet(self, load_weights=True):
        """Creates a new instance of the LipNet model and also loads pre-trained weights if they are available.
        
        Parameters
        ----------
        load_weights : {True, False}, bool, optional
            If True then the pre-trained LipNet weights are loaded.
        
        """
        assert isinstance(load_weights, bool)
