"""
Python implementation of ResNet, with a depth parameter included.
"""

import math
import torch
import models
import torch.nn as nn
import torch.nn.functional as F


class ResNetWithDepth(nn.Module):
    """
    Implementation of ResNet with a depth parameter.
    """

    def __init__(
        self, block=models.BottleneckNetwork, layers=[3, 8, 36, 3], n_classes=68
    ):
        """
        Instantiates a ResNet with depth parameter included.

        Parameters
        ----------
        TODO

        Returns
        -------
        None
        """
        super(ResNetWithDepth, self).__init__()
        self.n_inputs = 64
        self.conv1 = nn.Conv2d(
            3 + 68, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(7)
        self.fully_connected = nn.Linear(512 * block.expansion, n_classes)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, n_features, blocks, stride=1):
        """
        
        """
        layers = list()
        layer = nn.Sequential(*layers)

        return layer

    def forward(self, x):
        """
        Performs a forward pass of ``x`` through the ResNetWithDepth network.

        Parameters
        ----------
        x : torch.Tensor instance
            The input to the network.
        
        Returns
        -------
        output : torch.Tensor instance
            The output of the network.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        output = self.fully_connected(x)

        return output
