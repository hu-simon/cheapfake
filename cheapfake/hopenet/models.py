import os
import time

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class HopeNet(nn.Module):
    """
    Implements HopeNet, used for estimating the head pose from media (images and videos).
    """

    def __init__(self, block, layers, n_bins):
        """
        Instantiates a HopeNet object used for estimating head pose from media.

        Parameters
        ----------
        block : 
        layers : list (of ints) 
            List of layer sizes for each ``block`` object.
        n_bins : int
            The number of bins in the yaw, pitch, and roll outputs. Increase this number to have a finer estimate.

        Returns
        -------
        None
        """
        super(HopeNet, self).__init__()

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc_yaw = nn.Linear(512 * block.expansion, n_bins)
        self.fc_pitch = nn.Linear(512 * block.expansion, n_bins)
        self.fc_roll = nn.Linear(512 * block.expansion, n_bins)

        self.fc_finetune = nn.Linear(512 * block.expansion + 3, 3)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def _make_layer(self, block, n_planes, n_blocks, stride=1, downsample=None):
        """
        TODO

        Parameters
        ----------
        TODO

        Returns
        -------
        TODO
        """
        if stride != 1 or self.inplanes != n_planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    n_planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(n_planes * block.expansion),
            )

        layers = list()
        layers.append(block(self.inplanes, n_planes, stride, downsample))
        self.inplanes = n_planes * block.expansion
        for k in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Performs a forward pass of ``x`` through HopeNet.

        Parameters
        ----------
        x : torch.Tensor instance
            The input tensor to the network.

        Returns
        -------
        yaw, pitch, roll : torch.Tensor instance
            The output yaw, pitch, and roll from the input.
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

        yaw = self.fc_yaw(x)
        pitch = self.fc_pitch(x)
        roll = self.fc_roll(x)

        return yaw, pitch, roll


class EulerResNet(nn.Module):
    """
    Implements EulerResNet, which is used for regression for the the three Euler angles returned from HopeNet.
    """

    def __init__(self, block, layers, n_classes=1000):
        """
        Instantiates an EulerResNet object.

        Parameters
        ----------
        block : 
        layers : list (of ints)
            List containing layer sizes for each ``block`` instance.
        n_classes : int, optional
            The number of output classes in the regression network, by default 1000.

        Returns
        -------
        None
        """
        super(EulerResNet, self).__init__()

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc_euler = nn.Linear(512 * block.expansion, n_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def _make_layer(self, block, n_planes, n_blocks, stride=1, downsample=None):
        """
        TODO 

        Parameters
        ----------
        TODO

        Returns
        -------
        TODO
        """
        if stride != 1 or self.inplanes != n_planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    n_planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(n_planes * block.expansion),
            )

        layers = list()
        layers.append(block(self.inplanes, n_planes, stride, downsample))
        self.inplanes = n_planes * block.expansion

        for k in range(1, n_blocks):
            layers.append(block(self.inplanes, n_planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Performs a forward pass of the input ``x`` through EulerResNet.

        Parameters
        ----------
        x : torch.Tensor instance
            The input tensor to the network.

        Returns
        -------
        x : torch.Tensor instance
            The output tensor, containing the most probable estimate of the Euler angle.
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
        x = self.fc_euler(x)

        return x


class AlexNet(nn.Module):
    """
    Implements AlexNet, laid out as a HopeNet which classifies Euler angles in bins.

    Regression is then used on the output to output the expected value.
    """

    def __init__(self, n_bins, dropout_rate=0.5):
        """
        Instantiates an AlexNet object.

        Parameters
        ----------
        n_bins : int
            The number of bins, which are output by the network.
        dropout_rate : float, optional
            The dropout rate passed on to ``nn.Dropout()``, by default 0.5.

        Returns
        -------
        None
        """
        super(AlexNet, self).__init__()

        self.n_bins = n_bins
        self.dropout_rate = dropout_rate

        # Feature network.
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)

        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)

        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Classification network.
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)

        # Euler network.
        self.fc_yaw = nn.Linear(4096, n_bins)
        self.fc_pitch = nn.Linear(4096, n_bins)
        self.fc_roll = nn.Linear(4096, n_bins)

    def forward(self, x):
        """
        Performs a forward pass of the input ``x`` through the network.

        Parameters
        ----------
        x : torch.Tensor instance
            The input tensor to the network.

        Returns
        -------
        yaw, pitch, roll : torch.Tensor instance
            The output yaw, pitch, roll used to construct the pose of the network.
        """
        # Features network.
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.pool5(x)

        # Classification network.
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)

        yaw = self.fc_yaw(x)
        pitch = self.fc_pitch(x)
        roll = self.fc_roll(x)

        return yaw, pitch, roll
