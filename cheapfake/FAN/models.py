"""
Implements the 3D-FAN model.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def batch_and_conv(n_inputs, n_outputs, kernel_size=3, stride=1, padding=1, bias=False):
    """
    Convenience function that returns a BatchNorm2d and Conv2d layer togather.

    Parameters
    ----------
    n_inputs : int
        The number of inputs to the convolutional layer.
    n_outputs : int
        The number of outputs of the convolutional layer.
    kernel_size : int, optional
        The size of the kernel for the convolutional layer, by default 3.
    stride : int, optional
        The stride of the convolution layer, by default 1.
    padding : int, optional
        The padding for the convolutional layer, by default 1.
    bias : {True, False}, optional
        Boolean determining whether or not a bias term is added to the convolutional layer, by default False.

    Returns
    -------
    bn_layer : torch.nn.modules.batchnorm.BatchNorm2d instance
        An instance of the BatchNorm2d layer.
    conv_layer : torch.nn.modules.conv.Conv2d instance
        An instance of the Conv2d layer.
    """
    bn_layer = nn.BatchNorm2d(n_inputs)
    conv_layer = nn.Conv2d(
        n_inputs,
        n_outputs,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
    )

    return bn_layer, conv_layer


class ConvolutionalBlock(nn.Module):
    """
    Implementation of the Convolutional Block.
    """

    def __init__(self, n_inputs, n_outputs):
        """
        Instantiates a ConvolutionalBlock object.

        Parameters
        ----------
        n_inputs : int
            The number of inputs to the block.
        n_outputs : int
            The number of outputs to the block.

        Returns
        -------
        None
        """
        super(ConvolutionalBlock, self).__init__()
        self.bn1, self.conv1 = batch_and_conv(n_inputs, int(n_outputs / 2))
        self.bn2, self.conv2 = batch_and_conv(int(n_outputs / 2), int(n_outputs / 4))
        self.bn3, self.conv3 = batch_and_conv(int(n_outputs / 4), int(n_outputs / 4))

        # Downsample if the input and output sizes don't match.
        if n_inputs != n_outputs:
            self.downsample = nn.Sequential(
                [
                    nn.BatchNorm2d(n_inputs),
                    nn.ReLU(True),
                    nn.Conv2d(n_inputs, n_outputs, kernel_size=1, stride=1, bias=False),
                ]
            )
        else:
            self.downsample = None

    def forward(self, x):
        """
        Performs a forward pass of ``x`` through the Convolutional Block.

        Parameters
        ----------
        x : torch.Tensor instance
            The input to the Convolutional Block.

        Returns
        -------
        output : torch.Tensor instance
            The output of the Convolutional Block.
        """
        residual = x

        y1 = self.bn1(x)
        y1 = F.relu(y1, True)
        y1 = self.conv1(y1)

        y2 = self.bn2(y1)
        y2 = F.relu(y2, True)
        y2 = self.conv2(y2)

        y3 = self.bn3(y2)
        y3 = F.relu(y3, True)
        y3 = self.conv3(y3)

        output = torch.cat((y1, y2, y3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        output += residual

        return output


class BottleneckNetwork(nn.Module):
    """
    Implements the Bottleneck network architecture.
    """

    n_expansion = 4

    def __init__(self, n_inputs, n_features, stride=1, downsample=None):
        """
        Instantiates a Bottleneck network.

        Parameters
        ----------
        n_inputs : int
            The number of inputs to the Bottleneck network.
        n_features : int
            The number of features for each convolutional layer.
        stride : int, optional
            The number of strides for the convolutional layer, by default 1.
        downsample : torch.nn.Sequential instance
            Downsampling network, by default None, which means no downsampling is performed.

        Returns
        -------
        None
        """
        super(BottleneckNetwork, self).__init__()
        self.bn1, self.conv1 = batch_and_conv(
            n_inputs, n_features, kernel_size=1, padding=0
        )
        self.bn2, self.conv2 = batch_and_conv(n_features, n_features, stride=stride)
        self.bn3, self.conv3 = batch_and_conv(
            n_features, n_features * 4, kernel_size=1, padding=0
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        Performs a forward pass of ``x`` through the BottleNeck network.

        Parameters
        ----------
        x : torch.Tensor instance
            The input to the Bottleneck network.

        Returns
        -------
        output : torch.Tensor instance
            The output of the Bottleneck network.
        """
        residual = x

        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu(output)

        output = self.conv3(output)
        output = self.bn3(output)

        if self.downsample is not None:
            residual = self.downsample(residual)
        output += residual

        output = self.relu(output)

        return output


class HourglassNetwork(nn.Module):
    """
    Implementation of the Hourglass network.
    """

    def __init__(self, n_modules, depth, n_features):
        """
        Instantiates a Hourglass network.

        Parameters
        ----------
        n_modules : int
            The number of modules in the Hourglass network.
        depth : int
            The depth of the Hourglass network.
        n_features : int
            The number of features for the convolutional layer(s).
        
        Returns
        -------
        None
        """
        super(HourglassNetwork, self).__init__()
        self.n_modules = n_modules
        self.depth = depth
        self.n_features = n_features

    def _create_layers(self, depth):
        """
        Creates the layers necessary for the Hourglass network.

        Parameters
        ----------
        depth : int
            The depth of the Hourglass network.

        Returns
        -------
        None
        """
        self.add_module(
            "b1_" + str(depth), ConvolutionalBlock(self.n_features, self.n_features)
        )
        self.add_module(
            "b2_" + str(depth), ConvolutionalBlock(self.n_features, self.n_features)
        )

        if depth > 1:
            self._create_layers(depth - 1)
        else:
            self.add_module(
                "b2_plus_" + str(depth),
                ConvolutionalBlock(self.n_features, self.n_features),
            )

        self.add_module(
            "b3_" + str(depth), ConvolutionalBlock(self.n_features, self.n_features)
        )

    def _forward(self, depth, x):
        """
        Performs an internal forward pass of ``x`` through the Hourglass network, using the ``depth`` parameter.

        Parameters
        ----------
        depth : int
            The depth of the Hourglass network.
        x : torch.Tensor instance
            The input to the Hourglass network.

        Returns
        -------
        output : torch.Tensor instance
            The output of the internal forward pass. 
        """
        # Forward pass through the upper branch.
        upper_branch_1 = self._modules["b1_" + str(depth)](x)

        # Forward pass through the lower branch.
        lower_branch_1 = F.avg_pool2d(x, kernel_size=2, stride=2)
        lower_branch_1 = self._modules["b2_" + str(depth)](lower_branch_1)

        if depth > 1:
            lower_branch_2 = self._forward(depth - 1, lower_branch_1)
        else:
            lower_branch_2 = lower_branch_1
            lower_branch_2 = self._modules["b2_plus_" + str(depth)](lower_branch_2)

        lower_branch_3 = lower_branch_2
        lower_branch_3 = self._modules["b3_" + str(depth)](lower_branch_3)

        upper_branch_2 = F.interpolate(lower_branch_3, scale_factor=2, mode="nearest")

        output = upper_branch_1 + upper_branch_2

        return output

    def forward(self, x):
        """
        Performs a forward pass of ``x`` through the Hourglass network.

        Parameters
        ----------
        x : torch.Tensor instance
            The input to the Hourglass network.

        Returns
        -------
        output : torch.Tensor instance
            The output of the Hourglass network.
        """
        output = self._forward(self.depth, x)

        return output


class FaceAlignmentNetwork(nn.Module):
    """
    Implements the Face Alignment Network (FAN).
    """

    def __init__(self, n_modules=1):
        """
        Creates an instance of a FAN object.

        Parameters
        ----------
        n_modules : int, optional
            The number of modules in the FAN. 
        
        Returns
        -------
        None
        """
        super(FaceAlignmentNetwork, self).__init__()
        self.bn1, conv1 = batch_and_conv(3, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = ConvolutionalBlock(64, 128)
        self.conv3 = ConvolutionalBlock(128, 128)
        self.conv4 = ConvolutionalBlock(128, 256)

        for hourglass_module in range(self.n_modules):
            self.add_module("m" + str(hourglass_module), HourglassNetwork(1, 4, 256))
            self.add_module(
                "top_m_" + str(hourglass_module), ConvolutionalBlock(256, 256)
            )
            self.add_module(
                "conv_last" + str(hourglass_module),
                nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            )
            self.add_module("bn_end" + str(hourglass_module), nn.BatchNorm2d(256))
            self.add_module(
                "l" + str(hourglass_module),
                nn.Conv2d(256, 68, kernel_size=1, stride=1, padding=0),
            )

            if hourglass_module < self.n_modules - 1:
                self.add_module(
                    "bl" + str(hourglass_module),
                    nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                )
                self.add_module(
                    "al" + str(hourglass_module),
                    nn.Conv2d(68, 256, kernel_size=1, stride=1, padding=0),
                )

    def forward(self, x):
        """
        Performs a forward pass of ``x`` through the Face Alignment Network. 

        Parameters
        ----------
        x : torch.Tensor instance
            The input to the Face Alignment network.

        Returns
        -------
        outputs : list (of torch.Tensor instances)
            The output of the Face Alignment network, which is a heatmap of where the facial landmarks are.
        """
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.avg_pool2d(self.conv2d(x), 2, stride=2)
        x = self.conv3(x)
        x = self.conv4(x)

        previous = x

        outputs = list()
        for k in range(self.n_modules):
            hourglass_module = self._modules["m" + str(k)](previous)
            lower_leg = self._modules["top_m_" + str(k)](hourglass_module)
            lower_leg = F.relu(
                self._modules["bn_end" + str(k)](
                    self._modules["conv_last" + str(k)](lower_leg)
                ),
                True,
            )

            # Predict the heatmaps.
            output = self._modules["l" + str(k)](lower_leg)
            outputs.append(output)

            if k < self.n_modules - 1:
                lower_leg = self._modules["bl" + str(k)](lower_leg)
                output = self._modules["al" + str(k)](output)
                previous = previous + lower_leg + output

        return outputs
