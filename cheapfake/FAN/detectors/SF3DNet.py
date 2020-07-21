import torch
import torch.nn as nn
import torch.nn.functional as F


class S3FDNet(nn.Module):
    """
    Implementation of the S3FD Network.
    """

    def __init__(self):
        """
        Instantiates a S3FD network.
        """
        super(S3FDNet, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.fully_connected_1 = nn.Conv2d(
            512, 1024, kernel_size=3, stride=1, padding=3
        )
        self.fully_connected_2 = nn.Conv2d(
            1024, 1024, kernel_size=1, stride=1, padding=0
        )

        self.conv6_1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.conv6_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        self.conv7_1 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.conv7_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.conv3_norm = NormLayer(256, scale=10)
        self.conv4_norm = NormLayer(512, scale=8)
        self.conv5_norm = NormLayer(512, scale=5)

        self.conv3_confidence = nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)
        self.conv4_confidence = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
        self.conv5_confidence = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
        self.conv6_confidence = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
        self.conv7_confidence = nn.Conv2d(256, 2, kernel_size=3, stride=1, padding=1)
        self.fully_connected_2_confidence = nn.Conv2d(
            1024, 2, kernel_size=3, stride=1, padding=1
        )

        self.conv3_location = nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)
        self.conv4_location = nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1)
        self.conv5_location = nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1)
        self.conv6_location = nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1)
        self.conv7_location = nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)
        self.fully_connected_2_location = nn.Conv2d(
            1024, 4, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        """
        Performs a forward pass of ``x`` through the S3FD network.

        Parameters
        ----------
        x : torch.Tensor instance
            The input tensor to the network.

        Returns
        -------
        output : list (of torch.Tensor)
            The output of the network, of the confidences and the six key features predicted by the S3FD network.
        """
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        xc3 = x
        xc3 = self.conv3_norm(xc3)
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        xc4 = x
        xc4 = self.conv4_norm(xc4)
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        xc5 = x
        xc5 = self.conv5_norm(xc5)
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.fully_connected_1(x))
        x = F.relu(self.fully_connected_2(x))
        xfc1 = x

        x = F.relu(self.conv6_1(x))
        x = F.relu(self.conv6_2(x))
        xc6 = x

        x = F.relu(self.conv7_1(x))
        x = F.relu(self.conv7_2(x))
        xc7 = x

        conf1 = self.conv3_confidence(xc3)
        conf2 = self.conv4_confidence(xc4)
        conf3 = self.conv5_confidence(xc5)
        conf4 = self.fully_connected_2_confidence(xfc1)
        conf5 = self.conv6_confidence(xc6)
        conf6 = self.conv7_confidence(xc7)

        loc1 = self.conv3_location(xc3)
        loc2 = self, conv4_location(xc4)
        loc3 = self.conv5_location(xc5)
        loc4 = self.fully_connected_2_location(xfc1)
        loc5 = self.conv6_location(xc6)
        loc6 = self.conv7_location(xc7)

        chunk = torch.chunk(conf1, 4, 1)
        bmax = torch.max(torch.max(chunk[0], chunk[1]), chunk[2])
        conf1 = torch.cat([bmax, chunk[3]], dim=1)

        output = [
            conf1,
            loc1,
            conf2,
            loc2,
            conf3,
            loc3,
            conf4,
            loc4,
            conf5,
            loc5,
            conf6,
            loc6,
        ]

        return output


class NormLayer(nn.Module):
    """
    Implementation of the L2 Norm Layer used in the S3FD paper.
    """

    def __init__(self, n_channels, scale=1.0, epsilon=1e-10):
        """
        Instantiates a L2 Norm layer.

        Parameters
        ----------
        n_channels : int
            The number of channels in the input.
        scale : float, optional
            The scaling used for the weighted L2-norm, by default 1.0.
        epsilon : float, optional
            Parameter that prevents division by zero, by default 1e-10.

        Returns
        -------
        None
        """
        super(NormLayer, self).__init__()

        self.n_channels = n_channels
        self.scale = scale
        self.epsilon = epsilon
        self.weights = nn.Parameter(torch.Tensor(self.n_channels))
        self.weights.data *= 0.0
        self.weights.data += self.scale

    def forward(self, x):
        """
        Performs a forward pass of the input ``x`` through the L2 norm layer.

        Parameters
        ----------
        x : torch.Tensor instance
            The input tensor.

        Returns
        -------
        x : torch.Tensor instance
            The result of putting ``x`` through the L2 norm layer.
        """
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.epsilon
        x = x / norm * self.weight.view(1, -1, 1, 1)

        return x
