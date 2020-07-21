import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LipNet(nn.Module):
    """
    Implements the LipNet architecture.
    """

    def __init__(self, dropout_rate=0.5):
        """
        Instantiates a LipNet architecture.

        Parameters
        ----------
        dropout_rate : float, optional
            The dropout rate applied after each convolutional layer.
        """
        super(LipNet, self).__init__()
        self.dropout_rate = dropout_rate

        self.conv1 = nn.Conv3d(3, 32, (3, 5, 5), (1, 2, 2), (1, 2, 2))
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        self.conv2 = nn.Conv3d(32, 64, (3, 5, 5), (1, 1, 1), (1, 2, 2))
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        # Change the below number back to (64, 96, ...) since we had to modify it for the batch size.
        # self.conv3 = nn.Conv3d(64, 96, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        self.conv3 = nn.Conv3d(64, 60, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        self.fully_connected = nn.Linear(512, 27 + 1)

        # Change the below number back to (96 * 4 * 8, ...) since we had to modify it for the batch size.
        # self.gru1 = nn.GRU(96 * 4 * 8, 256, 1, bidirectional=True)
        self.gru1 = nn.GRU(60 * 4 * 8, 256, 1, bidirectional=True)
        self.gru2 = nn.GRU(512, 256, 1, bidirectional=True)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.dropout3d = nn.Dropout3d(self.dropout_rate)

        # Initialize the weights.
        self._init_weights()

    def _init_weights(self):
        """
        Initializes the weights of the network using the Kaiming Normal initialization.
        """
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv3.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.fully_connected.weight, nonlinearity="sigmoid")

        nn.init.constant_(self.conv1.bias, 0)
        nn.init.constant_(self.conv2.bias, 0)
        nn.init.constant_(self.conv3.bias, 0)
        nn.init.constant_(self.fully_connected.bias, 0)

        for gru in (self.gru1, self.gru2):
            # Change the below number back to (96 * 3 * 6 + 256) since we had to modify it for the batch size.
            # stddev = math.sqrt(2 / (96 * 3 * 6 + 256))
            stddev = math.sqrt(2 / 60 * 3 * 6 + 256)
            for k in range(0, 256 * 3, 256):
                nn.init.uniform_(
                    gru.weight_ih_l0[k : k + 256],
                    -math.sqrt(3) * stddev,
                    math.sqrt(3) * stddev,
                )
                nn.init.orthogonal_(gru.weight_hh_l0[k : k + 256])
                nn.init.constant_(gru.bias_ih_l0[k : k + 256], 0)
                nn.init.uniform_(
                    gru.weight_ih_l0_reverse[k : k + 256],
                    -math.sqrt(3) * stddev,
                    math.sqrt(3) * stddev,
                )
                nn.init.orthogonal_(gru.weight_hh_l0_reverse[k : k + 256])
                nn.init.constant_(gru.bias_ih_l0_reverse[k : k + 256], 0)

    def _preprocess_gru(self, x):
        """
        Preprocesses the input so that it is in the format expected by the RNN.

        Explicitly, (B, C, T, H, W) -> (T, B, C * H * W). Here, T is time, B is batches, C is channel, H is height, and W is weight.

        Parameters
        ----------
        x : torch.Tensor instance
            The input tensor to be processed.
        
        Returns
        -------
        output : torch.Tensor instance
            The output tensor, whose format is in the form expected by the RNN.

        Notes
        -----
        No checks are done to ensure that the input is in the form expected. Please use extreme caution when using this function.
        """
        output = x.permute(2, 0, 1, 3, 4).contiguous()
        output = output.view(output.size(0), output.size(1), -1)

        return output

    def forward(self, x):
        """
        Performs a forward pass of ``x`` through LipNet.

        Parameters
        ----------
        x : torch.Tensor instance
            The input to the network.
        
        Returns
        -------
        output : torch.Tensor instance
            The output of the network, from the forward pass of ``x`` through LipNet.
        """

        print("Starting forward pass...")
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool1(x)
        print("Finished convolution layer 1...")

        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool2(x)
        print("Finished convolution layer 2...")

        x = self.conv3(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool3(x)
        print("Finished convolution layer 3...")

        # Put the result into one that is expected by the RNN.
        x = self._preprocess_gru(x)

        self.gru1.flatten_parameters()
        self.gru2.flatten_parameters()
        print("Finished GRU layers 1 and 2...")

        x, h = self.gru1(x)
        x = self.dropout(x)
        x, h, = self.gru2(x)
        x = self.dropout(x)

        print("Start fully connected layer...")
        x = self.fully_connected(x)
        output = x.permute(1, 0, 2).contiguous()
        print("Finished forward pass...")

        return output
