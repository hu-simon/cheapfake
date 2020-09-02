"""
Python file containing the models.

Need to add support for adding in weights to the encoder model.
"""

import os
import time
import enum

import cv2
import torch
import face_alignment
import torch.nn as nn
import torch.nn.functional as F

import cheapfake.contrib.ResNetSE34L as audio_model
import cheapfake.lipnet.models as lipnet

# Uncomment the lines below if the files exist.
lipnet_options = __import__("lipnet_config")
face_alignment_options = __import__("face_alignment_config")

# Delete the line below if the files above exist.
#face_alignment_options = 0

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
            self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=4)
            #self.conv2 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1)
            #self.batchnorm2 = nn.BatchNorm2d(4)
            #self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(1292, 256)
            self.relu = nn.ReLU(inplace=True)
        elif network_type.value == 2:
            self.conv1 = torch.nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
            self.batchnorm1 = torch.nn.BatchNorm2d(4)
            self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=4)
            #self.conv2 = torch.nn.Conv2d(16, 25, kernel_size=3, stride=1, padding=1)
            #self.batchnorm2 = torch.nn.BatchNorm2d(25)
            #self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
            self.flatten = torch.nn.Flatten()
            self.fc1 = torch.nn.Linear(9728, 256)
            #self.fc2 = torch.nn.Linear(
            #    int((25 * 18 * 128) / 4), int((25 * 18 * 128) / 16)
            #)
            #self.fc3 = torch.nn.Linear(
            #    int((25 * 18 * 128) / 16), int((25 * 18 * 128) / 64)
            #)
            #self.fc4 = torch.nn.Linear(int((25 * 18 * 128) / 64), 256)
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
            #x = self.conv2(x)
            #x = self.batchnorm2(x)
            #x = self.maxpool2(x)
            x = self.flatten(x)
            #print(x.shape)
            x = self.fc1(x)
            x = self.relu(x)
        elif self.network_type.value == 2:
            x = self.conv1(x)
            x = self.batchnorm1(x)
            x = self.maxpool1(x)
            #x = self.conv2(x)
            #x = self.batchnorm2(x)
            #x = self.maxpool2(x)
            x = self.flatten(x)
            #print(x.shape)
            x = self.fc1(x)
            x = self.relu(x)
            #x = self.fc2(x)
            #x = self.relu(x)
            #x = self.fc3(x)
            #x = self.relu(x)
            #x = self.fc4(x)
            #x = self.relu(x)
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

        if self.landmarks_type == "2D":
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
        self.encoder_model = FeaturesEncoder(
            network_type=NetworkType.face_alignment
        ).to(self.device)

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
        landmarks = list()
        face_embeddings = list()
        for batch in x:
            landmark = self.face_alignment_model.get_landmarks_from_batch(
                batch.to(self.device)
            )

            # Compute the FAN embeddings.
            # Need to check for multiple landmarks before you put it in to a Torch tensor.
            landmark = torch.Tensor(landmark)
            if landmark.shape[1] != 1:
                landmark = landmark[:, :1, :, :]
            landmarks.append(landmark)
            landmark = landmark.permute(1, 3, 0, 2).float().to(self.device)
            # landmark = landmark[None, :, :, :].float().to(self.device)

            face_embedding = self.encoder_model(landmark.to(self.device))
            face_embeddings.append(face_embedding)

        landmarks = torch.stack(landmarks)
        landmarks = landmarks.squeeze(axis=2).float().to(self.device)

        face_embeddings = torch.stack(face_embeddings).float().to(self.device)

        return landmarks, face_embeddings
        # return landmarks


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
            dropout_rate=self.dropout_rate, verbose=self.verbose
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
        #else:
        #    print(
        #        "[WARNING] Invalid path to pre-trained weights, and thus no weights will be loaded."
         #   )

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
        x = x.permute(1, 0, 2)
        x = x[None, :, :, :].permute(1, 0, 2, 3)
        lipnet_embedding = self.encoder_model(x)
        lipnet_embedding = lipnet_embedding[:, None, :].float().to(self.device)

        return lipnet_embedding

    @staticmethod
    def _find_bounding_box(landmarks, tol=(2, 2, 2, 2)):
        """Computes the minimum bounding box containing ``landmarks``, with tolerance (in pixels) in the left, right, top, and bottom directions.

        Parameters
        ----------
        landmarks : torch.Tensor instance
            Torch tensor containing the xy-coordinates of the facial landmarks.
        tol : tuple, optional
            The tolerance (in pixels) in each direction (left, top, right, bottom) for the cropping operation, by default (2, 2, 2, 2).

        Returns
        -------
        bbox : tuple (of ints)
            Tuple (min_x, min_y, max_x, max_y) containing the coordinates of the minimum bounding box with tolerance (in pixels) in the left, top, right, and bottom directions.

        """
        assert isinstance(tol, tuple)
        assert len(tol) == 4, "Only four values can be specified for the tolerance."

        x_coords, y_coords = zip(*landmarks)
        bbox = (
            min(x_coords) - tol[0],
            min(y_coords) - tol[1],
            max(x_coords) + tol[2],
            max(y_coords) + tol[3],
        )
        bbox = tuple([int(item) for item in bbox])

        return bbox

    @staticmethod
    def _find_bounding_box_batch(batch_landmarks, tol=(2, 2, 2, 2)):
        """Computes the minimum bounding box containing ``batch_landmarks``, with tolerances (in pixels) in the left, right, top, and bottom directions, for a batch of images.

        It is assumed that the batch dimension is the first dimension.

        Parameters
        ----------
        batch_landmarks : torch.Tensor instance
            Torch tensor containing a batch of xy-coordinates of the facial landmarks.
        tol : tuple, optional
            The tolerance (in pixels) in each direction (left, top, right, bottom) for the cropping operation, by default (2, 2, 2, 2).
        
        Returns
        -------
        batch_bbox : list (of tuples)
            List containing tuples of xy-coordinates of the bounding boxes for the batch of facial landmarks.
        
        """
        batch_bbox = list()
        batch_landmarks = batch_landmarks[:, 48:68]
        for landmark in batch_landmarks:
            batch_bbox.append(AugmentedLipNet._find_bounding_box(landmark, tol))

        return batch_bbox

    @staticmethod
    def _crop_lips(frames, landmarks, tol=(2, 2, 2, 2), channels_first=True):
        """Crops the lip area from a batch of frames.

        Parameters
        ----------
        frames : torch.Tensor instance
            Torch tensor instance containing frames for cropping the lip area.
        landmarks : torch.Tensor instance
            Torch tensor containing the xy-coordinates of the facial landmarks.
        tol : tuple, optional
            The tolerance (in pixels) in each direction (left, top, right, bottom) for the cropping operation, by default (2, 2, 2, 2)
        channels_first : bool, optional
            If True then the input and output are assumed to have the channel dimension come before the spatial dimensions.
        
        Returns
        -------
        cropped_frames : torch.Tensor instance
            Torch tensor containing the cropped lip areas.

        """
        assert isinstance(tol, tuple)
        assert len(tol) == 4, "Only four values can be specified for the tolerance."
        assert isinstance(channels_first, bool)

        if channels_first:
            frames = frames.permute(0, 2, 3, 1)

        bboxes = AugmentedLipNet._find_bounding_box_batch(
            batch_landmarks=landmarks, tol=tol
        )

        output_shape = (frames.shape[0], 64, 128, 3)
        extracted_lips = torch.empty(output_shape)
        for idx, (bbox, frame) in enumerate(zip(bboxes, frames)):
            extracted_lip = frame[bbox[1] : bbox[3], bbox[0] : bbox[2], :]
            extracted_lips[idx] = torch.from_numpy(
                cv2.resize(
                    extracted_lip.cpu().numpy(),
                    dsize=(128, 64),
                    interpolation=cv2.INTER_CUBIC,
                )
            )

        return extracted_lips

    @staticmethod
    def _crop_lips_batch(
        batch_frames, batch_landmarks, tol=(2, 2, 2, 2), channels_first=True
    ):
        """Crops the lip area for a batch of batch of frames.

        It is assumed that the batch dimension is the first dimension.

        Parameters
        ----------
        batch_frames : torch.Tensor instance
            Torch tensor containing the batch of frames. 
        batch_landmarks : torch.Tensor instance 
            Torch tensor containing the batch of xy-coordinates of the facial landmarks.
        tol : tuple, optional
            The tolerance (in pixels) in each direction (left, top, right, bottom) for the cropping operation, by default (2, 2, 2, 2)
        channels_first : bool, optional
            If True then the input and output are assumed to have the channel dimension come before the spatial dimensions.

        Returns
        -------
        batch_extracted_lips : torch.Tensor instance
            Torch tensor containing a batch of cropped lip areas.

        """
        assert isinstance(tol, tuple)
        assert len(tol) == 4, "Only four values can be specified for the tolerance"
        assert isinstance(channels_first, bool)

        output_shape = (
            batch_frames.shape[0],
            batch_frames.shape[1],
            64,
            128,
            batch_frames.shape[2],
        )
        batch_extracted_lips = torch.empty(output_shape)
        for idx, (frames, landmarks) in enumerate(zip(batch_frames, batch_landmarks)):
            #print(landmarks.shape)
            batch_extracted_lips[idx] = AugmentedLipNet._crop_lips(
                frames, landmarks, tol=tol, channels_first=channels_first
            )

        return batch_extracted_lips


class AugmentedResNetSE34L(nn.Module):
    """Augmented ResNetSE34L that includes a feature embedding network that acts on the output of ResNetSE34L. 

    """

    def __init__(self, num_out=256, device=torch.device("cpu"), verbose=True, **kwargs):
        """Instantiates a new AugmentedResNetSE34L object.

        Parameters
        ----------
        num_out : int, optional
            The number of output features, by default 256.
        device : torch.device instance
            The device on which all procedures are carried out, by default torch.device("cpu").
        verbose : bool, optional
            If True then verbose output is printed to the system console.
        **kwargs
            Arguments passed onto i2ai.mmid.models.ResNetSE34L.
        
        """
        super(AugmentedResNetSE34L, self).__init__()

        assert isinstance(num_out, int)
        assert isinstance(device, torch.device)
        assert isinstance(verbose, bool)

        self.num_out = num_out
        self.device = device
        self.verbose = verbose

        self.resnet_model = audio_model.ResNetSE34L(nOut=num_out)
        self.resnet_model = self.resnet_model.to(device)

        # No encoder model for now...

    def forward(self, x):
        """Performs a forward pass of the input ``x`` through the network.

        Parameters
        ----------
        x : torch.Tensor instance
            Torch tensor containing the input to the network.

        Returns
        -------
        x : torch.Tensor instance
            Torch tensor containing the audio embeddings.

        """
        x = self.resnet_model(x)

        return x


class MultimodalClassifier(nn.Module):
    """Implementation of the Multimodal Classifier network.

    """

    def __init__(self, device=torch.device("cpu"), verbose=True):
        """Instantiates a new MultimodalClassifier object.

        Parameters
        ----------
        device : torch.device instance
            The device on which all procedures are carried out, by default torch.device("cpu").
        verbose : bool, optional
            If True then verbose output is printed to the system console.

        """
        super(MultimodalClassifier, self).__init__()

        assert isinstance(device, torch.device)
        assert isinstance(verbose, bool)

        self.device = device
        self.verbose = verbose

        # Build the network. Expected input is (batch_size, 3, 1, 256).
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(4)
        self.maxpool1 = nn.MaxPool2d(kernel_size=1, stride=2)
        self.conv2 = nn.Conv2d(4, 4, kernel_size=3, stride=2, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(4)
        self.maxpool2 = nn.MaxPool2d(kernel_size=1, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128, 2)
        #self.fc2 = nn.Linear(64, 32)
        #self.fc3 = nn.Linear(32, 2) 
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax()

    def forward(self, x):
        """Performs a forward pass of the input ``x`` through the network.

        Parameters
        ----------
        x : torch.Tensor instance
            Torch tensor containing the input to the network.
        
        Returns
        -------
        prediction : torch.Tensor instance
            Class label representing whether the video is real or fake.
        
        """
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        #x = self.relu(x)
        x = self.fc1(x)
        #x = self.relu(x)
        #x = self.fc2(x)
        #x = self.relu(x)
        #x = self.fc3(x)
        #x = self.relu(x)

        prediction = self.softmax(x)

        return prediction
