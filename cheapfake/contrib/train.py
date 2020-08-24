"""
Python script that trains the network.
"""

import os
import time
from pathlib import Path

import torch
import numpy as np
import torch.nn as nn

from tqdm import tqdm
import matplotlib.pyplot as plt

__all__ = ["train_model", "eval_model"]

"""
The models are going to have to be combined with the encoder. So instead of one large CheapFake model, we can have three separate models but each of them are going to have to 
"""


def _find_bounding_box(landmarks, tol=(2, 2, 2, 2)):
    """Finds the minimum bounding box containing the points, with tolerance in the left, right, top, and bottom directions (in pixels).

    Parameters
    ----------
    landmarks : numpy.ndarray or torch.Tensor instance
        Numpy array or Torch tensor containing the predicted xy-coordinates of the detected facial landmarks.
    tol : tuple, optional
        The tolerance (in pixels) in each direction (left, top, right, bottom) by default (2, 2, 2, 2). 
    
    Returns
    -------
    bbox : tuple (of ints)
        Tuple (min_x, min_y, max_x, max_y) containing the coordinates of the bounding box, with tolerance in the left, right, top and bottom directions. 

    """
    assert isinstance(tol, tuple)
    assert len(tol) == 4, "Need four values for the tolerance."

    x_coords, y_coords = zip(*landmarks)
    bbox = (
        min(x_coords) - tol[0],
        min(y_coords) - tol[1],
        max(x_coords) + tol[2],
        max(y_coords) + tol[3],
    )
    bbox = tuple([int(item) for item in bbox])

    return bbox


def _find_bounding_boxes(landmarks, tol=(2, 2, 2, 2)):
    """Finds the minimum bounding boxes for a batch of facial landmarks.

    Parameters
    ----------
    landmarks : numpy.ndarray or torch.Tensor instance
        Numpy array or Torch tensor containing the xy-coordinates of the detected facial landmarks, in batches.
    tol : tuple, optional
        The tolerance (in pixels) in each direction (left, top, right, bottom) by default (2, 2, 2,2).
    
    Returns
    -------
    bboxes : list (of tuples)
        List containing tuples containing the coordinates of the bounding boxes for the batch of landmarks.

    """
    bboxes = list()
    landmarks = landmarks[:, 48:68]
    for landmark in landmarks:
        bboxes.append(_find_bounding_box(landmarks, tol))

    return bboxes


def _crop_lips(frames, landmarks, tol=(2, 2, 2, 2), channels_first=True):
    """Crops the lip area from a batch of frames.

    Parameters
    ----------
    frames : torch.Tensor instance
        Torch tensor instance containing the frames to crop the lip areas from.
    landmarks : numpy.ndarray or torch.Tensor instance
        Numpy array or Torch tensor containing the xy-coordinates of the detected facial landmarks.
    tol : tuple, optional
        The tolerance (in pixels) in each direction (left, top, right, bottom) by default (2, 2, 2, 2).
    channels_first : bool, optional
        If True then the input and output are assumed to have shape (sample, channel, height, width), by default True. Otherwise the input and output are assumed to have shape (sample, height, width, channel).

    Returns
    -------
    cropped_frames : numpy.ndarray or torch.Tensor instance
        Numpy array or Torch tensor containing the cropped lips.

    """
    assert isinstance(frames, torch.Tensor)
    assert isinstance(landmarks, (torch.Tensor, np.ndarray))
    assert isinstance(tol, tuple)
    assert isinstance(channels_first, bool)

    if channels_first:
        frames = frames.permute(0, 2, 3, 1)

    bboxes = _find_bounding_boxes


def save_checkpoints(face_model, frames_model, audio_model, description, filename):
    """Saves the current state of the network weights to a checkpoint file.

    Parameters
    ----------
    face_model : torch.nn.Module instance
        A torch.nn.Module instance containing the model weights of the face alignment/embedding network.
    frames_model : torch.nn.Module instance
        A torch.nn.Module instance containing the model weights of the frames/lips embedding network
    audio_model : torch.nn.Module instance
        A torch.nn.Module instance containing the model weights of the audio embedding network.
    description : str
        String describing the saved checkpoint.
    filename : str
        The name of the file to be saved. The suffix should be included in the filename.

    """
    assert isinstance(face_model, torch.nn.Module)
    assert isinstance(frames_model, torch.nn.Module)
    assert isinstance(audio_model, torch.nn.Module)
    assert isinstance(description, str)
    assert isinstance(filename, str)

    model_state = {
        "description": description,
        "face_model": face_model.state_dict(),
        "frames_model": frames_model.state_dict(),
        "audio_model": audio_model.state_dict(),
    }

    torch.save(state, filename)


def train_model(
    face_model,
    frames_model,
    audio_model,
    dataloader,
    optimizer,
    criterion,
    num_epochs,
    device=torch.device("cpu"),
    verbose=True,
):
    """Trains the DeepFake detection model.

    Parameters
    ----------
    face_model : torch.nn.Module instance
        A torch.nn.Module instance used to create the face embeddings.
    frames_model : torch.nn.Module instance
        A torch.nn.Module instance used to create the frames/lips embeddings.
    audio_model : torch.nn.Module instance
        A torch.nn.Module instance used to create the audio embeddings.
    dataloader : torch.utils.data.dataloader.DataLoader instance
        Torch dataloader used for loading the training data.
    optimizer : torch.optim instance
        Torch optimizer function used for gradient descent.
    criterion : torch.nn Loss Function instance
        A torch.nn loss function used for gradient descent.
    num_epochs : int
        The number of epochs for training.
    device : torch.device instance
        The device on which all the computations are done.
    verbose : {True, False}, bool, optional
        If True, then training statistics are printed to the system console.
    
    """
    assert isinstance(frame_model, torch.nn.Module)
    assert isinstance(lip_model, torch.nn.Module)
    assert isinstance(audio_model, torch.nn.Module)
    assert isinstance(device, torch.device)
    assert isinstance(verbose, bool)

    frame_model = frame_model.to(device)
    lip_model = lip_model.to(device)
    audio_model = audio_model.to(device)

    checkpoint_path = "./checkpoints"
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    losses = list()

    if verbose is False:
        progress_bar = tqdm(total=len(dataloader))
    for epoch in range(n_epochs):
        face_model.train()
        frames_model.train()
        audio_model.train()

        for batch_idx, batch in enumerate(dataloader):
            frames, audio, audio_stft = batch

            frames = frames.float().to(device)
            audio = audio.float().to(device)
            audio_stft = audio.float().to(device)

            face_model.train()
            frames_model.train()
            audio_model.train()

            optim.zero_grad()

            landmarks, face_embedding = face_model(frames)
            extracted_lips = _crop_lips(frames, landmarks)
            frames_embedding = frames_model(extracted_lips)

            audio_embedding = audio_model(audio_stft.view(audio_stft.shape[0], -1))

            print(
                "\nFace Embedding Size: {}\nFrames Embedding Size:{}\nAudio Embeddings Size:{}".format(
                    face_embedding.size, frames_embedding.size, audio_embedding.size
                )
            )
