"""
Python script that trains the network.
"""

import os
import time
from pathlib import Path

import cv2
import torch
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader

import cheapfake.contrib.dataset as dataset
import cheapfake.contrib.models_contrib as models
import cheapfake.contrib.transforms as transforms
import cheapfake.contrib.ResNetSE34L as resnet_models

__all__ = ["train_model", "eval_model"]


def save_checkpoints(face_model, frame_model, audio_model, description, filename):
    """Saves the current state of the network weights to a checkpoint file.

    Parameters
    ----------
    face_model : torch.nn.Module instance
        A torch.nn.Module instance containing the model weights of the face alignment/embedding network.
    frame_model : torch.nn.Module instance
        A torch.nn.Module instance containing the model weights of the frames/lips embedding network
    audio_model : torch.nn.Module instance
        A torch.nn.Module instance containing the model weights of the audio embedding network.
    description : str
        String describing the saved checkpoint.
    filename : str
        The name of the file to be saved. The suffix should be included in the filename.

    """
    assert isinstance(face_model, torch.nn.Module)
    assert isinstance(frame_model, torch.nn.Module)
    assert isinstance(audio_model, torch.nn.Module)
    assert isinstance(description, str)
    assert isinstance(filename, str)

    model_state = {
        "description": description,
        "face_model": face_model.state_dict(),
        "frame_model": frame_model.state_dict(),
        "audio_model": audio_model.state_dict(),
    }

    torch.save(state, filename)


'''
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
            frames, _, audio_stft = batch

            frames = frames.float().to(device)
            audio_stft = audio_stft.float().to(device)

            face_model.train()
            frames_model.train()
            audio_model.train()

            optim.zero_grad()

            landmarks, face_embedding = face_model(frames)
            extracted_lips = _crop_lips(frames, landmarks)
            frame_embedding = frames_model(extracted_lips)
            audio_embedding = audio_model(audio_stft.view(audio_stft.shape[0], -1))

            print(
                "\nFace Embedding Size: {}\nFrames Embedding Size: {}\nAudio Embedding Size: {}".format(
                    face_embedding.shape, frame_embedding.shape, audio_embedding.shape
                )
            )

            # Compute the loss and then take the gradient step. Compute some verbose output if you want, especially if the user requests it.
'''


def train_model(
    face_model,
    frame_model,
    audio_model,
    dataloader,
    optimizer,
    criterion,
    num_epochs,
    checkpoint_path,
    device=torch.device("cpu"),
    verbose=True,
):
    """Trains the DeepFake detection model.

    Parameters
    ----------
    face_model : torch.nn.Module instance
        Torch module used to create the face embeddings.
    frame_model : torch.nn.Module instance
        Torch module used to create the lip embeddings.
    audio_model : torch.nn.Module instance
        Torch module used to create the audio embedddings.
    dataloader : torch.utils.data.dataloader.DataLoader instance
        Torch dataloader used to load the training data.
    optimizer : torch.optim instance
        Torch optimizer used for gradient descent.
    num_epochs : int
        The number of epochs for training.
    checkpoint_path : str
        The absolute path to the folder where checkpoints should be stored.
    device : torch.device instance, optional
        The device on which all procedures are carried out.
    verbose : bool, optional
        If True then training statistics are printed to the system console.

    """
    assert isinstance(face_model, torch.nn.Module)
    assert isinstance(frame_model, torch.nn.Module)
    assert isinstance(audio_model, torch.nn.Module)
    assert isinstance(num_epochs, int)
    assert isinstance(checkpoint_path, str)
    assert isinstance(device, torch.device)
    assert isinstance(verbose, bool)

    #face_model = face_model.to(device)
    #frame_model = frame_model.to(device)
    #audio_model = audio_model.to(device)

    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    losses = list()

    if verbose is False:
        progress_bar = tqdm(total=len(dataloader))
    for epoch in range(num_epochs):
        face_model.train()
        frame_model.train()
        audio_model.train()
        
        for batch_idx, batch in enumerate(dataloader):
            face_model.train()
            frame_model.train()
            audio_model.train()

            frames, _, audio_stft, label = batch
            frames = frames[:, :75]
            frames = frames.float().to(device)
            audio_stft = audio_stft.view(audio_stft.shape[0], -1).float().to(device)

            #optim.zero_grad()
            
            print("Going through FAN")
            landmarks, face_embeddings = face_model(frames)
            
            print("Going through LipNet")
            extracted_lips = models.AugmentedLipNet._crop_lips_batch(frames, landmarks)
            extracted_lips = extracted_lips.permute(0, -1, 1, 2, 3).float().to(device)
            frame_embeddings = frame_model(extracted_lips)
            
            print("Going through ResNet")
            audio_embeddings = audio_model(audio_stft)

            # Concatenate the embeddings together.
            concat_embeddings = (
                torch.cat(
                    (face_embeddings, frame_embeddings, audio_embeddings[:, None, :]),
                    axis=1,
                )
                .float()
                .to(device)
            )

            print(concat_embeddings.shape)


if __name__ == "__main__":
    random_seed = 41
    metadata_path = (
        "/home/shu/cheapfake/cheapfake/contrib/wide_balanced_metadata_fs03.csv"
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))

    dfdataset = dataset.DeepFakeDataset(
        metadata_path=metadata_path,
        frame_transform=transforms.BatchRescale(4),
        sequential_audio=True,
        sequential_frames=True,
        random_seed=random_seed,
        num_samples=1,
    )
    dfdataloader = DataLoader(dfdataset, batch_size=1, shuffle=True)
    checkpoint_path = "./checkpoints"

    optimizer = 0
    criterion = 0
    num_epochs = 5

    face_model = models.AugmentedFAN(device=device)
    frame_model = models.AugmentedLipNet(device=device)
    audio_model = models.AugmentedResNetSE34L(device=device)

    train_model(
        face_model=face_model,
        frame_model=frame_model,
        audio_model=audio_model,
        dataloader=dfdataloader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=num_epochs,
        checkpoint_path=checkpoint_path,
        device=device
    )

