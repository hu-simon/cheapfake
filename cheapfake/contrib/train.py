"""
Python script that trains the network.
"""

import os
import time
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

import cheapfake.contrib.dataset as dataset
import cheapfake.contrib.torch_gc as torch_gc
import cheapfake.contrib.models_contrib as models
import cheapfake.contrib.transforms as transforms
import cheapfake.contrib.ResNetSE34L as resnet_models

__all__ = ["train_model", "eval_model"]


def save_checkpoints(
    face_model, frame_model, audio_model, combination_model, description, filename
):
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
    assert isinstance(combination_model, torch.nn.Module)
    assert isinstance(description, str)
    assert isinstance(filename, str)

    model_state = {
        "description": description,
        "face_model": face_model.state_dict(),
        "frame_model": frame_model.state_dict(),
        "audio_model": audio_model.state_dict(),
        "combination_model": combination_model.state_dict(),
    }

    torch.save(model_state, filename)


# From Michaels' MMID code.
class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """

    def __init__(self):
        """ Initialize objects and reset for safety
        Parameters
        ----------
        Returns
        -------
        """
        self.reset()

    def reset(self):
        """ Resets the meter values if being re-used
        Parameters
        ----------
        Returns
        -------
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update meter values give current value and batchsize
        Parameters
        ----------
        val : float
            Value fo metric being tracked
        n : int
            Batch size
        Returns
        -------
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def eval_model(
    face_model,
    frame_model,
    audio_model,
    combination_model,
    dataloader,
    device=torch.device("cpu"),
    verbose=False,
):
    pass


def train_model(
    face_model,
    frame_model,
    audio_model,
    combination_model,
    dataloader,
    optimizer,
    criterion,
    num_epochs,
    checkpoint_path,
    save_freq,
    eval_freq,
    device=torch.device("cpu"),
    verbose=False,
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
    combination_model :torch.nn.Module instance
        Torch module used to combine the embeddings and produces class labels.
    dataloader : torch.utils.data.dataloader.DataLoader instance
        Torch dataloader used to load the training data.
    optimizer : torch.optim instance
        Torch optimizer used for gradient descent.
    num_epochs : int
        The number of epochs for training.
    checkpoint_path : str
        The absolute path to the folder where checkpoints should be stored.
    save_freq : int
        The frequency for saving model checkpoints.
    eval_freq : int
        The frequency for evaluating the model.
    device : torch.device instance, optional
        The device on which all procedures are carried out.
    verbose : bool, optional
        If True then training statistics are printed to the system console.

    """
    assert isinstance(face_model, torch.nn.Module)
    assert isinstance(frame_model, torch.nn.Module)
    assert isinstance(audio_model, torch.nn.Module)
    assert isinstance(combination_model, torch.nn.Module)
    assert isinstance(num_epochs, int)
    assert isinstance(checkpoint_path, str)
    assert isinstance(save_freq, int)
    assert isinstance(eval_freq, int)
    assert isinstance(device, torch.device)
    assert isinstance(verbose, bool)

    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    losses = list()

    meter = AverageMeter()

    if verbose is False:
        print("".join("-" * 80))
        print("Epoch 0")
        print("".join("-" * 80))
        progress_bar = tqdm(total=len(dataloader))
    for epoch in range(num_epochs):
        face_model.train()
        frame_model.train()
        audio_model.train()
        combination_model.train()

        for batch_idx, batch in enumerate(dataloader):
            face_model.train()
            frame_model.train()
            audio_model.train()
            combination_model.train()

            frames, audio_stft, label = batch
            frames = frames[:, :75]
            frames = frames.float().to(device)
            audio_stft = audio_stft.view(audio_stft.shape[0], -1).float().to(device)
            label = label.to(device)
            n_frames = frames.shape[0]

            try:
                landmarks, face_embeddings = face_model(frames)

                extracted_lips = models.AugmentedLipNet._crop_lips_batch(
                    frames, landmarks
                )
                extracted_lips = (
                    extracted_lips.permute(0, -1, 1, 2, 3).float().to(device)
                )
                frame_embeddings = frame_model(extracted_lips)

                audio_embeddings = audio_model(audio_stft)

                concat_embeddings = torch.cat(
                    (face_embeddings, frame_embeddings, audio_embeddings[:, None, :]),
                    axis=1,
                )
                concat_embeddings = concat_embeddings[:, :, None, :].float().to(device)
                prediction = combination_model(concat_embeddings)

                del frames
                del audio_stft
                del extracted_lips
                del face_embeddings
                del frame_embeddings
                del audio_embeddings
                del concat_embeddings

                torch.cuda.empty_cache()

                optimizer.zero_grad()

                loss = criterion(prediction, label)
                loss.backward()

                optimizer.step()
                losses.append(loss.item())
                meter.update(loss.item(), n_frames)

            except (ValueError, TypeError):
                print("No landmarks detected")
                pass
            finally:
                if verbose:
                    pass
                else:
                    progress_bar.update(1)
                torch.cuda.empty_cache()

        if (epoch + 1) % save_freq == 0:
            # Save the model.
            print("Saving model weights.")
            description = "Epoch: {}, Loss: {}".format(epoch, meter.avg)
            filename = "checkpoint_{}.pth".format(epoch)
            save_checkpoints(
                face_model,
                frame_model,
                audio_model,
                combination_model,
                description,
                filename,
            )
        """ Need to write the evaluation script first.
        if (epoch + 1) % eval_freq == 0:
            # Evaluate the model.
            print("Evaluating the model")
            description = 
        """

        if verbose is False:
            print("".join("-" * 80))
            print("Epoch {}".format(epoch + 1))
            print("".join("-" * 80))
            progress_bar.refresh()
            progress_bar.reset()

    losses = np.array(losses)
    np.save("./losses", losses)
    progress_bar.close()


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
        num_samples=100,
    )
    dfdataloader = DataLoader(dfdataset, batch_size=1, shuffle=True)
    checkpoint_path = "./checkpoints"

    face_model = models.AugmentedFAN(device=device)
    frame_model = models.AugmentedLipNet(device=device, verbose=False)
    audio_model = models.AugmentedResNetSE34L(device=device)
    combination_model = models.MultimodalClassifier(device=device).to(device)

    """
    # Implement data parallelism here.
    face_model = nn.DataParallel(face_model)
    frame_model = nn.DataParallel(frame_model)
    audio_model = nn.DataParallel(audio_model)
    combination_model = nn.DataParallel(combination_model)
    """

    params_list = (
        list(face_model.parameters())
        + list(frame_model.parameters())
        + list(audio_model.parameters())
        + list(combination_model.parameters())
    )
    optimizer = optim.SGD(params_list, lr=0.01)
    criterion = nn.BCELoss()
    num_epochs = 5

    train_model(
        face_model=face_model,
        frame_model=frame_model,
        audio_model=audio_model,
        combination_model=combination_model,
        dataloader=dfdataloader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=num_epochs,
        checkpoint_path=checkpoint_path,
        device=device,
        save_freq=1,
        eval_freq=1
    )

