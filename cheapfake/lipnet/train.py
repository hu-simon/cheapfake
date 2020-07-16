"""
Python script that trains the LipNet.
"""

import os
import time

import json
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

import cheapfake.lipnet.models as models
import cheapfake.lipnet.losses as losses
import cheapfake.lipnet.dataset as dataset

options = __import__("config")
os.environ["CUDA_VISIBLE_DEVICES"] = options.gpu


def _get_learning_rate(optimizer):
    """
    Gets the mean learning rates of an optimizer (torch.optim instance).

    Parameters
    ----------
    optimizer : torch.optim instance
        The optimizer object used to find the mean learning rates.

    Returns
    -------
    mean_lr : float
        The mean learning rate of the optimizer.
    """
    learning_rates = list()
    for param_group in optimizer.param_groups:
        learning_rates.append(param_group["lr"])

    return np.array(learning_rates).mean()


def train(model, writer):
    """
    Trains the network.

    Parameters
    ----------

    Returns
    -------
    """
    print("Loading dataset...")
    lipnet_dataset = dataset.LipNetDataset(
        options.video_path,
        options.annotations_path,
        options.train_list,
        options.video_padding,
        options.text_padding,
        "train",
    )
    print("Finished loading dataset...")
    print("Creating dataloader object...")
    dataloader = dataset.create_dataloader(
        lipnet_dataset, batch_size=options.batch_size, num_workers=options.num_workers
    )
    print("Finished creating dataloader object...")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=options.base_learning_rate,
        weight_decay=0.0,
        amsgrad=True,
    )

    criterion = nn.CTCLoss()
    train_word_error_rate = list()

    start_time = time.time()
    for epoch in range(options.max_epochs):
        for (iteration, data) in enumerate(dataloader):
            model.train()

            """ Uncomment below if using CUDA-enabled device.
            video = data.get("video").cuda()
            video_length = data.get("video_length").cuda()
            text = data.get("text").cuda()
            text_length = data.get("text_length").cuda()
            """

            video = data.get("video")
            video_length = data.get("video_length")
            text = data.get("text")
            text_length = data.get("text_length")

            print("".join(101 * "-"))
            print("STARTING PREDICTION")
            prediction = model(video)

            optimizer.zero_grad()
            loss = criterion(
                prediction.transpose(0, 1).log_softmax(-1),
                text,
                video_length.view(-1),
                text_length.view(-1),
            )
            loss.backward()
            if options.optimize:
                optimizer.step()

            total_iterations = iteration + epoch * len(dataloader)

            truth_text = [
                dataset.LipNetDataset.array_to_text(
                    text[_], start_index=1, ctc_mode=True
                )
                for _ in range(text.size(0))
            ]
            predicted_text = [
                dataset.LipNetDataset.array_to_text(
                    prediction.argmax(-1)[_], start_index=1, ctc_mode=True
                )
                for _ in range(prediction.size(0))
            ]
            train_word_error_rate.extend(
                losses.word_error_rate(truth_text, predicted_text)
            )

            if total_iterations % options.train_write_rate == 0:
                write_train_summary(
                    writer,
                    {
                        "start_time": start_time,
                        "total_iter": total_iterations,
                        "current_iter": iteration,
                        "dataloader": dataloader,
                        "loss": loss,
                        "train_word_error_rate": train_word_error_rate,
                        "truth_text": truth_text,
                        "predicted_text": predicted_text,
                        "epoch": epoch,
                    },
                )

            if (total_iterations % options.validation_write_rate == 0) and (
                total_iterations != 0
            ):
                write_validation_summary(
                    writer, {"total_iters": total_iters, "optimizer": optimizer}
                )


def write_train_summary(writer, info, verbose=True, mode="train"):
    """ Writes to the writer, a summary of the training results.
    
    Optionally, verbose comments about the training progress, is printed.

    Parameters
    ----------
    writer : tensorboardX.SummaryWriter instance
        Writer object that stores training information.
    info : dictionary
        Dictionary containing training information. Available information is
        {
            "start_time", 
            "total_iter",
            "current_iter", 
            "dataloader", 
            "loss", 
            "train_word_error_rate", 
            "truth_text", 
            "predicted_text", 
            "epoch", 
        }
    verbose : {True, False}, optional
        Boolean determining whether or not verbose information about the training progress is shown, by default True.
    mode : {"train", "eval"}, optional
        The current mode, either training or evaluation, by default "train".

    Returns
    -------
    None
    """
    v = 1.0 * (time.time() - info["start_time"]) / (info["total_iter"] + 1)
    eta = (len(info["dataloader"]) - info["current_iter"]) * v / 3600.0

    writer.add_scalar("train_loss", info["loss"], info["total_iter"])
    writer.add_scalar(
        "train_word_error_rate",
        np.array(info["train_word_error_rate"]).mean(),
        info["total_iter"],
    )

    if verbose:
        print("".join(101 * "-"))
        print("{:<50}|{:>50}".format("truth", "predicted"))
        print("".join(101 * "-"))

        for (truth, predict) in list(zip(info["truth_text"], info["predicted_text"]))[
            :3
        ]:
            print("{:<50}|{:>50}".format(truth, predict))
        print("".join(101 * "-"))
        print(
            "epoch: {}, total_iterations: {}, eta: {}, loss: {}, train_word_error_rate: {}".format(
                info["epoch"],
                info["total_iter"],
                eta,
                info["loss"],
                np.array(info["train_word_error_rate"]).mean(),
            )
        )
        print("".join(101 * "-"))


def write_validation_summary(writer, info, verbose=True, save=True):
    """ Writes to the writer, a summary of the validation results.

    Optionally, verbose comments about the training process, is printed.

    Parameters
    ----------
    writer : tensorboardX.SummaryWriter instance
        Writer object that stores training information.
    info : dictionary
        Dictionary containing training information. Available information is
        {
            "total_iters", 
            "optimizer", 
        }
    verbose : {True, False}, bool, optional
        Boolean determining whether or not verbose information about the validation progress is shown, by default True.
    save : {True, False}, bool, optional
        Boolean determining whether or not the validation results are saved. 

    Returns
    -------
    None
    """
    (loss, word_error_rate, char_error_rate) = eval(model, writer)
    if verbose:
        print(
            "iter: {}, learning_rate: {}, word_error_rate: {}, char_error_rate: {}".format(
                info["total_iters"],
                _get_learning_rate(info["optimizer"]),
                loss,
                word_error_rate,
                char_error_rate,
            )
        )
    writer.add_scalar("val_loss", loss, info["total_iter"])
    writer.add_scalar("word_error_rate", word_error_rate, info["total_iter"])
    writer.add_scalar("char_error_rate", char_error_rate, info["total_iter"])

    if save:
        savename = "{}_loss_{}_wer_{}_cer_{}.pt".format(
            options.save_prefix, loss, word_error_rate, char_error_rate,
        )
        (path, name) = os.path.split(savename)
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(model.state_dict(), savename)


def write_evaluation_summary(writer, info, verbose=True):
    """ Writers to the writer, a summary of the evaluation results.

    Optionally, verbose comments about the evaluation process, is printed.

    Parameters
    ----------
    writer : tensorboardX.SummaryWriter instance
        Writer object that stores training information.
    info : dictionary
        Dictionary containing the required information to plot the results. The dictionary contains: 
        {
            "start_time",
            "current_iter",
            "dataloader",
            "truth_text",
            "predicted_text",
            "word_error_rate",
            "char_error_rate"
        }
    verbose : {True, False}, bool, optional
        Boolean determining whether or not verbose information about the validation progress is shown, by default True.
    """
    v = 1.0 * (time.time() - info["start_time"]) / (info["current_iter"] + 1)
    eta = v * (len(info["dataloader"]) - info["current_iter"]) / 3600.0

    if verbose:
        print("".join(101 * "-"))
        print("{:<50}|{:>50}".format("truth, predicted"))
        print("".join(101 * "-"))
        for (truth, prediction) in list(
            zip(info["truth_text"], info["predicted_text"])
        )[:10]:
            print("{:<50}|{:>50}".format(truth, prediction))
        print("".join(101 * "-"))
        print(
            "test_iter: {}, eta: {}, word_error_rate {}, char_error_rate: {}".format(
                info["current_iter"],
                eta,
                np.array(info["word_error_rate"]).mean(),
                np.array(info["char_error_rate"]).mean(),
            )
        )
        print("".join(101 * "-"))


def eval(model, writer):
    """
    Evaluates the performance of the trained network.

    Parameters
    ----------
    TODO

    Returns
    -------
    TODO
    """
    # Finish this code.
    with torch.no_grad():
        model.eval()

        lipnet_dataset = dataset.LipNetDataset(
            options.video_path,
            options.annotations_path,
            options.train_list,
            options.video_padding,
            options.text_padding,
            "eval",
        )
        dataloader = dataset.create_dataloader(
            lipnet_dataset,
            batch_size=options.batch_size,
            num_workers=options.num_workers,
        )

        loss_list = list()
        word_error_rate = list()
        char_error_rate = list()

        criterion = nn.CTCLoss()
        start_time = time.time()

        for (iteration, data) in enumerate(dataloader):

            """ Uncomment if you are using a CUDA-enabled device.
            video = data.get("video").cuda()
            video_length = data.get("video_length").cuda()
            text = data.get("text").cuda()
            text_length = data.get("text_length").cuda()
            """

            video = data.get("video")
            video_length = data.get("video_length")
            text = data.get("text")
            text_length = data.get("text_length")

            prediction = model(video)

            loss = (
                criterion(
                    prediction.transpose(0, 1).log_softmax(-1),
                    text,
                    video_length.view(-1),
                    text_length.view(-1),
                )
                .detach()
                .cpu()
                .numpy()
            )
            loss_list.append(loss)

            truth_text = [
                dataset.LipNetDataset.array_to_text(text[_], start_index=1)
                for _ in range(text.size(0))
            ]
            predicted_text = [
                dataset.LipNetDataset.array_to_text(
                    prediction.argmax(-1)[_], start_index=1, ctc_mode=True
                )
                for _ in range(prediction.size(0))
            ]
            word_error_rate.extend(losses.word_error_rate(truth_text, predicted_text))
            char_error_rate.extend(losses.char_error_rate(truth_text, predicted_text))

            if iteration % options.test_write_rate == 0:
                write_evaluation_summary(
                    writer,
                    {
                        "start_time": start_time,
                        "dataloader": dataloader,
                        "truth_text": truth_text,
                        "predicted_text": predicted_text,
                        "word_error_rate": word_error_rate,
                        "char_error_rate": char_error_rate,
                    },
                )

        return (
            np.array(loss_list).mean(),
            np.array(word_error_rate).mean(),
            np.array(char_error_rate).mean(),
        )


if __name__ == "__main__":
    torch.manual_seed(options.random_seed)
    torch.cuda.manual_seed_all(options.random_seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_type = device.type

    model = models.LipNet().to(device)
    writer = SummaryWriter()

    # Load weights if they are available.
    if hasattr(options, "weights"):
        pretrained_dict = torch.load(options.weights, map_location=device)
        pretrained_dict["fully_connected.weight"] = pretrained_dict.pop("FC.weight")
        pretrained_dict["fully_connected.bias"] = pretrained_dict.pop("FC.bias")
        model_dict = model.state_dict()
        pretrain_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict.keys() and v.size() == model_dict[k].size()
        }
        missed_params = [
            k for k, v in model_dict.items() if not k in pretrained_dict.keys()
        ]
        print(
            "Loaded parmeters / Total parameters: {}/{}".format(
                len(pretrained_dict), len(model_dict)
            )
        )
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    train(model, writer)
