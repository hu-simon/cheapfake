from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn


__all__ = ['train_mmid', 'test_mmid']


def save_checkpoint(frame_model, audio_model, description, filename='checkpoint.pth.tar'):
    """ Saves input state dict to file
    Parameters
    ----------
    state : dict
        State dict to save. Can include parameters from model, optimizer, etc.
        as well as any other elements.
    is_best : bool
        If true will save current state dict to a second location
    filename : str
        File name for save
    Returns
    -------
    """
    state = {
        'description': description,
        'frame model': frame_model.state_dict(),
        'audio model': audio_model.state_dict()
    }
    torch.save(state, filename)


def get_stats(distribution):
    """ Calculates distribution stats
    Parameters
    ----------
    distribution : torch.tensor
        Embeddings L2 distance distributions
    Returns
    -------
    mean : float
        Distribution mean
    std : float
        Distribution standard deviation
    """
    total = len(distribution)
    mean = torch.sum(distribution)/total
    std = torch.sqrt(torch.sum((distribution-mean)**2)/total)
    return mean, std


def make_dist_plots(positive_dist, negative_dist, filename=''):
    positive_stats = get_stats(positive_dist)
    negative_stats = get_stats(negative_dist)
    fig = plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    lbl = 'mean: {:.2f} std: {:.2f}'.format(
        positive_stats[0].cpu().item(), positive_stats[1].cpu().item())
    sns.distplot(positive_dist.cpu().numpy(), label=lbl)
    plt.xlabel('Embedding distance')
    plt.title('Positive pairs')
    plt.legend()
    plt.subplot(1, 2, 2)
    lbl = 'mean: {:.2f} std: {:.2f}'.format(
        negative_stats[0].cpu().item(), negative_stats[1].cpu().item())
    sns.distplot(negative_dist.cpu().numpy(), label=lbl)
    plt.legend()
    plt.xlabel('Embedding distance')
    plt.title('Negative pairs')
    plt.savefig(filename)


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


def test_mmid(frame_model=None, audio_model=None, testloader=None,
              device='cpu', val_bs=1000):
    """ Test multimodal embeddings
    Parameters
    ----------
    frame_model : torch.Module
        Frame embeddigns model
    audio_model : torch.Module
        Audio embeddings model
    testloader : torch.utils.data.DataLoader
        Dataloader used for evaluation
    device : str
        Device to run eval
    val_bs : int
        Batch size for evaluation
    Returns
    -------
    true_scores : torch.tensor
        L2 distances between true matches
    false_scores : torch.tensor
        L2 distances between false matches
    """
    device = torch.device(device)
    frame_model.eval()
    frame_model.to(device)
    audio_model.eval()
    audio_model.to(device)
    print('Buidling_features')
    for idx, batch in enumerate(testloader):
        frames, audio, ids = batch
        frames, audio, ids = frames.to(
            device), audio.to(device), ids.to(device)
        with torch.no_grad():
            if idx == 0:
                frame_features = frame_model(frames).cpu()
                audio_features = audio_model(
                    audio.view(audio.shape[0], -1)).cpu()
                pids = ids
            else:
                frame_features = torch.cat(
                    [frame_features, frame_model(frames).cpu()], dim=0)
                audio_features = torch.cat([audio_features,
                                            audio_model(audio.view(audio.shape[0], -1)).cpu()], dim=0)
                pids = torch.cat([pids, ids], dim=0)
    print('Computing scores')
    bs = 10000  # smaller batchsize due to memoery limitation
    truth = torch.eq(ids, ids.unsqueeze(1))

    for idx in range(len(frame_features)//bs+1):
        upper = min(bs*(idx+1), len(frame_features))
        frame_batch = frame_features[bs*idx: upper].cuda()
        audio_batch = audio_features[bs*idx: upper].cuda()
        truth = torch.eq(pids[bs*idx: upper], pids[bs*idx: upper].unsqueeze(1))
        # tensors are already normed
        scores = torch.mm(frame_batch, audio_batch.transpose(0, 1))
        if idx == 0:
            true_scores = scores[truth == True].view(-1)
            false_scores = scores[truth == False].view(-1)
        else:
            true_scores = torch.cat(
                [true_scores, scores[truth == True].view(-1)], dim=0)
            false_scores = torch.cat(
                [false_scores, scores[truth == False].view(-1)], dim=0)
    return true_scores, false_scores


def train_mmid(frame_model=None, audio_model=None, trainloader=None,
               testloader=None, optim=None, scheduler=None,
               criterion=None, difficulty_scheduler=None, save_dir='.',
               n_epochs=0, e_saves=1, device='cpu', verbose=False):
    """ Training routing for deep fake detector
    Parameters
    ----------
    model : torch.Module
        Deep fake detector model
    dataloader : torch.utils.data.DataLoader
        Training dataset
    optim : torch.optim
        Optimizer for pytorch model
    scheduler : torch.optim.lr_scheduler
        Optional learning rate scheduler for the optimizer
    criterion : torch.nn.Module
        Objective function for optimization
    losses : list
        List to hold the lossses over each mini-batch
    averages : list
        List to hold the average loss over each epoch
    n_epochs : int
        Number of epochs for training
    device : str
        Device to run training procedure
    verbose : bool
        Verbose switch to print losses at each mini-batch
    """
    device = torch.device(device)
    frame_model = frame_model.to(device)
    audio_model = audio_model.to(device)
    meter = AverageMeter()
    best_loss = 999
    chpt_path = '{}/checkpoints'.format(save_dir)
    log_path = '{}/logs'.format(save_dir)
    Path(chpt_path).mkdir(parents=True, exist_ok=True)
    Path(log_path).mkdir(parents=True, exist_ok=True)
    losses = []
    positive_means = []
    positive_std = []
    negative_means = []
    negative_std = []

    if verbose is False:
        pbar = tqdm(total=len(trainloader))
    for epoch in range(n_epochs):
        frame_model.train()
        audio_model.train()
        tau = difficulty_scheduler()
        for i_batch, batch in enumerate(trainloader):
            frames, audio = batch
            frames, audio = frames.to(device), audio.float().to(device)
            frame_model.train()
            audio_model.train()
            optim.zero_grad()
            # Establish shared key
            frame_embeddings = frame_model(frames)
            audio_embeddings = audio_model(audio.view(audio.shape[0], -1))
            # print(predictions.shape)
            # print(predictions, lbls)
            loss = criterion(frame_embeddings, audio_embeddings, tau)
            loss.backward()
            optim.step()
            losses.append(loss.item())
            meter.update(loss.item(), frames.shape[0])
            if verbose:
                print(
                    '[{}/{}] Message loss:{:.4f} '.format(i_batch,
                                                          len(trainloader),
                                                          loss.item()))
            else:
                pbar.update(1)
        if (epoch+1) % e_saves == 0 or meter.avg < best_loss:
            print('Validating model')
            positive_scores, negative_scores = test_mmid(frame_model=frame_model,
                                                         audio_model=audio_model,
                                                         testloader=testloader,
                                                         device=device)
            positive_stats = get_stats(positive_scores)
            negative_stats = get_stats(negative_scores)
            #
            positive_means.append(positive_stats[0])
            positive_std.append(positive_stats[1])
            negative_means.append(negative_stats[0])
            negative_std.append(negative_stats[1])
            if meter.avg < best_loss:
                save_checkpoint(frame_model, audio_model, 'Epoch {} Loss:{} Positive score :{} Negative score:{}'.format(
                    epoch, loss.item(), positive_means, negative_means), '{}/best_model.pth.tar'.format(chpt_path))
                make_dist_plots(positive_scores, negative_scores,
                                '{}/best_results.jpg'.format(log_path))

            if (epoch+1) % e_saves == 0:
                save_checkpoint(frame_model, audio_model, 'Epoch {} Loss:{} Positive score :{} Negative score:{}'.format(
                    epoch, loss.item(), positive_means, negative_means), '{}/model_epoch_{}.pth.tar'.format(chpt_path, epoch+1))
                make_dist_plots(positive_scores, negative_scores,
                                '{}/results_{}.jpg'.format(log_path, epoch+1))
                losses.append(meter.avg)
        difficulty_scheduler.__step__()
        if verbose is False:
            pbar.refresh()
            pbar.reset()
        if scheduler is not None:
            scheduler.step(meter.avg)
        meter.reset()
    pbar.close()
