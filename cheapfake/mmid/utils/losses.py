import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# Losses inherited from https://github.com/adambielski/siamese-triplet
__all__ = ['OnlineContrastiveLoss', 'OnlineTripletLoss',
           'CurriculumPairMining', 'DifficutyScheduler']


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class OnlineContrastiveLoss(nn.Module):
    """
    Online Contrastive loss
    Takes a batch of embeddings and corresponding labels.
    Pairs are generated using pair_selector object that take embeddings and targets and return indices of positive
    and negative pairs
    """

    def __init__(self, margin, pair_selector):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector

    def forward(self, frame_embeddings, audio_embeddings, tau=0.):
        negative_pairs = self.pair_selector.get_pairs(
            frame_embeddings, audio_embeddings, tau)
        positive_loss = (frame_embeddings - audio_embeddings).pow(2).sum(1)
        negative_loss = F.relu(
            self.margin - (frame_embeddings - negative_pairs).pow(2).sum(
                1).sqrt()).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)
        return loss.mean()


class OnlineTripletLoss(nn.Module):
    """ BROKEN!!!!! NEEDS FIXING
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] -
                        embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] -
                        embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)


class CurriculumPairMining:
    def __init__(self):
        pass

    def get_pairs(self, inputs_1, inputs_2, tau=1.0):
        """
        Curriculum Pair Mining
        Takes a batch of face and voice embeddings and selects a negative pair
        given a diffuculty parameter tau where tau = 1.0 is the hardest example
        and tau = 0.0 is the easiest
        """
        def mask_diagonal(matrix):
            """
            mask_diagonal
            Removes diagonal elements from a matrix. Returns masked matrix
            and mapping between masked and original indices
            """
            bs = matrix.shape[0]
            t = torch.ones(matrix.shape, dtype=bool)
            ind = np.diag_indices(t.shape[0])
            t[ind[0], ind[1]] = torch.zeros(t.shape[0], dtype=bool)
            idx = t == True
            masked = differences[idx].reshape(bs, bs-1)
            masked2global = (idx == True).nonzero()[:, 1]
            return masked, masked2global.view(bs, bs-1)

        bs, ft = inputs_1.shape
        device = inputs_1.device
        differences = torch.pow(inputs_1.unsqueeze(
            1) - inputs_2, 2).sum(dim=2).detach()
        positive_pairs = torch.diagonal(differences, 0).to(device)
        negative_pairs, masked_2_global = mask_diagonal(differences)
        negative_pairs, sorted_idx = negative_pairs.sort(
            dim=1, descending=True)
        negative_pairs = negative_pairs.to(device)
        masked_2_global = masked_2_global.to(device)
        # get minimum index (maximum allowed difficulty) given tau
        nt = torch.tensor(np.round(tau*(bs-1)).astype(np.int)
                          ).expand(bs, 1).to(device)
        # get most dificult example
        ni = torch.argmin(
            negative_pairs - positive_pairs.view(bs, 1), dim=1).view(bs, 1).to(device)
        pi = torch.min(torch.cat([nt, ni], dim=1),
                       dim=1).values
        # convert index to the pre-sorted indices
        selected_examples = torch.gather(
            sorted_idx, 1, pi.view(bs, 1).expand(bs, bs-1))[:, 0]
        # get associated index pre-masking
        global_selected = torch.gather(
            masked_2_global, 1, selected_examples.view(bs, 1).expand(bs, bs-1))[:, 0]
        return inputs_2[global_selected]


class DifficutyScheduler:
    def __init__(self, init_val=0, step_size=0.1, frequency=1):
        self.val = init_val
        self.step = step_size
        self.frequency = frequency
        self.counter = 0

    def __step__(self):
        self.counter += 1
        if self.counter % self.frequency:
            self.val += self.step

    def __call__(self):
        return self.val
