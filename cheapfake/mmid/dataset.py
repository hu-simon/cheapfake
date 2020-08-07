import numpy as np
import librosa as libr

import torch
from torch.utils.data import Dataset
from torchvision import transforms


__all__ = ['VoxCeleb2Dataset']


class VoxCeleb2Dataset(Dataset):
    def __init__(self, df, video_transform=None, audio_transform=None,
                 video_loader=None, seconds=3, video_path='', audio_path='',
                 stochastic=True, return_id=False):
        self.files = df['File']
        self.video_path = video_path
        self.audio_path = audio_path
        self.video_loader = video_loader
        self.seconds = seconds
        self.stochastic = stochastic
        self.return_id = return_id

        if video_transform is None:
            self.video_transform = transforms.ToTensor()
        else:
            self.video_transform = video_transform

        self.audio_transform = audio_transform

    def __getitem__(self, idx):
        file = self.files[idx]
        frame, frame_id = self.video_loader.read_random_frames(
            path='{}/{}.mp4'.format(self.video_path, file), num_frames=1)
        (sig, rate) = libr.load('{}/{}.wav'.format(self.audio_path, file),
                                sr=None)

        if self.stochastic:
            fragment_length = int(self.seconds * rate)
            upper_bound = max(len(sig) - fragment_length, 1)
            fragment_start_index = np.random.randint(0, upper_bound)
        else:
            fragment_start_index = 0

        sig = sig[fragment_start_index:
                  fragment_start_index+fragment_length]

        if self.audio_transform is None:
            audio = torch.tensor(sig)
        else:
            audio = self.audio_transform(sig)
        if self.return_id:
            return self.video_transform(frame[0]), audio, torch.tensor(int(file.split('/')[-1]))
        else:
            return self.video_transform(frame[0]), audio

    def __len__(self):
        return len(self.files)
