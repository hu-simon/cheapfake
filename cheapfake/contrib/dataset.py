"""
Python file that creates the necessary objects for training cheapfake.

Leaving this here, which is also in models.py

Need to rethink the entire forward pass. The idea is as follows.

The resized frames should be sent into FAN, with the expected shape. This returns facial landmarks (if a face is detected). Using the resizing information, we can rescale the predicted facial landmarks to get an overall estimate of where the lips are located in the full-size image. Then, the lip region is extracted from the full-size image (with some buffer room) and gets resized to 64 x 128 which is the size expected by LipNet. The predicted facial features and the embeddings learning by the LipNet are sent into the MLP, along with the audio embeddings learned by the residual network.
"""

import os
import time
import random

import cv2
import glob
import torch
import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

import cheapfake.contrib.transforms as transforms
import cheapfake.contrib.video_processor as video_processor

"""
Test aligned vs. mis-aligned this gives us information. 

Also need to figure out how to load the 
"""


def _identity_transform(x):
    """Identity transformation that does not do anything to the input.

    """
    return x


class DeepFakeDataset(Dataset):
    """Python implementation of the DeepFakeDataset class.

    """

    def __init__(
        self,
        metadata_path,
        videofile_processor=None,
        frames_processor=None,
        audio_processor=None,
        frames_per_second=30,
        sample_rate=16000,
        n_seconds=3.0,
        num_samples=None,
        channel_first=True,
        frame_transform=None,
        audio_transform=None,
        return_tensor=True,
        verbose=False,
        random_seed=41,
        sequential_frames=False,
        sequential_audio=False,
        stochastic=True,
    ):
        """Instantiates a new DeepFakeDataset object.

        Parameters
        ----------
        metadata_path : str
            The absolute path to the folder containing the DFDC training metadata.
        videofile_processor : object, optional
            An object that performs the basic video processing tasks (see cheapfake.contrib.video_processor.VideoFileProcessor for naming conventions), by default None. If None, then the video processor defaults to cheapfake.contrib.video_processor.VideoFileProcessor.
        frames_processor : object, optional
            An object that performs the basic frame processing tasks (see cheapfake.contrib.video_processor.FramesProcessor for naming conventions), by default None. If None, then the frames processor defaults to cheapfake.contrib.video_processor.FramesProcessor.
        audio_processor : object, optional
            An object that performs the basic audio processing tasks (see cheapfake.contrib.video_processor.AudioProcessor for naming conventions), by default None. If None, then the audio processor defaults to cheapfake.contrib.video_processor.AudioProcessor.
        frames_per_second : int, optional
            The frame rate of the video, by default 30. 
        sample_rate : int, optional
            The sample rate of the audio, by default 16 kHz. 
        n_seconds : float, optional
            The number of seconds, passed onto __getitem__(), by default 3.0 seconds.
        num_samples : int, optional
            The number of data samples to draw from the dataframe, by default None. 
        channel_first : {True, False}, bool, optional
            If True then all input and output are assumed to have shape ``(T, C, H, W)`` where the channel dimension comes before the spatial dimensions, by default True. Otherwise the output has shape ``(T, H, W, C)``. 
        frame_transform : callable, optional
            A callable function that is used to transform extracted frames from videos, by default None. If None, then the resizing transform with factor 4 is applied to the frames. If a non-callable function is passed as input, then the default transform is used. The transform must be able to operate on batches (i.e., of shape ``(B, C, H, W)`` or ``(B, H, W, C)``).
        audio_transform : callable, optional
            A callable function that is used to transform extracted audio from videos, by default None. If None, then no transform is applied to the extracted audio. If a non-callable function is passed as input, then the default transform (i.e. no transform) is used. The transform must be able to operate on batches (i.e., of shape ``(B, C, H, W)`` or ``(B, H, W, C)``).
        return_tensor : {True, False}, bool, optional
            If True, then this parameter is passed to functions that can return tensors as output, by default True. 
        verbose : {False, True}, bool, optional
            If True then verbose output is sent to the system console. 
        random_seed : int, optional
            The random seed used for reproducibility, by default 41.
        sequential_frames : {False, True}, bool, optional
            If True then ``__getitem__`` returns sequential data after an initial index is chosen at random, by default True. Otherwise, ``__getitem__`` returns data that is not guaranteed to be sequential (i.e. each sample is chosen stochastically). 
        sequential_audio : {False, True}, bool, optional
            If True then ``__getitem__`` returns sequential data after an initial index is chosen at random, by default True. Otherwise, ``__getitem__`` returns data that is not guaranteed to be sequential (i.e. each sample is chosen stochastically). 
        stochastic : {True, False}, bool, optional
            If True then ``__getitem__`` returns data using a stochastic selection strategy, by default True. Otherwise, only the first ``int(n_seconds * frames_per_second)`` frames and ``int(n_seconds * sample_rate)`` audio samples are returned. 

        """
        assert isinstance(
            frames_per_second, int
        ), "Frames per second must be an integer."
        assert isinstance(sample_rate, int), "The sample rate must be an integer."
        assert isinstance(
            n_seconds, (float, int)
        ), "The number of seconds must be either a float or an integer."
        assert isinstance(num_samples, int)
        assert isinstance(return_tensor, bool)
        assert isinstance(verbose, bool)
        assert isinstance(random_seed, int)
        assert isinstance(sequential_frames, bool)
        assert isinstance(sequential_audio, bool)
        assert isinstance(stochastic, bool)

        self.metadata_path = metadata_path

        if videofile_processor is None:
            self.videofile_processor = video_processor.VideoFileProcessor(
                verbose=verbose
            )
        if frames_processor is None:
            self.frames_processor = video_processor.FramesProcessor(
                verbose=verbose, random_seed=random_seed
            )
        if audio_processor is None:
            self.audio_processor = video_processor.AudioProcessor(
                verbose=verbose, random_seed=random_seed
            )

        self.frames_per_second = frames_per_second
        self.sample_rate = sample_rate
        self.n_seconds = n_seconds
        self.num_samples = num_samples
        self.channel_first = channel_first

        if frame_transform is not None:
            assert (
                callable(frame_transform) == True
            ), "The frame transform must be a callable function."
            self.frame_transform = frame_transform
        else:
            self.frame_transform = transforms.BatchRescale(output_size=(64, 128))
        if audio_transform is not None:
            assert (
                callable(audio_transform) == True
            ), "The audio transform must be a callable function."
            self.audio_transform = audio_transform
        else:
            self.audio_transform = _identity_transform

        self.verbose = verbose
        self.return_tensor = return_tensor
        self.random_seed = random_seed
        self.sequential_frames = sequential_frames
        self.sequential_audio = sequential_audio
        self.stochastic = stochastic

        random.seed(random_seed)

        self.video_paths, self.video_labels = self._data_from_df(num_samples=self.num_samples)

        # self.video_paths = self._get_video_paths(root_path=self.root_path)

    def _data_from_df(self, num_samples=None):
        """Extracts the video paths and video labels (i.e. real or fake) corresponding to the data from the DFDC dataset.

        This function assumes that the video paths are contained in a Pandas DataFrame, under the header "File", and the video labels under the header "label"

        Paramters
        ---------
        num_samples : int, optional
            The number of video paths to extract from the Pandas dataframe, by default None. If None, then all video paths are extracted.

        Returns
        -------
        video_paths : pandas.core.series.Series instance
            Pandas series containing the absolute paths to the videos.

        """
        assert isinstance(num_samples, (int, type(None)))

        df = pd.read_csv(self.metadata_path)

        if num_samples is None:
            num_samples = len(df)

        video_paths = df["Files"][:num_samples]
        video_labels = df["label"][:num_samples]

        return video_paths, video_labels

    def _chunk_elements(self, elements, length=1, return_tensor=True):
        """Chunks a numpy.ndarray or torch.Tensor into smaller elements.

        Parameters
        ----------
        elements : numpy.ndarray or torch.Tensor
            Numpy array or Torch tensor containing the elements to be chunked.
        length : int, optional
            The number of elements in each chunk, by default 1. 
        return_tensor : {True, False}, bool, optional
            If True then the output is returned as a torch.Tensor instance, by default True. Otherwise, the output is returned as a numpy.ndarray instance. 

        Returns
        -------
        chunked_elements : numpy.ndarray or torch.Tensor instance
            Numpy array or Torch tensor containing the chunked elements, with each element of length ``length``.

        """
        assert isinstance(return_tensor, bool)
        assert isinstance(length, int)

        if type(elements) is torch.Tensor:
            elements = elements.numpy()

        chunked_elements = list()
        for k in range(0, len(elements), length):
            chunked_elements.append(elements[k : k + length])

        if chunked_elements[-1].shape != chunked_elements[0].shape:
            chunked_elements = chunked_elements[:-1]

        chunked_elements = np.array(chunked_elements)

        if return_tensor:
            chunked_elements = torch.from_numpy(chunked_elements)

        return chunked_elements

    def __len__(self):
        """Computes the number of videos.

        Returns
        -------
        int
            The number of videos.

        """
        return len(self.video_paths)

    def __getitem__(self, index):
        """Extracts frames and audio from a video instance.
        
        Frames (i.e. set of images) and audio from the video instance is extracted. A specific number of frames from the video stream are extracted, and a specific number of samples from the audio stream are extracted. There is no correlation between the number of frames extracted and the number of audio extracted. 

        All frames and audio samples from the video are extracted and chunked into sizes of 75 and 3 * 16000, respectively. 

        Parameters
        ----------
        index : int
            The index corresponding to the video instance.
        
        Returns
        -------
        frames : numpy.ndarray or torch.Tensor instance
            Numpy array or Torch tensor containing the frames extracted from the video.
        audio : numpy.ndarray or torch.Tensor instance
            Numpy array or Torch tensor containing the audio samples extracted from the video.
        audio_stft : numpy.ndarray or torch.Tensor instance
            Numpy array or Torch tensor containing the Short-Time Fourier Transform (STFT) of the audio signal.
        label : {0, 1}, int
            The label of the video (i.e. if the video is fake or real). 

        Notes
        -----
        Minimal robustness checks are in place. Use with caution.

        """
        video_path = self.video_paths[index]
        video_label = self.video_labels[index]

        frames = self.videofile_processor.extract_all_frames(video_path=video_path)
        audio = self.videofile_processor._extract_all_audio(video_path=video_path)

        frames = self.frames_processor.apply_transformation(
            frames, self.frame_transform
        )
        audio = self.audio_processor.apply_transformation(audio, self.audio_transform)

        if self.stochastic:
            frames = self.frames_processor.sample_frames_stochastic(
                frames,
                n_samples=self.n_seconds * self.frames_per_second,
                sequential=self.sequential_frames,
            )
            audio = self.audio_processor.sample_audio_stochastic(
                audio,
                n_samples=self.n_seconds * self.sample_rate,
                sequential=self.sequential_audio,
            )
        else:
            frames = frames[0 : self.n_seconds * self.frames_per_second]
            audio = audio[0 : self.n_seconds * self.sample_rate]

        audio_stft = self.audio_processor.extract_stft(
            audio, return_torch=self.return_tensor
        )

        #if self.return_tensor:
        #    audio = torch.from_numpy(audio)
        del audio
        
        return frames, audio_stft, video_label
