"""
Python file that creates the necessary objects for training cheapfake.

TODO
Add support for a more generic transform function.

IDEA
Instead of doing correlated audio/video we should just randomly sample! Then that would allow us to learn separate embeddings.

TODO
For the __getitem__() method, we are just going to spit out one sequence of 75 frames, and then one sequence of three seconds of audio. In this manner, we can split the image and audio inputs between the two networks without having to worry about syncing their size. Another experiment would be to sync this up, but let's worry about that later. 

Remove checks for non-whatever values. Probably not necessary and should just hope that the user is smart enough to figure it out...
"""

import os
import time
import random

import cv2
import glob
import torch
import librosa
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset

import cheapfake.contrib.video_processor as video_processor


def _identity_transform(x):
    """Identity transformation that does not do anything to the input.

    """
    return x


class DeepFakeDataset(Dataset):
    """Python implementation of the DeepFakeDataset class.

    """

    def __init__(
        self,
        root_path,
        videofile_processor=None,
        frames_processor=None,
        audio_processor=None,
        frames_per_second=30,
        sample_rate=16000,
        n_seconds=3,
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
        root_path : str
            The absolute path to the folder containing the DFDC training data.
        videofile_processor : object, optional
            An object that performs the basic video processing tasks (see cheapfake.contrib.video_processor.VideoFileProcessor for naming conventions), by default None. If None, then the video processor defaults to cheapfake.contrib.video_processor.VideoFileProcessor.
        frames_processor : object, optional
            An object that performs the basic frame processing tasks (see cheapfake.contrib.video_processor.FramesProcessor for naming conventions), by default None. If None, then the frames processor defaults to cheapfake.contrib.video_processor.FramesProcessor.
        audio_processor : object, optional
            An object that performs the basic audio processing tasks (see cheapfake.contrib.video_processor.AudioProcessor for naming conventions), by default None. If None, then the audio processor defaults to cheapfake.contrib.video_processor.AudioProcessor.
        frames_per_second : int, optional
            The frame rate of the video, by default 30. If a float is passed as input then it is cast as an int. If a non-float and non-int value is passed as input then the default value is used.
        sample_rate : int, optional
            The sample rate of the audio, by default 16 kHz. If a float is passed as input then it is cast as an int. If a non-float and non-int value is passed as input then the default value is used.
        n_seconds : int, optional
            The number of seconds, passed onto __getitem__(), by default 3 seconds. If a float is passed as input then it is cast as an int. If a non-float and non-int value is passed as input then the default value is used.
        channel_first : {True, False}, bool, optional
            If True then all input and output are assumed to have shape ``(T, C, H, W)`` where the channel dimension comes before the spatial dimensions, by default True. Otherwise the output has shape ``(T, H, W, C)``. If a non-boolean input is passed then the default value is used.
        frame_transform : callable, optional
            A callable function that is used to transform extracted frames from videos, by default None. If None, then the resizing transform is applied to the frames. If a non-callable function is passed as input, then the default transform is used. The transform must be able to operate on batches (i.e., of shape ``(B, C, H, W)`` or ``(B, H, W, C)``).
        audio_transform : callable, optional
            A callable function that is used to transform extracted audio from videos, by default None. If None, then no transform is applied to the extracted audio. If a non-callable function is passed as input, then the default transform (i.e. no transform) is used. The transform must be able to operate on batches (i.e., of shape ``(B, C, H, W)`` or ``(B, H, W, C)``).
        return_tensor : {True, False}, bool, optional
            If True, then this parameter is passed to functions that can return tensors as output, by default True. If a non-boolean parameter is passed as input, then the default value is used.
        verbose : {False, True}, bool, optional
            If True then verbose output is sent to the system console. If a non-boolean parameter is passed as input, then the default value is used.
        random_seed : int, optional
            The random seed used for reproducibility, by default 41. If a float is passed as input, then it is cast as an int. If a non-float and non-int value is passed as input, then the default value is used.
        sequential_frames : {False, True}, bool, optional
            If True then ``__getitem__`` returns sequential data after an initial index is chosen at random, by default True. Otherwise, ``__getitem__`` returns data that is not guaranteed to be sequential (i.e. each sample is chosen stochastically). If a non-boolean input is passed as input, then the default value is taken.
        sequential_audio : {False, True}, bool, optional
            If True then ``__getitem__`` returns sequential data after an initial index is chosen at random, by default True. Otherwise, ``__getitem__`` returns data that is not guaranteed to be sequential (i.e. each sample is chosen stochastically). If a non-boolean input is passed as input, then the default value is taken.
        stochastic : {True, False}, bool, optional
            If True then ``__getitem__`` returns data using a stochastic selection strategy, by default True. Otherwise, only the first ``n_seconds * frames_per_second`` frames and ``n_seconds * sample_rate`` audio samples are returned. If a non-boolean input is passed as input, then the default value is taken.

        """
        if type(frames_per_second) is float:
            frames_per_second = int(frames_per_second)
        elif (
            type(frames_per_second) is not float and type(frames_per_second) is not int
        ):
            frames_per_second = 30
        if type(sample_rate) is float:
            sample_rate = int(sample_rate)
        elif type(sample_rate) is not float and type(sample_rate) is not int:
            sample_rate = 16000
        if type(n_seconds) is float:
            n_seconds = int(n_seconds)
        elif type(n_seconds) is not float and type(n_seconds) is not int:
            n_seconds = 3
        if type(return_tensor) is not bool:
            return_tensor = True
        if type(verbose) is not bool:
            verbose = False
        if type(random_seed) is float:
            random_seed = int(random_seed)
        elif type(random_seed) is not float and type(random_seed) is not int:
            random_seed = 41
        if type(sequential_frames) is not bool:
            sequential_frames = False
        if type(sequential_audio) is not bool:
            sequential_audio = False
        if type(stochastic) is not bool:
            stochastic = True

        self.root_path = root_path

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
        self.channel_first = channel_first

        if frame_transform is not None:
            assert (
                callable(frame_transform) == True
            ), "The frame transform must be a callable function."
            self.frame_transform = frame_transform
        else:
            self.frame_transform = self._resize_frames
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

        self.video_paths = self._get_video_paths(root_path=self.root_path)

    def _get_video_paths(self, root_path, n_folders=None):
        """Extracts the video paths corresponding to the data from the DFDC dataset.

        Parameters
        ----------
        root_path : str
            The absolute path to the folder containing the DFDC data.
        n_folders : int, None
            The number of folders to consider, by default None. If None, then ``n_folders`` defaults to all the folders. If a float is passed as input, then it is converted to an int. If a non-float and non-integer is passed as input, then the default value is used.

        Returns
        -------
        video_paths : list
            List containing the absolute paths to the video files, from the DFDC data.
        
        """
        if type(n_folders) is float:
            n_folders = int(n_folders)
        if type(n_folders) is not float and type(n_folders) is not int:
            n_folders = None

        root_folders = [f for f in glob.glob(os.path.join(root_path, "*"))]
        if n_folders is None:
            n_folders = len(root_folders)
        root_folders = root_folders[:n_folders]

        video_paths = [glob.glob(os.path.join(folder, "*")) for folder in root_folders]
        video_paths = [item for sublist in video_paths for item in sublist]

        return video_paths

    def _resize_frames(
        self, frames, scale_factor=4.0, interpolation=cv2.INTER_LANCZOS4
    ):
        """Resizes the frames by a factor of ``scale_factor``.

        Parameters
        ----------
        frames : numpy.ndarray or torch.Tensor instance
            Numpy array or Torch tensor containing the frames.
        scale_factor : float, optional
            The scaling factor used to determine the new frame heights and widths, by default 4.0. If an int is sent as input, then it is cast as a float. If a non-float and non-int value is sent as input, then the default value is used.
        interpolation : int, optional
            OpenCV interpolation method, passed onto cv2.resize().
        
        Returns
        -------
        resized_frames : numpy.ndarray or torch.Tensor instance
            Numpy array or Torch tensor containing the resized frames.

        """
        if self.channel_first:
            frames = np.einsum("ijkl->iklj", frames)

        (new_height, new_width) = frames[0].shape[:2]
        new_height = int(new_height / scale_factor)
        new_width = int(new_width / scale_factor)

        resized_frames = np.empty((frames.shape[0], new_height, new_width, 3))
        for k, frame in enumerate(frames):
            resized_frames[k] = cv2.resize(
                frame, dsize=(new_width, new_height), interpolation=interpolation
            )

        if self.channel_first:
            resized_frames = np.einsum("ijkl->iljk", resized_frames)
        if self.return_tensor:
            resized_frames = torch.from_numpy(resized_frames)

        return resized_frames

    def _permute_for_lipnet(self, frames):
        """Permutes the dimensions of ``frames`` so that it fits within what is expected by LipNet.

        ``frames`` should have shape (sample, color, height, width) and by the end, ``frames`` will have shape (color, sample, height, width). 
    
        Parameters
        ----------
        frames : numpy.ndarray or torch.Tensor instance
            Numpy array or Torch tensor containing the frames, with shape (sample, color, height, width).
        
        Returns
        -------
        frames : numpy.ndarray
            Numpy array containing the frames, with shape (color, sample, height, width).

        Notes
        -----
        If a torch.Tensor instance is passed in, it is first converted to a Numpy array to take advantage of speed boosts.

        """
        if type(frames) is torch.Tensor:
            frames = frames.numpy()
        frames = np.einsum("ijkl->jikl", frames)

        return frames

    def _chunk_elements(self, elements, length=1, return_tensor=True):
        """Chunks a numpy.ndarray or torch.Tensor into smaller elements.

        Parameters
        ----------
        elements : numpy.ndarray or torch.Tensor
            Numpy array or Torch tensor containing the elements to be chunked.
        length : int, optional
            The number of elements in each chunk, by default 1. If a float is passed as input, then it is converted to an int. If a non-float or non-int value is passed as input, then the default value is used. If a value less than 1 is passed as input, then the default value is used.
        return_tensor : {True, False}, bool, optional
            If True then the output is returned as a torch.Tensor instance, by default True. Otherwise, the output is returned as a numpy.ndarray instance. If a non-boolean value is passed as input, the default value is used.

        Returns
        -------
        chunked_elements : numpy.ndarray or torch.Tensor instance
            Numpy array or Torch tensor containing the chunked elements, with each element of length ``length``.

        """
        if type(return_tensor) is not bool:
            return_tensor = True
        if type(length) is float:
            length = int(length)
        if type(length) is not float and type(length) is not int:
            length = 1
        if length < 1:
            length = 1

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

        Notes
        -----
        Minimal robustness checks are in place. Use with caution.

        """
        video_path = self.video_paths[index]

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

        frames = self._permute_for_lipnet(frames)
        audio_stft = self.audio_processor.extract_stft(
            audio, return_torch=self.return_tensor
        )

        if self.return_tensor:
            frames = torch.from_numpy(frames)
            audio = torch.from_numpy(audio)
            audio_stft = torch.from_numpy(audio_stft)

        return frames, audio, audio_stft
