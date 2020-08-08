"""
Python file that creates the necessary objects for training cheapfake.
"""

import os
import time

import glob
import torch
import librosa
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset

import cheapfake.contrib.video_processor as video_processor


class DeepFakeDataset(Dataset):
    """Python implementation of the DeepFakeDataset class.

    """

    def __init__(
        self,
        root_path,
        videofile_processor=None,
        frames_processor=None,
        audio_processor=None,
        frame_transform=None,
        audio_transform=None,
        return_tensor=True,
        verbose=False,
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
        frame_transform : callable, optional
            A callable function that is used to transform extracted frames from videos, by default None. If None, then no transform is applied to the extracted frames. If a non-callable function is passed as input, then the default transform (i.e. no transform) is used.
        audio_transform : callable, optional
            A callable function that is used to transform extracted audio from videos, by default None. If None, then no transform is applied to the extracted audio. If a non-callable function is passed as input, then the default transform (i.e. no transform) is used.
        return_tensor : {True, False}, bool, optional
            If True, then this parameter is passed to functions that can return tensors as output, by default True. If a non-boolean parameter is passed as input, then the default value is used.
        verbose : {False, True}, bool, optional
            If True then verbose output is sent to the system console. If a non-boolean parameter is passed as input, then the default value is used.

        """
        if type(return_tensor) is not bool:
            return_tensor = True
        if type(verbose) is not bool:
            verbose = False

        self.root_path = root_path

        if videofile_processor is None:
            self.videofile_processor = video_processor.VideoFileProcessor(
                verbose=verbose
            )
        if frames_processor is None:
            self.frames_processor = video_processor.FramesProcessor(verbose=verbose)
        if audio_processor is None:
            self.audio_processor = video_processor.AudioProcessor(verbose=verbose)

        self.frame_transform = frame_transform
        self.audio_transform = audio_transform

        self.verbose = verbose
        self.return_tensor = return_tensor

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

        chunked_elements = np.array(chunked_elements)

        if return_tensor:
            chunked_elements = torch.from_numpy(chunked_elements)

        return chunked_elements

    def __getitem__(self, index):
        """Extracts frames and audio from the next video instance.

        Frames and audio from each video instance are extracted, in chunks of 60 frames and 2 * 16000, for video and audio respectively.  

        Parameters
        ----------
        index : int
            The index corresponding to the next video instance.
        
        Returns
        -------
        frames : numpy.ndarray or torch.Tensor instance
            The frames extracted from the video.
        audio : numpy.ndarray or torch.Tensor instance
            The audio extracted from the video.

        Notes
        -----
        Eventually, we will want to leave it up to the user how many seconds and frames they want, but we can leave this for a later application/experiment.

        """
        video_path = self.video_paths[index]

        frames = self.videofile_processor.extract_all_frames(video_path=video_path)
        audio = self.videofile_processor._extract_all_audio(video_path=video_path)

        # Do transformations before feeding into chunks.
        frames = self._chunk_elements(
            frames, length=60, return_tensor=self.return_tensor
        )
        audio = self._chunk_elements(
            audio, length=16000 * 2, return_tensor=self.return_tensor
        )

        return frames, audio
