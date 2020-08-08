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
        return_tensors=True,
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
        return_tensors : {True, False}, bool, optional
            If True, then this parameter is passed to functions that can return tensors as output, by default True. If a non-boolean parameter is passed as input, then the default value is used.
        verbose : {False, True}, bool, optional
            If True then verbose output is sent to the system console. If a non-boolean parameter is passed as input, then the default value is used.

        """
        if type(return_tensors) is not bool:
            return_tensors = True
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
        self.return_tensors = return_tensors

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

        frames = videofile_processor.extract_all_frames(video_path=video_path)
        audio = videofile_processor._extract_all_audio(video_path=video_path)

        return frames, audio
