import os
import time
import warnings

import cv2
import numpy as np
import moviepy.editor

"""
TODO/NOTE

1. Test every function that has been developed in this file.
2. Idea: instead of warning people that they should use a better function for efficiency, it may be better to just do it for them!
3. For the methods that call on the "_from_indices" methods, two separate methods should be created: one internal, and one external. The internal method takes as input, a vidcap object so that it does not have to create a new one. The external method takes as input the video path and creates a new vidcap object. 

Idea for FramesProcessor. This function is going to essentially batch process the frames into Face Alignment Network, but need to figure out the logistics.

Idea for AudioProcessor. This function is essentially going to take the spectrogram of the audio signal and feed that into one of the networks that Michael has. Need to figure out the logistics there too. 
"""


class BetterFunctionWarning(UserWarning):
    """Warnings class for warning the user that there is a more efficient function for what they are trying to accomplish.
    """

    pass


class HighRateWarning(UserWarning):
    """Warnings class for warning the user that the sampling rate is higher than 44.1 kHz, the best audio rate for good results.
    """

    pass


class VideoFileProcessor:
    """Helper class used for processing video files."""

    def __init__(self, verbose=True):
        """Instantiates a VideoProcessor object.

        Parameters
        ----------
        verbose : {True, False}, bool, optional
            If True, verbose output is sent to the system console.

        """
        self.verbose = verbose

    def extract_all_frames(self, video_path, channel_first=True):
        """Extracts all frames from a video and returns it as a Numpy array.

        If ``channel_first`` is True then the shape of the output is ``(T, C, H, W)`` and ``(T, H, W, C)`` otherwise. Here T represents time, C represents channel, H represents height, and W represents width.

        Parameters
        ----------
        video_path : str
            The absolute path to the video file.
        channel_first : {True, False}, bool, optional
            If True then the output has shape ``(T, C, H, W)``, where the channel dimension comes before the spatial dimensions, by default True. Otherwise, the output has shape ``(T, H, W, C)``. If a non-boolean input is passed, then ``channel_first`` automatically defaults to True.

        Returns
        -------
        frames : numpy.ndarraay instance
            Numpy array containing all the frames in the video, with shape ``(T, C, H, W)`` or ``(T, H, W, C)`` depending on ``channel_first``. 

        """
        if type(channel_first) is not bool:
            channel_first = True

        vidcap = cv2.VideoCapture(video_path)
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        assert (
            frame_count > 0
        ), "Invalid video, number of frames must be greater than 0."

        frames = np.empty(
            (frame_count, frame_height, frame_width, 3), np.dtype("uint8")
        )

        count = 0
        success = True
        while count < frame_count and success:
            success, frames[count] = vidcap.read()
            count += 1
        vidcap.release()

        if channel_first:
            frames = np.einsum("ijkl->iljk", frames)

        return frames

    def extract_frames(self, video_path, num_frames=None, channel_first=True):
        """Extracts ``num_frames`` number of frames sequentially from a video, and returns it as a Numpy array.

        If ``channel_first`` is True then the shape of the output is ``(T, C, H, W)`` and ``(T, H, W, C)`` otherwise. Here T represents time, C represents channel, H represents height, and W represents width.

        Parameters
        ----------
        video_path : str
            The absolute path to the video file.
        num_frames : {None, 1, 2, ...}, int, optional
            The number of frames to extract, by default None. If None, then all frames of the video are extracted. If ``num_frames`` exceeds the total number of frames in the video, then ``num_frames`` collapses to the maximum number of frames that make up the video. Similarly, if ``num_frames <= 0`` then ``num_frames`` collapses to the maximum number of frames that make up the video.
        channel_first : {True, False}, bool, optional
            If True then the output has shape ``(T, C, H, W)``, where the channel dimension comes before the spatial dimensions, by default True. Otherwise, the output has shape ``(T, H, W, C)``. If a non-boolean input is passed, then ``channel_first`` automatically defaults to True.

        Returns
        -------
        frames : numpy.ndarray instance
            Numpy array containing ``num_frames`` frames from the video returned in either ``(T, C, H , W)`` or ``(T, H, W, C)`` format depending on ``channel_first``.

        """
        # Handle any improper parameters.
        if type(channel_first) is not bool:
            channel_first = True

        vidcap = cv2.VideoCapture(video_path)
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        assert frame_count > 0, "Number of frames must exceed 0."

        # Handle any None parameters.
        if num_frames is None or num_frames > frame_count or num_frames <= 0:
            num_frames = frame_count

        frames = np.empty((num_frames, frame_height, frame_width, 3), np.dtype("uint8"))

        count = 0
        success = True
        while count < num_frames and success:
            success, frames[count] = vidcap.read()
            count += 1
        vidcap.release()

        if channel_first:
            frames = np.einsum("ijkl->iljk", frames)

        return frames

    def _extract_frames_from_indices(
        self, video_path, indices=None, channel_first=True
    ):
        """Extracts frames, according to ``indices``, from a video and returns it as a Numpy array.

        If ``channel_first`` is True then the shape of the output is ``(T, C, H, W)`` and ``(T, H, W, C)`` otherwise. Here T represents time, C represents channel, H represents height, and W represents width.

        Parameters
        ----------
        video_path : str
            The absolute path to the video file.
        indices : array-like, optional
            Array-like object containing the indices of the frames to extract, by default None. If None, then ``indices`` is taken to be the entire video sequence.
        channel_first : {True, False}, bool, optional
            If True then the output has shape ``(T, C, H, W)``, where the channel dimension comes before the spatial dimensions, by default True. Otherwise, the output has shape ``(T, H, W, C)``. If a non-boolean input is passed, then ``channel_first`` automatically defaults to True.

        Returns
        -------
        frames : numpy.ndarray instance
            Numpy array containing the frames of the video, located at ``indices``. The output shape is either ``(T, C, H, W)`` or ``(T, H, W, C)`` depending on the value ``channel_first`` takes.

        """
        if type(channel_first) is not bool:
            channel_first = True

        if indices is not None:
            indices = np.asarray(indices)

        vidcap = cv2.VideoCapture(video_path)
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        assert frame_count > 0, "Invalid video."

        if indices is None:
            if self.verbose:
                warnings.warn(
                    "You are trying to extract all frames. Use extract_all_frames() for efficiency.",
                    BetterFunctionWarning,
                )
            indices = np.arange(frame_count)

        assert len(indices) > 0, "Number of indices must be strictly greater than 0."
        assert (
            len(indices) <= frame_count
        ), "Number of indices cannot exceed the maximum number of frames."
        assert (
            sum([index > frame_count for index in indices]) == 0
        ), "At least one of the indices exceeds the maximum number of frames."
        assert (
            sum([index < 0 for index in indices]) == 0
        ), "At least one of the indices is negative."

        count = 0
        index = 0
        success = True
        n_indices = len(indices)
        frames = np.empty((n_indices, frame_height, frame_width, 3), np.dtype("uint8"))
        while count < frame_count and success and index < n_indices:
            success = vidcap.grab()
            if count in indices:
                success, frames[index] = vidcap.retrieve()
                index += 1
            count += 1
        vidcap.release()

        if channel_first:
            frames = np.einsum("ijkl->iljk", frames)

        return frames

    def extract_frames_start_end(
        self, video_path, start=0, end=None, channel_first=True
    ):
        """Extracts ``end - start`` number of frames from a video and returns it as a Numpy array.

        If ``channels_first`` is True, then the shape of the output is ``(T, C, H, W)`` and ``(T, H, W, C)`` otherwise. Here, T represents time, C represents channel, H represents height, and W represents width.

        Parameters
        ----------
        video_path : str
            The absolute path to the video file.
        start : {0, 1, ...}, int, optional
            The start index, used to determine the location in the video to start frame extraction, by default 0.
        end : {None, 0, 1, ...}, int, optional
            The end index, used to determine the location in the video to end frame extraction, by default None. If None, then ``end`` collapses to the maximum frames in the video.
        channel_first : {True, False}, bool, optional
            If True, then the output has shape ``(T, C, H, W)``, where the channel dimension comes before the spatial dimensions, by default True. Otherwise, the output has shape ``(T, H, W, C)``. If a non-boolean input is passed, then ``channel_first`` automatically defaults to True.

        """
        assert start >= 0, "The start index must be greater than or equal to zero."
        if type(channel_type) is not bool:
            channel_type = True

        vidcap = cv2.VideoCapture(video_path)
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        assert frame_count > 0, "Invalid video."

        if end is None:
            if self.verbose:
                warnings.warn(
                    "You are trying to extract all frames. Use extract_all_frames for efficiency",
                    BetterFunctionWarning,
                )
            end = frame_count
        assert end > 0, "The end index must be greater than zero."
        assert end > start, "The end index must be greater than the start index."

        indices = np.arange(start, end)
        frames = self._extract_frames_from_indices(
            video_path=video_path, indices=indices, channel_first=channel_first
        )

        return frames

    def extract_all_audio(self, video_path, sample_rate=44100):
        """Extracts all the audio from a video file and returns it as a Numpy array.

        Parameters
        ----------
        video_path : str
            The absolute path to the video file.
        sample_rate : int, optional
            The sample rate of the audio, by default 44.1 kHz. If the sample rate is not an integer, then it is converted to one.

        Returns
        -------
        audio : numpy.ndarray instance
            Numpy array containing the audio extracted from the video.

        """
        if type(sample_rate) is not int:
            sample_rate = int(sample_rate)
        assert sample_rate > 0, "The sampling rate must be greater than 0."
        if self.verbose:
            if sample_rate > 44100:
                warnings.warn("Sampling rate is higher than 44.1 kHz.", HighRateWarning)

        audio = moviepy.editor.AudioFileClip(
            video_path, fps=sample_rate
        ).to_soundarray()[:, -1]

        return audio

    def extract_audio(self, video_path, num_samples=None, sample_rate=44100):
        """Extracts ``num_samples`` number of audio samples sequentially from a video and returns it as a Numpy array.

        Parameters
        ----------
        video_path : str
            The absolute path to the video file.
        num_samples : {None, 1, 2, ...}, int, optional
            The number of samples to extract, by default None. If None, then the entire audio is extracted. If ``num_samples`` exceeds the total number of audio samples, then ``num_samples`` collapses to the maximum number of samples that make up the audio. Similarly, if ``num_samples <= 0`` then ``num_samples`` collapses to the maximum number of frames that make up the video.
        sample_rate : int, optional
            The sample rate of the audio, by default 44.1 kHz. If the sample rate is not an integer, then it is converted to one.

        Returns
        -------
        audio : numpy.ndarray instance
            Numpy array containing ``num_samples`` number of samples from the video. 

        """
        assert sample_rate > 0, "The sampling rate must be greater than 0."
        if type(sample_rate) is not int:
            sample_rate = int(sample_rate)
        if self.verbose:
            if sample_rate > 44100:
                warnings.warn("Sampling rate is higher than 44.1 kHz.", HighRateWarning)

        audio = moviepy.editor.AudioFileClip(
            video_path, fps=sample_rate
        ).to_soundarray()[:, -1]

        # Handle None parameters.
        if num_samples is None or num_samples <= 0 or num_samples > len(audio):
            num_samples = len(audio)

        audio = audio[:num_samples]

        return audio

    def extract_audio_from_indices(self, video_path, indices=None, sample_rate=44100):
        """Extracts audio samples from a video, according to ``indices``, and returns it as a Numpy array.

        Parameters
        ----------
        video_path : str
            The absolute path to the video file.
        indices : array-like, optional
            Array-like object containing the indices of the audio samples to extract, by default None. If None, then ``indices`` is taken to be the entire video sequence.
        sample_rate : int, optional
            The sample rate of the audio, by default 44.1 kHz. If the sample rate is not an interger, then it is converted to one.
        
        Returns
        -------
        audio : numpy.ndarray instance
            Numpy array containing the audio samples from the video, located at ``indices``.
        
        """
        assert sample_rate > 0, "The sampling rate must be greater than 0."
        if type(sample_rate) is not int:
            sample_rate = int(sample_rate)

        if self.verbose:
            if sample_rate > 44100:
                warnings.warn("Sampling rate is higher than 44.1 kHz", HighRateWarning)

        indices = np.asarray(indices)

        audio = moviepy.editor.AudioFileClip(
            video_path, fps=sample_rate
        ).to_soundarray()[:, -1]

        if indices is None:
            return audio

        audio = audio[indices]

        return audio

    def extract_audio_start_end(self, video_path, start=0, end=None, sample_rate=44100):
        """Extracts ``end - start`` number of audio samples from a video and returns it as a Numpy array. 

        Parameters
        ----------
        video_path : str
            The absolute path to the video file.
        start : {0, 1, ...}, int, optional
            The start index, used to determine the start of the audio signal, by default 0.
        end : {None, 0, 1, ...}, int, optional
            The end index, used to determine the end of the audio signal, by default None. If None, then ``end`` defaults to the end of the entire audio signal.
        sample_rate : int, optional
            The sample rate of the audio, by default 44.1 kHz. If the sample rate is not an integer, then it is converted to one.

        Returns
        -------
        audio : numpy.ndarray instance
            Numpy array containing ``end - start`` samples from the audio signal extracted from the video.

        """
        assert start >= 0, "The start index must be greater than or equal to zero."
        assert sample_rate > 0, "The sampling rate must be greater than 0."
        if type(sample_rate) is not int:
            sample_rate = int(sample_rate)

        if self.verbose:
            if sample_rate > 44100:
                warnings.warn("Sampling rate is higher than 44.1 kHz.", HighRateWarning)

        audio = moviepy.editor.AudioFileClip(
            video_path, fps=sample_rate
        ).to_soundarray()[:, -1]

        if end is None:
            if self.verbose:
                warnings.warn(
                    "You are trying to extract all the audio samples. Use extract_all_audio for efficiency.",
                    BetterFunctionWarning,
                )
            end = len(audio)
        assert end > 0, "The end index must be greater than zero."
        assert end > start, "The end index must be greater than the start index."

        indices = np.arange(start, end)
        del audio
        audio = self._extract_frames


class FramesProcessor:
    """Helper class used for processing video frames."""

    pass


class AudioProcessor:
    """Helper class used for processing audio."""

    pass
