import os
import time

import cv2
import numpy as np
import moviepy.editor


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

    def extract_frames(self, video_path, num_frames=None, channel_first=True):
        """Extracts ``num_frames`` number of frames from a video, and returns it as a Numpy array.

        If ``channel_first`` is ``True`` then the shape of the output is ``(T, C, H, W)`` and ``(T, H, W, C)`` otherwise. Here, T represents time, C represents channel, H represents height, and W represents width.

        Parameters
        ----------
        video_path : str
            The absolute path to the video file.
        num_frames : {None, 1, 2, ...}, int, optional
            The number of frames to extract, by default None. If None, then all frames of the video are extracted. If ``num_frames`` exceeds the total number of frames in the video, then ``num_frames`` collapses to the maximum number of frames that make up the video. Similarly, if ``num_frames <= 0`` then ``num_frames`` collapses to the maximum number of frames that make up the video.
        channel_first : {True, False}, bool, optional
            If True then the output is returned with shape ``(T, C, H, W)`` where the channel dimension comes before the spatial dimensions, by default True. Otherwise, the output is of the format ``(T, H, W, C)``.

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
        frame_width = int(vidcap.get(cv2.CAP_PROP_WIDTH_COUNT))
        frame_height = int(vidcap.get(cv2.CAP_PROP_HEIGHT_COUNT))

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

    def extract_audio(self, video_path, num_samples=None):
        """Extracts ``num_samples`` number of audio samples from a video and returns it as a Numpy array.

        Parameters
        ----------
        video_path : str
            The absolute path to the video file.
        num_samples : {None, 1, 2, ...}, int, optional
            The number of samples to extract, by default None. If None, then the entire audio is extracted. If ``num_samples`` exceeds the total number of audio samples, then ``num_samples`` collapses to the maximum number of samples that make up the audio. Similarly, if ``num_samples <= 0`` then ``num_samples`` collapses to the maximum number of frames that make up the video.

        Returns
        -------
        audio : numpy.ndarray instance
            Numpy array containing ``num_samples`` number of samples from the video. 

        Notes
        -----
        It is assumed that the audio has a sample rate of 44.1 kHz. 

        """
        audio = moviepy.editor.AudioFileClip(video_path).to_soundarray()[:, -1]


class VideoProcessor:
    """Helper class used for processing video (i.e. frames)."""

    pass


class AudioProcessor:
    """Helper class used for processing audio."""

    pass
