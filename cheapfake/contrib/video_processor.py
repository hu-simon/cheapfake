import os
import time
import warnings

import cv2
import torch
import numpy as np
import moviepy.editor
import face_alignment


"""
The audio extraction functions need some sort of testing!

Need to enforce some sort of color flag so that the images are in RGB format.
"""


class VideoFileProcessor:
    """Helper class for processing video files."""

    def __init__(self, verbose=True, channel_first=True, return_rgb=True):
        """Instantiates a new VideoFileProcessor object.

        Parameters
        ----------
        verbose : {True, False}, bool, optional
            If True, then verbose output is sent to the system console, by default True.
        channel_first : {True, False}, bool, optional
            If True, then all outputs will have shapes where the channel dimension comes before the spatial dimensions, by default True.
        return_rgb : {True, False}, bool, optional
            If True, then all outputs will adhere to the RGB channel format, by default True. Otherwise, the outputs will adhere to the BGR channel format. 
        
        """
        self.verbose = verbose
        self.channel_first = channel_first
        self.return_rgb = return_rgb

    def _get_video_info(self, vidcap):
        """Extracts information about the video.

        Extracts the number of video frames and the height and width of each frame.

        Parameters
        ----------
        vidcap : cv2.VideoCapture instance
            VideoCapture object used for grabbing the frames.

        Parameters
        ----------
        video_info : dictionary
            Dictionary containing the number of video frames, and the height and width of each frame. The dictionary is structured as follows.
            {
                frame_count : int
                    The number of frames in the video.
                frame_height : int
                    The height of each frame, in pixels.
                frame_width : int
                    The width of each frame, in pixels.
            }
        
        """
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_info = {
            "frame_count": frame_count,
            "frame_height": frame_height,
            "frame_width": frame_width,
        }

        return video_info

    def _extract_all_frames(self, vidcap, channel_first=True, return_rgb=True):
        """Extracts all frames from a video and returns it as a Numpy array.

        If ``channel_first == True``, then the output has shape ``(T, C, H, W)`` and ``(T, H, W, C)`` otherwise. Here, T represents time, C represents channel, H represents height, and W represents width.

        Parameters
        ----------
        vidcap : cv2.VideoCapture instance
            VideoCapture object used for grabbing the frames.
        channel_first : {False, True}, bool, optional
            If True then the output has shape ``(T, C, H, W)``, where the channel dimension comes before the spatial dimensions, by default True. Otherwise the output has shape ``(T, H, W, C)``. If a non-boolean input is passed, then ``channel_first`` defaults to True.
        return_rgb : {True, False}, bool, optional
            If True, then the output adheres to the RGB color format, by default True. Otherwise, the output adheres to the BGR color format. If a non-boolean input is passed, then ``return_rgb`` defaults to True.
        
        Returns
        -------
        frames : numpy.ndarray instance
            Numpy array containing ``num_frames`` frames from the video returned in either ``(T, C, H, W)`` or ``(T, H, W, C)`` format depending on ``channel_first``. 

        """
        if type(channel_first) is not bool:
            channel_first = True
        if type(return_rgb) is not bool:
            return_rgb = True

        video_info = self._get_video_info(vidcap)
        frame_count = video_info["frame_count"]
        frame_width = video_info["frame_width"]
        frame_height = video_info["frame_height"]
        assert (
            frame_count > 0
        ), "Invalid video, the number of frames must be greater than 0."

        frames = np.empty(
            (frame_count, frame_height, frame_width, 3), np.dtype("uint8")
        )

        count = 0
        success = True
        while count < frame_count and success:
            success, frames[count] = vidcap.read()
            if return_rgb:
                frames[count] = cv2.cvtColor(frames[count], cv2.COLOR_BGR2RGB)
            count += 1
        vidcap.release()

        if channel_first:
            frames = np.einsum("ijkl->iljk", frames)

        return frames

    def extract_all_frames(self, video_path, channel_first=True, return_rgb=True):
        """Extracts all frames from a video and returns it as a Numpy array.

        If ``channel_first == True``, then the output has shape ``(T, C, H, W)`` and ``(T, H, W, C)`` otherwise. Here, T represents time, C represents channel, H represents height, and W represents width.

        Parameters
        ----------
        video_path : str
            The absolute path to the video file.
        channel_first : {False, True}, bool, optional
            If True then the output has shape ``(T, C, H, W)``, where the channel dimension comes before the spatial dimensions, by default True. Otherwise the output has shape ``(T, H, W, C)``. If a non-boolean input is passed, then ``channel_first`` defaults to True.
        return_rgb : {True, False}, bool, optional
            If True, then the output adheres to the RGB color format, by default True. Otherwise, the output adheres to the BGR color format. If a non-boolean input is passed, then ``return_rgb`` defaults to True.

        Returns
        -------
        frames : numpy.ndarray instance
            Numpy array containing ``num_frames`` frames from the video returned in either ``(T, C, H, W)`` or ``(T, H, W, C)`` format depending on ``channel_first``. 

        """
        vidcap = cv2.VideoCapture(video_path)
        frames = self._extract_all_frames(
            vidcap=vidcap, channel_first=channel_first, return_rgb=return_rgb
        )

        return frames

    def _extract_frames_from_indices(
        self, vidcap, indices=None, channel_first=True, return_rgb=True
    ):
        """Extracts frames, according to ``indices``, from a video and returns it as a Numpy array. 

        If ``channels_first`` is True then the shape of the output is ``(T, C, H, W)`` and ``(T, H, W, C)`` otherwise. Here T represents time, C represents channel, H represents height, and W represents width.

        Note that this function differs from ``extract_frames_from_indices`` in that this function takes the cv2.VideoCapture object as input to save memory.

        Parameters
        ----------
        vidcao : cv2.VideoCapture instance
            The absolute path to the video file.
        indices : array-like optional
            Array-like object containing the inddices of the frames to extract, by default None. If None, then ``indices`` is taken to be the entire video sequence. 
        channel_first : {True, False}, bool, optional
            If True then the output has shape ``(T, C, H, W)``, where the channel dimension comes before the spatial dimensions, by default True. Otherwise, the output has shape ``(T, H, W, C)``. If a non-boolean input is passed, then ``channel_first`` automatically defaults to True.
        return_rgb : {True, False}, bool, optional
            If True then the output adheres to the RGB color format, by default True. Otherwise, the output adheres to the BGR color format. If a non-boolean input is passed, then ``return_rgb`` defaults to True.

        Returns
        -------
        frames : numpy.ndarray instance
            Numpy array containing the frames of the video, located at ``indices``. The output shape is either ``(T, C, H, W)`` or ``(T, H, W, C)`` depending on the value ``channel_first`` takes.

        """
        if type(channel_first) is not bool:
            channel_first = True
        if type(return_rgb) is not bool:
            return_rgb = True

        if indices is None:
            frames = self._extract_all_frames(
                vidcap=vidcap, channel_first=channel_first
            )

            return frames
        else:
            indices = np.asarray(indices)

        video_info = self._get_video_info(vidcap)
        frame_count = video_info["frame_count"]
        frame_width = video_info["frame_width"]
        frame_height = video_info["frame_height"]

        assert (
            len(indices) > 0
        ), "The number of indices must be strictly greater than 0."
        assert (
            len(indices) <= frame_count
        ), "The number of indices cannot exceed the number of frames."
        assert (
            sum([index > frame_count for index in indices]) == 0
        ), "At least one of the indices exceeds the number of frames."
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
                if return_rgb:
                    frames[index] = cv2.cvtColor(frames[index], cv2.COLOR_BGR2RGB)
                index += 1
            count += 1
        vidcap.release()

        if channel_first:
            frames = np.einsum("ijkl->iljk", frames)

        return frames

    ''' Deprecated in favor of more recently updated function that calls private function. 
    def extract_frames_from_indices(self, video_path, indices=None, channel_first=True):
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
        assert (
            frame_count > 0
        ), "Invalid video, number of frames must be greater than 0."

        if indices is None:
            # Instead, just call extract_all_frames, since this is what it defaults to.
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
    '''

    def extract_frames_from_indices(
        self, video_path, indices=None, channel_first=True, return_rgb=True
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
        return_rgb : {True, False}, bool, optional
            If True, then the output adheres to the RGB color format, by default True. Otherwise, the output adheres to the BGR color format. If a non-boolean input is passed, then ``return_rgb`` defaults to True.

        Returns
        -------
        frames : numpy.ndarray instance
            Numpy array containing the frames of the video, located at ``indices``. The output shape is either ``(T, C, H, W)`` or ``(T, H, W, C)`` depending on the value ``channel_first`` takes.

        """
        vidcap = cv2.VideoCapture(video_path)
        frames = self._extract_frames_from_indices(
            vidcap=vidcap,
            indices=indices,
            channel_first=channel_first,
            return_rgb=return_rgb,
        )

        return frames

    def extract_frames(
        self, video_path, start=0, end=None, channel_first=True, return_rgb=True
    ):
        """Extracts frames starting from ``start`` and ending at ``end``, and returns it as a Numpy array.

        If ``channel_first == True``, then the output has shape ``(T, C, H, W)`` and ``(T, H, W, C)`` otherwise. Here, T represents time, C represents channel, H represents height, and W represents width.

        Parameters
        ----------
        video_path : str
            The absolute path to the video file.
        channel_first : {False, True}, bool, optional
            If True then the output has shape ``(T, C, H, W)``, where the channel dimension comes before the spatial dimensions, by default True. Otherwise the output has shape ``(T, H, W, C)``. If a non-boolean input is passed, then ``channel_first`` defaults to True.
        return_rgb : {True, False}, bool, optional
            If True, then the output adheres to the RGB color format, by default True. Otherwise, the output adheres to the BGR color format. If a non-boolean input is passed, then ``return_rgb`` defaults to True.

        Returns
        -------
        frames : numpy.ndarray
            Numpy array containing the frames extracted from the video.

        """
        assert start >= 0, "The start index must be greater than or equal to zero."
        if type(channel_first) is not bool:
            channel_type = True
        if type(return_rgb) is not bool:
            return_rgb = True

        vidcap = cv2.VideoCapture(video_path)
        video_info = self._get_video_info(vidcap=vidcap)
        frame_count = video_info["frame_count"]
        frame_width = video_info["frame_width"]
        frame_height = video_info["frame_height"]
        assert (
            frame_count > 0
        ), "Invalid video, number of frames must be greater than 0."

        if end is None:
            end = frame_count
        assert end > 0, "The end index must be greater than 0."
        assert end > start, "The end index must be greater than the start index."

        indices = np.arange(start, end)
        frames = self._extract_frames_from_indices(
            vidcap=vidcap,
            indices=indices,
            channel_first=channel_first,
            return_rgb=return_rgb,
        )

        return frames

    def _extract_all_audio(self, video_path, sample_rate=44100):
        """Extracts the entire audio sequence from a video and returns it as a Numpy array.

        Parameters
        ----------
        video_path : str
            The absolute path to the video file.
        sample_rate : int, optional
            The sample rate of the audio, by default 44.1 kHz. If the sample rate is not an integer, then it is converted to one.

        Returns
        -------
        audio_signal : numpy.ndarray instance
            Numpy array containing the entire audio sequence from the video.

        """
        assert (
            sample_rate >= 1
        ), "The sampling rate must be greater than or equal to 1 Hz."
        if type(sample_rate) is not int:
            sample_rate = int(sample_rate)

        if self.verbose:
            if sample_rate > 44100:
                warnings.warn(
                    "The sampling rate is higher than the suggested 44.1 kHz. Proceed with caution."
                )

        audio_signal = moviepy.editor.AudioFileClip(
            video_path, fps=sample_rate
        ).to_soundarray()[:, -1]

        return audio_signal

    def extract_audio_from_indices(self, video_path, indices=None, sample_rate=44100):
        """Extracts audio samples from a video, according to ``indices``, and returns it as a Numpy array.

        Parameters
        ----------
        video_path : str
            The absolute path to the video file.
        indices : array-like, optional
            Array-like object containing the indices of the audio samples to extract, by default None. If None, then audio from the entire video is extracted.
        sample_rate : int, optional
            The sample rate of the audio, by default 44.1 kHz. If the sample rate is not an integer, then it is converted to one.

        Returns
        -------
        audio_signal : numpy.ndarray instance
            Numpy array containing the audio samples from the video, located at ``indices``.

        """
        audio_signal = self._extract_all_audio(
            video_path=video_path, sample_rate=sample_rate
        )
        if indices is None:
            return audio_signal
        else:
            indices = np.asarray(indices)
            n_samples = len(indices)
            assert n_samples > 0, "The number of indices must be greater than 0."
            assert n_samples <= len(
                audio_signal
            ), "The number of indices cannot exceed the total number of audio samples."
            assert (
                sum([index > n_samples for index in indices]) == 0
            ), "At least one of the indices exceeds the number of samples."
            assert (
                sum([index < 0 for index in indices]) == 0
            ), "At least one of the indices is negative."

            return audio_signals[indices]


class FramesProcessor:
    """Helper class used for processing video frames."""

    def __init__(self, verbose=True):
        """Instantiates a FramesProcessor object.

        Parameters
        ----------
        verbose : {True, False}, bool, optional
            If True then verbose output is sent to the system console, by default True.

        """
        self.verbose = verbose

    def _extract_landmarks(self, model, frame, device="cpu"):
        """Extracts landmarks for a single image, using the Face Alignment Network with the model parameters given by ``model``.

        Note that ``model`` encodes information about the face detection model.

        Parameters
        ----------
        model : face_alignment.FaceAlignment instance
            The model used for extracting the facial landmarks.
        device : torch.device instance, optional
            The device on which all computations are carried out.  
        frame : numpy.ndarray instance
            Numpy array containing the image used for extracting facial landmarks.
        
        Returns
        -------
        landmarks : numpy.ndarray
            Numpy array containing the facial landmarks in ``frame``. Note that the output may contain landmarks for multiple face(s), if they are detected.
        
        """
        landmarks = model.get_landmarks_from_image(frame)

        return landmarks

    def extract_landmarks(self, frame, device="cpu", detector="sfd"):
        """Extract landmarks for a single image, using the Face Alignment Network with model parameters given by ``model``.

        Note that ``model`` encodes information about the face detection model.

        Parameters
        ----------
        frame : numpy.ndarray instance
            Numpy array containing the image used for extracting facial landmarks.
        device : torch.device instance, optional
            The device on which all computations are carried out.
        detector : {"sfd", "dlib", "blazeface"}, str, optional
            The face detection scheme to be used, by default "sfd". The three supported schemes are S3FD, DLib, and BlazeFace. If none of these options are sent as input, then the default scheme is used.

        Returns
        -------
        landmarks : numpy.ndarray instance
            Numpy array containing the facial landmarks in ``frame``. Note that there may be multiple landmarks in the scenario where multiple face(s) are detected.

        """
        model = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._2D, device=device, face_detector=detector
        )
        landmarks = self._extract_landmarks(model=model, frame=frame, device=device)

        return landmarks

    def _batch_extract_landmarks(self, model, frames, device="cpu"):
        """Extracts a batch of facial landmarks from a batch of frames, using the Face Alignment Network with model parameters given by ``model``.

        Note that ``model`` encodes information about the face detection model.

        Parameters
        ----------
        model : face_alignment.FaceAlignment instance
            The model used for extracting the facial landmarks.
        frames : torch.Tensor instance
            Tensor containing the batch of frames used to extract facial landmarks.
        device : torch.device instance, optional
            Device on which all computations are carried out, by default "cpu".

        Returns
        -------
        batch_landmarks : numpy.ndarray instance
            Numpy array containing the batch of facial features in ``frames``. Note that there may be multiple landmarks for each frame, in the scenario where multiple face(s) are detected.
        
        """
        batch_landmarks = model.get_landmarks_from_batch(frames)

        return batch_landmarks

    def batch_extract_landmarks(self, frames, device="cpu", detector="sfd"):
        """Extracts a batch of facial landmarks from a batch of frames, using the Face Alignment Network.

        Parameters
        ----------
        frames : numpy.ndarray instance
            Numpy array containing a batch of images used to extract facial landmarks.
        device : torch.device
            Device on which all computations are carried out, by default "cpu".
        detector : {"sfd", "dlib", "blazeface"}, str, optional
            The face detection scheme to be used, by default "sfd". The three supported schemes are S3FD, DLib, and BlazeFace. If none of these options are sent as input, then the default scheme is used.

        Returns
        -------
        batch_landmarks : numpy.ndarray instance
            Numpy array containing the batch of facial features in ``frames``. Note that there may be multiple landmarks for each frame, in the scenario where multiple face(s) are detected.

        """
        batch = np.stack(frames)
        batch = torch.Tensor(batch)
        # pythonbatch = batch.transpose(0, 3, 1, 2)
        model = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._2D, device=device, face_detector=detector
        )
        batch_landmarks = self._batch_extract_landmarks(
            model=model, frames=batch, device=device
        )

        return batch_landmarks


class AudioProcessor:
    """Helper class used for processing audio."""

