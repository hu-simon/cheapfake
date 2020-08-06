import os
import time
import warnings

import cv2
import torch
import librosa
import numpy as np
import moviepy.editor
import face_alignment
import librosa.display
import matplotlib.pyplot as plt


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

        Notes
        -----
        This function is not well-tested. Be cautious with use.

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

    def _extract_landmarks(self, model, frame, channel_first=True):
        """Extracts landmarks for a single image, using the Face Alignment Network.

        Note that ``model`` encodes information about the face detection scheme.

        Parameters
        ----------
        model : face_alignment.FaceAlignment instance
            The model used for extracting the facial landmarks.
        frame : numpy.ndarray instance
            Numpy array containing the image used for extracting the facial landmarks.
        channels_first : {True, False}, bool, optional
            If True, then the input is assumed to take the shape ``(T, C, H, W)``, by default True. Otherwise, the input is assumed to take the shape ``(T, H, W, C)``. If a non-boolean input is sent, then ``channel_first`` defaults to True.

        Returns
        -------
        landmarks : list
            List containing the facial landmarks in ``frame``. Note that the output may contain landmarks for multiple face(s), if they are detected. 

        """
        if type(channels_first) is not bool:
            channels_first = True

        if channels_first is False:
            frame = np.einsum("ijk->jki", frame)

        landmarks = model.get_landmarks_from_image(frame)

        return landmarks

    def extract_landmarks(
        self, frame, channels_first=True, device="cpu", detector="sfd"
    ):
        """Extracts landmarks for a single image, using the Face Alignment Network.

        Note that ``model`` encodes information about the face detection scheme. 

        Parameters
        ----------
        frame : numpy.ndarray instance
            Numpy array containing the image used for extracting the facial landmarks.
        channels_first : {True, False}, bool, optional
            If True, then the input is assumed to take the shape ``(T, C, H, W)``, by default True. Otherwise, the input is assumed to take the shape ``(T, H, W, C)``. If a non-boolean input is sent, then ``channels_first`` defaults to True.
        device : {"cpu", "cuda"}, str, optional
            The device on which all computations are carried out, by default "cpu".
        detector : {"sfd", "blazeface", "dlib"}, str, optional
            The face detection scheme, by default "sfd". The three supported schemes are S3FD, BlazeFace, and DLib. If none of these options are sent as input, then the default scheme is used.

        Returns
        -------
        landmarks : list
            Numpy array containing the facial landmarks in ``frame``. Note that there may be multiple landmarks in the scenario where multiple face(s) are detected.

        """
        if device == "cuda":
            frame = torch.from_numpy(frame).cuda()
        else:
            frame = torch.from_numpy(frame)

        model = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._2D, device=device, face_detector=detector
        )
        landmarks = self._extract_landmarks(
            model=model, frame=frame, channels_first=channels_first
        )

    def _batch_extract_landmarks(self, model, frames):
        """Extracts facial landmarks from a batch of frames, using the Face Alignment Network. 

        Note that ``model`` encodes information about the face detection model.

        Parameters
        ----------
        model : face_alignment.FaceAlignment instance
            The mode used for extracting the facial landmarks.
        frames : list (of numpy.ndarray instances)
            List containing the images used for extracting the facial landmarks.
        
        Returns
        -------
        batch_landmarks : list
            List containing the batch of facial features from ``frames``. Note that there may be multiple landmarks for each frame, in the scenario where multiple face(s) are detected.

        """
        batch_landmarks = model.get_landmarks_from_batch(frames)

        return batch_landmarks

    def batch_extract_landmarks(
        self, frames, device="cpu", detector="sfd", channels_first=True
    ):
        """Extracts facial landmarks from a batch of frames, using the Face Alignment Network.

        Note that ``model`` encodes information about the face detection model.

        Parameters
        ----------
        frames : list (of numpy.ndarray instances)
            List containing the images frames used to predict facial landmarks.
        device : {"cpu", "cuda"}, str, optional
            The device on which all computations are carried out, by default "cpu".
        detector : {"sfd", "blazeface", "dlib"}, str, optional
            The face detection scheme, by default "sfd". The three supported schemes are S3FD, BlazeFace, and DLib. If none of these options are sent as input, then the default scheme is used.
        channels_first : {True, False}, bool, optional
            If True, then the input is assumed to take the shape ``(T, C, H, W)``, by default True. Otherwise, the input is assumed to take the shapep ``(T, H, W, C)``. If a non-boolean input is sent, then ``channels_first`` defaults to True. 

        Returns
        -------
        batch_landmarks : list
            List containing the batch of facial features from ``frames``. Note that there may be multiple landmarks for frame, in the scenario where multiple face(s) are detected.

        """
        if type(channels_first) is not bool:
            channels_first = True

        if channels_first is False:
            frames = np.einsum("ijkl->iljk", frames)

        batch = np.stack(frames)
        if device == "cuda":
            batch = torch.from_numpy(batch).cuda()
        else:
            batch = torch.from_numpy(batch)

        model = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._2D, device=device, face_detector=detector
        )
        batch_landmarks = self._batch_extract_landmarks(model=model, frames=batch)

        return batch_landmarks


class AudioProcessor:
    """Helper class used for processing audio."""

    def __init__(self, verbose=True):
        """Instantiates a new AudioProcessor class.

        Parameters
        ----------
        verbose : {True, False}, bool, optional
            If True then verbose output is sent to the system console, by default True.
        
        """
        self.verbose = verbose

    def extract_spectrogram(
        audio_signal, sample_rate=44100, return_log=True, return_torch=True
    ):
        """Extracts the spectrogram from the audio_signal. 

        The output can either be in amplitude or dB, depending on ``return_log``.

        Parameters
        ----------
        audio_signal : numpy.ndarray instance
            The audio signal used to extract the spectrogram information.
        sample_rate : int, optional
            The sample rate of the audio signal, by default 44.1 kHz. If a float is passed as input, then it will be cast as an integer. If a non-integer or non-float input is passed as input, then the default audio sample rate is used.
        return_log : {True, False}, bool, optional
            If True then the output is the log-spectrogram, which is in dB units, by default True. Otherwise, the output is the spectrogram, which is in amplitude units. If a non-boolean input is passed then the default value is used.
        return_torch : {True, False}, bool, optional
            If True, then the output is converted to a torch.Tensor instance, by default True. If a non-boolean input is passed then the default value is used. 

        Returns
        -------
        spectrogram : numpy.ndarray or torch.Tensor instance
            The spectrogram, in amplitude or decibel units, of the audio signal returned as either a Numpy array or a Torch tensor.

        """
        # Handle invalid inputs.
        if type(sample_rate) is float:
            sample_rate = int(sample_rate)
        if type(sample_rate) is not float or type(sample_rate) is not int:
            sample_rate = 44100
        if type(return_log) is not bool:
            return_log = True
        if type(return_torch) is not bool:
            return_torch = True

        spectrogram = librosa.feature.melspectrogram(y=audio_signal, sr=sample_rate)

        if return_log:
            spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)

        if return_torch:
            spectrogram = torch.from_numpy(spectrogram)

        return spectrogram

    def plot_spectrogram(spectrogram, sample_rate=44100, show_colorbar=False):
        """Plots the spectrogram of an audio signal, sampled at ``sample_rate`` Hz.

        Parameters
        ----------
        spectrogram : numpy.ndarray or torch.Tensor instance
            The spectrogram, in amplitude or decibel units, of the audio signal as either a Numpy array or a Torch tensor.
        sample_rate : int, optional
            The sample rate of the audio signal, by default 44.1 kHz. If a float is passed as input, then it will be cast as an integer. If a non-integer or non-float input is passed then the default audio sample rate is used.
        show_colorbar : {False, True}, bool, optional
            If True then a colorbar is shown next to the spectrogram, by default False. If a non-boolean input is passed then the default value is used.

        """
        if type(sample_rate) is float:
            sample_rate = int(sample_rate)
        if type(sample_rate) is not float or type(sample_rate) is not int:
            sample_rate = 44100

        if type(spectrogram) is torch.Tensor:
            spectrogram = spectrogram.numpy()

        plt.figure(figsize=(12, 4))
        librosa.display.specshow(
            spectrogram, sr=sample_rate, x_axis="time", y_axis="mel"
        )
        if show_colorbar:
            plt.colorbar(format="%+02.0f dB")
        plt.show()
