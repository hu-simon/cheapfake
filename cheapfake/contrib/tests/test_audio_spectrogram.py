"""
Python file that tests the audio extraction features in video_processing.py

Eventually should be moved to a unittest.

Spectrogram stuff works, so we are good to go for putting it into the VGG model.
"""

import os
import time

import cheapfake.contrib.video_processor as video_processor


def test_audio_extract(video_path):
    videofileprocessor = video_processor.VideoFileProcessor(verbose=True)

    start_time = time.time()
    audio_signal = videofileprocessor._extract_all_audio(video_path=video_path)
    end_time = time.time()
    print("Entire operation took {} seconds.".format(end_time - start_time))
    print("The audio has shape {}.".format(audio_signal.shape))


def test_spectrogram_full_audio(video_path):
    """Computes and displays the spectrogram for the entire audio signal.

    """
    videofileprocessor = video_processor.VideoFileProcessor(verbose=True)
    audioprocessor = video_processor.AudioProcessor(verbose=True)
    audio_signal = videofileprocessor._extract_all_audio(video_path=video_path)

    start_time = time.time()
    spectrogram = audioprocessor.extract_spectrogram(
        audio_signal=audio_signal,
        sample_rate=16000,
        return_log=True,
        return_torch=False,
    )
    end_time = time.time()
    print("The entire operation took {} seconds.".format(end_time - start_time))

    # Plot the spectrogram.
    audioprocessor.plot_spectrogram(
        spectrogram=spectrogram, sample_rate=16000, show_colorbar=True
    )


def test_spectrogram_partial_audio(video_path):
    """Computes and displays the spectrogram for a second of audio.

    """
    videofileprocessor = video_processor.VideoFileProcessor(verbose=True)
    audioprocessor = video_processor.AudioProcessor(verbose=True)
    audio_signal = videofileprocessor._extract_all_audio(video_path=video_path)
    audio_signal = audio_signal[:16000]

    start_time = time.time()
    spectrogram = audioprocessor.extract_spectrogram(
        audio_signal=audio_signal,
        sample_rate=16000,
        return_log=True,
        return_torch=False,
    )
    end_time = time.time()
    print("The entire operation took {} seconds.".format(end_time - start_time))
    print("The shape of the partial spectrogram is {}.".format(spectrogram.shape))

    # Plot the spectrogram.
    audioprocessor.plot_spectrogram(
        spectrogram=spectrogram, sample_rate=16000, show_colorbar=True
    )


if __name__ == "__main__":
    video_path = "/Users/shu/Documents/Datasets/DFDC_small_subset/aagfhgtpmv/video/aagfhgtpmv.mp4"

    test_audio_extract(video_path)
    test_spectrogram_full_audio(video_path)
    test_spectrogram_partial_audio(video_path)
