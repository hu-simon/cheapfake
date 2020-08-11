"""
Python file that tests model integration between the VGGVox model and video_processor.

Should eventually be moved to a unittest.
"""

import os
import time

import torch
import numpy as np
import matplotlib.pyplot as plt

import cheapfake.contrib.video_processor as video_processor
from cheapfake.mmid.audio_models.VGGVox import VGGVox


def test_one(video_path):
    """In this test, we run the VGG model through one training sample.

    """
    videofileprocessor = video_processor.VideoFileProcessor()
    audioprocessor = video_processor.AudioProcessor()

    audio_signal = videofileprocessor._extract_all_audio(video_path=video_path)
    spectrogram = audioprocessor.extract_spectrogram(
        audio_signal=audio_signal, sample_rate=16000, return_log=True, return_torch=True
    )
    print(audio_signal.shape)
    stft = audioprocessor.extract_stft(audio_signal=audio_signal, return_torch=True)
    # print(stft.shape)

    model = VGGVox()
    result = model(torch.from_numpy(audio_signal)[:16000].float())

    print(result.shape)


if __name__ == "__main__":
    video_path = "/Users/shu/Documents/Datasets/DFDC_small_subset/aagfhgtpmv/video/aagfhgtpmv.mp4"

    test_one(video_path)
