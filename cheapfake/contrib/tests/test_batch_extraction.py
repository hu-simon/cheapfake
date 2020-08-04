"""
Python script that tests the batch extraction method for the FrameProcessor class.

Should eventually move these to unittests.
"""

import os
import time

import cv2
import torch
import numpy as np

import cheapfake.contrib.video_processor as video_processor


def test_one():
    """In this test, we predict an entire sequence of 75 frames.

    """
    video_path = "/Users/shu/Documents/Datasets/DFDC_small_subset/aagfhgtpmv/video/aagfhgtpmv.mp4"
    vidfileproc = video_processor.VideoFileProcessor(verbose=True)
    frames_all = vidfileproc.extract_all_frames(
        video_path=video_path, channel_first=True
    )
    frames_subset = frames_all[:2]
    framesproc = video_processor.FramesProcessor(verbose=True)
    batch_landmarks = framesproc.batch_extract_landmarks(frames_subset, device="cpu")

    print(len(batch_landmarks))


if __name__ == "__main__":
    test_one()
