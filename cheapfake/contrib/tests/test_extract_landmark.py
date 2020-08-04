"""
Python file that tests the extract_landmarks function from the FrameProcessor class.

This should eventually be moved to unittesting.
"""

import os
import time

import matplotlib.pyplot as plt
import cheapfake.contrib.video_processor as video_processor


def test_one():
    """This test should return the facial landmarks for a single image.

    """
    video_path = "/Users/shu/Documents/Datasets/DFDC_small_subset/aagfhgtpmv/video/aagfhgtpmv.mp4"
    vidfileproc = video_processor.VideoFileProcessor(verbose=True)
    frames_all = vidfileproc.extract_all_frames(
        video_path=video_path, channel_first=False
    )
    frame_single = frames_all[0]
    framesproc = video_processor.FramesProcessor(verbose=True)
    landmarks = framesproc.extract_landmarks(
        frame_single, device="cpu", detector="sfd"
    )[-1]

    plt.figure()
    plt.imshow(frame_single)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], 1)
    plt.show()


if __name__ == "__main__":
    test_one()
