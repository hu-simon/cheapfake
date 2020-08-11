"""
Python file that tests the extract_all_frames function in the VideoFileProcessor class.

Entire test should be moved to unittesting.

All two tests have passed.
"""

import os
import time
import warnings

import cv2
import numpy as np

import cheapfake.contrib.video_processor as video_processor


def test_one():
    """This test should pass, and should return a frame with size (300, 3, ..., ...)
    
    """
    print("Test #1")
    print("".join("-") * 7)

    video_path = "/Users/shu/Documents/Datasets/DFDC_small_subset/aagfhgtpmv/video/aagfhgtpmv.mp4"
    vidfileproc = video_processor.VideoFileProcessor(verbose=True)

    # Extract all frames.
    start_time = time.time()
    frames = vidfileproc.extract_all_frames(video_path=video_path)
    end_time = time.time()

    print("Frame array has shape {}".format(frames.shape))
    print("Entire operation took {} seconds".format(end_time - start_time))


def test_two():
    """This test should fail since the video does not exist.

    """
    print("\n")
    print("Test #2")
    print("".join("-") * 7)

    video_path = (
        "/Users/shu/Documents/Datasets/DFDC_small_subset/aagfhgtpmv/video/aagfhgtpm.mp4"
    )
    vidfileproc = video_processor.VideoFileProcessor(verbose=True)

    try:
        frames = vidfileproc.extract_all_frames(video_path=video_path)
    except:
        print("Video file does not exist.\n")


if __name__ == "__main__":
    print("".join("-") * 80)
    print("TESTING EXTRACT_ALL_FRAMES()")
    print("".join("-") * 80)

    test_one()
    test_two()

