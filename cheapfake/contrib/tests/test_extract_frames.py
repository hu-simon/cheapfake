"""
Python file that tests the extract_frames method in the VideoFileProcessor clas.

Entire test should bemoved to unittesting.
"""

import os
import time
import warnings

import cv2
import numpy as np

import cheapfake.contrib.video_processor as video_processor


def test_one():
    """This should return the frames from 0 to 20. Thus, it should have output shape (20, 3, ..., ...)

    """
    print("Test #1")
    print("".join("-") * 7)

    video_path = "/Users/shu/Documents/Datasets/DFDC_small_subset/aagfhgtpmv/video/aagfhgtpmv.mp4"
    vidfileproc = video_processor.VideoFileProcessor(verbose=True)

    # Extract the first ten frames.
    start_time = time.time()
    frames = vidfileproc.extract_frames(video_path=video_path, start=0, end=20)
    end_time = time.time()

    print("Frame array has shape {}".format(frames.shape))
    print("Entire operation took {} seconds".format(end_time - start_time))


def test_two():
    """This should return the frames from the entire video. Thus, it should have output shape (300, ..., ...)
    
    """
    print("Test #2")
    print("".join("-") * 7)

    video_path = "/Users/shu/Documents/Datasets/DFDC_small_subset/aagfhgtpmv/video/aagfhgtpmv.mp4"
    vidfileproc = video_processor.VideoFileProcessor(verbose=True)

    # Extract the first ten frames.
    start_time = time.time()
    frames = vidfileproc.extract_frames(video_path=video_path)
    end_time = time.time()

    print("Frame array has shape {}".format(frames.shape))
    print("Entire operation took {} seconds".format(end_time - start_time))


if __name__ == "__main__":
    print("".join("-") * 80)
    print("TESTING EXTRACT_FRAMES()")
    print("".join("-") * 80)

    test_one()
    test_two()
    # test_three()
    # test_four()
    # test_five()

