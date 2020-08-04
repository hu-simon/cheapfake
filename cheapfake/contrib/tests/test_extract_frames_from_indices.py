"""
Python file that tests the extract_frames_from_indices function in the VideoFileProcessor class.

Entire test should be moved to unittesting.
"""

import os
import time
import warnings

import cv2
import numpy as np

import cheapfake.contrib.video_processor as video_processor


def test_one():
    """The expected return is a frame with size (10, 3, ..., ...)

    """
    print("Test #1")
    print("".join("-") * 7)

    video_path = "/Users/shu/Documents/Datasets/DFDC_small_subset/aagfhgtpmv/video/aagfhgtpmv.mp4"
    vidfileproc = video_processor.VideoFileProcessor(verbose=True)

    # Extract the first ten frames.
    start_time = time.time()
    frames = vidfileproc.extract_frames_from_indices(
        video_path=video_path, indices=range(10)
    )
    end_time = time.time()

    print("Frame array has shape {}".format(frames.shape))
    print("Entire operation took {} seconds".format(end_time - start_time))


def test_two():
    """The expected return is the entire video, so the output should have shape (300, 3, ..., ...)
    
    """
    print("Test #2")
    print("".join("-") * 7)

    video_path = "/Users/shu/Documents/Datasets/DFDC_small_subset/aagfhgtpmv/video/aagfhgtpmv.mp4"
    vidfileproc = video_processor.VideoFileProcessor(verbose=True)

    # Extract the first ten frames.
    start_time = time.time()
    frames = vidfileproc.extract_frames_from_indices(video_path=video_path)
    end_time = time.time()

    print("Frame array has shape {}".format(frames.shape))
    print("Entire operation took {} seconds".format(end_time - start_time))


def test_three():
    """The expected return is an error, saying that the file cannot be opened.

    """
    print("Test #3")
    print("".join("-") * 7)

    video_path = (
        "/Users/shu/Documents/Datasets/DFDC_small_subset/aagfhgtpmv/video/aagfhgtpm.mp4"
    )
    vidfileproc = video_processor.VideoFileProcessor(verbose=True)

    try:
        frames = vidfileproc.extract_frames_from_indices(
            video_path=video_path, indices=range(100)
        )
    except:
        print("Video file cannot be opened.")


def test_four():
    """The expected return is an assertion error, saying that the number of indices must be greater than zero.
    
    """
    print("Test #4")
    print("".join("-") * 7)

    video_path = "/Users/shu/Documents/Datasets/DFDC_small_subset/aagfhgtpmv/video/aagfhgtpmv.mp4"
    vidfileproc = video_processor.VideoFileProcessor(verbose=True)

    start_time = time.time()
    try:
        frames = vidfileproc.extract_frames_from_indices(
            video_path=video_path, indices=[]
        )
        end_time = time.time()

        print("Frame array has shape {}".format(frames.shape))
        print("Entire operation took {} seconds".format(end_time - start_time))

    except AssertionError as error:
        print(error)


def test_five():
    """The expected return is an assertion error saying that the number of indices exceeds the number of frames.

    """
    print("Test #5")
    print("".join("-") * 7)

    video_path = "/Users/shu/Documents/Datasets/DFDC_small_subset/aagfhgtpmv/video/aagfhgtpmv.mp4"
    vidfileproc = video_processor.VideoFileProcessor(verbose=True)

    start_time = time.time()
    try:
        frames = vidfileproc.extract_frames_from_indices(
            video_path=video_path, indices=range(1000)
        )
        end_time = time.time()

        print("Frame array has shape {}".format(frames.shape))
        print("Entire operation took {} seconds".format(end_time - start_time))
    except AssertionError as error:
        print(error)


if __name__ == "__main__":
    print("".join("-") * 80)
    print("TESTING EXTRACT_FRAMES_FROM_INDICES()")
    print("".join("-") * 80)

    test_one()
    test_two()
    test_three()
    test_four()
    test_five()
