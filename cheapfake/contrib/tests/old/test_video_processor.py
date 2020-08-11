"""
Python file that tests all the functions in test_video_processor.
"""

import os
import time
import warnings

import cv2
import numpy as np
import moviepy.editor

import cheapfake.contrib.video_processor as video_processor


def test_extract_all_frames():
    """Extracts all frames from the video.
    
    """
    video_path = "/Users/shu/Documents/Datasets/DFDC_small_subset/aagfhgtpmv/video/aagfhgtpmv.mp4"
    vidfileproc = video_processor.VideoFileProcessor(verbose=True)

    # Extract all frames.
    start_time = time.time()
    frames = vidfileproc.extract_all_frames(video_path=video_path)
    end_time = time.time()

    # Check result.
    print("".join("-") * 80)
    print("TESTING EXTRACTING_ALL_FRAMES")
    print("".join("-") * 80)
    print("Frame array has shape {}".format(frames.shape))
    print("Entire operation took {} seconds".format(end_time - start_time))


if __name__ == "__main__":
    test_extract_all_frames()
