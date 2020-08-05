"""
Python script that tests resizing the images to (640 x 640) pixels and then feeding the input to 
FAN, to see if there is accuracy and speed up. If so, then we may be able to use this strategy to make S3FD faster!

Eventually should be moved into a unittest.
"""

import os
import time

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

import cheapfake.contrib.video_processor as video_processor


def test_one():
    """In this test, we resize a single image and then pass it to the FAN. The landmarks are then scaled and compared to the "ground-truth" landmark locations.

    """
    video_path = "/Users/shu/Documents/Datasets/DFDC_small_subset/aagfhgtpmv/video/aagfhgtpmv.mp4"
    vidfileproc = video_processor.VideoFileProcessor(
        verbose=True, channel_first=True, return_rgb=True
    )
    frames = vidfileproc.extract_all_frames(
        video_path=video_path, channel_first=False, return_rgb=True
    )
    frame = frames[0]
    frame_resized = cv2.resize(frame, dsize=(640, 640), interpolation=cv2.INTER_CUBIC)
    dh, dw = frame.shape[0] / 640.0, frame.shape[1] / 640.0

    print(frame_resized.shape)
    print(
        "The height and width resize factors are {} and {} respectively.".format(dh, dw)
    )

    framesproc = video_processor.FramesProcessor(verbose=True)

    start_time = time.time()
    landmarks_ground_truth = framesproc.extract_landmarks(frame)[-1]
    end_time = time.time()
    print("The full-size prediction took {} seconds.".format(end_time - start_time))

    start_time = time.time()
    landmarks_resized = framesproc.extract_landmarks(frame_resized)[-1]
    end_time = time.time()
    print("The resized prediction took {} seconds.".format(end_time - start_time))

    landmarks_resized[:, 0] *= dw
    landmarks_resized[:, 1] *= dh

    # print(landmarks_resized - landmarks_ground_truth)
    fig = plt.figure(figsize=(10, 7))

    plt.subplot(1, 2, 1)
    plt.imshow(frame)
    for landmark in landmarks_ground_truth:
        plt.scatter(landmark[0], landmark[1], 1, c="b")

    plt.subplot(1, 2, 2)
    plt.imshow(frame)
    for landmark in landmarks_resized:
        plt.scatter(landmark[0], landmark[1], 1, c="b")

    plt.show()


if __name__ == "__main__":
    test_one()
