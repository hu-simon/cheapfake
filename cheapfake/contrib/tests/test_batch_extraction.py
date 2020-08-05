"""
Python script that tests the batch extraction method for the FrameProcessor class.

Should eventually move these to unittests.

Idea: Maybe it is because the image is too large, so try making the image smaller to pass into S3FD and then try again with batches!
"""

import os
import time

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

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


def test_two():
    """In this test, we predict a sequence of three frames and then print them.

    """
    video_path = "/Users/shu/Documents/Datasets/DFDC_small_subset/aagfhgtpmv/video/aagfhgtpmv.mp4"
    vidfileproc = video_processor.VideoFileProcessor(verbose=True)
    frames_all = vidfileproc.extract_all_frames(
        video_path=video_path, channel_first=True, return_rgb=True
    )
    frames_subset = frames_all[:3]
    framesproc = video_processor.FramesProcessor(verbose=True)
    start_time = time.time()
    batch_landmarks = framesproc.batch_extract_landmarks(frames_subset, device="cpu")
    end_time = time.time()
    print("Predicting landmarks took {} seconds.".format(end_time - start_time))
    frames_subset = np.einsum("ijkl->iklj", frames_subset)

    fig = plt.figure(figsize=(10, 5))
    for k, landmarks in enumerate(batch_landmarks):
        plt.subplot(1, 3, k + 1)
        plt.imshow(frames_subset[k])
        for landmark in landmarks:
            plt.scatter(landmark[:, 0], landmark[:, 1], 1)
    plt.show()


def test_three():
    """In this test, we predict an entire sequence of 75 frames and see how long it takes.

    """
    video_path = "/Users/shu/Documents/Datasets/DFDC_small_subset/aagfhgtpmv/video/aagfhgtpmv.mp4"
    vidfileproc = video_processor.VideoFileProcessor(verbose=True)
    frames_all = vidfileproc.extract_all_frames(
        video_path=video_path, channel_first=True, return_rgb=True
    )
    frames_subset = frames_all[:30]
    framesproc = video_processor.FramesProcessor(verbose=True)
    start_time = time.time()
    batch_landmarks = framesproc.batch_extract_landmarks(frames_subset, device="cpu")
    end_time = time.time()

    print(
        "Processed {} frames in {} seconds.".format(
            len(batch_landmarks), end_time - start_time
        )
    )


def test_four():
    """In this test, we predict an entire sequence of 75 frames, with resizing, and see how long it takes.

    """
    video_path = "/Users/shu/Documents/Datasets/DFDC_small_subset/aagfhgtpmv/video/aagfhgtpmv.mp4"
    vidfileproc = video_processor.VideoFileProcessor(verbose=True)
    frames = vidfileproc.extract_all_frames(
        video_path=video_path, channel_first=False, return_rgb=True
    )
    frames_subset = frames[:75]

    frames_resized = np.asarray(
        [
            cv2.resize(frame, dsize=(640, 640), interpolation=cv2.INTER_CUBIC)
            for frame in frames_subset
        ]
    )
    dh, dw = frames_resized[0].shape[0] / 640.0, frames_resized[0].shape[1] / 640.0
    frames_resized = np.einsum("ijkl->iljk", frames_resized)

    framesproc = video_processor.FramesProcessor(verbose=True)

    start_time = time.time()
    landmarks_batch = framesproc.batch_extract_landmarks(frames_resized)
    end_time = time.time()

    fig = plt.figure(figsize=(10, 5))
    for k, landmarks in enumerate(landmarks_batch[:3]):
        plt.subplot(1, 3, k + 1)
        plt.imshow(frames_subset[k])
        for landmark in landmarks:
            plt.scatter(landmark[:, 0], landmark[:, 1], 1)
    plt.show()


def test_four_26():
    """In this test, we predict an entire sequence of 75 frames, with resizing and see how long it takes. This test is done on the .26 machine.

    """
    video_path = (
        "/home/shu/shu/Datasets/DFDC_small_subset/aagfhgtpmv/video/aagfhgtpmv.mp4"
    )
    vidfileproc = video_processor.VideoFileProcessor(verbose=True)
    frames = vidfileproc.extract_all_frames(
        video_path=video_path, channel_first=False, return_rgb=True
    )
    frames_subset = frames[:75]

    frames_resized = np.asarray(
        [
            cv2.resize(frame, dsize=(640, 640), interpolation=cv2.INTER_CUBIC)
            for frame in frames_subset
        ]
    )
    dh, dw = frames_resized[0].shape[0] / 640.0, frames_resized[0].shape[1] / 640.0
    frames_resized = torch.Tensor(np.einsum("ijkl->iljk", frames_resized)).cuda()

    framesproc = video_processor.FramesProcessor(verbose=True)

    start_time = time.time()
    landmarks_batch = framesproc.batch_extract_landmarks(frames_resized, device="cuda")
    end_time = time.time()

    fig = plt.figure(figsize=(10, 5))
    for k, landmarks in enumerate(landmarks_batch[:3]):
        plt.subplot(1, 3, k + 1)
        plt.imshow(frames_subset[k])
        for landmark in landmarks:
            plt.scatter(landmark[:, 0], landmark[:, 1], 1)
    plt.show()


if __name__ == "__main__":
    # test_one()
    # test_two()
    # test_three()
    test_four_26()
