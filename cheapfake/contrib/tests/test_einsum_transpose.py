"""
Python function that compares the speed of OpenCV and Numpy for BGR -> RGB.

Conclusion: just use OpenCV...
"""

import os
import time

import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def test_one():
    """
    image = cv2.imread(frame_path)

    start_time = time.time()
    opencv_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    end_time = time.time()
    print("OpenCV's operation took {} seconds.".format(end_time - start_time))

    start_time = time.time()
    numpy_image = image[:, :, ::-1]
    end_time = time.time()
    print("Numpy's operation took {} seconds.".format(end_time - start_time))

    plt.figure()
    plt.imshow(numpy_image)
    plt.show()
    """

    frames_path = [
        "/Users/shu/Documents/frame1.png",
        "/Users/shu/Documents/frame2.png",
        "/Users/shu/Documents/frame3.png",
        "/Users/shu/Documents/frame4.png",
        "/Users/shu/Documents/frame5.png",
    ]
    images = np.empty((len(frames_path), 1080, 1920, 3))
    for k, frame_path in enumerate(frames_path):
        images[k] = cv2.imread(frame_path)
    images = np.float32(images)

    opencv_images = np.empty((len(frames_path), 1080, 1920, 3))
    start_time = time.time()
    for k, image in enumerate(images):
        opencv_images[k] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    end_time = time.time()
    opencv_time = end_time - start_time
    # print("OpenCV's operation took {} seconds.".format(end_time - start_time))

    numpy_images = np.empty((len(frames_path), 1080, 1920, 3))
    start_time = time.time()
    for k, image in enumerate(images):
        numpy_images[k] = image[:, :, ::-1]
    end_time = time.time()
    numpy_time = end_time - start_time
    # print("Numpy's operation took {} seconds.".format(end_time - start_time))

    numpy_flip_images = np.empty((len(frames_path), 1080, 1920, 3))
    start_time = time.time()
    for k, image in enumerate(images):
        numpy_flip_images[k] = np.flip(image, axis=2)
    end_time = time.time()
    numpy_flip_time = end_time - start_time
    # print("Numpy's flip operation took {} seconds.".format(end_time - start_time))

    return opencv_time, numpy_time, numpy_flip_time


def test_two():
    """Tests to see if Numpy flip works.

    """
    """
    frames_path = [
        "/Users/shu/Documents/frame1.png",
        "/Users/shu/Documents/frame2.png",
        "/Users/shu/Documents/frame3.png",
        "/Users/shu/Documents/frame4.png",
        "/Users/shu/Documents/frame5.png",
    ]
    """
    frame_path = "/Users/shu/Documents/frame1.png"
    image = cv2.imread(frame_path)

    plt.figure()
    plt.imshow(np.flip(image, axis=2))
    plt.show()


if __name__ == "__main__":
    """
    opencv_total = 0
    numpy_total = 0
    numpy_flip_total = 0
    for _ in tqdm(range(100)):
        opencv_time, numpy_time, numpy_flip_time = test_one()
        opencv_total += opencv_time
        numpy_total += numpy_time
        numpy_flip_total += numpy_flip_time

    print("OpenCV's average is {} seconds.".format(opencv_total / 100.0))
    print("Numpy's average is {} seconds.".format(numpy_total / 100.0))
    print("Numpy's flip average is {} seconds.".format(numpy_flip_total / 100.0))
    """
    test_two()
