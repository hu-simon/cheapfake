import os
import time

import cv2
import numpy as np

import cheapfake.contrib.dataset as dataset
import cheapfake.contrib.transforms as transforms
import cheapfake.contrib.video_processor as video_processor


def test_one(root_path):
    dfdataset = dataset.DeepFakeDataset(
        root_path=root_path,
        return_tensor=False,
        sequential_frames=False,
        sequential_audio=True,
        stochastic=True,
    )
    frames, audio, _ = dfdataset.__getitem__(0)
    frames = np.einsum("ijkl->jikl", frames)
    frames = transforms.BatchRescale(output_size=(64, 128))(frames)
    print(frames.shape)


def test_two():
    rescale_transform = transforms.BatchRescale(output_size=(64, 128))


if __name__ == "__main__":
    root_path = "/Users/shu/Documents/Datasets/DFDC_small_subset_raw"
    # test_one(root_path=root_path)
    test_two()
