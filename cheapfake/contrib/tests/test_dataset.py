"""
Python file that tests the functionality of the dataset.py module.

Should eventually be moved to a unittest.
"""

import os
import time

import cv2
import torch
import numpy as np
import torchvision
import cheapfake.contrib.dataset as dataset
import cheapfake.contrib.video_processor as video_processor


def test_one(root_path):
    """Test to see why the audio is not being chunked properly.

    Conclusion: The reason the audio was not chunking properly before is because of stragglers due to the frame rate being approximated as 30 fps, when it is really 29.97 fps. 
    """
    dfdataset = dataset.DeepFakeDataset(root_path)
    videofileprocessor = video_processor.VideoFileProcessor(verbose=True)
    video_path = dfdataset.video_paths[0]

    frames = videofileprocessor.extract_all_frames(video_path=video_path)
    audio = videofileprocessor._extract_all_audio(video_path=video_path)

    print(frames.shape)
    print(audio.shape)

    chunked_audio = list()
    for k in range(0, len(audio), 32000):
        chunked_audio.append(audio[k : k + 32000])
    chunked_audio = chunked_audio[:-1]
    for chunk in chunked_audio:
        print(chunk.shape)

    chunked_audio = np.array(chunked_audio)
    print(chunked_audio.shape)
    chunked_audio = torch.from_numpy(chunked_audio)
    print(chunked_audio.shape)


def test_two(root_path):
    """Test to see if the audio and frames are chunked properly.
    
    """
    dfdataset = dataset.DeepFakeDataset(root_path)
    videofileprocessor = video_processor.VideoFileProcessor(verbose=True)
    video_path = dfdataset.video_paths[0]

    start_time = time.time()
    frames, audio = dfdataset.__getitem__(0)
    end_time = time.time()

    print(frames.shape)
    print(audio.shape)
    print("Entire operation took {} seconds".format(end_time - start_time))


def _resize_transform(images, scale_factor=4, return_torch=True):
    """Resizes the image by a factor of ``scale_factor``.

    Parameters
    ----------
    images : numpy.ndarray or torch.Tensor instance
        The input batch images to be resized.
    scale_factor : int, optional
        The scaling factor used to determine how much the image should be resized, by default 4.
    
    Returns
    -------
    images : numpy.ndarray or torch.Tensor instance
        The resized batch images.

    """
    images = np.einsum("ijkl->iklj", images)
    new_height = int(images[0].shape[0] / scale_factor)
    new_width = int(images[0].shape[1] / scale_factor)
    new_images = np.empty((images.shape[0], new_height, new_width, 3))
    for k, image in enumerate(images):
        new_images[k] = cv2.resize(
            image, dsize=(new_width, new_height), interpolation=cv2.INTER_LANCZOS4
        )

    return np.einsum("ijkl->iljk", new_images)


def test_three(video_path):
    """Tests resizing of the data using the _resize_transform function.

    """
    dfdataset = dataset.DeepFakeDataset(
        root_path=root_path, frame_transform=_resize_transform
    )

    start_time = time.time()
    frames, audio = dfdataset.__getitem__(0)
    end_time = time.time()

    print(frames.shape)
    print(audio.shape)
    print("Entire operation took {} seconds".format(end_time - start_time))


def test_four(video_path):
    """Tests tensor conversion of the data using the torchvision.transforms.ToTensor() function.

    """
    dfdataset = dataset.DeepFakeDataset(
        root_path=root_path, frame_transform=_to_tensor, n_seconds=3
    )
    start_time = time.time()
    frames, audio = dfdataset.__getitem__(0)
    end_time = time.time()

    print(frames.shape)
    print(audio.shape)
    print("Entire operation took {} seconds".format(end_time - start_time))


if __name__ == "__main__":
    root_path = "/Users/shu/Documents/Datasets/DFDC_small_subset_raw"
    # test_one(root_path)
    # test_two(root_path)
    test_three(root_path)
    # test_four(root_path)
