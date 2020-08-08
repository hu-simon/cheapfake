"""
Python file that tests the functionality of the dataset.py module.

Should eventually be moved to a unittest.
"""

import os
import time

import torch
import numpy as np
import cheapfake.contrib.dataset as dataset


def test_one(root_path):
    """Test to see if the frames and audio are chunked properly.

    """
    dfdataset = dataset.DeepFakeDataset(root_path)


def test_two(root_path):
    """Test to determine why the audio wont be chunked. 
    """


if __name__ == "__main__":
    root_path = "/Users/shu/Documents/Datasets/DFDC_small_subset_raw"
    test_one(root_path)
