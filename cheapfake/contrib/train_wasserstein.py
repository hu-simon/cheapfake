"""
Python script that trains the network using the Wasserstein loss.

Felt cute, might delete later.
"""

import os
import time
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from tqdm import tqdm

