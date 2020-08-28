"""
Python file that extracts frames from a video and then extracts the facial landmarks
 using the Face Alignment Network.
"""

import os
import time

import cv2
import glob
import torch
import numpy as np
import pandas as pd
import face_alignment

import cheapfake.contrib.transforms as transforms
import cheapfake.contrib.video_processor as video_processor


def process_video(
    path,
    frame_transform,
    frames_per_second=30,
    sample_rate=16000,
    num_seconds=3.0,
    num_samples=None,
    channel_first=True,
):
    frames = video_processor.FramesProcessor.extract_all_frames(video_path=path)
    audio = video_processor.AudioProcessor._extract_all_audio(video_path=path)

    frames = video_processor.FramesProcessor.apply_transformation(
        frames, frame_transform
    )


if __name__ == "__main__":
    metadata_path = "/home/shu/cheapfake/cheapfake/balanced_metadata_fs03.csv"
    df = pd.read_csv(metadata_path)

