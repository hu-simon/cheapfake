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
    frame_extractor,
    frame_processor,
    frame_transform,
    face_model,
    device=torch.device("cuda:0"),
    channel_first=True,
):
    frames = frame_extractor.extract_all_frames(video_path=path)
    frames = frame_processor.apply_transformation(
        frames, frame_transform
    )
    frames = frames.float().to(device)

    landmarks = face_model.get_landmarks_from_batch(frames)
    
    print(landmarks.shape)
    

if __name__ == "__main__":
    metadata_path = "/home/shu/cheapfake/cheapfake/contrib/balanced_metadata_fs03.csv"
    df = pd.read_csv(metadata_path)
    video_path = df["Files"][0]

    frame_transform = transforms.BatchRescale(4)
    frame_extractor = video_processor.VideoFileProcessor()
    frame_processor = video_processor.FramesProcessor()
    device = torch.device("cuda:0")
    model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device="cuda", face_detector="sfd")

    start_time = time.time()
    process_video(
        path=video_path, 
        face_model=model,
        frame_extractor=frame_extractor, 
        frame_processor=frame_processor,
        frame_transform=frame_transform,
    )
    end_time = time.time()

    print("Entire operation took {} seconds".format(end_time - start_time))
