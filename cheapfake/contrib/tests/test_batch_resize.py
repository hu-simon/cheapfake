"""
Python script that tests resizing schemes for batch landmark extraction.
"""

import os
import time

import cheapfake.contrib.video_processor as video_processor


def main_26():
    video_path = (
        "/home/shu/shu/Datasets/DFDC_small_subset/aagfhgtpmv/video/aagfhgtpmv.mp4"
    )
    videofileprocessor = video_processor.VideoFileProcessor(verbose=True)
    frames = videofileprocessor.extract_all_frames(
        video_path=video_path, channel_first=False, return_rgb=True
    )
    frames_subset = frames[:75]

    scale_factor = 4
    new_height = int(frames_subset[0].shape[0] / scale_factor)
    new_width = int(frames_subset[0].shape[1] / scale_factor)
    frames_resized = [
        cv2.resize(
            frame, dsize=(new_width, new_height), interpolation=cv2.INTER_LANCZOS4
        )
        for frame in frames_subset
    ]

    framesprocessor = video_processor.FramesProcessor(verbose=True)
    start_time = time.time()
    batch_landmarks = framesprocessor.batch_extract_landmarks(
        frames_resized, device="cuda", channels_first=False
    )
    end_time = time.time()
    print("Entire operation took {} seconds.".format(end_time - start_time))


if __name__ == "__main__":
    main_26()
