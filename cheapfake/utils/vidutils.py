import os
import re

import cv2
import glob
import numpy as np
from tqdm import tqdm
import cheapfake.utils.dlibutils as dlibutils


def extract_frames(video_path, channel_first=True):
    """Extracts the frames from a video, and returns it as a Numpy array.

    Parameters
    ----------
    video_path : str
        The absolute path to the .mp4 video file.
    channel_first : {True, False}, bool, optional
        If True, the output is in the format (T, C, H, W), where the channel dimension comes before the spatial dimensions, by default True. Otherwise, the output is in the format (T, H, W, C) so that the channel dimension is last.

    Returns
    -------
    frames : numpy.ndarray instance
        Numpy array containing the frames from the video.

    """
    vidcap = cv2.VideoCapture(video_path)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = np.empty((frame_count, frame_height, frame_width, 3), np.dtype("uint8"))

    count = 0
    success = True
    while count < frame_count and success:
        success, frames[count] = vidcap.read()
        count += 1
    vidcap.release()

    if channel_first:
        frames = np.einsum("ijkl->iljk", frames)

    return frames


def to_frames(path_to_video, path_to_frames, extension="png", debug=False):
    """
    Converts video to frames.
    """
    vidcap = cv2.VideoCapture(path_to_video)
    success, image = vidcap.read()

    video_length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 1

    while success:
        for k in tqdm(range(video_length)):
            cv2.imwrite(
                os.path.join(
                    path_to_frames, "frame{}.{}".format(frame_count, extension)
                ),
                image,
            )
            success, image = vidcap.read()
            frame_count += 1

    if debug:
        print(
            "Processed {} frames from {} to {}".format(
                frame_count, path_to_video, path_to_frames
            )
        )


def to_video(
    path_to_frames, path_to_video, fps, frame_extention="png", debug=False,
):
    """
    Converts a set of frames into a video.
    """
    frame_array = list()
    frame_files = dlibutils.sort_list(
        [
            f
            for f in glob.glob(
                os.path.join(path_to_frames, "*.{}".format(frame_extention))
            )
        ]
    )
    assert len(frame_files) != 0

    if debug:
        print(
            "Populating the buffer with the video frames. Total frames: {}".format(
                len(frames_files)
            )
        )
    for k, frame_file in enumerate(tqdm(frame_files)):
        image = cv2.imread(frame_file)
        height, width, layers = image.shape
        size = (width, height)
        assert layers >= 1
        frame_array.append(image)

    vidwriter = cv2.VideoWriter(
        path_to_video, cv2.VideoWriter_fourcc(*"DIVX"), fps, size
    )

    if debug:
        print("Writing the video to the file {}.".format(path_to_video))
    for _, frame in enumerate(frame_array):
        vidwriter.write(frame)

    if debug:
        print(
            "Processed {} frames from {} to video at {}, at {} frames per second.".format(
                len(frame_files, path_to_frames, path_to_video, fps)
            )
        )
    vidwriter.release()


def crop_video():
    raise NotImplementedError
