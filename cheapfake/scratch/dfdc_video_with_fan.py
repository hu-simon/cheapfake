import os
import time

import cv2
import glob
import json
import itertools
import collections
from skimage import io
from pprint import pprint
import concurrent.futures
import matplotlib.pyplot as plt

import face_alignment
import cheapfake.utils.vidutils as vidutils
import cheapfake.utils.dlibutils as dlibutils


def _get_prediction_types():
    """Obtains the prediction types and dictionary containing the slice indices.

    Returns
    -------
    pred_type : type instance
        Type instance that abstracts the prediction types.
    pred_dict : dictionary
        Dictionary containing the facial regions, along with the slice indices.

    """
    pred_type = collections.namedtuple("prediction_type", ["slice", "color"])
    pred_dict = {
        "face": pred_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
        "eyebrow_left": pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
        "eyebrow_right": pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
        "nose": pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
        "nostril": pred_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
        "eye_left": pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
        "eye_right": pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
        "lips": pred_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
        "teeth": pred_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4)),
    }

    return pred_type, pred_dict


def _evaluate(image):
    """Evaluates the images through FAN. 

    Parameters
    ----------
    image : numpy.ndarray instance
        Numpy array containing the image to be passed into the FAN.

    Returns
    -------
    predictions : numpy.ndarray instance  
        Numpy array containing the location of the facial landmarks.

    """
    model = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._3D, device="cpu", flip_input=True
    )
    predictions = model.get_landmarks(image)
    del model

    return predictions


''' Deprecated in favor of the function below.
def create_outline_single_thread(
    image_path, extract_path, show_image=False, verbose=True
):
    """Creates an outline for a single image, used for multithreading/multiprocessing.

    Parameters
    ----------
    model : face_alignment.FaceAlignment instance
        The FAN model used for predicting important feature locations. 
    image_path : str
        The absolute path to the frames.
    extract_path : str
        The absolute path to where the images with the features are stored.
    show_image : {False, True}, bool, optional
        Determines if the image is shown, by default False. Note that if this is True, then the program is blocked.
    verbose : {True, False}, bool, optional
        If True, then verbose information is output to the system console, by default True. 

    """
    input_image = io.imread(image_path)
    predictions = _evaluate(input_image)

    pred_type, pred_dict = _get_prediction_types()

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    for pred_type in pred_dict.values():
        ax.plot3D(
            predictions[pred_type.slice, 0] * 1.2,
            predictions[pred_type.slice, 1],
            predictions[pred_type.slice, 2],
            color="blue",
        )

    ax.axis("off")
    ax.view_init(elev=90.0, azim=90.0)
    ax.set_xlim(ax.get_xlim()[::-1])
    plt.savefig(extract_path)

    if verbose:
        print("Successfully processed {}".format(image_path))

    if show_image:
        plt.show()
'''


def create_outline_single_thread(model, image_path, extract_path, show_image=False):
    """Creates an outline for a single image, used for multithreading/multiprocessing.

    Parameters
    ----------
    model : face_alignment.FaceAlignment instance
        The FAN model used for predicting important feature locations.
    image_path : str
        The absolute path to the frames.
    extract_path : str
        The absolute path to where the images with the features are stored.
    
    """
    input_image = io.imread(image_path)
    predictions = model.get_landmarks(input_image)[-1]

    pred_type, pred_dict = _get_prediction_types()

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    for pred_type in pred_dict.values():
        ax.plot3D(
            predictions[pred_type.slice, 0] * 1.2,
            predictions[pred_type.slice, 1],
            predictions[pred_type.slice, 2],
            color="blue",
        )

    ax.axis("off")
    ax.view_init(elev=90.0, azim=90.0)
    ax.set_xlim(ax.get_xlim()[::-1])
    plt.savefig(extract_path)

    # For debugging, should be removed later.
    print("Successfully processed {}".format(image_path))

    if show_image:
        plt.show()


''' Deprecated in favor of the function below.
def create_outline(path_to_frames, show_image=False, verbose=True):
    """Creates an outline for a single video, using multithreading/multiprocessing.

    Parameters
    ----------
    path_to_frames : str
        The absolute path to the directory containing the frames.
    show_image : {False, True}, bool, optional
        Determines if the image is shown, by default False. Note that if this is True, then the program is blocked.
    verbose : {True, False}, bool, optional
        If True, then verbose information is output to the system console, by default True.

    """
    frames = dlibutils.sort_list(
        [f for f in os.listdir(path_to_frames) if not f.startswith(".")]
    )
    image_paths = dlibutils.sort_list(
        [os.path.join(path_to_frames, frame) for frame in frames]
    )
    extract_paths = [
        os.path.join(
            os.path.join("/".join(path_to_frames.split("/")[:-1]), "outline_frames"),
            frame,
        )
        for frame in frames
    ] * len(image_paths)
    show_image = [False] * len(image_paths)
    verbose = [True] * len(image_paths)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(
            create_outline_single_thread,
            image_paths,
            extract_paths,
            show_image,
            verbose,
        )
'''


def create_outline(model, path_to_frames):
    """Creates an outline for a single video, using multithreading/multiprocessing.

    Parameters
    ----------
    model : face_alignment.FaceAlignment instance
        The FAN model used for predicting important feature locations.
    path_to_frames : str
        The absolute path to the frames.

    """
    frames = dlibutils.sort_list(
        [f for f in os.listdir(path_to_frames) if not f.startswith(".")]
    )
    image_paths = dlibutils.sort_list(
        [os.path.join(path_to_frames, frame) for frame in frames]
    )
    extract_paths = [
        os.path.join(
            os.path.join("/".join(path_to_frames.split("/")[:-1]), "outline_frames"),
            frame,
        )
        for frame in frames
    ] * len(image_paths)
    models = [model] * len(image_paths)
    # models = itertools.repeat(model, len(image_paths))
    show_images = [False] * len(image_paths)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(
            create_outline_single_thread,
            models,
            image_paths,
            extract_paths,
            show_images,
        )


if __name__ == "__main__":
    root_path = "/Users/shu/Documents/Datasets/DFDC_small_subset"

    video_names = [f for f in os.listdir(root_path)]

    video_paths = [
        glob.glob(os.path.join(os.path.join(root_path, video_name), "video/*.mp4"))
        for video_name in video_names
    ]
    video_paths = list(itertools.chain.from_iterable(video_paths))

    frame_paths = [
        glob.glob(os.path.join(os.path.join(root_path, video_name), "frames"))
        for video_name in video_names
    ]
    frame_paths = list(itertools.chain.from_iterable(frame_paths))

    outline_paths = [
        glob.glob(os.path.join(os.path.join(root_path, video_name), "outline_frames"))
        for video_name in video_names
    ]
    outline_paths = list(itertools.chain.from_iterable(outline_paths))

    folder_paths = [glob.glob(os.path.join(root_path, "*"))]
    folder_paths = list(itertools.chain.from_iterable(folder_paths))

    model = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._3D, device="cpu", flip_input=True
    )

    create_outline(model, frame_paths[0])
