import os
import re
import sys

import cv2
import dlib
import random
import numpy as np

LIP_MARGIN = 0.5

"""
To deal with the isssue of not having the iterator work, we may just have to create separate chunks...that is unfortunate.
"""


def get_detector_predictor(path):
    """ Grabs the dlib detector and predictor objects. """
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(path)

    return detector, predictor


def sort_list(unsorted, sort_key=None):
    """ Sorts a list of files using the desired scheme. """
    if sort_key is None:
        sort_key = lambda f: int(re.sub("\D", "", f))

    sorted_list = sorted(unsorted, key=sort_key)
    return sorted_list


def to_bounding_box(rect):
    """ Takes a rectangle returned by the DLib detector and puts it into the format (x, y, w, h), which describes a bounding box.
    """
    bbox = (
        rect.left(),
        rect.top(),
        rect.right() - rect.left(),
        rect.bottom() - rect.top(),
    )
    return bbox


def to_np_array(features, dtype="int"):
    """ Converts a shape object containing the (x, y)-coordinates of the facial landmark regions, into a numpy array instance.
    """
    coords = [(features.part(k).x, features.part(k).y) for k in range(68)]
    coords = np.asarray(coords, dtype=dtype)
    return coords


def random_horizontal_flip(images, prob=0.5):
    """ Randomly flips a batch image across the horizontal axis.

    This function assumes that ``images`` has the shape (T, H, W, C) where T is time, H is height, W is width, and C is channel.

    Parameters
    ----------
    images : numpy.array instance
        Batch of images to be flipped.
    prob : float, optional
        The probability of the batch being flipped horizontally, by default 0.5.
    
    Returns
    -------
    images : numpy.array instance
        The batch of images that are potentially flipped.
    flipped : {True, False}
        Boolean revealing whether or not the images were flipped.
    """
    assert prob > 0 and prob < 1

    flipped = False

    if random.random() > prob:
        images = images[:, :, ::-1, ...]
        flipped = True

    return images, flipped


def normalize_images(images):
    """ Normalizes a batch of images, assuming that the highest color value is 255.0.

    Parameters
    ----------
    images : numpy.array instance
        Batch of images, to be normalized.

    Returns
    -------
    images : numpy.array instance
        Batch of normalized images.
    """
    images = images / 255.0

    return images


def chunk_elements(elements, length=1):
    """
    Chunks an array-like object ``elements`` into ``length`` sized chunks. 

    Parameters
    ----------
    elements : array-like
        An array-like object to be chunked into ``length`` sized chunks.
    length : int, optional
        The length of each chunk.

    Returns
    -------
    None

    Yields
    ------
    chunk : array-like
        An array-like object containing one chunk, of size ``length``, of ``elements``.

    Notes
    -----
    You are not always guaranteed that the final chunk is the same size as the others, unless ``length`` times the number of chunks is exactly equal to ``len(elements)``.
    """
    for k in range(0, len(elements), length):
        yield elements[k : k + length]


def prepare_payload(
    chunked_filenames, chunked_framenames, detector, predictor, save_path, rgb=True
):
    """
    Prepares a payload for feeding into the multiprocessing operation.

    TODO Documentation.
    """
    payloads = list()

    for k, (filename, framename) in enumerate(
        zip(chunked_filenames, chunked_framenames)
    ):
        data = {
            "id": k,
            "filenames": filename,
            "framenames": framename,
            "detector": detector,
            "predictor": predictor,
            "save_prefix": save_path,
            "rgb": rgb,
        }
        payloads.append(data)

    return payloads


def process_images(payload):
    """
    Indentifies and localizes facial features for a single image file.

    Parameters
    ----------
    payload : dict
        Dictionary containing the following values:
            {
                "id": str
                    The ID associated with the input.
                "filenames" : list (of strs)
                    List of strings representing the file names.
                "framenames" : list (of strs)
                    List of strings representing the frame names.
                "detector" : dlib.fhog_object_detector instance
                    The DLib detector used for detecting faces.
                "predictor" : dlib.shape_predictor instance
                    The DLib predictor used for detecting facial features.
                "save_prefix" : str
                    The path prefix for storing the original, cropped, and lip-cropped features.
            }
    
    Returns
    -------
    None
    """
    for k, filename in enumerate(payload["filenames"]):
        image = cv2.imread(filename)
        clean_image = image.copy()
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rectangles = payload["detector"](grayscale_image, 1)

        for _, rectangle in enumerate(rectangles):
            features = payload["predictor"](grayscale_image, rectangle)
            features = to_np_array(features)

            (x, y, w, h) = to_bounding_box(rectangle)
            cv2.rectangle(
                image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2
            )
            cropped_image = image[y : y + h, x : x + w]

            lip_features = features[48:68]
            lip_features_x = sorted(lip_features, key=lambda x: x[0])
            lip_features_y = sorted(lip_features, key=lambda x: x[1])
            x_margin = int((-lip_features_x[0][0] + lip_features_x[-1][0]) * LIP_MARGIN)
            y_margin = int((-lip_features_y[0][1] + lip_features_y[-1][1]) * LIP_MARGIN)
            crop_pos = (
                lip_features_x[0][0] - x_margin,
                lip_features_x[-1][0] + x_margin,
                lip_features_y[0][1] - y_margin,
                lip_features_y[-1][1] + x_margin,
            )
            cropped_lips = clean_image[
                crop_pos[2] : crop_pos[3], crop_pos[0] : crop_pos[1]
            ]

            # Save the results.
            cv2.imwrite(
                os.path.join(
                    payload["save_prefix"],
                    "cropped_frames/{}".format(payload["framenames"][k]),
                ),
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if payloads["rgb"] is False
                else image,
            )
            cv2.imwrite(
                os.path.join(
                    payload["save_prefix"],
                    "lip_frames/{}".format(payload["framenames"][k]),
                ),
                cv2.cvtColor(cropped_lips, cv2.COLOR_BGR2RGB)
                if payloads["rgb"] is False
                else image,
            )

        print("Processed file {}".format(filename))
        sys.stdout.flush()
