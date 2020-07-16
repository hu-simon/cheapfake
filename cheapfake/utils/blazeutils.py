import os

import cv2
import torch
import numpy as np


def load_weights_and_anchors(
    model,
    path,
    default_settings=True,
    min_score_thresh=0.5,
    min_suppression_threshold=0.3,
):
    """ Loads the BlazeFace weights and anchors, and sets some parameters. 

    Parameters
    ----------

    Returns
    -------
    """
    model.load_weights(os.path.join(path, "blazeface.pth"))
    model.load_anchors(os.path.join(path, "anchors.npy"))

    if default_settings is False:
        model.min_score_thresh = min_score_thresh
        model.min_suppression_threshold = min_suppression_threshold


def predict(model, image, rgb=True, interpolation=None):
    """ Obtains the bounding box predictions for the image.
    
    Parameters
    ----------

    Returns
    -------
    """
    if rgb is False:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if interpolation is None:
        interpolation = cv2.INTER_LANCZOS4

    image = cv2.resize(image, (128, 128), interpolation=interpolation)
    predictions = mode.predict_on_image(image)

    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()

    if predictions.ndim == 1:
        predictions = predictions.expand_dims(predictions, axis=0)

    return predictions
