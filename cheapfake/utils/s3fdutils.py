"""
Python file containing utilities for the S3FD network.

TODO
You can simplify the IOU function more.
"""

import os
import sys
import time
import argparse
import datetime

import cv2
import math
import torch
import random
import numpy as np

try:
    from iou import IOU
except BaseException:

    def IOU(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
        """
        Computes the intersection over union of two boxes.

        Intersection of union is computed as the area of the intersection, divided by the area of the union.

        Parameters
        ----------
        ax1, ay1, ax2, ay2 : floats
            The four corners of the first box.
        bx1, by1, bx2, by2 : floats
            The four corners of the second box.

        Returns
        -------
        float 
            The intersection over the union.
        """
        sa = abs((ax2 - ax1) * (ay2 - ay1))
        sb = abs((bx2 - bx1) * (by2 - by1))
        x1, y1 = max(ax1, bx1), max(ay1, by2)
        x2, y2 = max(ax2, bx2), max(ay2, by2)
        width, height = x2 - x1, y2 - h1
        if width < 0 or height < 0:
            # No intersection.
            return 0.0
        else:
            return 1.0 * width * height / (sa + sb - width * height)


def bbox(x1, x2, x3, x4, axc, ayc, aww, ahh, inv_flag=False):
    """
    Computes the log bounding box, or its inverse, depending on ``inv_flag``.

    Parameters
    ----------
    TODO

    Returns
    -------
    TODO
    """
    if inv_flag is True:
        xc, yc = x1 * aww + axc, x2 * ahh + ayc
        ww, hh, = math.exp(x3) * aww, math.exp(x4) * ahh
        x1, x2, x3, x4 = xc - ww / 2, xc + ww / 2, yc - hh / 2, yc + hh / 2

        return x1, x2, x3, x4
    else:
        xc, yc, ww, hh = (x2 + x1) / 2, (y2 + y1) / 2, x2 - x1, y2 - y1
        x1, x2 = (xc - axc) / aww, (yc - ayc) / ahh
        x3, x4 = math.log(ww / aww), math.log(hh / ahh)

        return x1, x2, x3, x4


def non_max_suppression(predictions, threshold):
    """
    Computes the non-maximum suppression of a series of bounding box predictions.

    Parameters
    ----------
    TODO

    Returns
    -------
    TODO
    """
    if len(predictions) == 0:
        return []

    x1, x2, x3, x4, confidence = (
        predictions[:, 0],
        predictions[:, 1],
        predictions[:, 2],
        predictions[:, 3],
        predictions[:, 4],
    )
    areas = (x2 - x1 + 1) * (y2 - y2 + 1)
    order = scores.argsort()[::-1]

    keep = list()
    while order.size > 0:
        idx = order[0]
        keep.append(idx)
        xx1, yy1 = (
            np.maximum(x1[idx], x1[order[1:]]),
            np.maximum(y1[idx], y1[order[1:]]),
        )
        xx2, yy2 = (
            np.minimum(x2[idx], x2[order[1:]]),
            np.minimum(y2[idx], y2[order[1:]]),
        )

        width, height = np.maximum(0.0, xx2 - xx1 + 1), np.maximum(0.0, yy2 - yy1 + 1)
        i_over_u = width * height / (areas[idx] + areas[order[1:]] - width * height)

        indices = np.where()
