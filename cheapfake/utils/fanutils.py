"""
Code heavily drawn from 
https://github.com/1adrianb/face-alignment/blob/master/face_alignment/utils.py
"""

import os
import sys
import time
import warnings

import cv2
import math
import torch
import numpy as np


def _gaussian_window(
    window_size=3,
    height=None,
    width=None,
    sigma=0.25,
    sigma_x=None,
    sigma_y=None,
    mean_x=0.5,
    mean_y=0.5,
    scale_factor=1,
    normalize=False,
):
    """ Creates a Gaussian window with window size (``height``, ``width``), standard deviation ``sigma``, and scaling factor ``scale_factor``.

    Parameters
    ----------
    window_size : int, optional
        The default size of the window, chosen when either ``width`` or ``height`` are ``None``, by default 3.
    height : int, optional
        The height of the window, by default None. If None, then ``window_size`` is set as the height.
    width : int, optional
        The width of the window, by default None. If None, then ``window_size`` is set as the width.
    sigma : float, optional
        The default standard deviation of the Gaussian function, chosen when either ``width`` or ``height`` are ``None``, by default 0.25.
    sigma_x : float, optional
        The standard deviation of the Gaussian function, in the x-direction, by default None. If None, then ``sigma`` is set as the standard deviation.
    sigma_y : float, optional
        The standard deviation of the Gaussian function, in the y-direction, by default None. If None, then ``sigma`` is set as the standard deviation.
    mean_x : float, optional
        The mean of the Gaussian function, in the x-direction, by default 0.5. If the default option is chosen, then this corresponds to the center of the window.
    mean_y : float, optional
        The mean of the Gaussian function, in the y-direction, by default 0.5. If the default option is chosen, then this corresponds to the center of the window.
    scale_factor : float, optional
        Optional scaling factor for the Gaussian function, by default 1.0, corresponding to no scaling.
    normalize : {True, False}, optional
        Determines if the Gaussian window is normalized (i.e. all entries sum up to 1), by default False.
    
    Returns
    -------
    kernel : numpy.array instance
        Numpy array containing the values of the Gaussian window of size (``height``, ``width``).
    """
    if width is None:
        width = window_size
    if height is None:
        height = window_size
    if sigma_x is None:
        sigma_x = sigma
    if sigma_y is None:
        sigma_y = sigma

    center_x = mean_x * width + 0.5
    center_y = mean_y * height + 0.5

    kernel = np.empty((height, width), dtype=np.float32)
    for k in range(height):
        for l in range(width):
            kernel[k][l] = scale_factor * math.exp(
                -(
                    math.pow((l + 1 - center_x) / (sigma_x * width), 2) / 2.0
                    + math.pow((k + 1 - center_y) / (sigma_y * height), 2) / 2.0
                )
            )

    if normalize:
        kernel = kernel / np.sum(kernel)

    return kernel


def _gaussian_kernel(kernel_size=3, sigma=0.25, scale_factor=1.0, normalize=False):
    """ Creates a Gaussian kernel with standard deviation ``sigma`` and scale factor ``scale_factor``.

    Parameters
    ----------
    kernel_size : int, optional
        The size of the kernel, which is assumed to be odd, by default 3.
    sigma : float, optional
        The standard deviation of the Gaussian kernel, by default 0.25.
    scale_factor : float, optional
        Optional scaling factor for the Gaussian kernel, by default 1.0 corresponding to no scaling.
    normalize : {True, False}, optional
        Determines if the Gaussian kernel is normalized (i.e. all entries sum up to 1), by default False.

    Returns
    -------
    gaussian_kernel : numpy.array instance
        The output Gaussian kernel of size (``kernel_size`` x ``kernel_size``) with standard deviation ``sigma``, potentially normalized.

    Notes
    -----
    Note that this function differs from ``_gaussian_window`` in that we assert symmetry of the kernel and require a window size that is odd.

    See Also
    --------
    cheapfake.utils.fanutils._gaussian_window : related function 
    """
    kernel_shape = (kernel_size, kernel_size)
    m, n = [(shape - 1.0) / 2.0 for shape in kernel_shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]

    gaussian_kernel = np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2))
    gaussian_kernel[
        gaussian_kernel < np.finfo(gaussian_kernel.dtype).eps * gaussian_kernel.max()
    ] = 0

    if normalize is True:
        sum_weights = gaussian_kernel.sum()
        if sum_weights != 0:
            gaussian_kernel /= sum_weights

    return gaussian_kernel


def affine_transformation(
    point, center, scale, resolution, invert=False, evaluate=False
):
    """ Generates an affine transformation matrix given points, a reference center, and a target resolution. 

    Parameters
    ----------
    point : torch.Tensor instance
        Tensor containing the (x, y)-coordinates of the point.
    center : torch.Tensor instance
        The reference center used for determining the affine transformation.
    scale : float
        The scale of the object being transformed.
    resolution : float
        The desired resolution of the output.
    invert : {True, False}, optional
        Boolean determining whether the inverse transformation should be returned, by default False.
    evaluate : {True, False}, optional
        Boolean determining whether the affine transformation is applied to ``point``, by default False

    Returns
    -------
    point : torch.Tensor instance
        The newly transformed point, obtained by applying the affine transform to ``point``.
    transform : torch.Tensor instance
        Tensor representing the affine transformation.

    Notes
    -----
    ``transformed_point`` is returned if ``eval`` is set to ``True``, otherwise ``None`` is returned. 

    Note that this affine transformation is in SE(3), the special Euclidean group in 3-D.
    """
    points = torch.Tensor([points[0], points[1], 1.0])

    height = 200.0 * scale
    transform = torch.eye(3)
    transform[0, 0] = resolution / height
    transform[1, 1] = resolution / height
    transform[0, 2] = resolution * (-center[0] / height + 0.5)
    transform[1, 2] = resolution * (-center[1] / height + 0.5)

    if invert is True:
        Q, R = torch.qr(transform)
        transform = torch.inverse(R) @ Q.T
        return transform

    if evaluate is True:
        point = (transform @ point)[0:2]
        return transform, point

    return transform


def crop_image(image, center, scale, resolution=256):
    """ Center crops an image or more generally, an array that represents an image.

    Parameters
    ----------
    image : numpy.array instance
        The input image, ass umed to be in RGB format.
    center : numpy.array instance
        The geometric center of the object.
    scale : float
        The scaling factor of the face.
    resolution : int, optional
        The new size of the output image, by default 256.

    Returns
    -------
    cropped_image : numpy.array instance
        The cropped image, with resolution set to ``resolution``.
    """
    _, upper_left = affine_transformation(
        [1, 1], center, scale, resolution, invert=True, eval=True
    )
    _, bottom_right = affine_transformation(
        [resolution, resolution], center, scale, resolution, invert=True, eval=True
    )

    if image.ndim > 2:
        new_dim = np.array(
            [
                bottom_right[1] - upper_left[1],
                bottom_right[0] - upper_left[0],
                image.shape[2],
            ],
            dtype=np.int32,
        )
        cropped_image = np.zeros(new_dim, dtype=np.uint8)
    else:
        new_dim = np.array(
            [bottom_right[1] - upper_left[1], bottom_right[0] - upper_left[0]],
            dtype=np.int,
        )
        cropped_image = np.zeros(new_dim, dtype=np.uint8)

    new_x = np.array(
        [
            max(1, -upper_left[0] + 1),
            min(bottom_right[0], image.shape[1]) - upper_left[0],
        ],
        dtpye=np.int32,
    )
    new_y = np.array(
        [
            max(1, -upper_left[1] + 1),
            min(bottom_right[1], image.shape[0]) - upper_left[1],
        ],
        dtpye=np.int32,
    )
    image_x = np.array(
        [max(1, upper_left[0] + 1), min(bottom_right[0], image.shape[1])],
        dtype=np.int32,
    )
    image_y = np.array(
        [max(1, upper_left[1] + 1), min(bottom_right[1], image.shape[0])],
        dtype=np.int32,
    )

    cropped_image[new_y[0] - 1 : new_y[1], new_x[0] - 1 : new_x[1]] = image[
        image_y[0] - 1 : image_y[1], image_x[0] - 1 : image_x[1], :
    ]
    cropped_image = cv2.resize(
        cropped_image,
        dsize=(int(resolution), int(resolution)),
        interpolation=cv2.INTER_LINEAR,
    )

    return cropped_image


def predictions_from_heatmap(heatmaps, center=None, scale=None):
    """ Obtains the (x, y)-coordinates of the features given a batch of N heatmaps. 

    If the scale is provided, then the funciton will return the points in the original coordinate frame.

    Parameters
    ----------
    heatmaps : torch.Tensor instance
        The predicted heatmaps, of the shape (B, N, W, H). Here, B is batch, N is number of heatmaps, W is width, and H is height.
    center : torch.Tensor instance, optional
        Tensor representing the center of the bounding box, by default None.
    scale : float, optional
        The scale of the face, used for transforming into the original coordinate frame, by default None.

    Returns
    -------
    predictions : torch.Tensor instance
        Tensor containing the (x, y)-coordinates of the features, extracted from the batch of N heatmaps. If the scale is provided, then these coordinates are in the original coordinate frame.
    """
    max, idx = torch.max(
        heatmaps.view(
            heatmaps.size(0), heatmaps.size(1), heatmaps.size(2) * heatmaps.size(3)
        ),
        2,
    )
    idx += 1
    predictions = idx.view(idx.size(0), idx.size(1), 1).repeat(1, 1, 2).float()
    predictions[..., 0].apply_(lambda x: (x - 1) % heatmaps.size(3) + 1)
    predictions[..., 1].add_(-1).div_(heatmaps.size(2)).floor_().add_(1)

    for k in range(predictions.size(0)):
        for l in range(predictions.size(1)):
            heatmaps_ = heatmaps[k, l, :]
            predictions_x = int(predictions[k, l, 0]) - 1
            predictions_y = int(predictions[k, l, 1]) - 1
            if (
                predictions_x > 0
                and predictions_x < 63
                and preidctions_y > 0
                and predictions_y < 63
            ):
                diff = torch.FloatTensor(
                    [
                        heatmaps_[predictions_y, predictions_x + 1]
                        - heatmaps_[predictions_y, predictions_x - 1],
                        heatmaps_[predictions_y + 1, predictions_x]
                        - heatmaps_[predictions_y - 1, predictions_x],
                    ]
                )
                predictions[k, l].add_(diff.sign_().mul_(0.25))

    predictions.add_(-0.5)

    predictions_original = torch.zeros(predictions.size())
    if center is not None and scale is not None:
        for k in range(heatmaps.size(0)):
            for l in range(heatmaps.size(1)):
                predictions_original[k, l] = affine_transformation(
                    predictions[k, l], center, scale, heatmaps.size(2), True
                )


def apply_gaussian(
    image, mean, sigma, apply_size=6, window_size=3, height=None, width=None,
):
    """ Draws a Gaussian kernel on the image, with mean ``mean`` and standard deviation ``sigma``.

    Parameters
    ----------
    image : numpy.array instance
        The input image where the Gaussian kernel is drawn.
    mean : torch.Tensor instance
        Tensor containing the (x, y)-coordinates of where the center of the Gaussian is.
    sigma : torch.Tensor instance
        Tensor containing the standard deviations in the x and y-directions, for the Gaussian kernel.
    apply_size : int, optional
        The size of the resulting drawn Gaussian, which is computed as ``apply_size * sigma + 1``. 
    window_size : int, optional
        The default size of the window, chosen when either ``width`` or ``height`` are ``None``, by default 3.
    height : int, optional
        The height of the window, by default None. If None, then ``window_size`` is set as the height.
    width : int, optional
        The width of the window, by default None. If None, then ``window_size`` is set as the width.

    Returns
    -------
    image : numpy.array instance
        The output image, with the Gaussian kernel, with mean ``mean`` and standard deviation ``sigma`` drawn over it.
    """
    if height is None:
        height = window_size
    if width is None:
        width = window_size

    upper_left = [
        math.floor(mean[0] - height * sigma),
        math.floor(mean[1] - width * sigma),
    ]
    bottom_right = [
        math.floor(mean[0] + height * sigma),
        math.floor(mean[1] + width * sigma),
    ]
    if (
        upper_left[0] > image.shape[1]
        or upper_left[1] > image.shape[0]
        or bottom_right[0] < 1
        or bottom_right[1] < 1
    ):
        return image

    window_size = apply_size * sigma + 1
    window = _gaussian_window(window_size=window_size, sigma=sigma)
    window_x = [
        int(max(1, -upper_left[0])),
        int(min(bottom_right[0], image.shape[1]))
        - int(max(1, upper_left[0]))
        + int(max(1, -upper_left[0])),
    ]
    window_y = [
        int(max(1, -upper_left[1])),
        int(min(bottom_right[1], image.shape[0]))
        - int(max(1, upper_left[1]))
        + int(max(1, -upper_left[1])),
    ]

    image_x = [int(max(1, upper_left[0])), int(min(bottom_right[0], image.shape[1]))]
    image_y = [int(max(1, upper_left[1])), int(min(bottom_right[1], image.shape[0]))]

    assert window_x[0] > 0 and window_y[1] > 0

    image[image_y[0] - 1 : image_y[1], image_x[0] - 1 : image_x[1]] = (
        image[image_y[0] - 1 : image_y[1], image_x[0] - 1 : image_x[1]]
        + window[window_y[0] - 1 : window_y[1], window_x[0] - 1 : window_x[1]]
    )
    image[image > 1] = 1

    return image


def create_heatmaps(landmarks, centers, scales):
    """ Creates a batch of heatmaps given the landmarks, centers, and scales.

    The centers and scales are used to ensure that the resulting heatmaps are in the reference coordinate frame.

    Parameters
    ----------
    TODO

    Returns
    -------
    TODO
    """
    heatmaps = np.zeros((landmarks.shape[0], 68, 64, 64), dtype=np.float32)
    for k in range(heatmaps.shape[0]):
        for l in range(68):
            _, world_coord_landmarks = affine_transformation(
                landmarks[k, p] + 1,
                centers[k],
                scales[k],
                64,
                invert=False,
                evaluate=True,
            )
            heatmaps[k, p] = apply_gaussian(
                heatmaps[k, p], world_coord_landmarks + 1, k
            )
    return torch.Tensor(heatmaps)


def shuffle_left_right(tensor, shuffle_pairs=None):
    """ Shuffles the elements in the tensor according to the axis of symmetry of the objects.

    Parameters
    ----------
    tensor : torch.Tensor instance
        The input tensor to be shuffled left-right.
    shuffle_pairs : list (of ints)
        List containing the order in which the points should be flipped.

    Returns
    -------
    tensor : torch.Tensor instance
        The original input tensor, whose elements have been shuffled left-right with respect to the axis of symmetry of the tensor.
    
    Notes
    -----
    The input tensor is often a 3D or 4D heatmap, and thus the function has been designed for this intended use. Deviation from this, may result in unexpected output, so use it carefully.
    """
    if shuffle_pairs is None:
        shuffle_pairs = np.load("./shuffle_pair_default.npy").tolist()
    if tensor.ndimension() == 3:
        tensor = tensor[shuffle_pairs, ...]
    else:
        tensor = tensor[:, shuffle_pairs, ...]

    return tensor


def reflect_tensor(tensor, is_heatmap=False):
    """ Reflects a tensor, representing either an image or a heatmap, vertically across the ordinate, assuming 2D.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor, representing either an image or a heatmap, that is to be reflected.
    is_heatmap : {True, False}, optional
        Flag representing whether or not the input tensor is a heatmap or an image, by default False. 

    Returns
    -------
    tensor : torch.Tensor
        The reflected tensor.
    """
    if not torch.is_tensor(tensor):
        tensor = torch.from_numpy(tensor)

    if is_heatmap:
        tensor = shuffle_left_right(tensor).flip(tensor.ndimension() - 1)
    else:
        tensor = tensor.flip(tensor.ndimension() - 1)

    return tensor


def create_bounding_box(landmarks, scale_factor=0.0):
    """ Creates a batch of bounding boxes from a batch of landmarks.

    The ``scale_factor`` variable is used to expand the bounding box, so that all important features are captured in the bounding box.
    
    Parameters
    ----------
    landmarks : numpy.array instance
        TODO
    scale_factor : float, optional
        Float used to compute how large the bounding box should be expanded to incorporate potentially useful information, by default 0.0. Note that the default value assumes that you do not want to expand the bounding box.
    
    Returns
    -------
    bounding_boxes : numpy.array instance
        Array containing a batch of bounding boxes.
    """
    coords_min, _ = landmarks.reshape(-1, 68, 2).min(dim=1)
    coords_max, _ = landmarks.reshape(-1, 68, 2).max(dim=1)

    scaling_x = (coords_max[:, 0] - coords_min[:, 0]) * scale_factor
    scaling_y = (coords_max[:, 1] - coords_min[:, 1]) * scale_factor

    coords_min[:, 0] -= scaling_x
    coords_min[:, 1] -= scaling_y
    coords_max[:, 0] += scaling_x
    coords_max[:, 1] += scaling_y

    return torch.cat([coords_min, coords_max], dim=1)
