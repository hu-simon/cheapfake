import os
import time

import cv2
import torch
import numpy as np
from scipy import io
from math import cos, sin

COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)


def filenames_from_path(path):
    """Grabs the relative path names from a .txt file containing the file names.

    Parameters
    ----------
    path : str
        The absolute/relative path to a .txt file containing the file names.
    
    Returns
    -------
    filenames : list (of strs)
        List containing the filenames from the .txt file.

    """
    with open(path) as f:
        filenames = f.read().splitlines()

    return filenames


def softmax_with_temperature(tensor, tau):
    """Computes the value of the softmax temperature function, using the temperature paramtere ``tau``.

    Parameters
    ----------
    tensor : torch.Tensor instance
        The input tensor to be put into the softmax function, with temperature.
    tau : float
        The temperature parameter.

    Returns
    -------
    result : torch.Tensor instance
        The output tensor containing the probabilities, obtained by putting ``tensor`` into the softmax function, with temperature.

    """
    result = torch.exp(tensor / tau)
    result = torch.div(result, torch.sum(result, 1).unsqueeze(1).expand_as(result))

    return result


def pose_params_from_mat(path_to_mat):
    """Grabs the pose parameters stored in a .mat file.
    
    The output is in the [pitch, yaw, roll] convention instead of the assumed [yaw, pitch, roll] convention used by HopeNet. All Euler angles are given in radians.

    Parameters
    ----------
    path_to_mat : str
        Absolute or relative path to the .mat file containing the pose parameters.
    
    Returns
    -------
    pose_params : list (of floats)
        List containing the pitch, yaw, roll, tdx, tdy parameters for the pose.
    
    Notes
    -----
    The Pose_300W_LP Dataset, obtained from source, contains the pose parameters stored in a .mat file.

    """
    mat = io.loadmat(path_to_mat)
    # Save only [pitch, yaw, roll, tdx, tdy].
    pose_params = mat["Pose_Para"][0][:5]

    return pose_params


def euler_from_mat(path_to_mat):
    """Grabs the Euler angles stored in a .mat file.

    The output is in the [pitch, yaw, roll] convention instead of the assumed [yaw, pitch, roll] convention used by HopeNet. All Euler angles are given in radians.
    
    Parameters
    ----------
    path_to_mat : str
        Absolute or relative path to the .mat file containing the pose parameters.

    Returns
    -------
    euler_angles : list (of floats)
        List containing the pitch, yaw, and roll, in radians.

    Notes
    -----
    The Pose_300W_LP Dataset, obtained from source, contains the pose parameters stored in a .mat file.
    
    """
    mat = io.loadmat(path_to_mat)
    # Save only pitch, yaw, roll.
    euler_angles = mat["Pose_Para"][0][:3]

    return euler_angles


def landmarks2d_from_mat(path_to_mat):
    """Grabs the 2D landmarks stored in a .mat file.

    The landmarks use the [x, y] convention for coordinates.

    Parameters
    ----------
    path_to_mat : str
        Absolute or relative path to the .mat file containing the coordinates of the 2D landmarks.

    Returns
    -------
    landmarks : list (of floats)
        List containing the coordinates of the 2D landmarks in [x, y] convention.

    Notes
    -----
    The Pose_300W_LP Dataset, obtained from source, contains the 2D landmark locations inside a .mat file.

    """
    mat = io.loadmat(path_to_mat)
    landmarks = mat["pt2d"]

    return landmarks


def rad_to_deg(angles, to_list=False, right_hand=False):
    """Converts Euler angles from radians to degrees.

    Parameters
    ----------
    angles : list
        List of Euler angles, in [yaw, pitch, roll] convention, in radians.
    to_list : {False, True}, bool, optional
        If False, then the result is not returned as a list, by default False. If True, then the result is returned as a numpy.array instance.
    right_hand : {False, True}, bool, optional  
        If False, then the result is returned assuming a left-handed coordinate system, by default False. If True, then the result is returned assuming a right-handed coordinate system.

    Returns
    -------
    angles : numpy.array instance or list (of floats)
        Euler angles, in [yaw, pitch, roll] convention, in degrees.

    """
    angles = np.asarray(angles)
    angles *= 180 / np.pi

    if right_hand:
        angles[0] *= -1

    if to_list:
        angles = list(angles)

    return angles


def deg_to_rad(angles, to_list=False, right_hand=False):
    """Converts Euler angles from degrees to radians.

    Parameters
    ----------
    angles : list
        List of Euler angles, in [yaw, pitch, roll] convention, in degrees.
    to_list : {False, True}, bool, optional
        If False, then the result is returned as a list, by default False. If True, then the result is returned as a numpy.array instance.
    right_hand : {False, True}, bool, optional
        If False, then the result is returned assuming a left-handed coordinate system, by default False. If true, then the result is returned assuming a right-handed coordinate system.

    Returns
    -------
    angles : numpy.array instance or list (of floats)
        Euler angles, in [yaw, pitch, roll] convention, in radians.

    """
    angles = np.asarray(angles)
    angles *= np.pi / 180

    if right_hand:
        angles[0] *= -1

    if to_list:
        angles = list(angles)

    return angles


def draw_axes3d(image, euler_angles, tdx=None, tdy=None, size=100.0):
    """Draws the 3D axes on an image.
    
    Parameters
    ----------
    image : numpy.array instance
        Array containing the image information.
    euler_angles : numpy.array instance or list (of floats)
        Euler angles in radians, using the [yaw, pitch, roll] convention.
    tdx, tdy : float, optional
        The offset in the x and y-directions, by default None. If None, then the width and height, respectively, of the image are used.
    size : float, optional
        The reference scale of the image, by default 100.0.

    Returns
    -------
    image : numpy.array instance
        Array containing the image information, with the 3D axes drawn on them.

    """
    yaw, pitch, roll = deg_to_rad(euler_angles, to_list=True, right_hand=True)

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = image.shape[:2]
        tdx = weight / 2
        tdy = height / 2

    # x-axis drawn in RED, pointing right.
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # y-axis drawn in GREEN, pointing down.
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # z-axis drawn in BLUE, pointing out of the screen.
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    # Make everything an int for pixel plotting purposes.
    tdx, tdy = int(tdx), int(tdy)
    x1, y1 = int(x1), int(y1)
    x2, y2 = int(x2), int(y2)
    x3, y3 = int(x3), int(y3)

    cv2.line(image, (tdx, tdy), (x1, y1), COLOR_RED, 3)
    cv2.line(image, (tdx, tdy), (x2, y2), COLOR_GREEN, 3)
    cv2.line(image, (tdx, tdy), (x3, y3), COLOR_BLUE, 3)

    return image


def draw_pose3d(image, euler_angles, tdx=None, tdy=None, size=150.0):
    """Draws the 3D pose cube on the image.

    Parameters
    ----------
    image : numpy.array instance
        Array containing the image information.
    euler_angles : numpy.array instance or list (of floats)
        Euler angles in radians, using the [yaw, pitch, roll] convention.
    tdx, tdy : float, optional
        The offset in the x and y-directions, by default None. If None, then the width and height, respectively, of the image are used.
    size : float, optional
        The reference scale of the image, by default 150.0.

    Returns
    -------
    image : numpy.array instance
        Array containing the image information, with the 3D pose cube drawn on it.

    """
    yaw, pitch, roll = deg_to_rad(euler_angles, to_list=True, right_hand=True)

    if tdx != None and tdy != None:
        face_x = tdx - 0.5 * size
        face_y = tdy - 0.5 * size
    else:
        height, width = image.shape[:2]
        face_x = width / 2 - 0.5 * size
        face_y = height / 2 - 0.5 * size

    x1 = size * (cos(yaw) * cos(roll)) + face_x
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + face_y

    x2 = size * (-cos(yaw) * sin(roll)) + face_x
    y2 = size * (sin(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + face_y

    x3 = size * (sin(yaw)) + face_x
    y3 = size * (-cos(yaw) * sin(pitch)) + face_y

    # Make everything an int for pixel plotting purposes.
    tdx, tdy = int(tdx), int(tdy)
    face_x, face_y = int(face_x), int(face_y)
    x1, y1 = int(x1), int(y1)
    x2, y2 = int(x2), int(y2)
    x3, y3 = int(x3), int(y3)

    # Base in RED.
    cv2.line(image, (face_x, face_y), (x1, y1), COLOR_RED, 3)
    cv2.line(image, (face_x, face_y), (x2, y2), COLO_RED, 3)
    cv2.line(image, (x1, y1), (x2 + x1 - face_x, y2 + y1 - face_y), COLOR_RED, 3)
    cv2.line(image, (x2, y2), (x2 + x1 - face_x, y2 + y1 - face_y), COLOR_RED, 3)

    # Pillars in BLUE.
    cv2.line(image, (face_x, face_y), (x3, y3), COLOR_BLUE, 3)
    cv2.line(image, (x1, y1), (x1 + x3 - face_x, y1 + y3 - face_y), COLOR_BLUE, 3)
    cv2.line(image, (x2, y2), (x2 + x3 - face_x, y2 + y3 - face_y), COLOR_BLUE, 3)
    cv2.line(
        image,
        (x1 + x2 - face_x, y1 + y2 - face_y),
        (x1 + x2 + x3 - 2 * face_x, y1 + y2 + y3 - 2 * face_y),
        COLOR_BLUE,
        3,
    )

    # Top in GREEN.
    cv2.line(
        image,
        (x1 + x3 - face_x, y1 + y3 - face_y),
        (x1 + x2 + x3 - 2 * face_x, y1 + y2 + y3 - 2 * face_y),
        COLOR_GREEN,
        3,
    )
    cv2.line(
        image,
        (x2 + x3 - face_x, y2 + y3 - face_y),
        (x1 + x2 + x3 - 2 * face_x, y1 + y2 + y3 - 2 * face_y),
        COLOR_GREEN,
        3,
    )
    cv2.line(image, (x3, y3), (x1 + x3 - face_x, y1 + y3 - face_y), COLOR_GREEN, 3)
    cv2.line(image, (x3, y3), (x2 + x3 - face_x, y2 + y3 - face_y), COLOR_GREEN, 3)

    return image
