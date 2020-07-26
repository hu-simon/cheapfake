import os
import re
import sys
import time

import cv2
import torch
import pandas
import torchvision
import numpy as np
from torch.utils.data.dataset import Dataset
import cheapfake.utils.hopeutils as hopeutils

from PIL import Image, ImageFilter


class Pose_300W_LP_Dataset(Dataset):
    def __init__(
        self,
        data_path,
        filename_path,
        transform,
        img_ext=".jpg",
        annot_ext=".mat",
        img_mode="RGB",
        flip_pose=False,
        blur_image=False,
    ):
        """Instantiates a Pose_300W_LP_Dataset object, containing data from the Pose_300_LP Dataset.

        Parameters
        ----------
        data_path : str
        filenames_path : str
        transform : 
        img_ext : {".jpg", ".png"}, str, optional
            The file extension of the image, by default ".jpg".
        annot_ext : {".mat", ".dat"}, str, optional
            The file extension of the annotations, by default ".mat".
        img_mode : {"RGB", "BGR"}, str, optional
            The coloring scheme of the images, by default RGB.
        flip_pose : {False, True}, bool, optional
            If True, the Euler angles (yaw, pitch, roll) are flipped by pi radians, by default False.
        blur_image : {False, True}, bool, optional
            If True, the image is blurred by a Gaussian kernel, by default False.

        Returns
        -------
        None

        """
        self.data_path = data_path
        self.filename_path = filename_path
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext
        self.img_mode = img_mode
        self.flip_pose = flip_pose
        self.blur_image = blur_image

        self.filename_list = hoputils.filenames_from_path(self.filename_path)
        self.length = len(self.filename_list)

        self.train_data = self.filename_list
        self.train_labels = self.filename_list

    def __len__(self):
        """Returns the length of the dataset (i.e. the number of data points).
        
        Parameters
        ----------
        None

        Returns
        -------
        int
            The length of the dataset.
        
        """
        return self.length

    def __getitem__(self, index):
        """Grabs the data in the dataset, at index ``index``.

        Parameters
        ----------
        index : int
            The index used to access the data in the dataset.

        Returns
        -------
        image : numpy.array instance
            Array containing the image information.
        labels : 
        labels_cont : 
        filename : str

        TODO
        
        """
        image = Image.open(
            os.path.join(self.data_path, self.train_data[index] + self.img_ext)
        )
        image = image.convert(self.image_mode)
        annot_path = os.path.join(
            self.data_path, self.train_labels[index] + self.annot_ext
        )

        landmark_coords = hopeutils.landmarks2d_from_mat(annot_path)
        x_min = min(landmark_coords[0, :])
        y_min = min(landmark_coords[1, :])
        x_max = max(landmark_coords[0, :])
        y_max = max(landmark_coords[1, :])

        k = np.random.random_sample() * 0.2 + 0.2
        x_min -= 0.6 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 0.6 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)

        x_min, y_min = int(x_min), int(y_min)
        x_max, y_max = int(x_max), int(y_max)

        image = image.crop(x_min, y_min, x_max, y_max)

        euler_angles = hopeutils.euler_from_mat(annot_path)
        yaw, pitch, roll = hopeutils.rad_to_deg(euler_angles, to_list=True)

        # TODO: Engineer a workaround that allows the user to define the probability instead.
        if self.flip_pose:
            rnd = np.random.random_sample()
            if rnd < 0.5:
                yaw *= -1
                roll *= -1
                image = image.transpose(Image.FLIP_LEFT_RIGHT)

        if self.blur_image:
            rnd = np.random.random_sample()
            if rnd < 0.05:
                image = image.filter(ImageFilter.BLUR)

        angle_bins = np.array(range(-99, 102, 3))
        angle_bins = np.digitize([yaw, pitch, roll], bins) - 1

        labels = torch.LongTensor(angle_bins)
        labels_cont = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            image = self.transform(image)

        return cropped_image, labels, labels_cont, self.train_data[index]


# Can replace this class with the previous one.
class Pose_300W_LP_Dataset_Random(Dataset):
    def __init__(
        self,
        data_path,
        filename_path,
        transform,
        img_ext=".jpg",
        annot_ext=".mat",
        img_mode="RGB",
        flip_pose=False,
        blur_image=False,
    ):
        """Instantiates a Pose_300W_LP_Dataset object, containing data from the Pose_300_LP Dataset, but with random downsampling.

        Parameters
        ----------
        data_path : str
        filenames_path : str
        transform : 
        img_ext : {".jpg", ".png"}, str, optional
            The file extension of the image, by default ".jpg".
        annot_ext : {".mat", ".dat"}, str, optional
            The file extension of the annotations, by default ".mat".
        img_mode : {"RGB", "BGR"}, str, optional
            The coloring scheme of the images, by default RGB.
        flip_pose : {False, True}, bool, optional
            If True, the Euler angles (yaw, pitch, roll) are flipped by pi radians, by default False.
        blur_image : {False, True}, bool, optional
            If True, the image is blurred by a Gaussian kernel, by default False.

        Returns
        -------
        None

        """
        self.data_path = data_path
        self.filename_path = filename_path
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext
        self.img_mode = img_mode
        self.flip_pose = flip_pose
        self.blur_image = blur_image

        self.filename_list = hoputils.filenames_from_path(self.filename_path)
        self.length = len(self.filename_list)

        self.train_data = self.filename_list
        self.train_labels = self.filename_list

    def __len__(self):
        """Returns the length of the dataset (i.e. the number of data points).
        
        Parameters
        ----------
        None

        Returns
        -------
        int
            The length of the dataset.
        
        """
        return self.length

    def __getitem__(self, index):
        """Grabs the data in the dataset, at index ``index``.

        Parameters
        ----------
        index : int
            The index used to access the data in the dataset.

        Returns
        -------
        image : numpy.array instance
            Array containing the image information.
        labels : 
        labels_cont : 
        filename : str

        TODO
        
        """
        image = Image.open(
            os.path.join(self.data_path, self.train_data[index] + self.img_ext)
        )
        image = image.convert(self.image_mode)
        annot_path = os.path.join(
            self.data_path, self.train_labels[index] + self.annot_ext
        )

        landmark_coords = hopeutils.landmarks2d_from_mat(annot_path)
        x_min = min(landmark_coords[0, :])
        y_min = min(landmark_coords[1, :])
        x_max = max(landmark_coords[0, :])
        y_max = max(landmark_coords[1, :])

        k = np.random.random_sample() * 0.2 + 0.2
        x_min -= 0.6 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 0.6 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)

        x_min, y_min = int(x_min), int(y_min)
        x_max, y_max = int(x_max), int(y_max)

        image = image.crop(x_min, y_min, x_max, y_max)

        euler_angles = hopeutils.euler_from_mat(annot_path)
        yaw, pitch, roll = hopeutils.rad_to_deg(euler_angles, to_list=True)

        ds = 1 + np.random.randint(0, 4) * 5
        original_size = image.size
        image = image.resize(
            (image.size[0] / ds, image.size[1] / ds), resample=Image.NEAREST
        )
        image = image.resize(
            (original_size[0], original_size[1]), resample=Image.NEAREST
        )

        if self.flip_pose:
            rnd = np.random.random_sample()
            if rnd < 0.5:
                yaw *= -1
                roll *= -1
                image = image.transpose(Image.FLIP_LEFT_RIGHT)

        if self.blur_image:
            rnd = np.random.random_sample()
            if rnd < 0.05:
                image = image.filter(ImageFilter.BLUR)

        angle_bins = np.array(range(-99, 102, 3))
        angle_bins = np.digitize([yaw, pitch, roll], bins) - 1

        labels = torch.LongTensor(angle_bins)
        labels_cont = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            image = self.transform(image)

        return image, labels, labels_cont, self.train_data[index]


class AFLW2000_Dataset(Dataset):
    def __init__(
        self,
        data_path,
        filename_path,
        transform,
        img_ext=".jpg",
        annot_ext=".mat",
        image_mode="RGB",
    ):
        self.data_path = data_path
        self.filename_path = filename_path
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext
        self.img_mode = img_mode

        self.filename_list = hoputils.filenames_from_path(self.filename_path)
        self.length = len(self.filename_list)

        self.train_data = self.filename_list
        self.train_labels = self.filename_list

    def __len__(self):
        """Returns the length of the dataset (i.e. the number of data points).

        Parameters
        ----------
        None

        Returns
        -------
        int
            The length of the dataset.

        """
        return self.length

    def __getitem__(self, index):
        """Grabs the entry in the dataset, at index ``index``.

        Parameters
        ----------
        index : int
            The index used to access the entry in the dataset.
        
        Returns
        -------
        image : numpy.array instance
            Array containing the image information.
        labels : 
        labels_cont : 
        filename : str

        """
        image = Image.open(
            os.path.join(self.data_path, self.train_data[index] + self.img_ext)
        )
        image = image.convert(self.img_mode)
        annot_path = os.path.join(
            self.data_path, self.train_labels[index] + self.annot_ext
        )

        landmark_coords = hopeutils.landmarks2d_from_mat(annot_path)
        x_min = min(landmark_coords[0, :])
        y_min = min(landmark_coords[1, :])
        x_max = max(landmark_coords[0, :])
        y_max = max(landmark_coords[1, :])

        k = 0.20
        x_min -= 2 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 2 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)

        x_min, y_min = int(x_min), int(y_min)
        x_max, y_max = int(x_max), int(y_max)

        image = image.crop(x_min, y_min, x_max, y_max)

        euler_angles = hopeutils.euler_from_mat(annot_path)
        yaw, pitch, roll = hopeutils.rad_to_deg(euler_angles, to_list=True)

        angle_bins = np.array(range(-99, 102, 3))
        angle_bins = np.digitize([yaw, pitch, roll], bins) - 1

        labels = torch.LongTensor(angle_bins)
        labels_cont = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            image = self.transform(image)

        return image, labels, labels_cont, self.train_data[index]
