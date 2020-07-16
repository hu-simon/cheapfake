"""
Python script for the LipNet datasets.
"""

import os
import re
import copy
import time
import random

import cv2
import glob
import json
import torch
import numpy as np
import cheapfake.utils.dlibutils as dlibutils


class LipNetDataset(torch.utils.data.Dataset):
    """
    Implementation of the LipNetDataset class.
    """

    letters = [re.sub("\n", "", line) for line in open("alphabet.txt", "r")]

    def __init__(
        self, video_path, annotations_path, file_list, video_padding, text_padding, mode
    ):
        """ Instantiates a LipNetDataset object.

        Parameters
        ----------
        video_path : str
        annotations_path : str
        file_list : list
        video_padding : int
        text_padding : int
        mode : str

        Returns
        -------
        None

        """
        self.video_path = video_path
        self.annotations_path = annotations_path
        self.file_list = file_list
        self.video_padding = video_padding
        self.text_padding = text_padding
        self.mode = mode

        with open(file_list, "r") as f:
            self.videos = [
                os.path.join(video_path, line.strip()) for line in f.readlines()
            ]

        self.data = list()
        for video in self.videos:
            items = video.split(os.path.sep)
            self.data.append((video, items[-4], items[-1]))

    def __len__(self):
        """ Returns the length of the data.

        Parameters
        ----------
        None

        Returns
        -------
        output : int
            The length of the data, i.e. the number of data points.
        """
        return len(self.data)

    def _load_video(self, path):
        """ Loads a video as an array. 

        Parameters
        ----------
        path : str
            Path to the directory with the video.

        Returns
        -------
        arr : numpy.array instance
            Array containing the video data.
        """
        files = os.listdir(path)
        files = list(filter(lambda file: file.find(".jpg") != -1, files))
        files = sorted(files, key=lambda file: int(os.path.splitext(file)[0]))
        arr = [cv2.imread(os.path.join(path, file)) for file in files]
        arr = list(filter(lambda im: not im is None, arr))
        arr = [
            cv2.resize(image, (128, 64), interpolation=cv2.INTER_LANCZOS4)
            for image in arr
        ]
        arr = np.stack(arr, axis=0).astype(np.float32)

        return arr

    def __getitem__(self, index):
        """ Returns the item, from the data, at ``index``.

        Parameters
        ----------
        index : int
            The index of the entry we want to return.
        
        Returns
        -------
        item : dict {"video", "text", "video_length", "text_length"}
            Dictionary containing the video, text, along with their associated lengths.
        """
        (video, speaker, name) = self.data[index]
        video = self._load_video(video)
        annotations = self._load_annotations(
            os.path.join(self.annotations_path, speaker, "align", name + ".align")
        )
        annotations_length = annotations.shape[0]
        annotations = self._add_padding(annotations, self.text_padding)

        if self.mode == "train":
            video, _ = dlibutils.random_horizontal_flip(video)

        video = dlibutils.normalize_images(video)
        video_length = video.shape[0]
        video = self._add_padding(video, self.video_padding)

        item = {
            "video": torch.FloatTensor(video.transpose(3, 0, 1, 2)),
            "video_length": video_length,
            "text": torch.LongTensor(annotations),
            "text_length": annotations_length,
        }

        return item

    def _add_padding(self, tensor, length):
        """
        Adds zero-padding to an array.

        Parameters
        ----------
        tensor : torch.Tensor instance
            The tensor that is to be zero-padded.
        length : int
            The amount of zero-padding to append to the tensor.

        Returns
        -------
        tensor : torch.Tensor instance
            The original tensor, padded with zeros.
        """
        tensor = [tensor[k] for k in range(tensor.shape[0])]
        pad_shape = tensor[0].shape
        for k in range(length - len(tensor)):
            tensor.append(np.zeros(pad_shape))

        tensor = np.stack(tensor, axis=0)

        return tensor

    def _load_annotations(self, path):
        """ Loads the annotations in a text file.

        Parameters
        ----------
        path : str
            Path to the annotation file to be loaded.
        
        Returns
        -------
        annotations : numpy.array instance
            The annotations loaded from the path.
        """
        with open(path, "r") as f:
            lines = [line.strip().split(" ") for line in f.readlines()]
            text = [line[2] for line in lines]
            text = list(filter(lambda s: not s.upper() in ["SIL", "SP"], text))

        annotations = LipNetDataset.text_to_array(" ".join(text).upper(), start_index=1)

        return annotations

    @staticmethod
    def text_to_array(text, start_index):
        """ Converts text to an array by assignment each element of the array as the shifted alpha-numeric position of the characters in ``text``. 

        For example, assuming ``letters`` as defined above, the phrase "GO TO" is converted to ``array([7, 0, 20, 15])`` provided ``start_index == 1``

        Parameters
        ----------
        text : str
            The string to be converted to an array.
        start_index : int
            The reference starting index, in case a further shift is used.

        Returns
        -------
        array : numpy.array instance
            Array containing the split text, where each character of ``text`` is assigned as an element of the array.
        """
        # Automatic upper-case in case user forgets!
        text = text.upper()

        array = list()
        for char in list(text):
            array.append(LipNetDataset.letters.index(char) + start_index)

        return np.array(array)

    @staticmethod
    def array_to_text(array, start_index, return_spaces=False, ctc_mode=False):
        """ Converts an array to text, essentially the inverse of ``text_to_array``.

        Parameters
        ----------
        array : numpy.array 
            Array containing the index of the characters to form the text.
        start_index : int
            The reference starting index, in case a further shift is used.
        return_spaces : {False, True}, optional
            Boolean determining if spaces are removed via ``.strip()`` or not, by default False.
        ctc_mode : {False, True}, optional
            Boolean representing whether or not, the input is from the CTC loss.
        
        Returns
        -------
        text : str
            The text compiled from the array.
        """
        text = list()
        if ctc_mode:
            prefix = -1
            for index in array:
                if prefix != index and index >= start_index:
                    if (
                        len(text) > 0
                        and text[-1] == " "
                        and LipNetDataset.letters[index - start_index] == " "
                    ):
                        pass
                    else:
                        text.append(LipNetDataset.letters[index - start_index])
                prefix = index
        else:
            for index in array:
                if index >= start_index:
                    text.append(LipNetDataset.letters[index - start_index])

        if return_spaces:
            text = "".join(text)
        else:
            text = "".join(text).strip()

        return text


def create_dataloader(
    dataset, batch_size=96, num_workers=4, shuffle=True, drop_last=False
):
    """
    Returns a torch.utils.data.DataLoader object to be used during training.

    Parameters
    ----------
    dataset : numpy.array instance
        Numpy array containing the data.
    batch_size : int, optional
        The size of each batch, a parameter passed on to torch.utils.data.DataLoader, by default 96.
    num_workers : int, optional
        The number of workers, a parameter passed on to torch.utils.data.DataLoader, by default 16.
    shuffle : {True, False}, bool, optional
        Determines whether the data is randomly shuffled, by default True.
    drop_last : {False, True}, bool, optional
        Determines whether the last data point is dropped, by default False.

    Returns
    -------
    dataloader : torch.utils.data.DataLoader instance
        A dataloader object used for yielding data.
    """
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=drop_last,
    )

    return dataloader
