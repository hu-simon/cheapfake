"""
Python file implementing the losses in LipNet.
"""

import os
import re
import time

import torch
import numpy as np


def levenshtein_distance(truth, prediction, cost=(1, 1, 2)):
    """ Computes the weighted Levenshtein distance between two sequences of characters.

    The weighted Levenshtein distance is defined as the minimum number of edits to transform one sequence to another, where each operation has an associated cost.

    By default, the weights are set to (1, 1, 2), as substitution is actually a deletion and then insertion.

    Parameters
    ----------
    truth : str
        The ground truth sequence.
    prediction : str
        The predicted sequence.
    cost : tuple (of ints), optional
        The associated costs with (insertion, deletion, substitution), by default (1, 1, 2). 

    Returns
    -------
    l_dist : int
        The minimum number of edits required to transform ``prediction`` into ``truth``.

    Notes
    -----
    Fun fact this is literally a distance in that it satisfies the triangle inequality: the Levenshtein distance between two strings is no greater than the sum of their Levenshtein distances from a third string.

    Time and space complexity for this implementation are both O(``len(truth) * len(prediction)``.
    """
    table = np.zeros((len(truth) + 1, len(prediction) + 1), dtype=np.int)
    for k in range(len(truth) + 1):
        for l in range(len(prediction) + 1):
            if k == 0:
                table[k, l] = l
            elif l == 0:
                table[k, l] = k
            else:
                table[k, l] = min(
                    table[k, l - 1] + cost[0],
                    table[k - 1, l] + cost[1],
                    table[k - 1, l - 1] + cost[2]
                    if truth[k - 1] != prediction[l - 1]
                    else table[k - 1, l - 1] + 0,
                )
    l_dist = table[len(truth), len(prediction)]

    return l_dist


def damerau_levenshtein_distance(truth, prediction, cost=(1, 1, 2, 2)):
    """ Computes the weighted Damerau-Levenshtein distance between two sequences of characters. 

    The weighted Damerau-Levenshtein distance is calculated assuming the following operations: insertion, deletion, substitution, and transposition between adjacent characters.

    Parameters
    ----------
    truth : str
        The ground truth sequence.
    prediction : str
        The predicted sequence.
    cost : tuple (of ints), optional
        The associated costs with (insertion, deletion, substitution, transposition), by default (1, 1, 2, 2).
    
    Returns
    -------
    dl_list : int
        The minimum number of edits required to transform ``prediction`` to ``truth`` using the Damerau-Levenshtein distance.
    """
    """
    table = np.zeros((len(truth) + 1), len(prediction) + 1, dtype=np.int)

    for k in range(len(truth) + 1):
        for l in range(len(prediction) + 1):
    """
    raise NotImplementedError


def hamming_distance(truth, prediction):
    """ Computes the Hamming distance between two sequences of characters.

    The Hamming distance is defined as the number of positions in a string, where characters in a sequence differ from each other.

    Parameters
    ----------
    truth : str
        The ground truth sequence.
    prediction : str 
        The predicted sequence.

    Returns
    -------
    h_dist : int
        The number of characters that two strings differ by.

    Notes
    -----
    The Hamming distance only operates on strings of the same length. 
    """
    assert len(truth) == len(
        prediction
    ), "The Hamming distance requires that the two sequences are of the same length."
    diffs = np.array(list(truth)) != np.array(list(prediction))
    h_dist = sum(diffs)

    return h_dist


def word_error_rate(truth, prediction):
    """ Computes the word error rate using the Levenshtein distance.

    (TODO Insert some information about how this is calculated here)

    Parameters
    ----------
    truth : str
        The ground truth sequence.
    prediction : str
        The predicted sequence.

    Returns
    -------
    wer : int
        The word error rate between ``truth`` and ``sequence``.

    Notes
    -----
    Since this function uses the Levenshtein distance, it is not required that the two sequences are of the same length.
    """
    pairs = [
        (char[0].split(" "), char[1].split(" ")) for char in zip(truth, prediction)
    ]
    wer = [
        1.0 * levenshtein_distance(word[0], word[1]) / len(word[0]) for word in pairs
    ]

    return wer


def character_error_rate(truth, prediction):
    """ Computes the character error rate using the Levenshtein distance.

    (TODO Insert some information about how this is calculated here)

    Parameters
    ----------
    truth : str
        The ground truth sequence.
    prediction : str
        The predicted sequence.

    Returns
    -------
    cer : int
        The character error rate between the two sequences, computed using the Levenshtein distance.
    """
    cer = [
        1.0 * levenstein_distance(char[0], char[1]) / len(char[0])
        for char in zip(truth, prediction)
    ]

    return cer
