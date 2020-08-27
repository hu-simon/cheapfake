"""
Python file containing garbage-collection utilities for detecting and plugging memory leaks.
"""

import gc
import torch


def human_readable(nbytes):
    """Converts bytes to a more human readable format, in gigabytes.

    Parameters
    ----------
    nbytes : int, optional
        The number of bytes.
    
    Returns
    -------
    nbytes : float
        The number of bytes, in gigabytes.

    """
    nbytes = nbytes * 1e-9

    return nbytes


def memory_report():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(
                type(obj),
                obj.size(),
                human_readable(obj.element_size() * obj.nelement()),
            )

