"""
Python file containing garbage-collection utilities for detecting and plugging memory leaks.
"""

import gc
import torch
import numpy as np

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

    return np.around(nbytes, 2)


def memory_report():
    total_memory = 0
    for obj in gc.get_objects():
        if isinstance(obj, torch.Tensor):
            print(
                type(obj),
                obj.size(),
                np.around(human_readable(obj.element_size() * obj.nelement()), 2),
            )
            total_memory += human_readable(obj.element_size() * obj.nelement())
    total_memory = np.around(total_memory, 2)
    print("Total memory used: {} GB".format(total_memory))
            

