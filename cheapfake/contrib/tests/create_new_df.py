"""
Python file that uses multiprocessing to create new dataframes.
"""

import os
import sys
import time
import multiprocessing
import concurrent.futures

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint
    
def process(filename, label, index):
    #print("Working on index {}".format(index))
    #sys.stdout.flush()
    vidcap = cv2.VideoCapture(filename)
    height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
    flag = False
    if height < width:
        flag = True
    
    return index, flag, filename, label
    
if __name__ == "__main__":
    metadata_path = "/home/shu/cheapfake/cheapfake/contrib/balanced_metadata_fs03.csv"
    dataframe = pd.read_csv(metadata_path)
    
    filelist = dataframe["Files"]
    labels = dataframe["label"]
    indices = list(range(len(filelist)))
    args = ((file, label, index) for (file, label, index) in zip(filelist, labels, indices))
    
    wide_videos = [None] * len(filelist)
    narrow_videos = [None] * len(filelist)
    wide_labels = [None] * len(filelist)
    narrow_labels = [None] * len(filelist)
    
    start_time = time.time()
    with concurrent.futures.ProcessPoolExecutor(multiprocessing.cpu_count() - 1) as executor:
        for idx, flag, name, label in tqdm(executor.map(process, *zip(*args)), total=len(filelist)):
            if flag is True:
                wide_videos[idx] = name
                wide_labels[idx] = label
            else:
                narrow_videos[idx] = name
                narrow_labels[idx] = label
    end_time = time.time()
    
    print("Entire process took {} seconds".format(end_time - start_time))
    
    wide_videos = [item for item in wide_videos if item is not None]
    wide_labels = [item for item in wide_labels if item is not None]
    narrow_videos = [item for item in narrow_videos if item is not None]
    narrow_labels = [item for item in narrow_labels if item is not None]
    
    wide_videos_df = pd.DataFrame({"Files": wide_videos, "label": wide_labels})
    narrow_videos_df = pd.DataFrame({"Files": narrow_videos, "label": narrow_labels})
    
    wide_videos_df.to_csv("./wide_balanced_metadata_fs03.csv")
    narrow_videos_df.to_csv("./narrow_balanced_metadata_fs03.csv")