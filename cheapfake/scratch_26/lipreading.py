"""
Python file that collects all of the LRS3 videos and extracts their frames.

This Python script currently works as intended. If we want to add more directories, then they can just go in the ``create_new_directories()`` function.
"""

import os
import time

import glob
import itertools
from ytbot.contrib.__main__ import YouTubeDownloader


def main():
    suffix_names = [
        f
        for f in os.listdir("/home/shu/shu/Datasets/LRS3/trainval")
        if not f.startswith(".")
    ]
    root_path = "/home/shu/shu/Datasets/VGG_Lipreading/trainval"
    paths = [os.path.join(root_path, suffix) for suffix in suffix_names]
    subdirectories = [
        "frames",
        "media/captions/raw",
        "media/captions/manual",
        "media/raw_video",
        "media/raw_audio",
        "media/comb_video",
    ]
    prods = list(itertools.product(paths, subdirectories))
    paths = ["/".join(item) for item in prods]

    for path in paths:
        try:
            os.makedirs(path)
        except OSError:
            print("[INFO] Failed to create {}".format(path))
        else:
            print("[INFO] Successesfully created {}".format(path))

    print("".join("-" * 80))
    print("[INFO] Downloading YouTube videos")
    print("".join("-" * 80))

    urls = ["https://youtube.com/watch?v=" + item for item in suffix_names]
    urls_subset = urls[:2]

    ytdownloader = YouTubeDownloader(urls, extract_path=root_path)
    ytdownloader.download_all_videos()


if __name__ == "__main__":
    main()
