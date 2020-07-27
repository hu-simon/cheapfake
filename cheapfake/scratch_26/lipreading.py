"""
Python file that collects all of the LRS3 videos and extracts their frames.

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

    urls = ["https://youtube.com/watch?v=" + suffix for suffix in suffix_names]

    ytdownloader = YouTubeDownloader(urls, extract_path=root_path)


if __name__ == "__main__":
    main()
