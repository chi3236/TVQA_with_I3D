import os
import h5py
import numpy as np
import json
import glob
import pysrt
from tqdm import tqdm
from PIL import Image
from utils import read_json_lines, load_json, save_json


import cv2 as cv

rgb_path = "/home/scw/Downloads/tvqa_new/frames_hq/"
of_path = "/home/scw/CLionProjects/optical_flow/cmake-build-release/optical_flow_hq/"
hf = h5py.File("whole_data.h5", "w")

def make_h5():
    group = hf.create_group("video")
    #j = 0
    for show_root_directory in os.listdir(rgb_path):
        print show_root_directory
        show_group = group.create_group(show_root_directory)
        for clip in tqdm(os.listdir(rgb_path + show_root_directory)):
            rgb_group = show_group.create_group("rgb_" + clip)
            of_group = show_group.create_group("of_" + clip)
            i = 0
            for rgb_image in os.listdir(rgb_path + show_root_directory + "/" + clip):
                image = Image.open(rgb_path + show_root_directory + "/" + clip + '/' + rgb_image)
                image = image.resize((224, 224))
                rgb_group.create_dataset('{:03d}'.format(i+1), data=np.array(image, dtype=np.float32), compression="gzip")
                i += 1

            i = 0
            for of_image in os.listdir(of_path + show_root_directory + "/" + clip):
                image = Image.open(of_path + show_root_directory + "/" + clip + '/' + of_image)
                image = image.resize((224, 224))
                of_group.create_dataset('{:03d}'.format(i+1), data=np.array(image, dtype=np.float32), compression="gzip")
                i += 1

if __name__ == '__main__':
    make_h5()




