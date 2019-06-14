import glob
import sys
import os
import tensorflow as tf
import h5py as hf
import numpy as np
import math
import re
import time
from utils import read_json_lines, load_json, save_json, load_pickle
sys.path.append("./kinetics-i3d")

qa_srt_file_path = "."
_CHECKPOINT_PATHS = {
    'rgb': './kinetics-i3d/data/checkpoints/rgb_scratch/model.ckpt',
    'rgb600': './kinetics-i3d/data/checkpoints/rgb_scratch_kin600/model.ckpt',
    'flow': './kinetics-i3d/data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': './kinetics-i3d/data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': './kinetics-i3d/data/checkpoints/flow_imagenet/model.ckpt',
}

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('num_classes', 400, 'Number of action classes')
tf.flags.DEFINE_integer('batch_size', 4, "Batch size")
tf.flags.DEFINE_integer('num_hidden', 150, "Number of hidden layer")
tf.flags.DEFINE_string("gpus", "0, 1", "GPUs to use")
tf.flags.DEFINE_integer("num_gpus", 2, "number of gpus")
tf.flags.DEFINE_integer("num_epoch", 100, "number of epoch")
#tf.flags.DEFINE_float('lr', 3e-4, 'learning_rate')
tf.flags.DEFINE_float('lr', 1e-1, 'learning_rate')
tf.flags.DEFINE_float("wd", 1e-5, 'weight decay')
#tf.flags.DEFINE_float("wd", 0.0, 'weight decay')
tf.flags.DEFINE_float('margin', 1.0, 'margin for ranking loss')
#tf.flags.DEFINE_bool("is_training", True, 'Is training')

video_file_path = "./whole_data.h5"
#glove_path = "./glove.840B.300d.txt"

#checkpoint_path = './log/checkpoint'
#checkpoint_prefix = os.path.join(checkpoint_path, "model")

image_path = "/home/scw/Downloads/tvqa_new/frames_hq/"
qa_srt_files = glob.glob(os.path.join(qa_srt_file_path, "*json"))
qa_file_list = []
#early_stopping = EarlyStopping(patience=3, verbose=True)
print("Loading json")
for i, qa_file in sorted(enumerate(qa_srt_files), key=lambda x: x[1]):
    qa_file_list.append(load_json(qa_file))
print("Done")
img_list = {}
y = 1
with open("image_list.txt", 'a') as f:
    for qa_file in qa_file_list:
        for qa in qa_file:
            if qa['vid_name'].startswith('bbt'):
                i = qa['located_frame'][0]
                while i <= qa['located_frame'][1]:
                    img_list['bbt_frames_' + qa['vid_name'].split('bbt_')[1] + '_' + str(i).zfill(5) + ' ' + image_path + 'bbt_frames/' + qa['vid_name'].split('bbt_')[1] + '/' + str(i).zfill(5) + '.jpg'] = 'o'
                    i += 1

            elif qa['vid_name'].startswith('castle'):
                i = qa['located_frame'][0]
                while i <= qa['located_frame'][1]:
                    img_list['castle_frames_' + qa['vid_name'] + '_' + str(i).zfill(5) + ' ' + image_path + 'castle_frames/' + qa['vid_name'] + '/' + str(i).zfill(5) + '.jpg'] = 'o'
                    i += 1

            elif qa['vid_name'].startswith('house'):
                i = qa['located_frame'][0]
                while i <= qa['located_frame'][1]:
                    img_list['house_frames_' + qa['vid_name'] + '_' + str(i).zfill(5) + ' ' + image_path + 'house_frames/' + qa['vid_name'] + '/' + str(i).zfill(5) + '.jpg'] = 'o'
                    i += 1

            elif qa['vid_name'].startswith('friends'):
                i = qa['located_frame'][0]
                while i <= qa['located_frame'][1]:
                    img_list['friends_frames_' + qa['vid_name'] + '_' + str(i).zfill(5) + ' ' + image_path + 'friends_frames/' + qa[
                        'vid_name'] + '/' + str(i).zfill(5) + '.jpg'] = 'o'
                    i += 1

            elif qa['vid_name'].startswith('grey'):
                i = qa['located_frame'][0]
                while i <= qa['located_frame'][1]:
                    img_list['grey_frames_' + qa['vid_name'] + '_' + str(i).zfill(5) + ' ' + image_path + 'grey_frames/' + qa[
                        'vid_name'] + '/' + str(i).zfill(5) + '.jpg'] = 'o'
                    i += 1
            else:
                i = qa['located_frame'][0]
                while i <= qa['located_frame'][1]:
                    img_list['met_frames_' + qa['vid_name'] + '_' + str(i).zfill(5) + ' ' + image_path + 'met_frames/' + qa[
                        'vid_name'] + '/' + str(i).zfill(5) + '.jpg'] = 'o'
                    i += 1
            print len(img_list)

    for line in sorted(img_list):
        f.write(line + '\n')