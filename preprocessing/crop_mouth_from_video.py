#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2020 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

""" Crop Mouth ROIs from videos for lipreading"""

import os
import cv2
import glob
import argparse
import numpy as np
import random
from collections import deque

from utils import *
from transform import *


def load_args(default_config=None):
    parser = argparse.ArgumentParser(description='Lipreading Pre-processing')
    # -- utils
    parser.add_argument('--video-direc', default=None, help='raw video directory')
    parser.add_argument('--landmark-direc', default=None, help='landmark directory')
    parser.add_argument('--filename-path', default='./lrw500_detected_face.csv', help='list of detected video and its subject ID')
    parser.add_argument('--save-direc', default=None, help='the directory of saving mouth ROIs')
    # -- mean face utils
    parser.add_argument('--mean-face', default='./20words_mean_face.npy', help='mean face pathname')
    # -- mouthROIs utils
    parser.add_argument('--crop-width', default=96, type=int, help='the width of mouth ROIs')
    parser.add_argument('--crop-height', default=96, type=int, help='the height of mouth ROIs')
    parser.add_argument('--start-idx', default=48, type=int, help='the start of landmark index')
    parser.add_argument('--stop-idx', default=68, type=int, help='the end of landmark index')
    parser.add_argument('--window-margin', default=12, type=int, help='window margin for smoothed_landmarks')
    # -- convert to gray scale
    parser.add_argument('--convert-gray', default=False, action='store_true', help='convert2grayscale')
    # -- test set only
    parser.add_argument('--trainset-only', default=False, action='store_true', help='process training set only')
    parser.add_argument('--testset-only', default=False, action='store_true', help='process test set only')
    parser.add_argument('--remove-frame', default=None, help='remove frames')
    parser.add_argument('--remove-pixel', default=None, help='remove pixels')
    
    args = parser.parse_args()
    return args

args = load_args()

# -- mean face utils
STD_SIZE = (256, 256)
mean_face_landmarks = np.load(args.mean_face)
stablePntsIDs = [33, 36, 39, 42, 45]


def crop_patch( video_pathname, landmarks):

    """Crop mouth patch
    :param str video_pathname: pathname for the video_dieo
    :param list landmarks: interpolated landmarks
    """

    frame_idx = 0
    frame_gen = read_video(video_pathname)
    while True:
        try:
            frame = frame_gen.__next__() ## -- BGR
        except StopIteration:
            break
        if frame_idx == 0:
            q_frame, q_landmarks = deque(), deque()
            sequence = []

        q_landmarks.append(landmarks[frame_idx])
        q_frame.append(frame)
        if len(q_frame) == args.window_margin:
            smoothed_landmarks = np.mean(q_landmarks, axis=0)
            cur_landmarks = q_landmarks.popleft()
            cur_frame = q_frame.popleft()
            # -- affine transformation
            trans_frame, trans = warp_img( smoothed_landmarks[stablePntsIDs, :],
                                           mean_face_landmarks[stablePntsIDs, :],
                                           cur_frame,
                                           STD_SIZE)
            trans_landmarks = trans(cur_landmarks)
            # -- crop mouth patch
            sequence.append( cut_patch( trans_frame,
                                        trans_landmarks[args.start_idx:args.stop_idx],
                                        args.crop_height//2,
                                        args.crop_width//2,))
        if frame_idx == len(landmarks)-1:
            while q_frame:
                cur_frame = q_frame.popleft()
                # -- transform frame
                trans_frame = apply_transform( trans, cur_frame, STD_SIZE)
                # -- transform landmarks
                trans_landmarks = trans(q_landmarks.popleft())
                # -- crop mouth patch
                sequence.append( cut_patch( trans_frame,
                                            trans_landmarks[args.start_idx:args.stop_idx],
                                            args.crop_height//2,
                                            args.crop_width//2,))
            return np.array(sequence)
        frame_idx += 1
    return None


def landmarks_interpolate(landmarks):
    
    """Interpolate landmarks
    param list landmarks: landmarks detected in raw videos
    """

    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    if not valid_frames_idx:
        return None
    for idx in range(1, len(valid_frames_idx)):
        if valid_frames_idx[idx] - valid_frames_idx[idx-1] == 1:
            continue
        else:
            landmarks = linear_interpolate(landmarks, valid_frames_idx[idx-1], valid_frames_idx[idx])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    # -- Corner case: keep frames at the beginning or at the end failed to be detected.
    if valid_frames_idx:
        landmarks[:valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
        landmarks[valid_frames_idx[-1]:] = [landmarks[valid_frames_idx[-1]]] * (len(landmarks) - valid_frames_idx[-1])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    assert len(valid_frames_idx) == len(landmarks), "not every frame has landmark"
    return landmarks

def rmv_frames(sequence, noise = 0.1):
#     print(noise)
    num = int(29 * noise)
#     print(num)
    for i in range(num):
        j = random.randrange(0, 29)
        sequence[j] *= 0
    return sequence

def rmv_pixels(sequence, noise = 0.1):
#     print(noise)
    num = int(96 * noise)
    for i in range(num):
        x = random.randrange(0, 96)
        for j in range(num):
            y = random.randrange(0, 96)
            sequence[:, x, y, :] *= 0
    return sequence

lines = open(args.filename_path).read().splitlines()
if args.trainset_only:
    lines = list(filter(lambda x: 'train' == x.split('/')[-2], lines)) 
elif args.testset_only:
    lines = list(filter(lambda x: 'test' == x.split('/')[-2], lines)) 

for filename_idx, line in enumerate(lines):

    filename, person_id = line.split(',')
    if filename_idx % 1000 == 0:
        print('idx: {} \tProcessing.\t{}'.format(filename_idx, filename))

    video_pathname = os.path.join(args.video_direc, filename+'.mp4')
    landmarks_pathname = os.path.join(args.landmark_direc, filename+'.npz')
    dst_pathname = os.path.join(args.save_direc, filename+'.npz')

    assert os.path.isfile(video_pathname), "File does not exist. Path input: {}".format(video_pathname)
#     assert os.path.isfile(landmarks_pathname), "File does not exist. Path input: {}".format(landmarks_pathname)
    if not os.path.isfile(landmarks_pathname):
        print("File does not exist. Path input: {}".format(landmarks_pathname))
        continue
    
    if os.path.exists(dst_pathname):
        continue

    multi_sub_landmarks = np.load( landmarks_pathname, allow_pickle=True)['data']
    landmarks = [None] * len( multi_sub_landmarks)
    for frame_idx in range(len(landmarks)):
        try:
            landmarks[frame_idx] = multi_sub_landmarks[frame_idx][int(person_id)]['facial_landmarks']
        except IndexError:
            continue

    # -- pre-process landmarks: interpolate frames not being detected.
    preprocessed_landmarks = landmarks_interpolate(landmarks)
    if not preprocessed_landmarks:
        continue

    # -- crop
    sequence = crop_patch(video_pathname, preprocessed_landmarks)
    assert sequence is not None, "cannot crop from {}.".format(filename)
    if args.remove_frame:
        address = "frame_" + args.remove_frame
        sequence = rmv_frames(sequence, int(args.remove_frame)/100.0)
    if args.remove_pixel:
        address = "pixel_" + args.remove_pixel
        sequence = rmv_pixels(sequence, int(args.remove_pixel)/100.0)
#     print(sequence.shape)
#     exit(0) 

    # -- save
    data = convert_bgr2gray(sequence) if args.convert_gray else sequence[...,::-1]
    if not os.path.exists(os.path.join(args.save_direc, address)):
        os.mkdir(os.path.join(args.save_direc, address))
    pathname = os.path.join(args.save_direc, address, filename+'.npz')
    save2npz(pathname, data=data)

print('Done.')
