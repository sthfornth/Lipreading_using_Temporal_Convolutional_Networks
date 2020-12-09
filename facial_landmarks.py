# This code is developed to be part of the Lip-reading Project code for 670 F2020
# preprocessing video (mp4) file to extract ROI per frame. 
# by Catherine Huang
# code based on the blog post: https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
import numpy as np
import cv2
import dlib
import imutils
import argparse
from imutils import face_utils
import os
# import matplotlib.pyplt as plt

# root_path = 
class_name = ''
vid_file_name = ''
root_path = ".\\yt_data\\"
load_root_path = "video\\" # change this to where the videos are stored
save_root_path = "yt_roi\\"
data_type = "test"

def get_all_files_path():
    files = []
    test_classes = os.listdir(root_path + load_root_path)
    print("number of class: {}\n".format(len(test_classes)))
    for c in test_classes:
        file_names = os.listdir(root_path + load_root_path+ c +"\\{}\\".format(data_type))
        files.append(file_names)
        # print(files)
    # print("")
    return test_classes, files

def video_to_frame(path):
    vid_obj = cv2.VideoCapture(path)

    frames = []
    success = 1

    while success:
        # vid_frame is a np array of (h x w x 3)
        success, vid_frame = vid_obj.read()
        frames.append(vid_frame)
    # print("video frames extraced {}".format(len(frames)))
    return frames

# return the croped region for the mouth ROI per frame 
def get_roi_per_frame(frame, frame_number):
    # convert to grayscale
    img = np.copy(frame)
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('orig', frame)
    # predict the face in frame
    rects = detector(gray_img, 1)
    roi = []

    # loop over the detected faces(rectangles)
    for (i, rect) in enumerate(rects):
        shape = predictor(gray_img, rect)
        shape = face_utils.shape_to_np(shape)
        

        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            if name == 'mouth':
                (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
                # print("roi x {} y {} w {} h {}".format(x, y, w, h))
                (x, y) = get_roi_center(x, y, w, h)
                # print("new x {} new y {}".format(x, y))
                roi = gray_img[y:y+96, x:x+96]
                # cv2.imshow("ROI", roi)
                cv2.imwrite("{}_ROI_{}_{}.png".format(class_name, str(frame_number).zfill(3), vid_file_name), roi)
                # cv2.waitKey(5000)
        
        # convert dlib's rectangle found by predictor to cv2 bounding box
        # (x, y, w, h) = face_utils.rect_to_bb(rect)
        # draw the bounding box
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # if multiple faces, annotate the face
        # cv2.putText(frame, "Face #{}".format(i+1), (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # loop over the (x,y) for facial landmarks and display
        # for (x,y) in shape:
        #     cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
    
    # cv2.imshow("landmarks", frame)
    # cv2.imwrite("{}_landmarks_{}.png".format(class_name, frame_number), frame)
    # cv2.imwrite("{}_orig_{}.png".format(class_name, frame_number), img)
    # cv2.waitKey(5000)
    # save the rois
    return roi

    # pass    

def get_roi_center(x, y, w, h):
    w_half = np.floor(w/2)
    h_half = np.floor(h/2)
    center_x = x+w_half
    center_y = y+h_half
    new_x = center_x - 48
    new_y = center_y - 48
    # prevent going overbord
    if new_x < 0:
        new_x = 0
    if new_y < 0:
        new_y = 0
    return (int(new_x), int(new_y))
    # pass

# save the ROIs to npz file contain frames x height x width
def save_roi(roi_data, index):
    # print(roi_data)
    check_create_folders()
    # print(roi_data[0])
    print(roi_data.shape)
    np.savez(root_path + save_root_path+ class_name + '\\{}\\'.format(data_type)+ '{}.npz'.format(vid_file_name), data = roi_data)

# check and create the folders to save the ROIs extracted from clips
def check_create_folders():
    # root folder (word/class) ie AROUND
    if not os.path.exists(root_path + save_root_path + class_name):
        os.makedirs(root_path + save_root_path+ class_name)
    # data type i.e. test, val or train
    if not os.path.exists(root_path + save_root_path + class_name + '\\{}\\'.format(data_type)):
        os.makedirs(root_path + save_root_path+ class_name + '\\{}\\'.format(data_type))


# test loading .npz file
# roi_load = np.load('ABOUT_00001.npz')
# print(roi_load.files)
# roi_data = roi_load['data']
# print(roi_data.shape)
# print(roi_data)

# print(get_all_files_path())
    
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
classes, files = get_all_files_path()

for c in range(len(classes)): # go through each class
    if classes[c] == "AFTER" or classes[c] == "ANSWER":
        continue
    else:
        class_name = classes[c]
        for f in range(len(files[c])): # go through each video file for that class
            vid_file_name = files[c][f]
            print(root_path + load_root_path + files[c][f])
            yt_vid_frames = video_to_frame(root_path + load_root_path + classes[c] + "\\{}\\".format(data_type) + files[c][f])
            # print(len(yt_vid_frames))        
            rois = np.ones((29,96,96))
            try:
                for i in range(0, 29):
                    # print(i)
                    rois[i, :, :] = get_roi_per_frame(yt_vid_frames[i], i)
                save_roi(rois, f)
            except:
                # if issue found, like can't find face or what not, go to the next file
                continue

# if __name__:"__main__":
#     # load arguments
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--folder", type=str, help="path to the main folder that hold the data")
#     parser.add_argument("--data_type", type=str, help="the type of data file, test, val, or train data")
#     parser.add_argument("--roi_size", type= int, help="single int for the width and height of the ROI")
#     args = parser.parse_args()
 
