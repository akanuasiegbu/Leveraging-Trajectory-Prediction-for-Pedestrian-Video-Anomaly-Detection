"""
# Note that when I ran this orginally was using jupyter notebook
# WOuld need to make sure right directory is being looked looked at
# Also would want to add directoory and want correct directory to save new
# videos """

"""
loc : Directory that is looked at. (It's two directory out)
save_vid_loc : Directory that video is saved too
"""

import os
from os import listdir
from os.path import isfile, join, isdir
import cv2
from load_data import Files_Load
from experiments_code.config import hyparams, loc, exp

def vid_to_frames(vid_loc,  pic_loc):
    """
    Thats the location of the videos
    """

    frame_index = 0
    video_capture = cv2.VideoCapture(vid_loc)
    while True:
    
        ret, frame = video_capture.read()
        if ret != True:
            break

        cv2.imwrite(pic_loc + '/' +'{:02d}.jpg'.format(frame_index) , frame)
        frame_index += 1




def convert_spec_frames_to_vid(loc, save_vid_loc, vid_name, frame_rate):
    """
    loc : Directory that is looked at. 
    save_vid_loc : Directory that video is saved too
    frame_rate: frame rate for produced video
    """
    all_frames = []
    # loop over all the images in the folder
    for c in sorted(listdir(loc)):
        # if c[0] not in ['0', '1', '2', '3','4', '5','6','7','8','9']:
                # continue
        img_path = join(loc, c)
        img = cv2.imread(img_path)
        height,width,layers = img.shape
        size = (width, height)
        all_frames.append(img)

    out = cv2.VideoWriter( join(save_vid_loc, '{}.avi'.format(vid_name)), cv2.VideoWriter_fourcc(*'DIVX'), frame_rate, size)

    for i in range(0,len(all_frames)):
        out.write(all_frames[i])
    out.release()

def conver_data_vid(loc, save_vid_loc):
    """
    Used this orginally in jupyter to convert ucsd dataset to videos
    loc : Directory that is looked at. (It's two directory out)
    save_vid_loc : Directory that video is saved too
    """

    # loc = '/home/akanu/Dataset/Anomaly/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test'
    loc = None
    save_vid_loc = None # make
    for f in sorted(listdir(loc)):
        # if f[0] != 'T' or f[-2:] == 'gt':
            # continue
        directory_path = join(loc, f)
        if isdir(directory_path):
            all_frames = []
            # loop over all the images in the folder
            for c in sorted(listdir(directory_path)):
                # if c[0] not in ['0', '1', '2', '3','4', '5','6','7','8','9']:
                        # continue
                img_path = join(directory_path, c)
                img = cv2.imread(img_path)
                height,width,layers = img.shape
                size = (width, height)
                all_frames.append(img)

            out = cv2.VideoWriter( join(save_vid_loc, '{}.avi'.format(f)), cv2.VideoWriter_fourcc(*'DIVX'), 24, size)

            for i in range(0,len(all_frames)):
                out.write(all_frames[i])
            out.release()

def make_dir(dir_list):
    try:
        print(os.makedirs(join( '/mnt/roahm/users/akanu/dataset/Anomaly/Avenue_Dataset/',
                                *dir_list )) )
    except OSError:
        print('Creation of the directory {} failed'.format( join(os.path.dirname(os.getcwd()),
                                                            *dir_list) ) )
    else:
        print('Successfully created the directory {}'.format(   join(os.path.dirname(os.getcwd()),
                                                                *dir_list) ) )



if __name__ =='__main__':
    train_file = loc['data_load'][exp['data']]['train_vid']
    test_file = loc['data_load'][exp['data']]['test_vid']    
    test = False

    if test:
        file = test_file
    else:
        file = train_file

    for vid in sorted(listdir(file)):

        if test:
            dir_list = ['frames_of_vid', 'test', '{:02d}'.format(int(vid[:-4]))]
        else:
            dir_list = ['frames_of_vid', 'train', '{:02d}'.format(int(vid[:-4]))]

        vid_loc = test_file + '/' + vid

        make_dir(dir_list)
        pic_loc = join( '/mnt/roahm/users/akanu/dataset/Anomaly/Avenue_Dataset/', *dir_list )

        vid_to_frames(vid_loc, pic_loc)    