import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from keras.preprocessing.sequence import pad_sequences
import os
import pickle
# from collections import OrderedDict
# from tensorflow.python.ops import math_ops
# does change show

def Files_Load(train_file,test_file):
    """
    This file seperates the unique locations of each file and
    the has and return output of the locations and file text.

    train_file: locations of the training bounding boxes
    test_file : locations of the testing bounding boxes
    This just put folder directory in a list for train and test

    return: dict with 'files_train', 'files_test', 'train_txt', 'test_txt'
    dict has list of the files 
    examples '/home/akanu/dataset/dataset/Anomaly/Avenue_Dataset/bounding_box_tlbr/Txt_Data/Train_Box/16.txt'

    or '01.txt'


    Potential fixes to make: remove the number that is returned by dict
    01.txt etc seems unnecessary

    """
    box_train_txt = os.listdir(train_file)
    box_train_txt.sort()
    box_test_txt = os.listdir(test_file)
    box_test_txt.sort()

    loc_files_train, loc_files_test = [], []

    for txt in box_train_txt:
        loc_files_train.append(os.path.join(train_file, txt))
        # loc_files_train.append(train_file + txt)
    for txt in box_test_txt:
        loc_files_test.append(os.path.join(test_file, txt))
        # loc_files_test.append(test_file + txt)



    locations = {}
    locations['files_train'] = loc_files_train
    locations['files_test'] = loc_files_test
    locations['txt_train'] = box_train_txt
    locations['txt_test'] = box_test_txt

    return locations

# def load_pkl(loc_files ):
def load_pkl(data_loc, data_name):
    
    # data_loc = '/home/akanu/output_bitrap/avenue/gaussian_avenue_640_360_trained_normal_data.pkl'
    # data_loc = '/home/akanu/output_bitrap/avenue/gaussian_avenue_in_5_out_10.pkl'
    temp, datadict = {}, {}
    
    with open(data_loc, 'rb') as f:
        data = pickle.load(f)
    # outputs = { 'X_global': all_X_globals, 'video_file': all_video_files,'abnormal':  all_abnormal,
    #                         'id_x':all_id_x, 'id_y': all_id_y , 'frame_x':all_frame_x, 'frame_y':all_frame_y,
    #                         'pred_trajs': all_pred_trajs, 'gt_trajs':all_gt_trajs,'distributions':distribution}
    # # creates empty list
    for key in data.keys():
        if key == 'distributions':
            continue
        
        temp[key] = []
        datadict[key] = 0
    
    # adds data
    for key in temp.keys():
        data_keyed = data[key]

        for data_elem in data_keyed:
            if key == 'pred_trajs':
                # Might need to figure out a way to fix this here
                preds = []
                for pred in data_elem:
                    preds.append(pred[0])
                temp[key].append(np.array(preds).reshape(-1,4) ) # picking one of the 20 right here
            elif key == 'video_file':
                if data_name =='avenue':
                    temp[key].append('{:02d}.txt'.format(data_elem))
                elif data_name == 'st':
                    elem = str(data_elem)
                    if len(elem) == 5:
                        temp[key].append('0{}_{}.txt'.format(elem[0], elem[1:]))
                    elif len(elem) == 6:
                        temp[key].append('{}_{}.txt'.format(elem[0:2], elem[2:]))
                    else:
                        print('Error in load_pkl')
                        quit()

            else:
                temp[key].append(data_elem)

    # puts data in array
    for key in temp.keys():
        # print(key)
        datadict[key] = np.array(temp[key])

    temp = None #clears temp array
    
    #  rename keys
    datadict['x_ppl_box'] = datadict.pop('X_global')
    datadict['y_ppl_box'] = datadict.pop('gt_trajs') #.reshape(-1,4)

    # add frame_ppl_id key
    frame_ppl_id = []
    for i,j,k,l in zip(data['frame_x'], datadict['frame_y'], datadict['id_x'], datadict['id_y']):
        frame = np.append(i,j)
        ids = np.append(k,l)

        frame_ppl_id.append( np.column_stack((frame, ids)) )

    datadict['frame_ppl_id'] = np.array(frame_ppl_id)
    return datadict



def Boxes(loc_files, txt_names, input_seq, pred_seq,data_consecutive, pad ='pre', to_xywh = False, testing= False, window =1 ):
    """
    This file process the bounding box data and creates a numpy array that
    can be put into a tensor



    
    loc_files: List that contains that has text files save
    txt_names: Txt file names. For visualization process
    time_step: Sequence length input. Also known as frames looked at for input
    input_seq: number of sequences that are inputted
    pred_seq: number of sequences predicted from the input
    data_consecutive: ensure that input and output pedestrains are in consecutive frames
    pad: inputs 'pre' or 'post'. supplments short data with ones in front
    to_xywh: if given file is in tlbr and want to convert to xywb

    return:
    x_person_box: Has bounding box locations
    y_person_box: Label for bounding box locations
    frame_person_id: Contains frame Number and person_Id of entire sequence,
                     Last element is prediction frame. For visulization process
    video_file: Points to video file used. This gives me the frame video used
                and also the person id.

    abnormal: tells you wheather the frame is abnormal or not

    testing: tells if loading testing txt files or training
    """
    #intilization 
    x_ppl_box, y_ppl_box, frame_ppl_id, video_file = [], [], [], [] #Has bounding box locations inside
    abnormal_gt, abnormal_ped_pred, abnormal_ped_input = [] , [], []
    frame_x, frame_y, id_x, id_y = [], [], [], []

    #For splitting process
    split = 0
    find_split = 0

    # Tells me how many in sequence was short.
    # Do I want to go back and count for train and test separately
    short_len = 0


    datadict = {}

    for loc, txt_name in zip(loc_files, txt_names):
        data = pd.read_csv(loc, ' ' )
        # Note that person_box is 1 behind ID
        max_person = data['Person_ID'].max()
        for num in range(1,max_person+1):
            temp_box = data[data['Person_ID'] == num ]['BB_tl_0	BB_tl_1	BB_br_0	BB_br_1'.split()].values
            if to_xywh:
                temp_box = temp_box.astype(float)
                temp_box[:,2] = np.abs(temp_box[:,2] - temp_box[:,0] )
                temp_box[:,3] = np.abs(temp_box[:,3] - temp_box[:,1])

                temp_box[:,0] = temp_box[:,0] + temp_box[:,2]/2
                temp_box[:,1] = temp_box[:,1] + temp_box[:,3]/2

            person_seq_len = len(temp_box)
            # To split the frame and id to seperate code this is where I would make
            # initial change


            temp_frame_id = data[data['Person_ID'] == num ]['Frame_Number Person_ID'.split()].values
            temp_frame_all = data[data['Person_ID'] == num ]['Frame_Number'].values
            temp_id_all = data[data['Person_ID'] == num ]['Person_ID'].values
            ###########################################################
            # If regenerate training data, and then retrain I can simply this
            # Block
            if testing: 
                # Have this here because I changed anomaly to abnormal_ped
                # And did not need to retrain model, so testing data has slighting diff format same data.
                # So currently testing data has slightly different format
                # different column title and additional column (abnormal_gt)
                # For training we know abnormal_gt is 0 because training is normal
                abnormal_frame_ped = data[data['Person_ID'] == num]['abnormal_ped'].values
                abnormal_gt_frame = data[data['Person_ID'] == num]['abnormal_gt'].values
            else:
                abnormal_frame_ped = data[data['Person_ID'] == num]['anomaly'].values
            ###################################################################

            if person_seq_len >= (input_seq + pred_seq):
    
                # checks that frames are loaded sequentially to begin with
                # Meaning that frame 1, frame 2, frame 3,....
                fix = temp_frame_id[:,0]
                cons_check = np.argsort(fix) == np.arange(0, len(fix), 1)
                assert np.all(cons_check) == True, print('Frames are not ordered correctly /n sort data')

                    
                    
                for i in range(0, person_seq_len - input_seq - pred_seq + 1, window):
    
                    

                    start_input = i
                    end_input = i + input_seq
                    end_output = i + input_seq + pred_seq
                
                    
                    temp_fr_person_id = temp_frame_id[start_input:end_output]

                    x_and_y = temp_frame_all[start_input:end_output]


                    if data_consecutive:
                        # This ensures that data inputted is from consecutive frames 
                        first_frame_input = temp_frame_all[start_input]
                        proposed_last_output = temp_frame_all[start_input] + input_seq + pred_seq
                        check_seq = np.cumsum(x_and_y) == np.cumsum(np.arange(first_frame_input, proposed_last_output, 1))
                        if not np.all(check_seq):
                            continue
                    
                    #input seq frame and pred seq frames
                    x_frames = temp_frame_all[start_input:end_input]
                    y_frames = temp_frame_all[end_input:end_output]

                    # input seq id and pred seq id
                    x_ids = temp_id_all[start_input:end_input]
                    y_ids = temp_id_all[end_input:end_output]


                    temp_input_box = temp_box[start_input:end_input]
                    temp_pred_box = temp_box[end_input:end_output]
                    
                    assert temp_input_box.shape == (input_seq,4)
                    assert temp_pred_box.shape == (pred_seq,4)
                    # temp_person_box = temp_box[i:(i+time_steps)]
                   

                    x_ppl_box.append(temp_input_box)
                    y_ppl_box.append(temp_pred_box)

                    frame_ppl_id.append(temp_fr_person_id)
                    frame_x.append(x_frames)
                    frame_y.append(y_frames)

                    id_x.append(x_ids)
                    id_y.append(y_ids)

                    video_file.append(txt_name) # Load one file at a time so this works
                    
                    abnormal_ped_input.append(abnormal_frame_ped[start_input:end_input])
                    abnormal_ped_pred.append(abnormal_frame_ped[end_input:end_output]) #Finds if predicted frame is abnormal

                    if testing:
                        # abnormal_gt.append(abnormal_gt_frame[i+time_steps])
                        abnormal_gt.append(abnormal_gt_frame[end_input:end_output])
                    else:
                        abnormal_gt.append(np.zeros([pred_seq,1]))

            elif person_seq_len == 1:
                # want it to skip loop
                continue
            # This would add noise to data
            elif person_seq_len < input_seq + pred_seq:
                continue

                
                # x_and_y = temp_frame_all


                # if data_consecutive:
                #     # This ensures that data inputted is from consecutive frames 
                #     first_frame_input = temp_frame_all[0]
                #     proposed_last_output = temp_frame_all[start_input] + person_seq_len
                #     check_seq = np.cumsum(x_and_y) == np.cumsum(np.arange(first_frame_input, proposed_last_output, 1))
                #     if not np.all(check_seq):
                #         continue

                # temp_person_box_pad = pad_sequences(temp_box.T, maxlen = input_seq + pred_seq, padding = pad).T
                # temp_frame_all_pad = pad_sequences(temp_frame_all.T,  maxlen = input_seq + pred_seq, padding = pad).T
                # temp_id_all_pad = pad_sequences(temp_id_all.T,  maxlen = input_seq + pred_seq, padding = pad).T
                
                # temp_fr_person_id_pad = pad_sequences(temp_frame_id.T,  maxlen = input_seq + pred_seq, padding = pad).T
                # abnormal_frame_ped_pad = pad_sequences(abnormal_frame_ped.T,  maxlen = input_seq + pred_seq, padding = pad).T
                # abnormal_gt_frame_pad = pad_sequences(abnormal_gt_frame.T,  maxlen = input_seq + pred_seq, padding = pad).T

                # # Indexing
                # start_input = 0
                # end_input = 0 + input_seq
                # end_output = 0 + input_seq + pred_seq
                
                # # input frames and predicted frames if frame 
                # x_frames = temp_frame_all_pad[start_input:end_input]
                # y_frames = temp_frame_all_pad[end_input:end_output]

                # # input ids and predicted ids are set to zero
                # # would be able to sort them outd

                
                # x_ids = temp_id_all_pad[start_input:end_input]
                # y_ids = temp_id_all_pad[end_input:end_output]

                # x_ppl_box.append(temp_person_box_pad[start_input:end_input,:])
                # y_ppl_box.append(temp_person_box_pad[end_input:end_output,:])
            
                # frame_ppl_id.append(temp_fr_person_id_pad[start_input:end_output,:])

                # frame_x.append(x_frames)
                # frame_y.append(y_frames)

                # id_x.append(x_ids)
                # id_y.append(y_ids)
            
                # video_file.append(txt_name)
                # abnormal_ped.append(abnormal_frame_ped_pad[end_input:end_output]) #Finds if predicted frame is abnormal

                # abnormal_gt.append(abnormal_gt_frame_pad[end_input:end_output])

            else:
                print('error')

    datadict['x_ppl_box'] = np.array(x_ppl_box)
    datadict['y_ppl_box'] = np.array(y_ppl_box)
    datadict['frame_ppl_id'] = np.array(frame_ppl_id) # delete later not needed once otuer changes 
    datadict['video_file'] = np.array(video_file)
    datadict['abnormal_ped_input'] = np.array(abnormal_ped_input, dtype=np.int8)
    datadict['abnormal_ped_pred'] = np.array(abnormal_ped_pred, dtype=np.int8)
    datadict['abnormal_gt_frame'] = np.array(abnormal_gt, dtype=np.int8)
    datadict['id_x'] = np.array(id_x)
    datadict['id_y'] = np.array(id_y)
    datadict['frame_x'] = np.array(frame_x)
    datadict['frame_y'] = np.array(frame_y)

    return  datadict


def test_split_norm_abnorm(testdict):
    abnormal_index = np.nonzero(testdict['abnormal_ped_pred'])
    normal_index = np.where(testdict['abnormal_ped_pred'] == 0)
    normal_dict = {}
    abnormal_dict = {}

    for key in testdict.keys():
        normal_dict[key] = testdict[key][normal_index]
        abnormal_dict[key] = testdict[key][abnormal_index]


    return abnormal_dict, normal_dict


def norm_train_max_min(data, max1, min1, undo_norm=False):

    """
    data_dict: data input in the form of a dict. input is of same structure as
                output of  Boxes function

    data: data that is not in dictornary format that needs to be unnormailized
    max1 : normalizing parameter
    min1: normalizing parameter
    undo_norm: unnormailize data boolean

    return: depends on undo_norm boolean:
            if undo_norm is true return unnormailized data
            if undo_norm is false normailize


            
    """


    if undo_norm:
        # If data comes in here it is not in a dictornay format
        data = data*(max1-min1) + min1
        return data

    else:
        # If data comes in here it should be in the same 
        # format as Boxes function output partially 
        xx = (data['x_ppl_box'] - min1)/(max1 - min1)
        yy = (data['y_ppl_box'] - min1)/(max1 - min1)
        return xx,yy
