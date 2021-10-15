import numpy as np
from load_data import norm_train_max_min
import cv2
import os
from custom_functions.utils import make_dir
from os.path import join


# def seperate_misclassifed_examples(bm_model,test_x, indices, test_y, threshold=0.5):
def seperate_misclassifed_examples(y_pred,indices, test_y, threshold=0.5):
    """
    This function takes in the binary model and seperates out the index
    Values. Note that index values on the second row of test_x data.
    This index values allow for mapping back to test dataset.
    For an easy check compare numbers to confusion matrix with correct
    threshold.

    bm_model: binary classifer
    test_x: data we want to classify. These are iou points and index values
    indices: this is the index values for each iou point that corrrespond to
             dict
    test_y: the ground truth of data
    return:
        index values seperate into FN and FP maybe a dictornay ??
    """
    # Now this seperate into True and False
    # y_pred = bm_model.predict(test_x) > threshold  #might need to just have this not be indexed

    y_pred = y_pred > threshold

    TN, TP, FP, FN = [], [], [], []
    index ={}

    # Key here is that the indices for the test data points
    # are on the second row
    for gt,pred,map_index in zip (test_y.reshape(-1),y_pred,indices):
        if gt == False and pred == False:
            # This is for TN
            TN.append(map_index)
        elif gt == True and pred == False:
            # This one is FN
            FN.append(map_index)
        elif gt == False and pred == True:
            ## THis one is FP
            FP.append(map_index)
        else:
            # This one is for TP
            TP.append(map_index)

    # print(TN)
    # print(FP)
    # print(FN)
    # print(TP)

    index['TN'] = np.array([int(i) for i in TN])
    index['TP'] = np.array([int(i) for i in TP])
    index['FP'] = np.array([int(i) for i in FP])
    index['FN'] = np.array([int(i) for i in FN])

    return index


def sort_TP_TN_FP_FN_by_vid_n_frame(testdict, conf_dict ):
    """
    The goal of this function is seperate TP, TN, FP, FN indices
    by video and order each frame increasing to decreasing. Makes it
    easier to anlayze results as its more computially efficent to
    generate plots.

    testdict: this dict is in same format as Boxes function dict.
              dict contains 5 keys x_ppl_box, y_ppl_box, video_file,
              frame_ppl_id, abnormal.

              For me to use a traindict would need unsorted dict inputed

    conf_dict: contains indices that are correlated to the testdict.
                Indices are split based on TP, TN, FP, FN


    return:
        TP_TN_FP_FN: indices split up into specifc videos
        boxes_dict: testdict data parsed and split up into specific videos
                    and into confusion matrix keys
                    Right before last level key next keys contain 
                    x_ppl_box, y_ppl_box, video_file,
                    frame_ppl_id, abnormal.
    """

    TP_TN_FP_FN = {}
    boxes_dict = {}

    for conf_key in conf_dict.keys():
        if len(conf_dict[conf_key]) == 0:
            continue

        # Need to seperate by video first
        # First line is the index of a specfic confusion matrix value
        # Those index map back to testdict unsorted
        unsorted_index_by_vid = conf_dict[conf_key]
    
        sorted_index = unsorted_index_by_vid.argsort()
        sorted_index_by_vid = unsorted_index_by_vid[sorted_index]

        sorted_video_list = testdict['video_file'][sorted_index_by_vid]

        prev = sorted_video_list[0]

        TP_TN_FP_FN[conf_key] = {}
        TP_TN_FP_FN[conf_key][prev] = []
        boxes_dict[conf_key] = {}

        #inital loop the prev is alawys equal to current
        # Then this next loop seperates into videos and ordered frames
        for current, j in zip(sorted_video_list,sorted_index_by_vid ):
            if prev != current:
                TP_TN_FP_FN[conf_key][current] = []
                TP_TN_FP_FN[conf_key][current].append(j)

            else:
                TP_TN_FP_FN[conf_key][current].append(j)

            prev = current

        # This look goes back and puts elements together
        for vid_key in TP_TN_FP_FN[conf_key].keys():
            boxes_dict[conf_key][vid_key] = {}
            index_per_vid= TP_TN_FP_FN[conf_key][vid_key]

            for attr_key in testdict.keys():
                temp = testdict[attr_key][ index_per_vid ]
                boxes_dict[conf_key][vid_key][attr_key] = temp

    return TP_TN_FP_FN,boxes_dict


def cycle_through_videos(model,both, data, max1, min1,pic_loc, loc_videos, xywh=False):
    """
    NOTE: this function does not plot trajectories, it just plots
           the predicted next step from the given trajactory.
           
    The goal of the function is to allow data to be inputted as dict,
    that contains all videos and data is able to be plotted in order.
    A error might occur its probably because of data keys

    model: Note that this should be lstm model
    both: both both bitrap and lstm predicted
    data: Should contain dictornay keys to different videos. Each video
           should have same dictornay keys  'x_ppl_box' 'y_ppl_box'
               'video_file' 'abnormal'
               Note 'frame_ppl_id':  has format ( examples, seq input +seq output, (frame, ppl_id))
               
    max1:  scaling factor
    min1:  scaline factor
    loc_videos: this are videos that are used to make plots, need the folder that
                that contains all the videos
    pic_loc: need to save to different location depending on confusion
                    type and/or can input generic location to allow for
                    plotting all videos at the same sort_TP_TN_FP_FN_by_vid_n_frame
    """


    # Need to save to a different location depening on confusion type
    frame_count = 0
    for vid_key in data.keys():
#             print(vid_key)
            # Need to specifc folder location inside of loop
            vid_data = data[vid_key]
            # FOR SORTING
            frame = np.array( vid_data['frame_y'] )
            sort_index = frame.argsort()
            
            if model == 'bitrap':
                y_pred = vid_data['pred_trajs'][sort_index]
            else:
                x_scal,y_scal = norm_train_max_min(data= vid_data, max1 = max1,min1 =min1, undo_norm=False)
                x_scal,y_scal = x_scal[sort_index], y_scal[sort_index]
                y_scal_pred = model.predict(x_scal)
                y_pred = norm_train_max_min(data=y_scal_pred, max1 = max1,min1 =min1,undo_norm=True)
                if both:
                    y_pred_bitrap = vid_data['pred_trajs'][sort_index]
            ## Sorting
            # size = len(vid_data['frame_ppl_id'])
            
            
            ## Sorting

            size = len(vid_data['id_x'])
            frame_y = vid_data['frame_y'][sort_index]
            frame_x = vid_data['frame_x'][sort_index]
            id_x = vid_data['id_x'][sort_index]
            id_y  = vid_data['id_y'][sort_index]
            # frame_ppl = vid_data['frame_ppl_id'][sort_index]
            y_true = vid_data['y_ppl_box'][sort_index].reshape(-1,4) # not normailized
            
            
    
            # last_frame = frame_ppl[-1,-1, 0]
            last_frame = frame_y[-1]
            next_frame_index, j = 0, 0

            loc_vid = os.path.join(loc_videos, vid_key[:-4]+ '.avi')
            video_capture = cv2.VideoCapture(loc_vid)

            # there could be information lost here
            # Converts xywh frame to tlbr frame for plotting
            if xywh:
                y_pred[:,0] =  y_pred[:,0] - y_pred[:,2]/2
                y_pred[:,1] =  y_pred[:,1] - y_pred[:,3]/2 # Now we are at tlwh
                y_pred[:,2:] = y_pred[:,:2] + y_pred[:,2:]

                y_true[:,0] =  y_true[:,0] - y_true[:,2]/2
                y_true[:,1] =  y_true[:,1] - y_true[:,3]/2 # Now we are at tlwh
                y_true[:,2:] = y_true[:,:2] + y_true[:,2:]

                if both:
                    y_pred_bitrap[:,0] =  y_pred_bitrap[:,0] - y_pred_bitrap[:,2]/2
                    y_pred_bitrap[:,1] =  y_pred_bitrap[:,1] - y_pred_bitrap[:,3]/2 # Now we are at tlwh
                    y_pred_bitrap[:,2:] = y_pred_bitrap[:,:2] + y_pred_bitrap[:,2:]


            for i in range(0, last_frame+1):
                ret, frame = video_capture.read()
                # if i == frame_ppl[j, -1,0 ]: #finds the frames
                if i == frame_y[j]:
                    # while i == frame_ppl[j,-1,0]:
                    while i == frame_y[j]:

                        y_fr_act = y_true[j]
                        y_fr_pred = y_pred[j]
                        if both:
                            y_fr_pred_bitrap = y_pred_bitrap[j]
                        # id1 = frame_ppl[j,-1,1]
                        id_per = id_y[j]

                        gt_frame = frame.copy()     # this is the image
                        pred_frame = frame.copy()   # this is the image
                        both_frame = frame.copy()   # this is the image

                        # # Ground Truth
                        # cv2.rectangle(gt_frame, (int(y_fr_act[0]), int(y_fr_act[1])), (int(y_fr_act[2]), int(y_fr_act[3])),(0,255,0), 2)

                        # # Predicted
                        # cv2.rectangle(pred_frame, (int(y_fr_pred[0]), int(y_fr_pred[1])), (int(y_fr_pred[2]), int(y_fr_pred[3])),(0,255,255), 2)

                        # Combined frame
                        cv2.rectangle(both_frame, (int(y_fr_act[0]), int(y_fr_act[1])), (int(y_fr_act[2]), int(y_fr_act[3])),(0,255,0), 2)
                        if model == 'bitrap':
                            cv2.rectangle(both_frame, (int(y_fr_pred[0]), int(y_fr_pred[1])), (int(y_fr_pred[2]), int(y_fr_pred[3])),(255,0, 0), 2)
                        else:
                            cv2.rectangle(both_frame, (int(y_fr_pred[0]), int(y_fr_pred[1])), (int(y_fr_pred[2]), int(y_fr_pred[3])),(0,255,255), 2)

                        if both:
                            cv2.rectangle(both_frame, (int(y_fr_pred_bitrap[0]), int(y_fr_pred_bitrap[1])), (int(y_fr_pred_bitrap[2]), int(y_fr_pred_bitrap[3])),(255,0,0), 2)

                        # Need to change This
                        vid_str_info = vid_key[:-4] + '___' + str(i) + '__' + str(id_per)
                        # vid_str_info has video number, frame number, person_Id

                        # cv2.imwrite( os.path.join(pic_loc, vid_str_info + '_gt.jpg'), gt_frame)
                        # cv2.imwrite( os.path.join(pic_loc, vid_str_info + '_pred.jpg'), pred_frame)
                        cv2.imwrite( os.path.join(pic_loc, vid_str_info + '_both.jpg'), both_frame)


                        frame_count += 1
                        print('vid:{} index:{} frame: {}'.format(
                                                                vid_key,
                                                                j,
                                                                frame_y[j])
                                                                )
                                                                # frame_ppl[j,-1,0]))

                        next_frame_index += 1
                        j = next_frame_index

                        if j == size:
                            break


def helper_TP_TN_FP_FN(datadict, traj_model, ped, both, max1, min1):
    
    """
    This uses function in the TP_TN_FP_FN file for plotting
    datadict: 
    traj_model: lstm, etc
    ped: dict with x is two columns contains predictions, indices
         y contains the ground truth information 
    both: plot bitrap and lstm model on top of each other
    """
  

    # seperates them into TP. TN, FP, FN

    # Note that y_pred should not be threshold yet, granted if it is no
    # error cuz would change by threshold again assuming using same threshold 
    conf_dict = seperate_misclassifed_examples( y_pred = ped['x'],
                                                indices = ped['index'],
                                                test_y = ped['y'],
                                                threshold=0.5
                                                )

    
    print('length of  TP {} '.format(len(conf_dict['TP'])))
    print('length of  TN {} '.format(len(conf_dict['TN'])))
    print('length of  FP {} '.format(len(conf_dict['FP'])))
    print('length of  FN {} '.format(len(conf_dict['FN'])))
    # quit()
    
    # what am I actually returning
    TP_TN_FP_FN, boxes_dict = sort_TP_TN_FP_FN_by_vid_n_frame(datadict, conf_dict )


    # Does not return result, but saves images to folders
    make_dir(loc['visual_trajectory_list'])
    pic_loc = join(     os.path.dirname(os.getcwd()),
                        *loc['visual_trajectory_list']
                        )

    # need to make last one robust "test_vid" : "train_vid"
    # can change

    loc_videos = loc['data_load'][exp['data']]['test_vid']
    # print(boxes_dict.keys())
    # quit()
    for conf_key in boxes_dict.keys():
        temp = loc['visual_trajectory_list'].copy()
        temp.append(conf_key)
        make_dir(temp)

    for conf_key in boxes_dict.keys():
        pic_loc_conf_key =  join(pic_loc, conf_key)
        cycle_through_videos(traj_model, both, boxes_dict[conf_key], max1, min1, pic_loc_conf_key, loc_videos, xywh=True)
