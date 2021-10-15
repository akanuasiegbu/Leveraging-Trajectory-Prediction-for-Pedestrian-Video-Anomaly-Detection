import numpy as np
import os
import cv2


def ind_seq(data, video, frame):
    """
    Takes the data and find the video and frame so that 
    sequence can be inputted and creates smaller dict of one 
    sequence

    data: Format is keys are different videos
    video: video of interest. useful if in TN, TP, FN, FP. Ny that
            I mean that only inputed then the next immediate key is
            video numbers. For generic data this wouldn't work.
            As I would need to find the videos. Probably go back and 
            delete this funciton later on
    return output: same format as x_ppl_box,y_ppl_box etx
    """

    output ={}
    data = data['{}.txt'.format(video)]
    frames = data['frame_ppl_id'][:,-1,0]
    found_index_of_frame = np.where(frames==frame)

    
    for key in data.keys():
        output[key] = data[key][found_index_of_frame]

    
    return output

def ind_seq_dict(data, video, frame, idx):
    """
    This can sort dict directly. Note this is not compute
    """
    vid_loc = data['video_file'].reshape(-1,1) #videos locations
    frame_loc = data['frame_y'].reshape(-1,1) # frame locations
    id_y = data['id_y'].reshape(-1,1)
    vid_frame_id_y = []
    for i,j,k in zip(vid_loc, frame_loc, id_y):
        vid_frame_id_y.append( str(i[0]) + '_' + str(j[0])  + '_' + str(k[0]))
    
    # found1_index = np.where(vid_frame_id_y == '{}.txt'.format(video) + '_' + '{}'.format(frame) + '_' + '{}'.format(id))
    
    i = 0
    find = '{}.txt'.format(video) + '_' + '{}'.format(frame) + '_' + '{}'.format(idx)
    for j in vid_frame_id_y:
        if j == find:
            found_index = i 
        else:
            i += 1
    output = {}
    for key in data.keys():
        output[key] = data[key][found_index]

    return output


    

# def plot_sequence(model, one_ped_seq, max1, min1, vid_key,pic_loc, loc_videos, xywh=False):
def plot_sequence(one_ped_seq, max1, min1, vid_key,pic_loc, loc_videos, xywh=False):
    """
    This will plot the sequences of the of one pedestrain
    Not computially efficent if you want to plot lots of pedestrain
    model: lstm bm_model
    one_ped_seq: one pedestrain sequence: 'x_ppl_box', 'y_ppl_box',
    'frame_ppl_id', 'video_file', 'abnormal'. From ind_seq function
    max1:  scaling factor
    min1:  scaline factor
    loc_videos: this are videos that are used to make plots
    pic_loc: need to save to different location depending on confusion
                    type and/or can input generic location to allow for
                    plotting all videos at the same sort_TP_TN_FP_FN_by_vid_n_frame
    """
    data = one_ped_seq
    x_input = data['x_ppl_box']
    # x_scal,y_scal = norm_train_max_min(data_dict = data, max1 = max1,min1 =min1)
    # last_frame = data['frame_ppl_id'][-1,-2,0]
    last_frame = data['frame_y']
    # last_frame = data['frame_x'][-1]

    

    # next_frame_index, j, frame_count = 0, 0, 0

    # Need to make sure I have video path
    next_frame_index, j = 0, 0
    #NEED TO UNCOMMENT
    loc_vid = loc_videos  # COMMENT THIS FOR GENERAL APPROACH
    # loc_vid = os.path.join(loc_videos, vid_key[:-4]+ '.avi')
    video_capture = cv2.VideoCapture(loc_vid)

    # there could be information lost here
    # print('X input while in xywh')
    # print(x_input)

    if xywh:
        x_input[:,0]  =  x_input[:,0]  -  x_input[:,2]/2
        x_input[:,1]  =  x_input[:,1]  -  x_input[:,3]/2 # Now we are at tlwh
        x_input[:,2:] =  x_input[:,:2] +  x_input[:,2:]


    x_input = x_input.squeeze()
    # print('X_input after converted into tlbr')
    # print(x_input)

    frame_ppl = data['frame_ppl_id'].squeeze()

    # for i in range(0, last_frame+1):
    for i in range(0, last_frame):
        # goes through each frame of the video
        ret, frame = video_capture.read()
        
        if i == frame_ppl[j,0 ]: #finds the frames
            while i == frame_ppl[j,0]:

                x_box = x_input[j]
                print(x_box)
                id1 = frame_ppl[j,1]

                # input_frame = frame.copy()
                gt_frame = frame.copy()

                # Since camera is statiornay I can plot other bbox as well on same video
                # Input Data


                # NEED TO UNCOMMENT
                # cv2.rectangle(gt_frame, (int(x_box[0]), int(x_box[1])), (int(x_box[2]), int(x_box[3])),(0,255,0), 2)
                # cv2.rectangle(gt_frame, (int(x_box[0]), int(x_box[1])), (int(x_box[2]), int(x_box[3])),(0,255,255), 2) # yellow
                
                # use for input data
                cv2.rectangle(gt_frame, (int(x_box[0]), int(x_box[1])), (int(x_box[2]), int(x_box[3])),(255,255,255), 2) # white

                
                # Need to change This
                vid_str_info = vid_key[:-4] + '___' + str(i) + '__' + str(id1)
                cv2.imwrite( os.path.join(pic_loc, vid_str_info + '_input.jpg'), gt_frame)
                
                # frame_count += 1
                next_frame_index += 1
                j = next_frame_index
                
                if j == last_frame:
                    break





def plot_frame(gt_boxes, pred_boxes, vid_key,pic_loc, loc_videos, last_frame):

    """
    pred_boxes and gt_boxes
    """
    

    

    # Need to make sure I have video path

    next_frame_index, j = 0, 0
    #NEED TO UNCOMMENT
    loc_vid = loc_videos  # COMMENT THIS FOR GENERAL APPROACH
    # loc_vid = os.path.join(loc_videos, vid_key[:-4]+ '.avi')
    video_capture = cv2.VideoCapture(loc_vid)



    # for i in range(0, last_frame+1):
    for i in range(0, last_frame+1):
        # goes through each frame of the video
        ret, frame = video_capture.read()
        
        if i == last_frame: #finds the frames


                # input_frame = frame.copy()
                gt_frame = frame.copy()

                for gt_bbox, pred_box in zip(gt_boxes, pred_boxes):
                    cv2.rectangle(gt_frame, (int(gt_bbox[0]), int(gt_bbox[1])), (int(gt_bbox[2]), int(gt_bbox[3])),(0,255,0), 2)

                    cv2.rectangle(gt_frame, (int(pred_box[0]), int(pred_box[1])), (int(pred_box[2]), int(pred_box[3])),(0,255,255), 2) # yellow
                    cv2.putText(gt_frame, str(pred_box[-2]),(int(pred_box[0]), int(pred_box[1] -20)),0, 5e-3 * 200, (0,255,0),2)
                    cv2.putText(gt_frame, str(pred_box[-1]),(int(pred_box[2] - 25), int(pred_box[3] + 30)),0, 5e-3 * 200, (0,255,255),2)



                
                # use for input data
                # cv2.rectangle(gt_frame, (int(x_box[0]), int(x_box[1])), (int(x_box[2]), int(x_box[3])),(255,255,255), 2) # white

                
                # Need to change This
                vid_str_info = vid_key + '___{}'.format(i)
                cv2.imwrite( os.path.join(pic_loc, vid_str_info + '_input.jpg'), gt_frame)
                
                # frame_count += 1
                next_frame_index += 1
                j = next_frame_index
                
                if j == last_frame:
                    break

                