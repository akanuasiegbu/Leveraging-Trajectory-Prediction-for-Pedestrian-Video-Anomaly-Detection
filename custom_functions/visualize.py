# Should Maybe delete this one as not using it much
from load_data import norm_train_max_min
import numpy as np
import cv2

def visual_ouput(model=None,max1 = None,min1=None, vid=None,pic_loc =None, output_dict=None,xywh = False):
    """
    Right now this only works for one video at a time

    model : lstm model
    max1 : scaling factor
    """

    xx,yy = norm_train_max_min(data_dict = output_dict, max1 = max1,min1 =min1)
    size = len(output_dict['frame_ppl_id'])


    ###########################################################################
    #This part here can probably basically be a function that's called open
    # This function is easier because its only one index so if you sort
    # index your good

    # For a more general case I need sort by video_file and then by
    # frame
    #sort index by frames
    frame = []
    for i in range(0,size):
        #sort index by frames
        # I go to last element because I Combined
        # the x and y frame and person id into one matrix
        frame.append(output_dict['frame_ppl_id'][i,-1,0])

    frame = np.array(frame)
    sort_index = frame.argsort()
    #Sorted from first to last frame
    xx_scal, yy_scal = xx[sort_index], yy[sort_index]
    vid_file = output_dict['video_file'][sort_index] # not really needed if using one video
    frame_ppl = output_dict['frame_ppl_id'][sort_index]
    y_true = output_dict['y_ppl_box'][sort_index] # not normailized
    ###########################################################################

    # Note that predicted outout is already sorted
    y_pred_scal = model.predict(xx_scal)
    y_pred = norm_train_max_min(data=y_pred_scal, max1 = max1,min1 =min1,undo_norm=True)


    last_frame = frame[sort_index][-1]
    next_frame_index , j = 0 , 0

    #  Start Video
    ### Should change this so that location of video is variable
    loc_videos ="/home/akanu/Dataset/Anomaly/Avenue_Dataset/testing_videos/{:02}.avi".format(vid)
    video_capture = cv2.VideoCapture(loc_videos)

    if xywh:
        y_pred[:,0] = y_pred[:,0] - y_pred[:,2]/2
        y_pred[:,1] = y_pred[:,1] - y_pred[:,3]/2 # Now we are at tlwh
        y_pred[:,2:] = y_pred[:,:2] + y_pred[:,2:]

        y_true[:,0] = y_true[:,0] - y_true[:,2]/2
        y_true[:,1] = y_true[:,1] - y_true[:,3]/2 # Now we are at tlwh
        y_true[:,2:] = y_true[:,:2] + y_true[:,2:]

    for i in range(0,last_frame+1):
        ret, frame = video_capture.read()
        if i == frame_ppl[j,-1,0]: #finds the frame
            while i == frame_ppl[j,-1,0]:
                y_fr_act = y_true[j]
                y_fr_pred = y_pred[j]
                id1 = frame_ppl[j,-1,1]
#                 print("{}  ,{}".format(y_fr_act.shape, y_fr_pred.shape))

                gt_frame = frame.copy()
                pred_frame = frame.copy()
                both_frame = frame.copy()


                # Ground Truth
                cv2.rectangle(gt_frame, (int(y_fr_act[0]), int(y_fr_act[1])), (int(y_fr_act[2]), int(y_fr_act[3])),(0,255,0), 2)
#                 cv2.putText(gt_frame, str(frame_ppl[j,-1,1]),(int(y_fr_act[0]), int(y_fr_act[1])),0, 5e-3 * 200, (255,255,0),2)

                # Predicted
                cv2.rectangle(pred_frame, (int(y_fr_pred[0]), int(y_fr_pred[1])), (int(y_fr_pred[2]), int(y_fr_pred[3])),(0,255,255), 2)
#                 cv2.putText(pred_frame, str(frame_ppl[j,-1,1]),(int(y_fr_pred[2]), int(y_fr_pred[3])),0, 5e-3 * 200, (255,255,0),2)

                # Combined frame
                cv2.rectangle(both_frame, (int(y_fr_act[0]), int(y_fr_act[1])), (int(y_fr_act[2]), int(y_fr_act[3])),(0,255,0), 2)
#                 cv2.putText(both_frame, str(frame_ppl[j,-1,1]),(int(y_fr_act[0]), int(y_fr_act[1])),0, 5e-3 * 200, (255,255,0),2)
                cv2.rectangle(both_frame, (int(y_fr_pred[0]), int(y_fr_pred[1])), (int(y_fr_pred[2]), int(y_fr_pred[3])),(0,255,255), 2)
#                 cv2.putText(both_frame, str(frame_ppl[j,-1,1]),(int(y_fr_pred[2]), int(y_fr_pred[3])),0, 5e-3 * 200, (255,255,0),2)


                cv2.imwrite(pic_loc.format(vid,i,id1) + '_gt.jpg', gt_frame)
                cv2.imwrite(pic_loc.format(vid,i,id1) + '_pred.jpg', pred_frame)
                cv2.imwrite(pic_loc.format(vid,i,id1) + '_both.jpg', both_frame)




                next_frame_index +=1
                j = next_frame_index
#                 print('saved')
#                 print(j)
                if j == size:
                    return
