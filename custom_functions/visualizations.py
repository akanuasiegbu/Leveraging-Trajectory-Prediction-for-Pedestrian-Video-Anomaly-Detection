from os import makedirs
import cv2
import os
import numpy as np
from os.path import join
from matplotlib import pyplot as plt
from experiments_code.config import hyparams, loc, exp
from custom_functions.utils import make_dir

def generate_images_with_bbox(testdicts,out_frame, visual_path):
    # testdict:
    # outframe:
    # visual path: 
    # Function does not return anything
    visual_plot_loc = join( os.path.dirname(os.getcwd()), *visual_path )
    if exp['data'] =='avenue':
        for i in range(1,22):
            path = visual_path.copy()
            path.append('{:02d}'.format(i))
            make_dir(path)
            
            if not hyparams['errortype']=='error_flattened':

                path_timeseries = visual_path.copy()
                path_timeseries.append('{:02d}_time_series'.format(i))
                make_dir(path_timeseries)


    elif exp['data']=='st':
        for txt in np.unique(testdicts[0]['video_file']):
            path = visual_path.copy()
            path.append('{}'.format(txt[:-4]))
            make_dir(path)

            if not hyparams['errortype']=='error_flattened':
                path_timeseries = visual_path.copy()
                path_timeseries.append('{}_time_series'.format(txt[:-4]))
                make_dir(path_timeseries)
            
    # This plots the data for visualizations
    pic_locs = loc['data_load'][exp['data']]['pic_loc_test']
    plot_vid( out_frame, pic_locs, visual_plot_loc, exp['data'] )
    


def plot_frame_from_image(pic_loc, bbox_preds, save_to_loc, vid, frame, idy, prob,abnormal_frame, abnormal_ped, gt_bboxs = None):
    """
    pic_loc: this is where the orginal pic is saved
    bbox_pred: this is where the bbox is saved
    """
    img = cv2.imread(pic_loc)
    for bbox_pred in bbox_preds:
        cv2.rectangle(img, (int(bbox_pred[0]), int(bbox_pred[1])), (int(bbox_pred[2]), int(bbox_pred[3])),(0,255,255), 2)
    
    for gt_bbox in gt_bboxs:
        # Doing this way also takes care of multiple bounding boxes
        cv2.rectangle(img, (int(gt_bbox[0]), int(gt_bbox[1])), (int(gt_bbox[2]), int(gt_bbox[3])),(0,255,0), 2)


    # cv2.putText(img, '{:.4f}'.format(prob),(25,65),0, 5e-3 * 100, (255,255,0),2)
    
    # # This is for abnormal frame, this uses the last bbox of set to plot 
    # if abnormal_frame:
    #     cv2.putText(img, 'Abnormal Frame',(25,25),0, 5e-3 * 150, (0,0,255),2)
    # else:
    #     cv2.putText(img, 'Normal Frame',(25, 25),0, 5e-3 * 150, (0,255, 0 ),2)
    
    # # This is for abnormal person
    # if abnormal_ped:
    #     cv2.putText(img, '1',(25,45),0, 5e-3 * 100, (0,0,255),2)

    # else:
    #     cv2.putText(img, '0',(25,45),0, 5e-3 * 100, (255,0, 0),2)



    # if abnormal_frame and abnormal_ped:
    
    cv2.imwrite(save_to_loc + '/' + '{}'.format(vid) + '/' + '{}__{}_{}_{:.4f}.jpg'.format(vid, frame, idy, prob), img)


def plot_error_in_time(prob_with_time, time_series_plot_loc, vid, frame, idy):
    fig,ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(prob_with_time, '-*', label='error summed')
    ax.plot(np.diff(prob_with_time), '-o', label='error diff')
    ax.set_xlabel('Time')
    ax.set_ylabel('Error Summed')
    ax.set_title('video:{} frame:{} idy:{}'.format(vid, frame, idy ))
    ax.legend()

    img_path = join(    time_series_plot_loc,  '{}_time_series'.format(vid),
                        '{}_{}_{}.jpg'.format(vid,frame,idy))

    fig.savefig(img_path)
    plt.close(fig)


def plot_vid(out_frame, pic_locs, visual_plot_loc, data):

    """
    out_frame: this is the dict
    pic_loc: pic that will be plottd
    visual_plot_loc: this is where the video will be saved at
    """
    for bbox_preds, vid, frame, idy, prob, prob_with_time, abnormal_frame, abnormal_ped, gt_bbox in zip(    out_frame['pred_bbox'], out_frame['vid'], 
                                                                                                            out_frame['frame'], out_frame['id_y'],
                                                                                                            out_frame['prob'],
                                                                                                            out_frame['prob_with_time'],
                                                                                                            out_frame['abnormal_gt_frame_metric'],
                                                                                                            out_frame['abnormal_ped_pred'],
                                                                                                            out_frame['gt_bbox'] ):
        if data =='avenue':
            pic_loc = join(  pic_locs, '{:02d}'.format(int(vid[0][:-4])) )
            pic_loc =  pic_loc + '/' +'{:02d}.jpg'.format(int(frame))
        elif data =='st':
            pic_loc = join(  pic_locs, '{}'.format(vid[0][:-4]) )
            pic_loc =  pic_loc + '/' +'{:03d}.jpg'.format(int(frame))

        plot_frame_from_image(  pic_loc = pic_loc,  
                                bbox_preds = bbox_preds ,
                                save_to_loc = visual_plot_loc, 
                                vid = vid[0][:-4],
                                frame = int(frame[0]), 
                                idy = int(idy[0]),
                                prob = prob[0],
                                abnormal_frame = abnormal_frame,
                                abnormal_ped = abnormal_ped,
                                gt_bboxs = gt_bbox)
        if not hyparams['errortype']=='error_flattened':
            plot_error_in_time(prob_with_time, visual_plot_loc, vid[0][:-4], int(frame[0]), int(idy[0]))