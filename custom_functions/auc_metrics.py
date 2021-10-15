# This is in develpment rn
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import os, sys, time
from os.path import join


from experiments_code.config import hyparams, loc, exp

from custom_metrics import bb_intersection_over_union, bb_intersection_over_union_np
from coordinate_change import xywh_tlbr, tlbr_xywh
from TP_TN_FP_FN import *
from load_data import norm_train_max_min, load_pkl
from load_data_binary import compute_iou
from custom_functions.iou_utils import giou, diou, ciou



def anomaly_metric(prob, avg_or_max, pred_trajs, gt_box, vid_loc, frame_loc, person_id, abnormal_gt, abnormal_person, prob_in_time,
                    prob_l2=None, prob_iou=None, prob_giou=None, prob_ciou=None, prob_diou=None):
    """
    This functon helps calculate abnormality by averaging or taking the max of pedestrains
    that belong to the same frame and video.

    This is allows for aggravating of pedestrains together     
    """
    
    vid_frame = np.append(vid_loc, frame_loc, axis=1)
    vid_frame_person = np.append(vid_frame, person_id, axis =1)

    unique, unique_inverse, unique_counts = np.unique(vid_frame_person, axis=0, return_inverse=True, return_counts=True)

    repeat_inverse_id = np.where(unique_counts >= 1)[0] 
    # makes sense cuz sometimes might be one pedestrain if the start off at  the front of seq

    calc_prob, frame, vid, id_y, gt_abnormal, std = [], [], [], [], [],[]
    std_iou_or_l2, bbox_list, abnormal_ped, gt = [], [], [], []
    prob_l2_list, prob_iou_list, prob_giou_list, prob_ciou_list, prob_diou_list  = [], [], [], [], []
    prob_with_time = []
    for i in repeat_inverse_id:
        same_vid_frame_person = np.where(unique_inverse == i)[0]
        temp_prob = prob[same_vid_frame_person]
        temp_pred_trajs = pred_trajs[same_vid_frame_person]
        gt_box_temp = gt_box[same_vid_frame_person]

        if avg_or_max == 'avg':
            calc_prob.append(np.mean(prob[same_vid_frame_person]))
            bbox = temp_pred_trajs
            # bbox = np.mean(temp_pred_trajs, axis = 0) # For averageing
            if exp['plot_images']:
                if hyparams['errortype']=='error_flattened':
                    prob_with_time.append(-1)
                else:    
                    prob_with_time.append(prob_in_time[same_vid_frame_person])
                prob_l2_list.append(prob_l2[same_vid_frame_person])
                prob_iou_list.append(prob_iou[same_vid_frame_person])
                prob_giou_list.append(prob_giou[same_vid_frame_person])
                prob_ciou_list.append(prob_ciou[same_vid_frame_person])
                prob_diou_list.append(prob_diou[same_vid_frame_person])

                
        if avg_or_max == 'max':
            calc_prob.append(np.max(prob[same_vid_frame_person]))
            max_loc = np.where( temp_prob == np.max(temp_prob))[0]

            if len(max_loc) > 1:
                bbox = temp_pred_trajs[max_loc[0]]

            else:
                bbox = temp_pred_trajs[max_loc]
               


        frame.append(vid_frame_person[same_vid_frame_person[0]][1])
        vid.append(vid_frame_person[same_vid_frame_person[0]][0])
        id_y.append(vid_frame_person[same_vid_frame_person[0]][2])
        gt_abnormal.append(abnormal_gt[same_vid_frame_person[0]])
        # print('std axis = 0 probably doesnt make sense for the vertical direction so might need to change')
        std.append( np.std(pred_trajs[same_vid_frame_person].reshape(-1,4), axis=0) ) # Might need to change this for the vertical direction
        # if hyparams['metric'] == 'iou':
        std_iou_or_l2.append(np.std(prob[same_vid_frame_person]))
        abnormal_ped.append(abnormal_person[same_vid_frame_person][0]) #same person
        bbox_list.append(np.array(bbox).reshape(-1,4))
        gt.append(np.array(gt_box_temp).reshape(-1,4))

        # print('I dont think the gt bbox shape is correct')
        # quit()


    out = {}
    out['prob'] = np.array(calc_prob).reshape(-1,1)
    out['frame'] = np.array(frame, dtype=int).reshape(-1,1)
    out['vid'] = np.array(vid).reshape(-1,1)
    out['id_y'] = np.array(id_y, dtype =int).reshape(-1,1)
    out['abnormal_gt_frame_metric'] = np.array(gt_abnormal).reshape(-1,1)
    out['std'] = np.array(std).reshape(-1,4)
    out['std_iou_or_l2'] = np.array(std_iou_or_l2).reshape(-1,1)
    out['pred_bbox'] = xywh_tlbr(np.array(bbox_list, dtype=object)) #  Note that this can be diff shapes in diff index
    out['abnormal_ped_pred'] = np.array(abnormal_ped).reshape(-1,1)
    out['gt_bbox'] = xywh_tlbr(np.array(gt))
    
    if exp['plot_images']:
        out['prob_with_time'] = np.squeeze( np.array(prob_with_time) )
        out['prob_l2'] = np.array(prob_l2_list)
        out['prob_iou'] = np.array(prob_iou_list)
        out['prob_giou'] = np.array(prob_giou_list)
        out['prob_ciou'] = np.array(prob_ciou_list)
        out['prob_diou'] = np.array(prob_diou_list)
        
    return out


def iou_as_probability(testdicts, models, errortype, max1 =None, min1 =None):
    """
    Note that to make abnormal definition similar to orgininal definition
    Need to switch ious because low iou indicates abnormal and high 
    iou indicate normal
    testdict: 
    model: traj prediction model
    iou_prob: probability where high indicates abnormal pedestrain and low
              indicates normal pedestrain
    """ 
    # model here is lstm model
    # need to normalize because lstm expects normalized
    if models =='bitrap':
        ious = []
        for testdict in testdicts:
            y = testdict['y_ppl_box'] # this is the gt 
            # gt_bb_unorm_tlbr = xywh_tlbr(np.squeeze(y))
            gt_bb_unorm_tlbr = xywh_tlbr(y)
            predicted_bb_unorm_tlbr = xywh_tlbr(testdict['pred_trajs'])
            iou = bb_intersection_over_union_np(    predicted_bb_unorm_tlbr,
                                                    gt_bb_unorm_tlbr )
            # need to squeeze to index correctly 
            ious.append(np.squeeze(iou))
        
        iou  = np.concatenate(ious)

    else:
        ious = []
        for testdict, model in zip(testdicts, models):
        # Need to fix this for lstm network
            x,y = norm_train_max_min(   testdict,
                                        # max1 = hyparams['max'],
                                        # min1 = hyparams['min']
                                        max1 = max1,
                                        min1 = min1
                                        )

            ious.append(compute_iou(x, y, max1, min1,  model))
            iou = np.concatenate(ious)

    iou_prob =  1 - iou


    if errortype == 'error_diff':
        output = np.sum(np.diff(iou_prob, axis =1), axis=1)
    elif errortype == 'error_summed':
        output = np.sum(iou_prob, axis=1)
    elif errortype == 'error_flattened':
        output = iou_prob.reshape(-1,1)
    else:
        pass

    return  output, iou_prob



def l2_error(testdicts, models, errortype, max1 = None, min1 =None ):     
    """
    testdicts: allows for multitime combinations
    models: allows for multitime combinations of lstm
    errortype: 'error_diff', 'error_summed', 'error_flattened'
    max1: hyperparameter
    min1: hyperparameter
    """
    preds = []
    if models == 'bitrap':
        for testdict in testdicts:
            preds.append(xywh_tlbr(testdict['pred_trajs']))

        
    else:
        for testdict, model in zip(testdicts, models):
            x,y = norm_train_max_min( testdict, max1 = max1, min1 = min1)
            shape =  x.shape
            pred = model.predict(x)

            preds.append(xywh_tlbr(pred.reshape(shape[0] ,-1,4)))

    
    trues = []
    for testdict in testdicts:
        trues.append(xywh_tlbr(testdict['y_ppl_box']))
    
    summed_list, output_in_time = [], []
    for pred, true in zip(preds, trues):
        diff = (true-pred)**2
        summed = np.sum(diff, axis =2)
        summed = np.sqrt(summed)
        

        if errortype == 'error_diff':
            error = np.sum(np.diff(summed, axis =1), axis=1)
        elif errortype == 'error_summed':
            error = np.sum(summed, axis=1)
        elif errortype == 'error_flattened':
            error = summed.reshape(-1,1)
        else:
            pass

        summed_list.append(error)
        output_in_time.append(summed)
   
        

    
    output_in_time = np.concatenate(output_in_time)
    l2_error  = np.concatenate(summed_list)

    return l2_error.reshape(-1,1), output_in_time


def giou_as_metric(testdicts, models, errortype, max1 = None, min1 =None):
    # giou expects tlbr
    # Fliped 
    if models =='bitrap':
        gious = []
        for testdict in testdicts:
            gt_bb = xywh_tlbr(testdict['y_ppl_box'])
            predicted_bb = xywh_tlbr(testdict['pred_trajs'])
            gious_temp = giou( gt_bb, predicted_bb)

            # need to squeeze to index correctly 
            gious.append(np.squeeze( 1- gious_temp)) #######################
        
        gious  = np.concatenate(gious)


    # Need to add the LSTM part here

    if errortype == 'error_diff':
        output = np.sum(np.diff(gious, axis =1), axis=1)
    elif errortype == 'error_summed':
        output = np.sum(gious, axis=1)
    elif errortype == 'error_flattened':
        output = gious.reshape(-1,1)

    
    return output

def ciou_as_metric(testdicts, models, errortype, max1 = None, min1 =None):
    # flipped
    if models =='bitrap':
        cious = []
    for testdict in testdicts:
        gt_bb = testdict['y_ppl_box']
        predicted_bb = testdict['pred_trajs']
        cious_temp = ciou( gt_bb, predicted_bb)

        # need to squeeze to index correctly 
        cious.append(np.squeeze(1-cious_temp)) ###########################3

    cious  = np.concatenate(cious)


    # Need to add the LSTM part here

    if errortype == 'error_diff':
        output = np.sum(np.diff(cious, axis =1), axis=1)
    elif errortype == 'error_summed':
        output = np.sum(cious, axis=1)
    elif errortype == 'error_flattened':
        output = cious.reshape(-1,1)

    
    return output



def diou_as_metric(testdicts, models, errortype, max1 = None, min1 =None):
    # Needs it in xywh form
    # Subtracred one to turn them into probablity of sort 
    if models =='bitrap':
        dious = []
    for testdict in testdicts:
        gt_bb = testdict['y_ppl_box']
        predicted_bb = testdict['pred_trajs']
        dious_temp = diou( gt_bb, predicted_bb)

        # need to squeeze to index correctly 
        dious.append(np.squeeze(1-dious_temp))

    dious  = np.concatenate(dious)


    # Need to add the LSTM part here

    if errortype == 'error_diff':
        output = np.sum(np.diff(dious, axis =1), axis=1)
    elif errortype == 'error_summed':
        output = np.sum(dious, axis=1)
    elif errortype == 'error_flattened':
        output = dious.reshape(-1,1)

    
    return output


def giou_ciou_diou_as_metric(testdicts, models, metric, errortype, max1=None, min1=None):
    
    gious_dious_cious = []
    if models == 'bitrap':
        for testdict in testdicts:
            gt_bb = testdict['y_ppl_box']
            predicted_bb = testdict['pred_trajs']
            if metric == 'giou':
                gt_bb = xywh_tlbr(testdict['y_ppl_box'])
                predicted_bb = xywh_tlbr(testdict['pred_trajs'])
                temp = giou( gt_bb, predicted_bb)
            elif metric == 'ciou':
                temp = ciou( gt_bb, predicted_bb)
            elif metric == 'diou':
                temp = diou( gt_bb, predicted_bb)
            # need to squeeze to index correctly 
            gious_dious_cious.append(np.squeeze(1-temp))

        output_in_time  = np.concatenate(gious_dious_cious)

    else:
        for testdict, model in zip(testdicts, models):
        # Need to fix this for lstm network
        ############ Might be good to replace this
            x,y = norm_train_max_min(   testdict,
                                        # max1 = hyparams['max'],
                                        # min1 = hyparams['min']
                                        max1 = max1,
                                        min1 = min1
                                        )

            shape =  x.shape
            predicted_bb = model.predict(x)
    
            predicted_bb_unorm = norm_train_max_min(predicted_bb, max1, min1, True)
            predicted_bb_unorm_xywh = predicted_bb_unorm.reshape(shape[0] ,-1,4)
            ############################33

            gt_bb_unorm_xywh = norm_train_max_min(y, max1, min1, True)

            if metric == 'giou':
                gt_bb_unorm_tlbr = xywh_tlbr(gt_bb_unorm_xywh)
                predicted_bb_unorm_tlbr = xywh_tlbr(predicted_bb_unorm_xywh)
                temp = giou( gt_bb_unorm_tlbr, predicted_bb_unorm_tlbr)   
            elif metric == 'ciou':
                temp = ciou( gt_bb_unorm_xywh, predicted_bb_unorm_xywh)
            elif metric == 'diou':
                temp = diou( gt_bb_unorm_xywh, predicted_bb_unorm_xywh)

            gious_dious_cious.append(np.squeeze(1 - temp))
        
        output_in_time = np.concatenate(gious_dious_cious)

    if errortype == 'error_diff':
        output = np.sum(np.diff(output_in_time, axis =1), axis=1)
    elif errortype == 'error_summed':
        output = np.sum(output_in_time, axis=1)
    elif errortype == 'error_flattened':
        output = output_in_time.reshape(-1,1)
    else:
        pass 
    
    return output, output_in_time




def combine_time(testdicts, models, errortype, modeltype, max1 =None, min1 = None):
    """
    This function is selecting the first frame to be the frame of interest in multi output 
    when using pedestrain motion in time to determine abnormal events.

    Function also flattens the person and tells allows me to combine pedestrains at the same frame 
    from different trajactories. 
    """
    
    vid_loc, person_id, frame_loc, abnormal_gt_frame, abnormal_event_person, gt_bbox, pred_trajs = [],[],[],[],[], [],[]

    for testdict in testdicts:
        if errortype =='error_diff' or errortype == 'error_summed':
            person_id.append(testdict['id_y'][:,0] )
            frame_loc.append(testdict['frame_y'][:,0] )# frame locations
            abnormal_gt_frame.append(testdict['abnormal_gt_frame'][:,0])
            abnormal_event_person.append(testdict['abnormal_ped_pred'][:,0])
            gt_bbox.append(testdict['y_ppl_box'])
            print('Note that this does not work fully yet for testdicts in terms of gt_bboxes')
            vid_loc = testdict['video_file'].reshape(-1,1) #videos locations


        else:
            person_id.append(testdict['id_y'].reshape(-1,1))
            frame_loc.append(testdict['frame_y'].reshape(-1,1) )# frame locations
            abnormal_gt_frame.append(testdict['abnormal_gt_frame'].reshape(-1,1))
            abnormal_event_person.append(testdict['abnormal_ped_pred'].reshape(-1,1))
            gt_bbox.append(testdict['y_ppl_box'].reshape(-1,4))
            temp_vid_loc = testdict['video_file'].reshape(-1,1) #videos locations

        
            for vid in temp_vid_loc:
                vid_loc.append(np.repeat(vid,testdict['y_ppl_box'].shape[1]))


        if modeltype =='bitrap':
            if errortype =='error_diff' or errortype == 'error_summed':
                pred_trajs.append(testdict['pred_trajs'] )
            elif errortype == 'error_flattened':
                pred_trajs.append(testdict['pred_trajs'].reshape(-1,4) )
        else:
            for testdict, model in zip(testdicts, models):
                # Might be good to replace this
                x,_ = norm_train_max_min( testdict, max1 = max1, min1 = min1)
                shape =  x.shape
                predicted_bb = model.predict(x)
                predicted_bb_unorm_xywh = norm_train_max_min(predicted_bb, max1, min1, True)
                predicted_bb_unorm_xywh = predicted_bb_unorm_xywh.reshape(shape[0] ,-1,4)
                ########################

                if errortype =='error_diff' or errortype == 'error_summed':
                    pred_trajs.append(predicted_bb_unorm_xywh)

                    # print('check if the size of pred_tras is correct')
                    # quit()

                elif errortype =='error_flattened':
                    pred_trajs.append(predicted_bb_unorm_xywh.reshape(-1,4) )# This is in xywh
            


    # Initilzation 
    pkldict = {}
    pkldict['id_y'] = np.concatenate(person_id).reshape(-1,1)
    pkldict['frame_y'] = np.concatenate(frame_loc).reshape(-1,1)
    pkldict['abnormal_gt_frame'] = np.concatenate(abnormal_gt_frame).reshape(-1,1)
    pkldict['abnormal_ped_pred'] = np.concatenate(abnormal_event_person).reshape(-1,1)
    pkldict['vid_loc'] = np.concatenate(vid_loc).reshape(-1,1)
    pkldict['gt_bbox'] = np.concatenate(gt_bbox)
    pkldict['pred_trajs'] = np.concatenate(pred_trajs)


    return pkldict

if __name__ == '__main__':

    pkldicts = []
    pkldicts.append(load_pkl(loc['pkl_file']['avenue_template'].format(20,10)))
    # pkldicts.append(load_pkl(loc['pkl_file']['avenue_template'].format(5,5)))
    
    # pkldict = load_pkl(loc['pkl_file']['avenue_template'].format())
    

    error = combine_time(pkldicts, errortype='error_summed', models = 'bitrap')

    # error = l2_error(pkldicts, 'bitrap' , 'error_flattened')

