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



if __name__ == '__main__':

    pkldicts = []
    pkldicts.append(load_pkl(loc['pkl_file']['avenue_template'].format(20,10)))

