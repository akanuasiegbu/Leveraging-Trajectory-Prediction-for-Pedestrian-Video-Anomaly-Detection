import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
import os, sys, time
from os.path import join
from custom_functions.utils import make_dir, SaveTextFile, write_to_txt, SaveAucTxt, SaveAucTxtTogether

# Is hyperparameters and saving files config file
from config import hyparams, loc, exp

from custom_metrics import bb_intersection_over_union, bb_intersection_over_union_np
from coordinate_change import xywh_tlbr, tlbr_xywh
from TP_TN_FP_FN import *

# Data Info
from data import data_lstm, tensorify, data_binary
from load_data import norm_train_max_min, load_pkl
from load_data_binary import compute_iou
# Plots
from metrics_plot import *
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import more_itertools as mit

# Models
from models import lstm_network, lstm_train
import wandb
from custom_functions.ped_sequence_plot import ind_seq_dict, plot_sequence, plot_frame

# from custom_functions.convert_frames_to_videos import convert_spec_frames_to_vid

from custom_functions.auc_metrics import l2_error, iou_as_probability, anomaly_metric, combine_time
from custom_functions.auc_metrics import giou_as_metric, ciou_as_metric, diou_as_metric, giou_ciou_diou_as_metric

from verify import order_abnormal

from custom_functions.visualizations import plot_frame_from_image, plot_vid, generate_images_with_bbox
from custom_functions.search import pkl_seq_ind
from custom_functions.pedsort  import incorrect_frame_represent



def gpu_check():
    """
    return: True if gpu amount is greater than 1
    """
    return len(tf.config.experimental.list_physical_devices('GPU')) > 0


def ped_auc_to_frame_auc_data(model, testdicts, metric, avg_or_max, modeltype, test_bin=None):
    """
    Note that this does not explictly calcuate frame auc
    but removes select points to reduce to frame AUC data.

    testdict: From orginal data test dict
    test_bin: binary classifer test dict
    model: lstm model
    metric: l2, giou,l2
    modeltype: type of model


    return:
    test_auc_frame: people that are in the frame as proability and index of frame
    remove_list: indices of pedestrain removed
    """
    if not test_bin:
        # calc iou prob
        # for iou, giou, ciou and diou. used i-iou, 1-giou, 1-ciou, 1-diou etc 
        # because abnormal pedestrain would have a higher score
        if metric == 'iou':
            prob, prob_along_time = iou_as_probability(testdicts, model, errortype = hyparams['errortype'], max1 = max1, min1= min1)
        
        elif metric == 'l2':
            
            prob, prob_along_time = l2_error(testdicts = testdicts, models = model, errortype =hyparams['errortype'], max1=max1,min1= min1)
        
        elif metric == 'giou' or metric =='ciou' or metric =='diou':
            prob, prob_along_time = giou_ciou_diou_as_metric(testdicts = testdicts, models = model, metric=metric,errortype = hyparams['errortype'], max1 = max1,min1 = min1)


        prob_iou, prob_l2, prob_giou, prob_ciou, prob_diou = None, None, None, None, None
        if exp['plot_images']:
            prob_iou, _ = iou_as_probability(testdicts, model, errortype = hyparams['errortype'], max1 = max1, min1= min1)

            prob_l2, _ = l2_error(testdicts = testdicts, models = model, errortype =hyparams['errortype'], max1=max1,min1= min1)

            prob_giou , _ = giou_ciou_diou_as_metric(testdicts = testdicts, models = model, metric='giou',errortype = hyparams['errortype'], max1 = max1,min1 = min1)
            prob_ciou , _ = giou_ciou_diou_as_metric(testdicts = testdicts, models = model, metric='ciou',errortype = hyparams['errortype'], max1 = max1,min1 = min1)
            prob_diou , _ = giou_ciou_diou_as_metric(testdicts = testdicts, models = model, metric='diou',errortype = hyparams['errortype'], max1 = max1,min1 = min1)


        pkldicts = combine_time(    testdicts, models=model, errortype=hyparams['errortype'], 
                                    modeltype = modeltype, max1 =max1, min1=min1)


        out = anomaly_metric(   prob, 
                                avg_or_max,
                                pred_trajs = pkldicts['pred_trajs'],
                                gt_box = pkldicts['gt_bbox'], 
                                vid_loc = pkldicts['vid_loc'],
                                frame_loc = pkldicts['frame_y'],
                                person_id = pkldicts['id_y'],
                                abnormal_gt = pkldicts['abnormal_gt_frame'],
                                abnormal_person = pkldicts['abnormal_ped_pred'],
                                prob_in_time = prob_along_time,
                                prob_iou = prob_iou,
                                prob_l2 = prob_l2,
                                prob_giou = prob_giou,
                                prob_ciou = prob_ciou,
                                prob_diou = prob_diou
                                )

        prob = out['prob']
        vid_loc = out['vid']
        frame_loc = out['frame']
        
        # frame_loc, vid_loc, abnormal_gt_frame_metric, std, std_iou
        test_index = np.arange(0, len(out['abnormal_gt_frame_metric']), 1)
    
    else:
        test_bin_index = test_bin['x'][:,1]

        test_bin_index = test_bin_index.astype(int) 

        vid_loc = testdict['video_file'][test_bin_index].reshape(-1,1) #videos locations
        frame_loc = testdict['frame_y'][test_bin_index].reshape(-1,1) # frame locations
    
    # encoding video and frame 
    vid_frame = np.append(vid_loc, frame_loc, axis=1)


    
    # Treating each row vector as a unique element 
    # and looking for reapeats
    unique, unique_inverse, unique_counts = np.unique(vid_frame, axis=0, return_inverse=True, return_counts=True)


    #  finds where repeats happened and gives id for input into unique_inverse
    
    # this works because removing those greater than 1
    repeat_inverse_id = np.where(unique_counts>1)[0] 

    
    # Pedestrain AUC equals Frame AUC
    if len(repeat_inverse_id) == 0:
        # print('Ped AUC = Frame AUC')
        if not test_bin:
            test_auc_frame = 'not possible'
        else: 
            test_auc_frame = test_bin

    # Convert Pedestrain AUC to Frame AUC
    else:
        # print('Ped AUC != Frame AUC')
        # print(repeat_inverse_id.shape)
        # # find pairs given repeat_inverse_id
        remove_list_temp = []
        for i in repeat_inverse_id:
            # find all same vid and frame
            if not test_bin:
                same_vid_frame = np.where(unique_inverse == i)[0]
                # Note that this is treating iou_as_prob
                y_pred = prob[same_vid_frame]

            else:
                same_vid_frame = np.where(unique_inverse == i )[0]
                y_pred = model.predict(test_bin['x'][:,0][same_vid_frame])
                # find max y_pred input other indices to remove list

            # This saves it for both cases below  
            max_loc = np.where( y_pred == np.max(y_pred))[0]
            if len(max_loc) > 1: 
                temp_1 = max_loc[1:]
                temp_2 = np.where(y_pred != np.max(y_pred))[0]
                remove_list_temp.append(same_vid_frame[temp_1])
                remove_list_temp.append(same_vid_frame[temp_2])
            else:
                temp = np.where(y_pred != np.max(y_pred))[0]
                remove_list_temp.append(same_vid_frame[temp])
                    
        
        remove_list = [item for sub_list in remove_list_temp for item in sub_list]

             
        remove_list = np.array(remove_list).astype(int)

        # print('Length of removed elements is :{}'.format(len(remove_list)))
        # print(test_bin['x'].shape)
        test_auc_frame = {}
        if not test_bin:
            test_auc_frame['x'] = np.delete( prob.reshape(-1,1), remove_list, axis = 0 )
            test_auc_frame['y'] = np.delete(out['abnormal_gt_frame_metric'], remove_list, axis = 0)
            test_auc_frame['x_pred_per_human'] = prob.reshape(-1,1)
            test_auc_frame['y_pred_per_human'] = out['abnormal_gt_frame_metric']
            test_auc_frame['std_per_frame'] = np.delete(out['std'], remove_list, axis = 0)
            test_auc_frame['std_per_human'] = out['std']
            test_auc_frame['index'] = test_index.reshape(-1,1)

            # if hyparams['metric'] == 'iou':
            test_auc_frame['std_iou_or_l2_per_frame'] = np.delete(out['std_iou_or_l2'], remove_list, axis = 0)
            test_auc_frame['std_iou_or_l2_per_human'] = out['std_iou_or_l2']

            out_frame = {}
            for key in out.keys():
                out_frame[key] = np.delete(out[key], remove_list, axis = 0)
            # test_auc_frame['y'] = np.delete(testdict['abnormal_gt_frame'].reshape(-1,1) , remove_list, axis=0)
            # test_auc_frame['abnormal_ped_pred'] = np.delete(testdict['abnormal_ped_pred'].reshape(-1,1) , remove_list, axis=0)
        else:
            test_auc_frame['x'] = np.delete(test_bin['x'], remove_list, axis=0)
            test_auc_frame['y'] = np.delete(test_bin['y'], remove_list, axis=0)

    return test_auc_frame, remove_list, out_frame, out 

# def find_abnormal_range(out):
#     poten = []
#     quick_search = np.where(out['abnormal_gt_frame_metric'] ==1)[0]

#     poten.append(quick_search[0])
#     start = quick_search[0]
#     if num in quick_search[1:]:
#         temp  = start +1
#         if temp == num:
#             save = temp
#         else:

def find_ranges(iterable):
    """Yield range of consecutive numbers."""
    for group in mit.consecutive_groups(iterable):
        group = list(group)
        if len(group) == 1:
            yield group[0]
        else:
            yield group[0], group[-1]


def count_frame_level_human_accuracy(out_frame):
    y_true = out_frame['abnormal_gt_frame_metric']
    y_pred = out_frame['abnormal_ped_pred'] 
    tn, fp, fn,tp = confusion_matrix(y_true, y_pred).ravel()

    print('tn:{}, fp:{}, fn:{}, tp:{} '.format(tn, fp, fn, tp))


def plot_frame_wise_scores(out_frame):
    vid_to_split = np.unique(out_frame['vid'])

    out = {}
    for vid in vid_to_split:
        vid_index = np.where(out_frame['vid'] == vid)[0]
        # frames = np.array(out_frame['frame'], dtype=int)
        frames = out_frame['frame']
        framesort = np.argsort(frames[vid_index].reshape(-1))
        out[vid] = {}
        for key in out_frame.keys():
            out[vid][key] = out_frame[key][vid_index][framesort]

    
    # for key in out.keys():
    for key in out.keys():
        fig,ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
        # abnorm = np.where(out[key]['abnormal_gt_frame_metric'] == 1)[0]
        # norm = np.where(out[key]['abnormal_gt_frame_metric'] == 0)[0]
        # ax.scatter(out[key]['frame'][abnorm], out[key]['prob'][abnorm], marker='.', color ='r')
        # ax.scatter(out[key]['frame'][norm], out[key]['prob'][norm], marker='.', color ='b')
        ax.plot(out[key]['frame'],out[key]['prob'])

        index = np.where(out[key]['abnormal_gt_frame_metric'] ==1)[0]
        index_range =list(find_ranges(index))
        start = []
        end = []

        for i in index_range:
            if len(i) == 2:
                start.append(out[key]['frame'][i[0]])
                end.append(out[key]['frame'][i[1]])
            else:
                temp = out[key]['frame'][i[0]]
                start.append(temp)
                end.append(temp)
        

        for s,e in zip(start,end):
            ax.axvspan(s,e, facecolor='r', alpha=0.5)
        # ax.axvspan(299, 306, facecolor='b', alpha=0.5)
        # ax.axvspan(422, 493, facecolor='b', alpha=0.5)
        # ax.axvspan(562, 604, facecolor='b', alpha=0.5)

        ax.set_xlabel('Frames')
        ax.set_ylabel('Anomaly Score' )
        fig.savefig('testing_{}.jpg'.format(key[:-4]))  




    
def frame_traj_model_auc(model, testdicts, metric, avg_or_max, modeltype):
    """
    This function is meant to find the frame level based AUC
    model: any trajactory prediction model (would need to check input matches)
    testdicts: is the input data dict
    metric: iou or l2, giou, ciou,diou metric
    avg_or_max: used when looking at same person over frame, for vertical comparssion

    return:
    auc_human_frame: human and frame level auc
    """

    # Note that this return ious as a prob 
    test_auc_frame, remove_list, out_frame, out_human = ped_auc_to_frame_auc_data(model, testdicts, metric, avg_or_max, modeltype)
    # return
    # count_frame_level_human_accuracy(out_frame)

    # Plot abnormal Grapph scores
    # plot_frame_wise_scores(out_frame)
    # quit()
    nc = [  loc['nc']['date'] + '_per_frame',
            loc['nc']['model_name'],
            loc['nc']['data_coordinate_out'],
            loc['nc']['dataset_name'],
            hyparams['input_seq'],
            hyparams['pred_seq']
            ] # Note that frames is the sequence input
    
    if test_auc_frame == 'not possible':
        quit()
    
    # 1 means  abnormal, if normal than iou would be high
    wandb_name = ['rocs', 'roc_curve']
    
    path_list = loc['metrics_path_list'].copy()
    visual_path = loc['visual_trajectory_list'].copy()
    for path in [path_list, visual_path]:
        path.append('{}_{}_in_{}_out_{}_K_{}'.format(loc['nc']['date'], exp['data'], hyparams['input_seq'],
                                                            hyparams['pred_seq'],exp['K'] ))
        path.append('{}_{}_{}_in_{}_out_{}_{}'.format(  loc['nc']['date'],
                                                        metric,
                                                        avg_or_max, 
                                                        hyparams['input_seq'], 
                                                        hyparams['pred_seq'],
                                                        hyparams['errortype'] ) )

    make_dir(path_list)
    plot_loc = join( os.path.dirname(os.getcwd()), *path_list )
    joint_txt_file_loc = join( os.path.dirname(os.getcwd()), *path_list[:-1] )

    # This plots the result of bbox in the images 
    if exp['plot_images']:
        generate_images_with_bbox(testdicts,out_frame, visual_path)

    generate_metric_plots(test_auc_frame, metric, nc, plot_loc)   
    # else:
    #     file_avg_metrics = SaveTextFile(plot_loc, metric)
    #     file_avg_metrics.save(output_with_metric, auc_frame_human)
    #     file_with_auc = SaveAucTxt(joint_txt_file_loc, metric)
    #     file_with_auc.save(auc_frame_human)

    #     print(joint_txt_file_loc)

    y_true = test_auc_frame['y']
    y_pred = test_auc_frame['x']
    y_true_per_human = test_auc_frame['y_pred_per_human']
    y_pred_per_human = test_auc_frame['x_pred_per_human']


    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    AUC_frame = auc(fpr, tpr)
    fpr, tpr, thresholds = roc_curve(y_true_per_human, y_pred_per_human)
    AUC_human = auc(fpr,tpr)
    auc_human_frame = np.array([AUC_human, AUC_frame])
    return auc_human_frame





# def trouble_shot(testdict, model, frame, ped_id, vid):
def plot_traj_gen_traj_vid(testdict, model):
    """
    testdict: the dict
    model: model to look at
    frame: frame number of interest (note assuming that we see the final frame first and)
            then go back and plot traj (int)
    ped_id: pedestrain id (int)
    vid: video number (string)

    """

    frame = 181
    # This is helping me plot the data from tlbr -> xywh -> tlbr
    ped_loc = loc['visual_trajectory_list'].copy()
    ped_id = 6
    
    vid = '01_0014'
    person_seq = ind_seq_dict(testdict, '{}'.format(vid), frame,  ped_id) # this is a slow search I would think  
    
    ped_loc[-1] =  '{}'.format(vid) + '_' + '{}'.format(frame)+ '_' + '{}'.format(ped_id)
    make_dir(ped_loc)
    pic_loc = join(     os.path.dirname(os.getcwd()),
                        *ped_loc
                        )


    # test_auc_frame, remove_list, y_pred_per_human = ped_auc_to_frame_auc_data(model, testdict)
    
    # temp_dict = {}
    # for i in testdict.keys():
    #     indices = np.array(test_auc_frame['x'][:,1], dtype=int)
    #     temp_dict[i] = testdict[i][indices]

    person_seq = ind_seq_dict(testdict, '{}'.format(vid), frame,  ped_id) # this is a slow search I would think
    # person_seq = ind_seq_dict(temp_dict, '{}'.format(vid), frame,  ped_id) # this is a slow search I would think


    # bbox_pred = norm_train_max_min(bbox_pred_norm, max1, min1, undo_norm=True)
    bbox_pred = np.expand_dims(person_seq['pred_trajs'], axis=0)
    iou_unorm = bb_intersection_over_union_np(  xywh_tlbr(bbox_pred),
                                                xywh_tlbr(np.expand_dims(person_seq['y_ppl_box'], axis=0) )
                                                )

    print('iou not normalized  which is correct{}'.format(iou_unorm))
    
    # # see vid frame i
    print('vid:{} frame:{} id:{}'.format(vid, frame, ped_id))
    print('abnormal indictor {}'.format(person_seq['abnormal_ped_pred']))

    # quit()
    plot_sequence(  person_seq,
                    max1,
                    min1,
                    '{}.txt'.format(vid),
                    pic_loc = pic_loc,
                    loc_videos = loc_videos,
                    xywh= True
                    )

    gen_vid(vid_name = '{}_{}_{}'.format(vid, frame, ped_id),pic_loc = pic_loc, frame_rate = 1)
    print('should see this rn if quit works')
    quit()


   

def gen_vid(vid_name, pic_loc, frame_rate):
    # vid_name = '04_670_61'
    # image_loc = '/home/akanu/results_all_datasets/experiment_traj_model/visual_trajectory_consecutive/{}'.format(vid_name)
    save_vid_loc = loc['visual_trajectory_list']
    save_vid_loc[-1] = 'short_generated_videos'

    make_dir(save_vid_loc)
    save_vid_loc = join(    os.path.dirname(os.getcwd()),
                            *save_vid_loc
                            )
    convert_spec_frames_to_vid( loc = pic_loc, 
                                save_vid_loc = save_vid_loc, 
                                vid_name = vid_name, frame_rate = frame_rate )

    

def main():

    load_lstm_model = False
    model_loc = join(   os.path.dirname(os.getcwd()),
                        *loc['model_path_list']
                        ) # create save link
    



    print(model_loc)
    
    nc = [  #loc['nc']['date'],
            '07_05_2021',
            loc['nc']['model_name'],
            loc['nc']['data_coordinate_out'],
            loc['nc']['dataset_name'],
            hyparams['input_seq'],
            hyparams['pred_seq'],
            ] # Note that frames is the sequence input


    global max1, min1
    
    max1 = None
    min1 = None

    if exp['model_name'] =='lstm_network':
        traindict, testdict = data_lstm(    loc['data_load'][exp['data']]['train_file'],
                                        loc['data_load'][exp['data']]['test_file'],
                                        hyparams['input_seq'], hyparams['pred_seq']
                                        , 
                                        )

        max1 = traindict['x_ppl_box'].max() if traindict['y_ppl_box'].max() <= traindict['x_ppl_box'].max() else traindict['y_ppl_box'].max()
        min1 = traindict['x_ppl_box'].min() if traindict['y_ppl_box'].min() >= traindict['x_ppl_box'].min() else traindict['y_ppl_box'].min()


        traindict_win, testdict_win = data_lstm(    loc['data_load'][exp['data']]['train_file'],
                                loc['data_load'][exp['data']]['test_file'],
                                hyparams['input_seq'], hyparams['pred_seq'],
                                window = hyparams['input_seq'] 
                                )

    # To load the pkl files from BiTrap
    # pkldicts = [ load_pkl(loc['pkl_file'][exp['data']], exp['data']) ]

    # plot_traj_gen_traj_vid(pkldicts[0], 'bitrap')
    
  

    if load_lstm_model:        
        model_path = os.path.join(  model_loc,
                        '{}_{}_{}_{}_{}_{}.h5'.format(*nc)
                        )
        print(model_path)
        lstm_model = tf.keras.models.load_model(    model_path,  
                                                    custom_objects = {'loss':'mse'} , 
                                                    compile=True
                                                    )
    else:
        # returning model right now but might change that in future and load instead
        lstm_model = lstm_train(traindict, max1, min1)

    #Load Data
    # pkldicts_temp = load_pkl('/home/akanu/output_bitrap/avenue_unimodal/gaussian_avenue_in_5_out_1_K_1.pkl')
    
    

    # pkldicts = []
    # pkldicts.append(load_pkl(loc['pkl_file']['avenue_template'].format(5,5)))

    # modelfile = '/home/akanu/results_all_datasets/experiment_traj_model/saved_model_consecutive/07_05_2021_lstm_network_xywh_avenue_{}_{}.h5'.format(hyparams['input_seq'], hyparams['pred_seq'])

        

    frame_traj_model_auc([lstm_model], [testdict], hyparams['metric'], hyparams['avg_or_max'], exp['model_name'])
    # frame_traj_model_auc( 'bitrap', pkldicts, hyparams['metric'], hyparams['avg_or_max'], exp['model_name'])
    print('Input Seq: {}, Output Seq: {}'.format(hyparams['input_seq'], hyparams['pred_seq']))
    print('Metric: {}, avg_or_max: {}'.format(hyparams['metric'], hyparams['avg_or_max']))
    # # Note would need to change mode inside frame_traj


    # classifer_train(traindict, testdict, lstm_model)
     

    # plot_traj_gen_traj_vid(pkldict,lstm_model)

    
def run_quick(window_not_one = False):
    """
    window: changes the window size
    """
    
    global max1, min1

    max1 = None
    min1 = None
    # change this to run diff configs
    in_lens = [3,5,13,25]
    out_lens = [3, 5,13,25]
    errors_type = ['error_summed', 'error_flattened']

    for in_len, out_len in zip(in_lens, out_lens):
        hyparams['input_seq'] = in_len
        hyparams['pred_seq'] = out_len
        print('{} {}'.format(hyparams['input_seq'], hyparams['pred_seq']))
        # continue
        if exp['data']=='st' and exp['model_name']=='bitrap':
            if window_not_one:
                pklfile = loc['pkl_file']['st_template_skip'].format(hyparams['input_seq'], hyparams['pred_seq'],exp['K'], hyparams['input_seq'])
            else:
                pklfile = loc['pkl_file']['st_template'].format(hyparams['input_seq'], hyparams['pred_seq'], exp['K'])

        elif exp['data']=='avenue' and exp['model_name']=='bitrap':
            if window_not_one:
                pklfile = loc['pkl_file']['avenue_template_skip'].format(hyparams['input_seq'], hyparams['pred_seq'],exp['K'], hyparams['input_seq'])
                print('I am here window not one')
            else:
                pklfile = loc['pkl_file']['avenue_template'].format(hyparams['input_seq'], hyparams['pred_seq'],exp['K'])

        elif exp['data']=='avenue' and exp['model_name']=='lstm_network':
            if in_len in [3,13,25]:
                modelfile = '/home/akanu/results_all_datasets/experiment_traj_model/saved_model_consecutive/07_05_2021_lstm_network_xywh_avenue_{}_{}.h5'.format(hyparams['input_seq'], hyparams['pred_seq'])
            else:
                modelfile = '/home/akanu/results_all_datasets/experiment_traj_model/saved_model_consecutive/05_18_2021_lstm_network_xywh_avenue_{}_{}.h5'.format(hyparams['input_seq'], hyparams['pred_seq'])
            
        elif exp['data']=='st' and exp['model_name']=='lstm_network':
            modelfile = '/home/akanu/results_all_datasets/experiment_traj_model/saved_model_consecutive/07_05_2021_lstm_network_xywh_st_{}_{}.h5'.format(hyparams['input_seq'], hyparams['pred_seq'])

        if exp['model_name'] == 'lstm_network':
            model = tf.keras.models.load_model(     modelfile,  
                                                    custom_objects = {'loss':'mse'}, 
                                                    compile=True
                                                    )

            traindict, testdict = data_lstm(    loc['data_load'][exp['data']]['train_file'],
                                                loc['data_load'][exp['data']]['test_file'],
                                                hyparams['input_seq'], hyparams['pred_seq'] 
                                                )
            # This sets the max1 and min1
            max1 = traindict['x_ppl_box'].max() if traindict['y_ppl_box'].max() <= traindict['x_ppl_box'].max() else traindict['y_ppl_box'].max()
            min1 = traindict['x_ppl_box'].min() if traindict['y_ppl_box'].min() >= traindict['x_ppl_box'].min() else traindict['y_ppl_box'].min()

            if window_not_one:
                # Changes the window to run
                traindict, testdict = data_lstm(    loc['data_load'][exp['data']]['train_file'],
                                                    loc['data_load'][exp['data']]['test_file'],
                                                    hyparams['input_seq'], hyparams['pred_seq'],
                                                    window = hyparams['input_seq']
                                                    )

        
        elif exp['model_name'] == 'bitrap':
            print(pklfile)                                                                                
            pkldicts = load_pkl(pklfile, exp['data'])
            model = 'bitrap'
        
        # for error in  ['error_diff', 'error_summed', 'error_flattened']:
        for error in errors_type:
            hyparams['errortype'] = error
            auc_metrics_list = []
            print(hyparams['errortype'])
            for metric in ['iou', 'giou', 'l2']:
                hyparams['metric'] = metric
                print(hyparams['metric'])
                if exp['model_name'] == 'bitrap':
                    auc_metrics_list.append(frame_traj_model_auc( 'bitrap', [pkldicts], hyparams['metric'], hyparams['avg_or_max'], exp['model_name']))
                elif exp['model_name'] == 'lstm_network':
                    auc_metrics_list.append(frame_traj_model_auc( [model], [testdict], hyparams['metric'], hyparams['avg_or_max'], exp['model_name']))
            
            path_list = loc['metrics_path_list'].copy()
            path_list.append('{}_{}_in_{}_out_{}_K_{}'.format(loc['nc']['date'], exp['data'], hyparams['input_seq'],
                                                                hyparams['pred_seq'],exp['K'] ))
            joint_txt_file_loc = join( os.path.dirname(os.getcwd()), *path_list )

            print(joint_txt_file_loc)
            auc_together=np.array(auc_metrics_list)


            auc_slash_format = SaveAucTxtTogether(joint_txt_file_loc)
            auc_slash_format.save(auc_together)

            # auc_slash_format = SaveAucTxt(joint_txt_file_loc)



   
   
def overlay_roc_curves():
    # Load pkl files 
    in_lens = [3,5,13,25]
    out_lens = [3,5,13,25]
    result_table = {}
    result_table['type']=[]
    result_table['fpr']=[]
    result_table['tpr']=[]
    result_table['auc']=[]
    exp['model_name']=='bitrap'
    global max1, min1
    max1=None
    min1=None
    compare = ['l2', 'l2']
    errortype =['error_summed', 'error_flattened']
    # This is for bitrap model
    for in_len, out_len in zip(in_lens, out_lens):
            hyparams['input_seq'] = in_len
            hyparams['pred_seq'] = out_len
            hyparams['error_type'] = errortype[0]
            print('{} {}'.format(hyparams['input_seq'], hyparams['pred_seq']))
            # continue
            if exp['data']=='st':
                pklfile = loc['pkl_file']['st_template'].format(hyparams['input_seq'], hyparams['pred_seq'], exp['K'])

            elif exp['data']=='avenue':
                pklfile = loc['pkl_file']['avenue_template'].format(hyparams['input_seq'], hyparams['pred_seq'],exp['K'])


            pkldicts = load_pkl(pklfile, exp['data'])

            # Return input to roc curve
            test_auc_frame, _, __, ___ = ped_auc_to_frame_auc_data('bitrap', [pkldicts], compare[0], 'avg', 'bitrap')

            y_true = test_auc_frame['y']
            y_pred = test_auc_frame['x']
            
            # Return tpr and fpr
            fpr, tpr, thresholds = roc_curve(y_true, y_pred)

            AUC = auc(fpr, tpr)

            result_table['type'].append('bitrap_{}_{}'.format(in_len, out_len))
            result_table['fpr'].append(fpr)
            result_table['tpr'].append(tpr)
            # Input to obtain auc result
            result_table['auc'].append(AUC)
            

    #  This is for LSTM Baseline
    for in_len, out_len in zip(in_lens, out_lens):
            hyparams['input_seq'] = in_len
            hyparams['pred_seq'] = out_len
            print('{} {}'.format(hyparams['input_seq'], hyparams['pred_seq']))
            hyparams['error_type'] = errortype[1]
            # continue
            exp['model_name']=='lstm_network'
            if exp['data']=='avenue':
                if in_len == 5 and out_len==5:
                    modelfile = '/home/akanu/results_all_datasets/experiment_traj_model/saved_model_consecutive/05_18_2021_lstm_network_xywh_avenue_{}_{}.h5'.format(hyparams['input_seq'], hyparams['pred_seq'])

                else:
                    modelfile = '/home/akanu/results_all_datasets/experiment_traj_model/saved_model_consecutive/07_05_2021_lstm_network_xywh_avenue_{}_{}.h5'.format(hyparams['input_seq'], hyparams['pred_seq'])
                
            elif exp['data']=='st':
                modelfile = '/home/akanu/results_all_datasets/experiment_traj_model/saved_model_consecutive/07_05_2021_lstm_network_xywh_st_{}_{}.h5'.format(hyparams['input_seq'], hyparams['pred_seq'])

            model = tf.keras.models.load_model(     modelfile,  
                                                    custom_objects = {'loss':'mse'}, 
                                                    compile=True
                                                    )

            traindict, testdict = data_lstm(    loc['data_load'][exp['data']]['train_file'],
                                                loc['data_load'][exp['data']]['test_file'],
                                                hyparams['input_seq'], hyparams['pred_seq'] 
                                                )

            max1 = traindict['x_ppl_box'].max() if traindict['y_ppl_box'].max() <= traindict['x_ppl_box'].max() else traindict['y_ppl_box'].max()
            min1 = traindict['x_ppl_box'].min() if traindict['y_ppl_box'].min() >= traindict['x_ppl_box'].min() else traindict['y_ppl_box'].min()

            # Return input to roc curve
            test_auc_frame, _, __, ___ = ped_auc_to_frame_auc_data([model], [testdict], compare[1], 'avg', 'lstm_network')

            y_true = test_auc_frame['y']
            y_pred = test_auc_frame['x']
            
            # Return tpr and fpr
            fpr, tpr, thresholds = roc_curve(y_true, y_pred)

            AUC = auc(fpr, tpr)

            result_table['type'].append('lstm_{}_{}'.format(in_len, out_len))
            result_table['fpr'].append(fpr)
            result_table['tpr'].append(tpr)
            # Input to obtain auc result
            result_table['auc'].append(AUC)


    fig = plt.figure(figsize=(8,6))
    for i in range(0, len(result_table['auc'])):
        plt.plot(result_table['fpr'][i], 
             result_table['tpr'][i], 
             label="{}, AUC={:.4f}".format(result_table['type'][i], result_table['auc'][i]))

    plt.plot([0,1], [0,1], color='orange', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("False Positive Rate", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)


    plt.title('BiTrap vs LSTM ROC Curve Analysis', fontweight='bold', fontsize=15)
    plt.legend(prop={'size':13}, loc='lower right')

    fig.savefig('roc_plot_{}_{}_{}_{}_{}.jpg'.format(errortype[0], compare[0], errortype[1], compare[1], exp['data']))
    # Load model and load correct dataset format

    # Save output with roc values 



    

if __name__ == '__main__':
    # print('GPU is on: {}'.format(gpu_check() ) )
    start = time.time()
    main()
    # run_quick(window_not_one=True)
    end = time.time()
    # print(end - start)


    print('Done') 