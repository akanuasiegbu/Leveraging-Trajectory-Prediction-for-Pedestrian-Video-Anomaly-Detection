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
from data import data_lstm, tensorify
from load_data import norm_train_max_min, load_pkl
from load_data_binary import compute_iou
# Plots
from metrics_plot import *
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
# import more_itertools as mit

# Models
from models import lstm_network, lstm_train
import wandb
from custom_functions.ped_sequence_plot import ind_seq_dict, plot_sequence, plot_frame

# from custom_functions.convert_frames_to_videos import convert_spec_frames_to_vid

from custom_functions.auc_metrics import l2_error, iou_as_probability #, anomaly_metric, combine_time
from custom_functions.auc_metrics import giou_as_metric, ciou_as_metric, diou_as_metric, giou_ciou_diou_as_metric

from verify import order_abnormal

from custom_functions.visualizations import plot_frame_from_image, plot_vid, generate_images_with_bbox
from custom_functions.search import pkl_seq_ind
from custom_functions.pedsort  import incorrect_frame_represent
from custom_functions.anomaly_detection import frame_traj_model_auc


def gpu_check():
    """
    return: True if gpu amount is greater than 1
    """
    return len(tf.config.experimental.list_physical_devices('GPU')) > 0


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

    

    pkldicts = []
    pkldicts.append(load_pkl('/mnt/roahm/users/akanu/projects/anomalous_pred/output_bitrap/avenue_unimodal/gaussian_avenue_in_5_out_5_K_1.pkl', 'avenue'))

    # modelfile = '/home/akanu/results_all_datasets/experiment_traj_model/saved_model_consecutive/07_05_2021_lstm_network_xywh_avenue_{}_{}.h5'.format(hyparams['input_seq'], hyparams['pred_seq'])

        

    # frame_traj_model_auc([lstm_model], [testdict], hyparams['metric'], hyparams['avg_or_max'], exp['model_name'], max1, min1)
    frame_traj_model_auc( 'bitrap', pkldicts, hyparams['metric'], hyparams['avg_or_max'], exp['model_name'], max1, min1)
    print('Input Seq: {}, Output Seq: {}'.format(hyparams['input_seq'], hyparams['pred_seq']))
    print('Metric: {}, avg_or_max: {}'.format(hyparams['metric'], hyparams['avg_or_max']))


    # classifer_train(traindict, testdict, lstm_model)
     

    # plot_traj_gen_traj_vid(pkldicts[0], 'bitrap')
    # plot_traj_gen_traj_vid(pkldict,lstm_model)

    
  
   

    

if __name__ == '__main__':
    # print('GPU is on: {}'.format(gpu_check() ) )
    start = time.time()
    # main()
    auc_calc_lstm()
    # run_quick(window_not_one=True)
    end = time.time()
    # print(end - start)


    print('Done') 