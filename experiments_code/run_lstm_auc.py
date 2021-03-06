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


def auc_calc_lstm(load_lstm_model=True, pretrained_model_loc = None ):
    # load_lstm_model = False
    # model_loc = join(   os.path.dirname(os.getcwd()),
    #                     *loc['model_path_list']
    #                     ) # create save link
    
    # print(model_loc)

    global max1, min1
    
    max1 = None
    min1 = None

    # nc = [  #loc['nc']['date'],
    #         '07_05_2021',
    #         exp['model_name'],
    #         loc['nc']['data_coordinate_out'],
    #         loc['nc']['dataset_name'],
    #         hyparams['input_seq'],
    #         hyparams['pred_seq'],
    #         ] # Note that frames is the sequence input

    traindict, testdict = data_lstm(    loc['data_load'][exp['data']]['train_file'],
                                    loc['data_load'][exp['data']]['test_file'],
                                    hyparams['input_seq'], hyparams['pred_seq']
                                    , 
                                    )

    max1 = traindict['x_ppl_box'].max() if traindict['y_ppl_box'].max() <= traindict['x_ppl_box'].max() else traindict['y_ppl_box'].max()
    min1 = traindict['x_ppl_box'].min() if traindict['y_ppl_box'].min() >= traindict['x_ppl_box'].min() else traindict['y_ppl_box'].min()

    if load_lstm_model:        
        
        # model_path = os.path.join(  model_loc,
        #         '{}_{}_{}_{}_{}_{}.h5'.format(*nc)
        #         )
        # print(model_path)
        # lstm_model = tf.keras.models.load_model(    model_path,   
        #                                         custom_objects = {'loss':'mse'} , 
        #                                         compile=True
        #                                         )


        lstm_model = tf.keras.models.load_model(    pretrained_model_loc,   
                                                    custom_objects = {'loss':'mse'} , 
                                                    compile=True
                                                    )
    else:
        # returning model right now but might change that in future and load instead
        lstm_model = lstm_train(traindict, max1, min1)

    frame_traj_model_auc([lstm_model], [testdict], hyparams['metric'], hyparams['avg_or_max'], exp['model_name'], max1, min1)
    print('Input Seq: {}, Output Seq: {}'.format(hyparams['input_seq'], hyparams['pred_seq']))
    print('Metric: {}, avg_or_max: {}'.format(hyparams['metric'], hyparams['avg_or_max']))



if __name__ == '__main__':
    exp['data'] =  'avenue' #st, avenue
    exp['model_name'] ='lstm_network'
        
        

    hyparams['input_seq'] = 13
    hyparams['pred_seq'] = 13
    hyparams['metric'] = 'l2' #giou,l2, ciou diou,iou
    hyparams['avg_or_max'] = 'avg' #avg 
    hyparams['errortype'] = 'error_flattened' #or 'error_summed' or 'error_flattened'

    pretrained_model_loc = '/mnt/roahm/users/akanu/projects/anomalous_pred/results_all_datasets/experiment_traj_model/saved_model_consecutive/07_05_2021_lstm_network_xywh_avenue_13_13.h5'

    auc_calc_lstm(pretrained_model_loc=pretrained_model_loc)
