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
from coordinate_change import xywh_tlbr, tlbr_xywh
# Data Info
from load_data import load_pkl
# Plots
from metrics_plot import *
from matplotlib import pyplot as plt
# Models
import wandb

from custom_functions.anomaly_detection import frame_traj_model_auc
from custom_functions.visualizations import anomaly_score_frame_plot_from_figure_4


def auc_calc_bitrap(load_pkl_file):

    max1 = None
    min1 = None
    pkldicts = []

    pkldicts.append(load_pkl(load_pkl_file, 'avenue'))
    auc_human_frame, out_frame = frame_traj_model_auc( 'bitrap', pkldicts, hyparams['metric'], hyparams['avg_or_max'], exp['model_name'], max1, min1)
    print('Input Seq: {}, Output Seq: {}'.format(hyparams['input_seq'], hyparams['pred_seq']))
    print('Metric: {}, avg_or_max: {}'.format(hyparams['metric'], hyparams['avg_or_max']))
    print('auc_frame:{} '.format(auc_human_frame[1]))
    
    anomaly_score_figure_4_dir = '../experiments_code/anomaly_score_plots'
    anomaly_score_frame_plot_from_figure_4(out_frame, anomaly_score_figure_4_dir)


if __name__ == '__main__':
    exp['data'] =  'avenue' #st, avenue
    exp['model_name'] ='bitrap'
        
        

    hyparams['input_seq'] = 5
    hyparams['pred_seq'] = 5
    hyparams['metric'] = 'l2' #giou,l2, ciou diou,iou
    hyparams['avg_or_max'] = 'avg' #avg 
    hyparams['errortype'] = 'error_flattened' #or 'error_summed' or 'error_flattened'


    load_pkl_file ='/mnt/roahm/users/akanu/projects/anomalous_pred/output_bitrap/avenue_unimodal/gaussian_avenue_in_5_out_5_K_1.pkl'
    auc_calc_bitrap(load_pkl_file)
