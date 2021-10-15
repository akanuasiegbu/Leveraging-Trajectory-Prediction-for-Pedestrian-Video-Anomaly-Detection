import datetime

exp = { '1': False,
        '2': False,
        '3_1': False,
        '3_2': False,
        # want an adaptive model saved based on arch size for model_loc
        'data': 'st', #st, avenue,hr-st
        'data_consecutive': True,
        'model_name': 'bitrap', #lstm_network, bitrap, bitrap_640_360
        'K': 1,
        'plot_images':False # Plot images
        }


hyparams = {
    'epochs': 350,
    'batch_size': 32,
    'buffer_size': 10000,
 
    'frames': 5,
    'input_seq':25,
    'pred_seq':25,
    'metric': 'l2', #giou,l2, ciou diou,iou
    'avg_or_max': 'avg', #avg or max
    'errortype': 'error_flattened', #'error_diff' or 'error_summed' or 'error_flattened'

    'to_xywh': True, # This is assuming file is in tlbr format

    'networks': {
        'lstm':{
            'loss':'mse',
            'lr': 8.726e-06,
            'early_stopping': True,
            'mointor':'loss',
            'min_delta': 0.00005,
            'patience': 15,
            'val_ratio': 0.3,
        },


        # 'binary_classifier':{
        #     'neurons': 30,
        #     'dropout':0.3,
        #     'lr': 0.0001, #0.00001
        #     'save_model': False,
        #     'early_stopping':False,
        #     'mointor': 'loss',
        #     'min_delta':0.0000005,
        #     'batch_size': 128,
        #     'patience':30,
        #     'wandb': False, # want to move this out of here and create a seperate wandb file
        #     'seed':40,
        #     'abnormal_split':0.5, # Split for the data into normal and abnormal
        #     'val_ratio':0.3, #guarantee the ratio of normal and abnormal frames
        #                     # are the same for validatin set and training set.
        #                     # so val_ratio. Think val_ratio(normal) + val_ratio(abnormal ) = val_ratio(normal + abnormal)

        # }
    }

}

# name_exp = None
if exp['1']:
    name_exp = '1'
elif exp['2']:
    name_exp = '2'
elif exp['3_1']:
    name_exp ='3_1'
elif exp['3_2']:
    name_exp = '3_2'
else:
    name_exp = 'traj_model'

now = datetime.datetime.now()
date = now.strftime("%m_%d_%Y")
time = now.strftime("%H:%M:%S")

if exp['data_consecutive']:
    model_path_list = ['results_all_datasets', 'experiment_{}'.format(name_exp), 'saved_model_consecutive']
    metrics_path_list = ['results_all_datasets', 'experiment_{}'.format(name_exp), 'metrics_plot_consecutive']
    visual_trajectory_list = ['results_all_datasets', 'experiment_{}'.format(name_exp), 'visual_trajectory_consecutive', '{}_{}_{}_{}_{}'.format(date, exp['data'], time, hyparams['input_seq'], hyparams['pred_seq'])]
    
    if exp['model_name'] == 'bitrap' or exp['model_name'] == 'bitrap_640_360' or exp['model_name'] == 'bitrap_1080_1020':
         model_path_list = ['results_all_datasets', 'experiment_{}'.format(name_exp), 'saved_model_consecutive']
         metrics_path_list = ['results_all_datasets', 'experiment_{}'.format(name_exp), 'metrics_plot_consecutive_bitrap']
         visual_trajectory_list = ['results_all_datasets', 'experiment_{}'.format(name_exp), 'visual_trajectory_consecutive_bitrap', '{}_{}_{}_{}'.format(date, exp['data'], hyparams['input_seq'], hyparams['pred_seq'])]

else:
    model_path_list = ['results_all_datasets', 'experiment_{}'.format(name_exp), 'saved_model']
    metrics_path_list ['results_all_datasets', 'experiment_{}'.format(name_exp), 'metrics_plot']
    visual_trajectory_list = ['results_all_datasets', 'experiment_{}'.format(name_exp), 'visual_trajectory', '{}_{}_{}_{}_{}'.format(date, exp['data'], time, hyparams['input_seq'], hyparams['pred_seq'])]


loc =  {
    # if I'm running a test where don't want to save anything
    # how do I do that. Maybe move them to tmp
    
    'model_path_list': model_path_list,
    'metrics_path_list': metrics_path_list, 
    'visual_trajectory_list': visual_trajectory_list,
    
    'nc':{
        'model_name': exp['model_name'],
        'model_name_binary_classifer': 'binary_network',
        'data_coordinate_out': 'xywh',
        'dataset_name': exp['data'], # avenue, st             
        'date': date,
        },   

    'data_load':{
            'avenue':{
                # These are good because these locations are perm unless I manually move them
                'train_file': "/mnt/roahm/users/akanu/projects/anomalous_pred/output_deepsort/avenue/train_txt/",
                'test_file': "/mnt/roahm/users/akanu/projects/anomalous_pred/output_deepsort/avenue/test_txt/",
                'train_vid': '/mnt/roahm/users/akanu/dataset/Anomaly/Avenue_Dataset/training_videos',
                'test_vid': '/mnt/roahm/users/akanu/dataset/Anomaly/Avenue_Dataset/testing_videos',
                'pic_loc_test': '/mnt/roahm/users/akanu/dataset/Anomaly/Avenue_Dataset/frames_of_vid/test/'
                },

            'st':{
                'train_file':"/mnt/roahm/users/akanu/projects/anomalous_pred/output_deepsort/st/train_txt/",
                "test_file": "/mnt/roahm/users/akanu/projects/anomalous_pred/output_deepsort/st/test_txt/",
                'train_vid': '/mnt/workspace/datasets/shanghaitech/training/videos',
                'test_vid':  '/mnt/roahm/users/akanu/projects/Deep-SORT-YOLOv4/tensorflow2.0/deep-sort-yolov4/input_video/st_test',
                'pic_loc_test':'/mnt/workspace/datasets/shanghaitech/testing/frames'
                },
            'hr-st':{
                'train_file':"/mnt/roahm/users/akanu/projects/anomalous_pred/output_deepsort/HR-ShanghaiTech/train_txt/",
                "test_file": "/mnt/roahm/users/akanu/projects/anomalous_pred/output_deepsort/HR-ShanghaiTech/test_txt/",
                'train_vid': '/mnt/workspace/datasets/shanghaitech/training/videos',
                'test_vid':  '/mnt/roahm/users/akanu/projects/Deep-SORT-YOLOv4/tensorflow2.0/deep-sort-yolov4/input_video/st_test',
                },
            },
    
    'pkl_file':{
        'avenue': "/home/akanu/output_bitrap/avenue_unimodal/gaussian_avenue_in_{}_out_{}_K_{}.pkl".format(hyparams['input_seq'],
                                                                                             hyparams['pred_seq'],
                                                                                             exp['K']),

        'avenue_template': "/home/akanu/output_bitrap/avenue_unimodal/gaussian_avenue_in_{}_out_{}_K_{}.pkl",
        'avenue_template_skip': "/home/akanu/output_bitrap/avenue_unimodal/gaussian_avenue_in_{}_out_{}_K_{}_skip_{}.pkl",

        'st': "/home/akanu/output_bitrap/st_unimodal/gaussian_st_in_{}_out_{}_K_{}.pkl".format(hyparams['input_seq'],
                                                                                             hyparams['pred_seq'],
                                                                                             exp['K']),

        'st_template': "/home/akanu/output_bitrap/st_unimodal/gaussian_st_in_{}_out_{}_K_{}.pkl",
        'st_template_skip': "/home/akanu/output_bitrap/st_unimodal/gaussian_st_in_{}_out_{}_K_{}_skip_{}.pkl",
                                                                                             
                                    
    }


}
