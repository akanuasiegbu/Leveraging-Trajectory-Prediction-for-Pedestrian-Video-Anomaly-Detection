import os, sys, time
from os.path import join
import numpy as np
from config import hyparams, loc, exp
from sklearn.metrics import roc_curve, auc

def make_dir(dir_list):
    try:
        print(os.makedirs(join( os.path.dirname(os.getcwd()),
                                *dir_list )) )
    except OSError:
        print('Creation of the directory {} failed'.format( join(os.path.dirname(os.getcwd()),
                                                            *dir_list) ) )
    else:
        print('Successfully created the directory {}'.format(   join(os.path.dirname(os.getcwd()),
                                                                *dir_list) ) )


class SaveTextFile(object):
    def __init__(self, save_path, metric, header_=True):
        nc = [  hyparams['metric'],
                loc['nc']['date'],
                loc['nc']['model_name'],
                loc['nc']['data_coordinate_out'],
                loc['nc']['dataset_name'],
                hyparams['input_seq'],
                hyparams['pred_seq']
                ]
        self._directory =  join(save_path, '{}_{}_{}_{}_{}_{}_{}.txt'.format(*nc) )
        metric_fill = [metric]*4
        header = 'Type {}_frame std_sum_frame std_x_frame std_y_frame std_w_frame std_h_frame std_{}_frame {}_human std_sum_human std_x_human std_y_human std_w_human std_h_human std_{}_human'.format(*metric_fill)

        with open(self._directory, 'a') as with_as_write:
            with_as_write.write('{}\n'.format(header))

    def save(self, text, auc):
        titles = 'Abnormal Normal'.split()
        titles =  np.array(titles).reshape(-1)
        with open(self._directory, 'a') as with_as_write:

            for title, data in zip(titles, text):
                with_as_write.write('{} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.5f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.5f}\n'.format(title, *data))

            with_as_write.write('{} {:.4f}\{:.4f}\n'.format('auc_frame\human', *auc))
            

class SaveAucTxt(object):
    def __init__(self, save_path, metric):
        nc = [  loc['nc']['date'],
                loc['nc']['model_name'],
                loc['nc']['data_coordinate_out'],
                loc['nc']['dataset_name'],
                hyparams['input_seq'],
                hyparams['pred_seq'],
                hyparams['errortype']

                ]
        self.metric = metric
        self._directory =  join(save_path, '{}_{}_{}_{}_{}_{}_{}.txt'.format(*nc) )
        # header = 'metric Human Frame'

        # with open(self._directory, 'a') as with_as_write:
        #     with_as_write.write('{}\n'.format(header))

    def save(self, auc):

        if self.metric =='giou' or self.metric=='ciou' or self.metric =='diou' or self.metric =='iou':
            metric = '1-'+self.metric
        else:
            metric = self.metric
        with open(self._directory, 'a') as with_as_write:
            with_as_write.write('{} {:.4f} {:.4f}\n'.format(metric, auc[0], auc[1]))

class SaveAucTxtTogether(object):
    def __init__(self, save_path):
        nc = [  loc['nc']['date'],
            loc['nc']['model_name'],
            loc['nc']['data_coordinate_out'],
            loc['nc']['dataset_name'],
            hyparams['input_seq'],
            hyparams['pred_seq'],
            hyparams['errortype']
            ]
        self._directory =  join(save_path, '{}_{}_{}_{}_{}_{}_{}_together.txt'.format(*nc) )

    def save(self, auc):

        with open(self._directory, 'a') as with_as_write:
            # with_as_write.write('{:.4f}/{:.4f}/{:.4f}\n'.format(*auc[:,0])) # Human
            # with_as_write.write('{:.4f}/{:.4f}/{:.4f}\n'.format(*auc[:,1]))
            with_as_write.write('&{:.3f} &{:.3f} &{:.3f}\n'.format(*auc[:,0])) # Human
            with_as_write.write('&{:.3f} &{:.3f} &{:.3f}\n'.format(*auc[:,1]))




def write_to_txt(test_auc_frame):
    
    # Frame Level
    abnormal_index_frame = np.where(test_auc_frame['y'] == 1)[0]
    normal_index_frame = np.where(test_auc_frame['y'] == 0)[0]
    index_frame = [abnormal_index_frame, normal_index_frame]
    norm =[]
    abnorm = []    
    abnorm_norm = [abnorm, norm]
    AUC_list = []

    for indices, lists in zip(index_frame, abnorm_norm):
        lists.append(np.mean(test_auc_frame['x'][indices])) # for metric used 
        lists.append(np.mean(np.sum(test_auc_frame['std_per_frame'][indices], axis = 1))) # for std summerd
        
        for i in range(0,4):
            lists.append(np.mean(test_auc_frame['std_per_frame'][indices][:,i])) # for std of each axis
        
        lists.append(np.mean(test_auc_frame['std_iou_or_l2_per_frame'][indices] ))

        y_true = test_auc_frame['y']

    y_pred = test_auc_frame['x']
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    AUC = auc(fpr, tpr)
    AUC_list.append(AUC)
    
    # Human based
    abnormal_index = np.where(test_auc_frame['y_pred_per_human'] == 1)[0]
    normal_index = np.where(test_auc_frame['y_pred_per_human'] == 0)[0]
    index = [abnormal_index, normal_index]

    for indices, lists in zip(index, abnorm_norm):
        lists.append(np.mean(test_auc_frame['x_pred_per_human'][indices])) # for metric used 
        lists.append(np.mean(np.sum(test_auc_frame['std_per_human'][indices], axis = 1))) # for std summerd
        
        for i in range(0,4):
            lists.append(np.mean(test_auc_frame['std_per_human'][indices][:,i])) # for std of each axis
        
        lists.append(np.mean(test_auc_frame['std_iou_or_l2_per_human'][indices] ))

    temp = np.array( abnorm_norm, dtype=object )
    output = np.concatenate( (temp[0].reshape(-1,1), temp[1].reshape(-1,1)) , axis=1 ).T

    # y_true_per_human = testdict['abnormal_ped_pred']
    y_true_per_human = test_auc_frame['y_pred_per_human']
    y_pred_per_human = test_auc_frame['x_pred_per_human']
    fpr, tpr, thresholds = roc_curve(y_true_per_human, y_pred_per_human)
    AUC = auc(fpr, tpr)

    AUC_list.append(AUC)
    AUC_list = np.array(AUC_list).reshape(-1)

    return output, AUC_list















