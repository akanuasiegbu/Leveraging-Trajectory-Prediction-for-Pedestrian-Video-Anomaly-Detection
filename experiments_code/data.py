# This is the higher level loading
# The lower level functions are in the custom functions
# folder
from load_data import Files_Load, Boxes
from pedsort import pedsort


from sklearn.model_selection import train_test_split

from config import hyparams, exp
import tensorflow as tf
import numpy as np

from load_data import norm_train_max_min
from load_data_binary import *


def data_lstm(train_file, test_file, input_seq, pred_seq, window=1):

    # returns a dict file
    loc = Files_Load(train_file,test_file)
    traindict = Boxes(  loc_files = loc['files_train'], 
                        txt_names = loc['txt_train'],
                        input_seq = input_seq,
                        pred_seq = pred_seq,
                        data_consecutive = exp['data_consecutive'], 
                        pad = 'pre', 
                        to_xywh = hyparams['to_xywh'],
                        testing = False,
                        window = window
                        )

    testdict = Boxes(   loc_files = loc['files_test'], 
                        txt_names = loc['txt_test'],
                        input_seq = hyparams['input_seq'],
                        pred_seq = hyparams['pred_seq'], 
                        data_consecutive = exp['data_consecutive'],
                        pad = 'pre',
                        to_xywh = hyparams['to_xywh'],
                        testing = True,
                        window = window 
                        )
                        
    return traindict, testdict

def tensorify(train, val, batch_size):

    """
    Mainly using this function to make training and validation sets
    train: dict that contains x and y
    val:dict that contains x and y

    return
    train_univariate: training tensor set
    val_univariate: validation tensor set
    """

    buffer_size = hyparams['buffer_size']
    batch_size = hyparams['batch_size']

    train_univariate = tf.data.Dataset.from_tensor_slices((train['x'], np.array(train['y'].reshape(-1,hyparams['pred_seq']*4), dtype=np.float32)))
    train_univariate = train_univariate.cache().shuffle(buffer_size).batch(batch_size)
    val_univariate = tf.data.Dataset.from_tensor_slices((val['x'],np.array(val['y'].reshape(-1,hyparams['pred_seq']*4), dtype=np.float32)))
    val_univariate = val_univariate.cache().shuffle(buffer_size).batch(batch_size)

    return train_univariate, val_univariate



def data_binary(traindict, testdict, lstm, max1, min1):
    # I could have three main files that I load or
    # I could use an if statemet to switch bewteen the
    # experiments? What is the best method to use?
    # OR I could create another function that does loading?

    # Note that I am using the testing dable than Mac OS?
    # The answer is simple – more control to the user while providing betteta
    # If I do end up changing my normlization only need
    # to change in main file. Design intent


    # Test Data
    x,y = norm_train_max_min(   testdict,
                                # max1 = hyparams['max'],
                                # min1 = hyparams['min']
                                max1 = max1,
                                min1 = min1
                                )

    iou = compute_iou(x,y, max1, min1, lstm)
    
    # Note that indices returned are in same order
    # as testdict unshuffled

    # Note that indexing works only if data is loaded the same way
    # Every time . Otherwise I could create an lstm model then I would train it.
    # If loaded again then I would need to make


    ## Find indices
    indices = return_indices(   testdict['abnormal'],
                                seed = hyparams['networks']['binary_classifier']['seed'],
                                abnormal_split = hyparams['networks']['binary_classifier']['abnormal_split']
                                )



    # Gets the train and test data
    # returns a dict with keys: x, y
    train, test = binary_data_split(iou, indices)
    
    
    
    # Simple mapping
    if exp['1']:
        exp_1, exp_3 = True, False
    elif exp['2']:
        exp_1, exp_3 = False, False
    elif exp['3_1']:
        exp_1, exp_3 = True, True
    elif exp['3_2']:
        exp_1, exp_3 = False, True
    else:
        print("check experiment mapping ")
        quit()

    if exp_1:
        if exp_3:
            # Makes same amount of normal and abnormal in train
            train = reduce_train(   train['x'],
                                    train['y'],
                                    seed = hyparams['networks']['binary_classifier']['seed'])

            

        train, val = train_val_same_ratio(  train['x'],
                                            train['y'],
                                            hyparams['networks']['binary_classifier']['val_ratio'],
                                            seed = hyparams['networks']['binary_classifier']['seed']
                                            )


    else:
        # does not reshuffle just normalizes
        x,y = norm_train_max_min(   traindict,
                                    # max1 = hyparams['max'],
                                    # min1 = hyparams['min']
                                    max1 = max1,
                                    min1 = min1
                                    )

        iou = compute_iou(x, y , max1, min1, lstm)


        # Note training_set_from_lstm has iou in first column
        # second colum is the index( Put in a filler -1)
        synethic_index = -np.linspace(0,len(iou)-1, len(iou))
        synethic_index[0] = -0.01 
        # training_set_from_lstm = np.append( iou.reshape((-1,1)),
        #                                     -np.ones((len(iou), 1)),
        #                                     axis=1
        #                                     )

        training_set_from_lstm = np.append( iou.reshape((-1,1)),
                                            synethic_index.reshape((-1,1)),
                                            axis=1
                                            )


        temp_combined_train = {}

        # print('train[x] {}'.format(train['x'].shape))
        # print('training from lstm set {}'.format(training_set_from_lstm.shape))
        temp_combined_train['x']= np.append(    train['x'],
                                                training_set_from_lstm,
                                                axis=0
                                                )

        # print('shape of tem[ combined from lstm set {}'.format(temp_combined_train['x'].shape))


        #since training set appended is coming from the orginal data
        # we know that all the labels are zeros because its normal

        temp_combined_train['y'] = np.append(train['y'],
                                                np.zeros(len(iou), dtype=np.int8)
                                                )

        if exp_3:
            temp_combined_train = reduce_train( temp_combined_train['x'],
                                                temp_combined_train['y'],
                                                seed = hyparams['networks']['binary_classifier']['seed']
                                                )

        train, val = train_val_same_ratio(  temp_combined_train['x'],
                                            temp_combined_train['y'],
                                            val_ratio=hyparams['networks']['binary_classifier']['val_ratio'],
                                            seed = hyparams['networks']['binary_classifier']['seed']
                                            )
        
    return train, val, test