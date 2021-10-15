import numpy as np
from math import floor
from custom_metrics import bb_intersection_over_union, bb_intersection_over_union_np
from coordinate_change import xywh_tlbr, tlbr_xywh
from load_data import norm_train_max_min



def return_indices(data, seed, abnormal_split):
    """
    Note that function returns index values that will
    allow for the creation of a train and test set that has an specificed ratio
    of normal and abnormal examples. Rest of abnormal and normal are then used
    in the training set depending on experiment. If you shuffle dict first without keeping
    track of index somehow then this function will produce meaningless
    results that are progated.

    data: 1 and 0's whose location in index corresponds to location
            in acutal dataset
    abnormal_split: percentage of abnormal frames to put in test frame

    returns: list that contains indices
            [train_abn_indices, train_n_indices, test_abn_indices, test_n_indices]
    """

    np.random.seed(seed)

    #Find normal and abnormal frames 
    abnorm_index = np.where(data == 1)
    norm_index = np.where(data == 0)

    # Randomify
    rand_an = np.random.permutation(len(abnorm_index[0]))
    rand_n = np.random.permutation(len(norm_index[0]))
    
    # Permutates the found abnormal and normal indices
    abnorm_index = abnorm_index[0][rand_an]
    norm_index = norm_index[0][rand_n]

    # Split length
    len_abn_split = floor(len(abnorm_index)*abnormal_split)

    # Testing set indices
    test_abn_indices = abnorm_index[:len_abn_split]
    test_n_indices = norm_index[:len_abn_split]

    train_abn_indices = abnorm_index[len_abn_split:]
    train_n_indices = norm_index[len_abn_split:]
        
    # Dict 
    indices = {}
    indices['train_abn'] = train_abn_indices
    indices['train_n'] = train_n_indices
    indices['test_abn'] = test_abn_indices
    indices['test_n'] = test_n_indices

    # return [train_abn_indices, train_n_indices, test_abn_indices, test_n_indices]
    return indices

def compute_iou(x,y, max1, min1, model):
    """
    This function takes in input x and y 
    that is unnormed. It then normilized x and y. 
    And passes through function


    x: normed testing data
    y: normed tested data
    model: lstm_model or other model that estimates box loc
    """
    shape =  x.shape
    predicted_bb = model.predict(x)
    # predicted_bb_unorm = norm_train_max_min(xywh_tlbr(predicted_bb), max1, min1, True)
    # gt_bb_unorm = norm_train_max_min(xywh_tlbr(y), max1, min1, True)

    predicted_bb_unorm = norm_train_max_min(predicted_bb, max1, min1, True)
    predicted_bb_unorm_tlbr = xywh_tlbr(predicted_bb_unorm.reshape(shape[0] ,-1,4))

    gt_bb_unorm = norm_train_max_min(y, max1, min1, True)
    gt_bb_unorm_tlbr = xywh_tlbr(gt_bb_unorm)



    iou = bb_intersection_over_union_np(    predicted_bb_unorm_tlbr,
                                            gt_bb_unorm_tlbr )
    return np.squeeze(iou)


def binary_data_split(iou, indices):
    """
    This function takes normed data and returns training data with IOU and return_indices
    Indices can be used to track back to location in unshuffled testdict.
    So that visulzations of what can be given as to what happened.

    iou: iou values from compute iou as input
    indices:    Note that this is a dict now
                Note that indices passed are relative to orginal
                dict locations. If shuffle dict first without returning
                indices.

    return: train and test dict with keys:x, y
            Note that the second coloumn of train_x and test_x
            contain indices corresponding the location in unshuffled
            dict
    """

    train_x = np.array( [np.append(iou[indices['train_abn']], iou[indices['train_n']] ),
                        np.append(indices['train_abn'], indices['train_n']) ]
                        )

    # indices turn to floats here when appended , come back and fix maybe not as important

    train_y = np.append(    np.ones(len(indices['train_abn']), dtype=np.int8 ),
                            np.zeros(len(indices['train_n']), dtype=np.int8 ) 
                            )


    test_x = np.array( [np.append(iou[indices['test_abn']], iou[indices['test_n']] ),
                        np.append(indices['test_abn'], indices['test_n'])]
                        )
    test_y = np.append( np.ones(len(indices['test_abn']), dtype=np.int8),
                        np.zeros(len(indices['test_n']), dtype=np.int8) 
                      )

    train, test = {}, {}
    train['x'] = train_x.T # transpose to make new features a row
    train['y'] = train_y
    test['x'] = test_x.T #transpose to make new features a row
    test['y'] = test_y
    # return train_x,train_y, test_x, test_y
    return train, test




# def same_ratio_split_train_val(train_x,train_y, val_ratio = 0.3):
def train_val_same_ratio(train_x,train_y, val_ratio, seed):
    """
    This function forces the training and validation sets
    to have the same ratio for abnormal and normal cases

    train_x: training x data for binary classifer. Shape (somenumber, 2)
             First column is the iou values. second column is the 
             index values that correspond to locations in train dict.

    train_y: training y data for binary classifer. shape (somenumber,)
             has indicator variable for abnormal. 1 means abonormal

    val_ratio: ratio to split between validation and training set

    return: train and val dict with keys:x, y
    """
    np.random.seed(seed)

    # Find normal and abnormal index
    abnorm_index = np.where(train_y == 1)
    norm_index = np.where(train_y == 0)

    # I changed the train_x input features to be rows
    # so changed from (2, somenumber) -> (somenumber, 2)

    # Randomify
    rand_an = np.random.permutation(len(abnorm_index[0]))
    rand_n = np.random.permutation(len(norm_index[0]))
    abnorm_index = abnorm_index[0][rand_an]
    norm_index = norm_index[0][rand_n]

    # Split lengths
    len_val_ab = int(len(abnorm_index)*val_ratio)
    len_val_n = int(len(norm_index)*val_ratio)


    val_x = np.append(train_x[abnorm_index[:len_val_ab], :] ,
                      train_x[norm_index[:len_val_n],:], axis = 0)

    val_y = np.append(train_y[abnorm_index[:len_val_ab]],
                      train_y[norm_index[:len_val_n]])

    train_x = np.append(train_x[abnorm_index[len_val_ab:],:] ,
                      train_x[norm_index[len_val_n:],:], axis=0)

    train_y = np.append(train_y[abnorm_index[len_val_ab:]],
                      train_y[norm_index[len_val_n:]])

    train, val = {}, {}
    val['x'] = val_x
    val['y'] = val_y
    train['x'] = train_x 
    train['y'] = train_y
    
    return train, val



# def one_weight_ratio_train(train_x, train_y):
def reduce_train(train_x, train_y,seed):
    """
    This function splits the training data to an equal amount of abnormal
    and normal sequences. Returns same type of data as inputted.
    rows and col are same format. Think about as removing the excess normal Values
    that are not used.

    train_x: training x data for binary classifer. Shape (somenumber, 2)
             First column is the iou values. second column is the 
             index values that correspond to locations in train dict.
    train_y: training y data for binary classifer. shape (somenumber,)
             has indicator variable for abnormal. 1 means abonormal

    return: train_x_even_split, train_y_even_split
    """
    np.random.seed(seed)

    # Find normal and abnormal
    abnorm_index = np.where(train_y ==1)[0]
    norm_index = np.where(train_y == 0)[0]

    # Randomify
    rand_norm = np.random.permutation(len(norm_index))
    norm_index = norm_index[rand_norm]
    
    # Apply to data
    train_x_even_split = np.append(train_x[abnorm_index, :],
                                train_x[norm_index,:][:len(abnorm_index),:], # double check
                                  axis=0)
    train_y_even_split = np.append(train_y[abnorm_index],
                                  train_y[norm_index][:len(abnorm_index)])
    
    train={}
    train['x'] = train_x_even_split
    train['y'] = train_y_even_split
    
    # return train_x_even_split, train_y_even_split
    return train