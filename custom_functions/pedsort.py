import numpy as np

def pedsort(datadict):
    """
    This function saves each pedestrain bounding boxes in a video as a key
    To combine with multiple (x) videos would need to call this function x times
    and save result into another dict. Makes function moduluar

    datadict: dict with input keys 'x_ppl_box', 'y_ppl_box', 'frame_ppl_id'
              'video_file', 'abnormal_ped', 'abnormal_gt_frame'. Same keys as loaded in load_data.py
    ped: Returned dict containing ped bounding boxes for unique tracking id's
    """

    ped = {}

    # For each pedestrain want to create a new key
    per_id = np.unique(datadict['frame_ppl_id'][:,-1,1])
    for i in per_id:
        ped[str(i)] = {}
        index = np.where(datadict['frame_ppl_id'][:,-1,1] == i)
        index = index[0][np.argsort(index[0])]
        # should ensure it sorted by frame
        temp = {}
        for key in datadict.keys():
            temp[key] = datadict[key][index]

        ped[str(i)] = temp
    return ped


def incorrect_frame_represent(outframe):

    ped_wrong_represent ={}

    abnormal_frame = np.where(outframe['abnormal_gt_frame_metric'] == 1)[0]

    ped_represent= outframe['abnormal_ped_pred'][abnormal_frame]
    ped_index = np.where(ped_represent == 0)[0]
    abnormal_frame_incorrect_ped_represent = abnormal_frame[ped_index]


    for key in outframe.keys():
        ped_wrong_represent[key] = outframe[key][abnormal_frame_incorrect_ped_represent]

    
    return ped_wrong_represent

    