

def pkl_seq_ind(data, frame, vid, idy):

    """
    This search function finds a pedestrain in the pkl format
    """
    vid_loc = data['vid']
    frame_loc = data['frame']
    id_y = data['id_y']

    vid_frame_id_y = []
    for i,j,k in zip(vid_loc, frame_loc, id_y):
        vid_frame_id_y.append( str(i[0]) + '_' + str(j[0])  + '_' + str(k[0]))

    i = 0
    find = '{}.txt'.format(vid) + '_' + '{}'.format(frame) + '_' + '{}'.format(idy)
    for here in vid_frame_id_y:
        if here == find:
            found_index = i 
        else:
            i += 1
    output = {}
    for key in data.keys():
        output[key] = data[key][found_index]

    return output


def return_metrics(gt_box, pred_box, errortype):

    prob_iou = iou_as_probability(testdicts, model, errortype = hyparams['errortype'], max1 = max1, min1= min1)
    prob_l2 = l2_error(testdicts = testdicts, models = model, errortype =hyparams['errortype'], max1=max1,min1= min1)
    prob = giou_ciou_diou_as_metric(testdicts = testdicts, models = model, metric=metric,errortype = hyparams['errortype'], max1 = max1,min1 = min1)


    