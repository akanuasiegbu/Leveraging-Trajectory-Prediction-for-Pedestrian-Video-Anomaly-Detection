import numpy as np
from custom_metrics import bb_intersection_over_union, bb_intersection_over_union_np
from load_data import norm_train_max_min, load_pkl
from experiments_code.config import hyparams, loc, exp
from coordinate_change import xywh_tlbr, tlbr_xywh


"""
# vectorized and/or used code from this repo
https://github.com/generalized-iou/g-darknet/blob/4f0b31470a60822f4e2cc60bb06126f00aa20657/scripts/iou_utils.py
https://github.com/Zzh-tju/CIoU/blob/606e6c71e370faacb4cac9aecf0442b1f09468a3/layers/modules/multibox_loss.py#L109
"""

def intersection(x,y):
    y = y.astype(float)
    x = x.astype(float)
    top = np.max((x[...,0:1],y[...,0:1]), axis=0,keepdims=True)
    left = np.max((x[...,1:2],y[...,1:2]), axis=0,keepdims=True)
    bottom = np.min((x[...,2:3],y[...,2:3]), axis=0,keepdims=True)
    right = np.min((x[...,3:4],y[...,3:4]), axis=0,keepdims=True)
    height = np.maximum(bottom-top, np.zeros(top.shape))
    width = np.maximum(right-left, np.zeros(top.shape))

    return np.multiply(height,width)

def union(x,y):
    boxAArea = (x[...,2:3] - x[...,0:1] ) * (x[...,3:4] - x[...,1:2] )
    boxBArea = (y[...,2:3] - y[...,0:1] ) * (y[...,3:4] - y[...,1:2] )

    return boxAArea + boxBArea - intersection(x,y)

def c(x,y):

    top = np.min((x[...,0:1],y[...,0:1]), axis=0,keepdims=True)
    left = np.min((x[...,1:2],y[...,1:2]), axis=0,keepdims=True)
    bottom = np.max((x[...,2:3],y[...,2:3]), axis=0,keepdims=True)
    right = np.max((x[...,3:4],y[...,3:4]), axis=0,keepdims=True)
    height = np.maximum(bottom-top, np.zeros(top.shape))
    width = np.maximum(right-left, np.zeros(top.shape))


    return np.multiply(height,width)


def iou(x, y):
    '''
        Need to convert them to tlbr first befor inputting
        input: 2 boxes (x,y)
        output: Itersection/Union
    '''
    U = union(x,y)
    I = intersection(x,y)
    
    return np.divide(I, U, where=U!=0)





def giou(x, y):
    '''
        Need to convert them to tlbr first befor inputting
        input: 2 boxes x,y 
        output: Itersection/Union - (c - U)/c
    '''
    I = intersection(x,y)
    U = union(x,y)
    C = c(x,y)
    iou_term = np.divide(I, U, where=U>0)
    giou_term = np.divide(C-U, C, where=C>0)
    #print("  I: %f, U: %f, C: %f, iou_term: %f, giou_term: %f"%(I,U,C,iou_term,giou_term))
    return iou_term - giou_term

def sigmoid(bbox):
    return 1/(1+np.exp(-bbox))

def ciou(bboxes1, bboxes2):
    

    x = xywh_tlbr(bboxes1)
    y = xywh_tlbr(bboxes2)
    I = intersection(x,y)
    Uni = union(x,y)

    w1 = bboxes1[..., 2:3]
    h1 = bboxes1[..., 3:4]
    w2 = bboxes2[..., 2:3]
    h2 = bboxes2[..., 3:4]
    center_x1 = bboxes1[..., 0:1]
    center_y1 = bboxes1[..., 1:2]
    center_x2 = bboxes2[..., 0:1]
    center_y2 = bboxes2[..., 1:2]

    c_l = np.min((center_x1 - w1 / 2,center_x2 - w2 / 2),  axis=0, keepdims=True)
    c_r = np.max((center_x1 + w1 / 2,center_x2 + w2 / 2),  axis=0, keepdims=True)
    c_t = np.min((center_y1 - h1 / 2,center_y2 - h2 / 2),  axis=0, keepdims=True)
    c_b = np.max((center_y1 + h1 / 2,center_y2 + h2 / 2),  axis=0, keepdims=True)

    inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2
    c_diag = np.clip((c_r - c_l),a_min=0, a_max =None)**2 + np.clip((c_b - c_t),a_min=0, a_max =None)**2

    
    u_term = inter_diag.squeeze()/ c_diag.squeeze()
    iou_term = np.divide(I, Uni, where=Uni>0)
    v = (4 / (np.pi ** 2)) * ((np.arctan(w2 / h2) - np.arctan(w1 / h1))**2)

    S = iou_term>0.5 # the ones greater than 0.5 are the trues which are 1
    alpha= S*v/(1-iou_term+v)

    # using the metric and not the loss
    cious = iou_term.squeeze() - u_term.squeeze() - alpha.squeeze() * v.squeeze()

    return cious


def diou(bboxes1, bboxes2):
    
    x = xywh_tlbr(bboxes1)
    y = xywh_tlbr(bboxes2)
    I = intersection(x,y)
    Uni = union(x,y)

    w1 = bboxes1[..., 2:3]
    h1 = bboxes1[..., 3:4]
    w2 = bboxes2[..., 2:3]
    h2 = bboxes2[..., 3:4]
    center_x1 = bboxes1[..., 0:1]
    center_y1 = bboxes1[..., 1:2]
    center_x2 = bboxes2[..., 0:1]
    center_y2 = bboxes2[..., 1:2]

    c_l = np.min((center_x1 - w1 / 2,center_x2 - w2 / 2),  axis=0, keepdims=True)
    c_r = np.max((center_x1 + w1 / 2,center_x2 + w2 / 2),  axis=0, keepdims=True)
    c_t = np.min((center_y1 - h1 / 2,center_y2 - h2 / 2),  axis=0, keepdims=True)
    c_b = np.max((center_y1 + h1 / 2,center_y2 + h2 / 2),  axis=0, keepdims=True)

    inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2
    c_diag = np.clip((c_r - c_l),a_min=0, a_max =None)**2 + np.clip((c_b - c_t),a_min=0, a_max =None)**2

    
    u_term = inter_diag.squeeze()/ c_diag.squeeze()
    iou_term = np.divide(I, Uni, where=Uni>0)


    # using the metric and not the loss                                                                                                                                                                                                                                                                                                                                                                                             
    dious = iou_term.squeeze() - u_term.squeeze() 

    return dious



if __name__ == '__main__':
    pkldicts = load_pkl(loc['pkl_file']['avenue'])
    # gt =xywh_tlbr(pkldicts['y_ppl_box'][0:1])
    # pred = xywh_tlbr(pkldicts['pred_trajs'][0:1])

    gt = pkldicts['y_ppl_box']
    pred = pkldicts['pred_trajs']
    # gt[0][0] = np.array([477, 114, 511,237])
    # pred[0][0] = np.array([477.41, 112.95, 512.34,237.22])
    gt[0][0] = np.array([494, 175.5, 34, 123])
    pred[0][0] = np.array([494.875, 175.085, 34.93, 124.27])

    ciou(gt, pred)
    # gious = giou(gt, pred)
    # ious = bb_intersection_over_union_np(gt, pred)
    # print(ious)

