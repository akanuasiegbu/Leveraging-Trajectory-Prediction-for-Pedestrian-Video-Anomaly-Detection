"""
Both Coordinate_change.py and custom_metrics.py doing same/similiar

"""
import tensorflow.keras.backend as kb
import numpy as np


def bb_intersection_over_union(y, x):

    xA = kb.max((x[:,0:1],y[:,0:1]), axis=0,keepdims=True)
    yA = kb.max((x[:,1:2],y[:,1:2]), axis=0,keepdims=True)
    xB = kb.min((x[:,2:3],y[:,2:3]), axis=0,keepdims=True)
    yB = kb.min((x[:,3:4],y[:,3:4]), axis=0,keepdims=True)

    interArea1 = kb.max((kb.zeros_like(xB), (xB-xA) ), axis=0, keepdims=True)
    interArea2 = kb.max((kb.zeros_like(xB), (yB-yA) ), axis=0, keepdims=True)
    interArea = interArea1*interArea2
    boxAArea = (x[:,2:3] - x[:,0:1] ) * (x[:,3:4] - x[:,1:2])
    boxBArea = (y[:,2:3] - y[:,0:1] ) * (y[:,3:4] - y[:,1:2])

    iou = interArea / (boxAArea + boxBArea - interArea)
    iou_mean = -kb.mean(iou)
    return iou_mean



def bb_intersection_over_union_np(y, x, mean =False):
    y = y.astype(float)
    x = x.astype(float)
    xA = np.max((x[...,0:1],y[...,0:1]), axis=0,keepdims=True)
    yA = np.max((x[...,1:2],y[...,1:2]), axis=0,keepdims=True)
    xB = np.min((x[...,2:3],y[...,2:3]), axis=0,keepdims=True)
    yB = np.min((x[...,3:4],y[...,3:4]), axis=0,keepdims=True)

    interArea1 = np.max((np.zeros_like(xB), (xB-xA ) ), axis=0, keepdims=True)
    interArea2 = np.max((np.zeros_like(xB), (yB-yA) ), axis=0, keepdims=True)
    interArea = interArea1*interArea2
    boxAArea = (x[...,2:3] - x[...,0:1] ) * (x[...,3:4] - x[...,1:2] )
    boxBArea = (y[...,2:3] - y[...,0:1] ) * (y[...,3:4] - y[...,1:2] )

    iou = interArea / (boxAArea + boxBArea - interArea)

    if mean:
        iou = -np.mean(iou)
    return iou
