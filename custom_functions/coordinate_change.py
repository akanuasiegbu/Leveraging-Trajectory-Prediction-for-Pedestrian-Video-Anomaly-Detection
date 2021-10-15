# Convert tlbr to xywh and reverse

import numpy as np

def xywh_tlbr(data):
    """
    Takes a coordinate xywh as input
    Return tlbr as output
    Note that input must be of size (1,4) to use
    """
    try:
        elments = data.astype(float)
        elments[...,0] = elments[...,0] - elments[...,2]/2
        elments[...,1] = elments[...,1] - elments[...,3]/2
        elments[...,2] = elments[...,0] + elments[...,2]
        elments[...,3] = elments[...,1] + elments[...,3]
    except:
        elments = []
        for elem in data:
            elem[...,0] = elem[...,0] - elem[...,2]/2
            elem[...,1] = elem[...,1] - elem[...,3]/2
            elem[...,2] = elem[...,0] + elem[...,2]
            elem[...,3] = elem[...,1] + elem[...,3]
            elments.append(elem)


    return elments

def tlbr_xywh(data):
    """
    Takes a coordinate tlbr as input
    Return xywh as output
    Note that input must be of size (1,4) to use
    Ex. np.array([[10,12,22,20]]) -> [[10, 12, 22, 20]]
    """
    data = data.astype(float)
    data[...,2] = np.abs(data[...,2] - data[...,0] )
    data[...,3] = np.abs(data[...,3] - data[...,1])

    data[...,0] = data[...,0] + data[...,2]/2
    data[...,1] = data[...,1] + data[...,3]/2


    return data
