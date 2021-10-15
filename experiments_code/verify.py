"""
Intend this to be a hodgepodge of functions used to verify functionality
Will move to other locations if I end up using them alot
"""

import numpy as np
from config import hyparams, loc, exp
from matplotlib import pyplot as plt
from os.path import join


def order_abnormal( prob_iou, gt_label, ylabel, plot_loc, nc):

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols =1, figsize=(20,20))
    index = np.argsort(prob_iou, axis = 0)

    ax1.plot(np.arange(0, len(index)), np.squeeze(gt_label[index]), '+', ms =0.8)
    ax1.set_xlabel('Frames')
    ax1.set_ylabel('Abnormal Indicator')
    ax1.set_title('{}_{}_{}'.format(hyparams['metric'], hyparams['input_seq'], hyparams['pred_seq']))


    ax2.plot(np.arange(0, len(index)), np.squeeze(prob_iou[index]), '.', ms =1)
    ax2.set_xlabel('Frames')
    ax2.set_ylabel(ylabel)
    ax2.set_title('{}_{}_{}'.format(hyparams['metric'], hyparams['input_seq'], hyparams['pred_seq']))
    plt.tight_layout()


    img_path = join( plot_loc, 
            '{}_{}_{}_{}_{}_{}_ordered.jpg'.format(
            *nc
            ))
    fig.savefig(img_path)



