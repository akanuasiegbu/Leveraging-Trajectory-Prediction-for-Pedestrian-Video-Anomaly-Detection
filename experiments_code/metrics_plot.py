"""
This will contain more than just metrics plots
"""


from matplotlib import pyplot as plt
from os.path import join
from config import hyparams, loc
from sklearn.metrics import roc_curve, auc
import wandb 
import numpy as np


def generate_metric_plots(test_auc_frame, metric , nc, plot_loc):
    
    print("Number of abnormal people after maxed {}".format(len(np.where(test_auc_frame['y'] == 1 )[0] ) ))

   


    #### Per bounding box
    nc_per_human = nc.copy()
    nc_per_human[0] = loc['nc']['date'] + '_per_bounding_box'
    # y_pred_per_human = iou_as_probability(testdict, model)

    # abnormal_index = np.where(testdict['abnormal_ped_pred'] == 1)
    # normal_index = np.where(testdict['abnormal_ped_pred'] == 0)
    
    abnormal_index = np.where(test_auc_frame['y_pred_per_human'] == 1)[0]
    normal_index = np.where(test_auc_frame['y_pred_per_human'] == 0)[0]
    
    abnormal_index_frame = np.where(test_auc_frame['y'] == 1)[0]
    normal_index_frame = np.where(test_auc_frame['y'] == 0)[0]
    
    if metric == 'iou':
        ylabel = '1-IOU'

    elif metric == 'l2':
        ylabel = 'L2 Error'

    elif metric =='giou':
        ylabel ='giou'

    elif metric =='ciou':
        ylabel = 'ciou'

    elif metric =='diou':
        ylabel = 'diou'

    index = [abnormal_index, normal_index]
    ped_type = ['abnormal_ped', 'normal_ped']
    xlabel = ['Detected Abnormal Pedestrains', 'Detected Normal Pedestrains']
    titles =['Abnormal', 'Normal']


    ##############
    # # DELETE OR MOVE TO A Different place
    index = [abnormal_index_frame, normal_index_frame ]
    xlabel = ['Abnormal Frames', 'Detected Normal Frames']
    ped_type = ['abnormal_ped_frame', 'normal_ped_frame']
    wandb_name = ['rocs', 'roc_curve']
    
    y_true = test_auc_frame['y']
    y_pred = test_auc_frame['x']

    # Uncomment to make iou plots
    ################################################

    for indices, ped_, x_lab, title in zip(index, ped_type, xlabel, titles ):
        plot_iou(   prob_iou = test_auc_frame['x_pred_per_human'][indices],
                    gt_label = test_auc_frame['y_pred_per_human'][indices],
                    xlabel = x_lab,
                    ped_type = ped_,
                    plot_loc = plot_loc,
                    nc = nc_per_human,
                    ylabel = ylabel,
                    title = title
                    )        
    # xlabel = ['Detected Abnormal Pedestrains', 'Detected Normal Pedestrains']
    index = [abnormal_index_frame, normal_index_frame ]
    xlabel = ['Abnormal Frames', 'Detected Normal Frames']
    ped_type = ['abnormal_ped_frame', 'normal_ped_frame']

    for indices, ped_, x_lab, title in zip(index, ped_type, xlabel, titles ):
        plot_iou(   prob_iou = np.sum(test_auc_frame['std_per_frame'][indices], axis = 1),
                    gt_label = test_auc_frame['y'][indices],
                    xlabel = x_lab,
                    ped_type = '{}_std'.format(ped_),
                    plot_loc = plot_loc,
                    nc = nc,
                    ylabel = 'Standard Deviation Summed',
                    title = title
                    )

    for indices, ped_, x_lab, title in zip(index, ped_type, xlabel, titles ):
        for i, axis in zip(range(0,4), ['Mid X', 'Mid Y', 'W', 'H']):
            plot_iou(   prob_iou = test_auc_frame['std_per_frame'][indices][:,i],
                        gt_label = test_auc_frame['y'][indices],
                        xlabel = x_lab,
                        ped_type = '{}_std_axis_{}'.format(ped_, i),
                        plot_loc = plot_loc,
                        nc = nc,
                        ylabel = 'Standard Deviation {}'.format(axis),
                        title = '{}_axis_{}'.format(title, i)
                        )

    for indices, ped_, x_lab, title in zip(index, ped_type, xlabel, titles ):
        plot_iou(   prob_iou = test_auc_frame['std_iou_or_l2_per_frame'][indices],
                    gt_label = test_auc_frame['y'][indices],
                    xlabel = x_lab,
                    ped_type = '{}_std_{}'.format(ped_, hyparams['metric']),
                    plot_loc = plot_loc,
                    nc = nc,
                    ylabel = 'Standard Deviation {}'.format(hyparams['metric']),
                    title = title 
                    )


    for indices, ped_, x_lab, title in zip(index, ped_type, xlabel, titles ):
        plot_iou(   prob_iou = test_auc_frame['x'][indices],
                    gt_label = test_auc_frame['y'][indices],
                    xlabel = x_lab,
                    ped_type = ped_,
                    plot_loc = plot_loc,
                    nc = nc,
                    ylabel = ylabel,
                    title = title
                    )        

                

    
    ###################################################
    # This is where the ROC Curves are plotted 
    y_true = test_auc_frame['y']
    y_pred = test_auc_frame['x']

    # y_true_per_human = testdict['abnormal_ped_pred']
    y_true_per_human = test_auc_frame['y_pred_per_human']
    y_pred_per_human = test_auc_frame['x_pred_per_human']

    roc_plot( y_true_per_human, y_pred_per_human, plot_loc, nc_per_human, wandb_name)
    roc_plot( y_true, y_pred, plot_loc, nc, wandb_name)


def loss_plot(history, plot_loc, nc, save_wandb):
    """
    history:  trained model with details for plots.
    plot_loc: directory to save images for metrics 
    nc: naming convention
    """
    fig,ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(history.history['loss'], '-', 
                    color='black', label='train_loss')

    ax.plot(history.history['val_loss'], '-', 
                    color='red', label='val_loss')
    
    ax.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    img_path = join(    plot_loc, 
                        '{}_loss_{}_{}_{}_{}_{}.jpg'.format(
                        *nc
                        ))
    fig.savefig(img_path)
    # might have a problem if I try saving lstm model loss
    if save_wandb:
        if hyparams['networks']['binary_classifier']['wandb']:    
            wandb.log({"losses": wandb.Image(img_path)})
    

    print('Saving Done for Loss')

def accuracy_plot(history, plot_loc, nc):
    """
    history:  trained model with details for plots
    plot_loc: directory to save images for metrics 
    nc: naming convention
    """
    fig,ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(history.history['accuracy'], '-', 
                    color='black', label='train_acc')

    ax.plot(history.history['val_accuracy'], '-', 
                    color='red', label='val_acc')
    
    ax.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    img_path = join( plot_loc, 
                '{}_acc_{}_{}_{}_{}.jpg'.format(
                *nc
                ))
    fig.savefig(img_path)
    if hyparams['networks']['binary_classifier']['wandb']:    
        wandb.log({"acc": wandb.Image(img_path)})

    print('Saving Done for Acc')


# def roc_plot(model,data, plot_loc, nc, wandb_name):
def roc_plot(y_true, y_pred, plot_loc, nc, wandb_name=None):
    """
    y_true: true y_values
    y_pred: predicted y_values
    plot_loc: directory to save images for metrics 
    nc: naming convention
    wandb_name: string that controls name of files saved
                in wandb

    """
    fig,ax = plt.subplots(nrows=1, ncols=1)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    AUC = auc(fpr, tpr)
    print('AUC is {}'.format(AUC))
    
    ax.plot(fpr, tpr, linewidth=2, label ='AUC = {:.4f}'.format(AUC) )
    ax.plot([0, 1], [0, 1], 'k--')
    ax.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Input Length {} Output Length {}'.format(hyparams['input_seq'], hyparams['pred_seq']))
    # index = np.argsort(y_pred, axis =0)
    # y_true = y_true[index]
    # print(y_pred[index])
    # print(y_true[index])

    # print('sum of first half {}'.format(y_true[:int(len(index) *.5)]))
    # print(fpr)
    # print(tpr)
    # print(thresholds)
    img_path = join( plot_loc, 
                '{}_roc_{}_{}_{}_{}_{}.jpg'.format(
                *nc
                ))
    fig.savefig(img_path)
    
    # if hyparams['networks']['binary_classifier']['wandb']:    
    #     wandb.log({"rocs": wandb.Image(img_path)})
    #     wandb.log({wandb_name[0]: wandb.Image(img_path)})
    
   
        # # For Wandb
        # y_pred_0 = np.array([1-i for i in y_pred]).reshape(-1,1)
        # y_pred = np.append(y_pred_0, y_pred,axis=1)
        # # wromg one wandb.log({"roc_curve" : wandb.plot.roc_curve( data['y'], y_pred, labels=["normal", "abnormal"] ) } )
        # wandb.log({wandb_name[1] : wandb.plot.roc_curve( data['y'], y_pred, labels=["normal", "abnormal"] ) } )





def plot_iou(prob_iou, gt_label, xlabel, ped_type, plot_loc, nc, ylabel, title, split = False):
    """
    envisioned this to show how the abnormal pedestrains iou look
    prob_iou: this is the prob iou
    xlabel: xlabel for plot
    ped_type: 'normal_ped' , abnormal_ped
    """
    fig,ax = plt.subplots(nrows=1, ncols=1)
    
    if split == True:
        abnormal = np.where(gt_label == 1)[0]
        normal = np.where(gt_label == 0)[0]

        ax.plot(np.arange(0, len(abnormal)), prob_iou[abnormal], '.', color = 'r', label ='abnormal')
        ax.plot(np.arange(len(abnormal), len(normal)+len(abnormal)), prob_iou[normal], '.', color = 'g',  label = 'normal' )
    else:
        if 'Abnormal' in title:
            color = 'r'
        elif 'Normal'  in title:
            color = 'g'
        else:
            color = 'k' 
        ax.plot(np.arange(0,len(prob_iou)), prob_iou, '.', color = color )
    if 'Standard Deviation' in ylabel:
        pass
    else:
        ax.plot(np.arange(0,len(prob_iou)), 0.5*np.ones([len(prob_iou),1]), '-b', label='midpoint' )
    ax.plot(np.arange(0,len(prob_iou)), np.mean(prob_iou)*np.ones([ len(prob_iou), 1]), '-k', label='mean = {:.4f}'.format(np.mean(prob_iou)) )
    ax.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('{}_{}_{}'.format(title, hyparams['input_seq'], hyparams['pred_seq']))


    img_path = join( plot_loc, 
            '{}_{}_{}_{}_{}_{}_{}.jpg'.format(
            *nc, ped_type
            ))
    fig.savefig(img_path)




