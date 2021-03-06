import os, sys, time
import tensorflow as tf
from tensorflow import keras

from config import hyparams, exp, loc
from tensorflow.keras.layers import Lambda
from data import data_lstm, tensorify
import wandb 
from wandb.keras import WandbCallback
from load_data import norm_train_max_min
from sklearn.model_selection import train_test_split

from os.path import join
from custom_functions.utils import make_dir
from metrics_plot import loss_plot

# To Do List
## Import time and make

def lstm_train(traindict, max1, min1):
    """
    All this is doing is training the lstm network.
    After training make plots to see results. (Make a plotter class or functions)
    """
    train_x,train_y = norm_train_max_min(   data = traindict,
                                            # max1=hyparams['max'],
                                            # min1=hyparams['min']
                                            max1 = max1,
                                            min1 = min1
                                        )
    
    train, val = {}, {}
    train['x'], val['x'],train['y'],val['y'] = train_test_split(    train_x,
                                                                    train_y,
                                                                    test_size = hyparams['networks']['lstm']['val_ratio']
                                                                    )
    
    # train test split in tensorify function
    train_data,val_data = tensorify(    train, 
                                        val,
                                        batch_size = hyparams['batch_size']
                                        )

    #naming convention
    nc = [  loc['nc']['date'],
            loc['nc']['model_name'],
            loc['nc']['data_coordinate_out'],
            loc['nc']['dataset_name'],
            hyparams['input_seq'],
            hyparams['pred_seq']
            ] # Note that frames is the sequence input

    # folders not saved by dates
    make_dir(loc['model_path_list']) # Make directory to save model

    # create save link
    model_loc = join(   os.path.dirname(os.getcwd()),
                        *loc['model_path_list']
                        ) 

   

    history, model = lstm_network(  train_data,
                                    val_data,
                                    model_loc=model_loc, 
                                    nc = nc,
                                    epochs=hyparams['epochs']
                                    )

    make_dir(loc['metrics_path_list'])
    plot_loc = join(    os.path.dirname(os.getcwd()),
                        *loc['metrics_path_list']
                        )
    # loss plot is saved to plot_loc
    loss_plot(history, plot_loc, nc, save_wandb=False)

    return model


def lstm_network(train_data, val_data, model_loc, nc,  epochs=300):
    """
    train_data: train_data_tensor
    val_data: validation_data tensor
    model_loc : location to save models too
    nc: naming convention. list that contains [model, type,dataset,seq]
        model: lstm
        type: xywh, tlbr
        dataset: ped1,ped2,st, avenue
        seq: size of sequence, 20, 5 etc int of sequence
        example: lstm_xywh_ped1_20.h5

    """
    with tf.device('/device:GPU:0'):
        lstm_20 = keras.Sequential()
        lstm_20.add(keras.layers.InputLayer(
            input_shape=(hyparams['input_seq'], 4)))
        lstm_20.add(keras.layers.LSTM(4, return_sequences=True))
        lstm_20.add(keras.layers.LSTM(3, return_sequences=True))
        lstm_20.add(keras.layers.LSTM(6, return_sequences=True))
        lstm_20.add(keras.layers.LSTM(4, return_sequences=True))
        lstm_20.add(keras.layers.LSTM(4, return_sequences=True))
        lstm_20.add(keras.layers.LSTM(4))
        lstm_20.add(keras.layers.Dense(hyparams['pred_seq']*4))
        opt = keras.optimizers.Adam(learning_rate=hyparams['networks']['lstm']['lr'])
        checkpoint_cb = keras.callbacks.ModelCheckpoint(os.path.join(model_loc, '{}_{}_{}_{}_{}_{}.h5'.format(*nc)),
                                                        save_best_only=True)

        if hyparams['networks']['lstm']['early_stopping'] == True:
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='loss',
                min_delta=hyparams['networks']['lstm']['min_delta'],
                patience=hyparams['networks']['lstm']['patience'])

            cb = [early_stopping, checkpoint_cb]
        else:
            cb = [checkpoint_cb]

        # if use iou metric need to conver to tlbr
        lstm_20.compile(optimizer=opt, loss=hyparams['networks']['lstm']['loss'])
        lstm_20_history = lstm_20.fit(train_data,
                                      validation_data=val_data,
                                      epochs=epochs,
                                      callbacks=cb)
        return lstm_20_history , lstm_20

def custom_loss(weight_ratio):
    """
    Note that weight_ratio is postive/negative
    """
    def loss(y_true, y_pred):
        when_y_1 = y_true*tf.keras.backend.log(y_pred)*(1/weight_ratio)
        # when_y_1 = y_true*tf.keras.backend.log(y_pred)*(1/1)
        neg_y_pred = Lambda(lambda x: -x)(y_pred)
        when_y_0 = ( 1+Lambda(lambda x: -x)(y_true))*tf.keras.backend.log(1+neg_y_pred )

        weighted_cross_entr = Lambda(lambda x: -x)(when_y_0+when_y_1)
        return weighted_cross_entr
    return loss


if __name__ == '__main__':
    traindict, testdict = data_lstm(    loc['data_load'][exp['data']]['train_file'],
                                loc['data_load'][exp['data']]['test_file'],
                                hyparams['input_seq'], hyparams['pred_seq']
                                , 
                                )

    max1 = traindict['x_ppl_box'].max() if traindict['y_ppl_box'].max() <= traindict['x_ppl_box'].max() else traindict['y_ppl_box'].max()
    min1 = traindict['x_ppl_box'].min() if traindict['y_ppl_box'].min() >= traindict['x_ppl_box'].min() else traindict['y_ppl_box'].min()
    lstm_model = lstm_train(traindict, max1, min1)
