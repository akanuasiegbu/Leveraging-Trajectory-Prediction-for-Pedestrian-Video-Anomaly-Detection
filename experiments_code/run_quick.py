def run_quick(window_not_one = False):
    """
    window: changes the window size
    """
    
    global max1, min1

    max1 = None
    min1 = None
    # change this to run diff configs
    in_lens = [3,5,13,25]
    out_lens = [3, 5,13,25]
    errors_type = ['error_summed', 'error_flattened']

    for in_len, out_len in zip(in_lens, out_lens):
        hyparams['input_seq'] = in_len
        hyparams['pred_seq'] = out_len
        print('{} {}'.format(hyparams['input_seq'], hyparams['pred_seq']))
        # continue
        if exp['data']=='st' and exp['model_name']=='bitrap':
            if window_not_one:
                pklfile = loc['pkl_file']['st_template_skip'].format(hyparams['input_seq'], hyparams['pred_seq'],exp['K'], hyparams['input_seq'])
            else:
                pklfile = loc['pkl_file']['st_template'].format(hyparams['input_seq'], hyparams['pred_seq'], exp['K'])

        elif exp['data']=='avenue' and exp['model_name']=='bitrap':
            if window_not_one:
                pklfile = loc['pkl_file']['avenue_template_skip'].format(hyparams['input_seq'], hyparams['pred_seq'],exp['K'], hyparams['input_seq'])
                print('I am here window not one')
            else:
                pklfile = loc['pkl_file']['avenue_template'].format(hyparams['input_seq'], hyparams['pred_seq'],exp['K'])

        elif exp['data']=='avenue' and exp['model_name']=='lstm_network':
            if in_len in [3,13,25]:
                modelfile = '/home/akanu/results_all_datasets/experiment_traj_model/saved_model_consecutive/07_05_2021_lstm_network_xywh_avenue_{}_{}.h5'.format(hyparams['input_seq'], hyparams['pred_seq'])
            else:
                modelfile = '/home/akanu/results_all_datasets/experiment_traj_model/saved_model_consecutive/05_18_2021_lstm_network_xywh_avenue_{}_{}.h5'.format(hyparams['input_seq'], hyparams['pred_seq'])
            
        elif exp['data']=='st' and exp['model_name']=='lstm_network':
            modelfile = '/home/akanu/results_all_datasets/experiment_traj_model/saved_model_consecutive/07_05_2021_lstm_network_xywh_st_{}_{}.h5'.format(hyparams['input_seq'], hyparams['pred_seq'])

        if exp['model_name'] == 'lstm_network':
            model = tf.keras.models.load_model(     modelfile,  
                                                    custom_objects = {'loss':'mse'}, 
                                                    compile=True
                                                    )

            traindict, testdict = data_lstm(    loc['data_load'][exp['data']]['train_file'],
                                                loc['data_load'][exp['data']]['test_file'],
                                                hyparams['input_seq'], hyparams['pred_seq'] 
                                                )
            # This sets the max1 and min1
            max1 = traindict['x_ppl_box'].max() if traindict['y_ppl_box'].max() <= traindict['x_ppl_box'].max() else traindict['y_ppl_box'].max()
            min1 = traindict['x_ppl_box'].min() if traindict['y_ppl_box'].min() >= traindict['x_ppl_box'].min() else traindict['y_ppl_box'].min()

            if window_not_one:
                # Changes the window to run
                traindict, testdict = data_lstm(    loc['data_load'][exp['data']]['train_file'],
                                                    loc['data_load'][exp['data']]['test_file'],
                                                    hyparams['input_seq'], hyparams['pred_seq'],
                                                    window = hyparams['input_seq']
                                                    )

        
        elif exp['model_name'] == 'bitrap':
            print(pklfile)                                                                                
            pkldicts = load_pkl(pklfile, exp['data'])
            model = 'bitrap'
        
        # for error in  ['error_diff', 'error_summed', 'error_flattened']:
        for error in errors_type:
            hyparams['errortype'] = error
            auc_metrics_list = []
            print(hyparams['errortype'])
            for metric in ['iou', 'giou', 'l2']:
                hyparams['metric'] = metric
                print(hyparams['metric'])
                if exp['model_name'] == 'bitrap':
                    auc_metrics_list.append(frame_traj_model_auc( 'bitrap', [pkldicts], hyparams['metric'], hyparams['avg_or_max'], exp['model_name']))
                elif exp['model_name'] == 'lstm_network':
                    auc_metrics_list.append(frame_traj_model_auc( [model], [testdict], hyparams['metric'], hyparams['avg_or_max'], exp['model_name']))
            
            path_list = loc['metrics_path_list'].copy()
            path_list.append('{}_{}_in_{}_out_{}_K_{}'.format(loc['nc']['date'], exp['data'], hyparams['input_seq'],
                                                                hyparams['pred_seq'],exp['K'] ))
            joint_txt_file_loc = join( os.path.dirname(os.getcwd()), *path_list )

            print(joint_txt_file_loc)
            auc_together=np.array(auc_metrics_list)


            auc_slash_format = SaveAucTxtTogether(joint_txt_file_loc)
            auc_slash_format.save(auc_together)

            # auc_slash_format = SaveAucTxt(joint_txt_file_loc)




def overlay_roc_curves():
    # Load pkl files 
    in_lens = [3,5,13,25]
    out_lens = [3,5,13,25]
    result_table = {}
    result_table['type']=[]
    result_table['fpr']=[]
    result_table['tpr']=[]
    result_table['auc']=[]
    exp['model_name']=='bitrap'
    global max1, min1
    max1=None
    min1=None
    compare = ['l2', 'l2']
    errortype =['error_summed', 'error_flattened']
    # This is for bitrap model
    for in_len, out_len in zip(in_lens, out_lens):
            hyparams['input_seq'] = in_len
            hyparams['pred_seq'] = out_len
            hyparams['error_type'] = errortype[0]
            print('{} {}'.format(hyparams['input_seq'], hyparams['pred_seq']))
            # continue
            if exp['data']=='st':
                pklfile = loc['pkl_file']['st_template'].format(hyparams['input_seq'], hyparams['pred_seq'], exp['K'])

            elif exp['data']=='avenue':
                pklfile = loc['pkl_file']['avenue_template'].format(hyparams['input_seq'], hyparams['pred_seq'],exp['K'])


            pkldicts = load_pkl(pklfile, exp['data'])

            # Return input to roc curve
            test_auc_frame, _, __, ___ = ped_auc_to_frame_auc_data('bitrap', [pkldicts], compare[0], 'avg', 'bitrap')

            y_true = test_auc_frame['y']
            y_pred = test_auc_frame['x']
            
            # Return tpr and fpr
            fpr, tpr, thresholds = roc_curve(y_true, y_pred)

            AUC = auc(fpr, tpr)

            result_table['type'].append('bitrap_{}_{}'.format(in_len, out_len))
            result_table['fpr'].append(fpr)
            result_table['tpr'].append(tpr)
            # Input to obtain auc result
            result_table['auc'].append(AUC)
            

    #  This is for LSTM Baseline
    for in_len, out_len in zip(in_lens, out_lens):
            hyparams['input_seq'] = in_len
            hyparams['pred_seq'] = out_len
            print('{} {}'.format(hyparams['input_seq'], hyparams['pred_seq']))
            hyparams['error_type'] = errortype[1]
            # continue
            exp['model_name']=='lstm_network'
            if exp['data']=='avenue':
                if in_len == 5 and out_len==5:
                    modelfile = '/home/akanu/results_all_datasets/experiment_traj_model/saved_model_consecutive/05_18_2021_lstm_network_xywh_avenue_{}_{}.h5'.format(hyparams['input_seq'], hyparams['pred_seq'])

                else:
                    modelfile = '/home/akanu/results_all_datasets/experiment_traj_model/saved_model_consecutive/07_05_2021_lstm_network_xywh_avenue_{}_{}.h5'.format(hyparams['input_seq'], hyparams['pred_seq'])
                
            elif exp['data']=='st':
                modelfile = '/home/akanu/results_all_datasets/experiment_traj_model/saved_model_consecutive/07_05_2021_lstm_network_xywh_st_{}_{}.h5'.format(hyparams['input_seq'], hyparams['pred_seq'])

            model = tf.keras.models.load_model(     modelfile,  
                                                    custom_objects = {'loss':'mse'}, 
                                                    compile=True
                                                    )

            traindict, testdict = data_lstm(    loc['data_load'][exp['data']]['train_file'],
                                                loc['data_load'][exp['data']]['test_file'],
                                                hyparams['input_seq'], hyparams['pred_seq'] 
                                                )

            max1 = traindict['x_ppl_box'].max() if traindict['y_ppl_box'].max() <= traindict['x_ppl_box'].max() else traindict['y_ppl_box'].max()
            min1 = traindict['x_ppl_box'].min() if traindict['y_ppl_box'].min() >= traindict['x_ppl_box'].min() else traindict['y_ppl_box'].min()

            # Return input to roc curve
            test_auc_frame, _, __, ___ = ped_auc_to_frame_auc_data([model], [testdict], compare[1], 'avg', 'lstm_network')

            y_true = test_auc_frame['y']
            y_pred = test_auc_frame['x']
            
            # Return tpr and fpr
            fpr, tpr, thresholds = roc_curve(y_true, y_pred)

            AUC = auc(fpr, tpr)

            result_table['type'].append('lstm_{}_{}'.format(in_len, out_len))
            result_table['fpr'].append(fpr)
            result_table['tpr'].append(tpr)
            # Input to obtain auc result
            result_table['auc'].append(AUC)


    fig = plt.figure(figsize=(8,6))
    for i in range(0, len(result_table['auc'])):
        plt.plot(result_table['fpr'][i], 
             result_table['tpr'][i], 
             label="{}, AUC={:.4f}".format(result_table['type'][i], result_table['auc'][i]))

    plt.plot([0,1], [0,1], color='orange', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("False Positive Rate", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)


    plt.title('BiTrap vs LSTM ROC Curve Analysis', fontweight='bold', fontsize=15)
    plt.legend(prop={'size':13}, loc='lower right')

    fig.savefig('roc_plot_{}_{}_{}_{}_{}.jpg'.format(errortype[0], compare[0], errortype[1], compare[1], exp['data']))
    # Load model and load correct dataset format

    # Save output with roc values 

