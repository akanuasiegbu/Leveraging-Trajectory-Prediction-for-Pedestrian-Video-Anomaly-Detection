



def  frame_based_plot():
    """
    Helps make plots for figure
    """
    ped_loc = loc['visual_trajectory_list'].copy()
    frame = 72
    
    vid_key = '01_0073'


    loc_videos = '/mnt/roahm/users/akanu/projects/Deep-SORT-YOLOv4/tensorflow2.0/deep-sort-yolov4/input_video/st_test/{}.avi'.format(vid_key)



    
    
    ped_loc[-1] =  'figures_for_ped_frame_explantion'
    make_dir(ped_loc)
    pic_loc = join(     os.path.dirname(os.getcwd()),
                        *ped_loc
                        )
    gt_boxes_72 = [[460, 102, 496, 195], [100, 108, 142, 204]]
    pred_boxes_72 = [[457.57,100.26, 494.68, 200.66, 0, 0.16],
                  [94.57,113.28, 131.58, 212.4, 0, 0.41]]

    # gt_boxes_200 = [[775, 114, 817, 191], [483, 121, 533, 217], [241, 151, 301, 245]]
    
    # pred_boxes_200 = [  [771.81, 94.96, 810.71,193.9, 0 , 0.37],
    #                     [489.02, 113.96, 529.73, 223.17, 0, 0.27],
    #                     [234.12, 131.83, 276.3, 251.4, 1, 0.55]
    #                     ]

    plot_frame( gt_boxes_72,
                pred_boxes_72, 
                vid_key, 
                pic_loc, 
                loc_videos,
                frame
                )
    
