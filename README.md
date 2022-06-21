# Leveraging-Trajectory-Prediction-for-Pedestrian-Video-Anomaly-Detection
Asiegbu Miracle Kanu-Asiegbu, Ram Vasudevan, and Xiaoxiao Du 

## Installation 
  * scipy==1.4.1 
  * matplotlib==3.3.1 
  * Pillow==7.2.0 
  * scikit_learn==0.23.2
  * opencv-python==4.4.0.42
  * jupyter 
  * jupyterthemes==0.20.0 
  * hyperas==0.4.1 
  * pandas==1.1.2
  * seaborn==0.11.0
  * tensorflow_addons==0.11.2
  * tensorflow_datasets
  * wandb==0.10.12
  * more_itertools==8.8.0 


 You can also use docker with 'docker/Dockerfile'. Note that I set the PYTHONPATH inside docker file would need to adjust that path
 "ENV PYTHONPATH "/mnt/roahm/users/akanu/projects/anomalous_pred/custom_functions:/home/akanu".

 
 ## Step 1: Download Dataset
 * You use extracted bounding box Avenue and ShanghaiTech trajectoryies.
 * To want to recreate the input bounding box trajecoty 
 * Download [Avenue](http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html) and [ShanghaiTech](https://svip-lab.github.io/dataset/campus_dataset.html) dataset 
   * Use [Deep-SORT-YOLOv4](https://github.com/LeonLok/Deep-SORT-YOLOv4/tree/a4b7d2e1263e6f1af63381a24436c5db5a4b6e91) commit number a4b7d2e
  
 ## Step 2: Training
 We used two two models for our experiments Long Short Term Memory (LSTM) Model and BiTrap model.
 * Users can train their LSTM models on Avenue and ShanghaiTech
   * Training Avenue:  ```python models.py ``` 
     * In config.py change ```hyparams['input_seq'] and hyparams['pred_seq'] to match input/output trajectory length
   * Training ShanghaiTech:  ```python models.py ``` 
     * In config.py change ```hyparams['input_seq'] and hyparams['pred_seq'] to match input/output trajectory length
* Users can train BiTrap models refer to repo[here]
 For training BiTrap models refer forked repo [here](https://github.com/akanuasiegbu/bidireaction-trajectory-prediction).
 
 
 
 ## Step 3: Inference 
##### Pretrained BiTrap Model:
Trained BiTrap models for Avenue and ShanghiTech can be found [here](https://drive.google.com/drive/folders/1942GF9FIzoqTVOHyW2Qo86s3R1OOSnsg?usp=sharing) 

##### Pretrained LSTM Models: 
Trained LSTM models for Avenue and ShanghiTech can be found [here](https://drive.google.com/drive/folders/1kZUxDyCETg7FY_-WeICILeyUYUDJ-nEH)



##### BiTrap Inference
To obtain BiTrap PKL files containing the pedestrain trajectory use commands below.
Test on Avenue dataset:
```
cd bitrap
python tools/test.py --config_file configs/avenue.yml CKPT_DIR **DIR_TO_CKPT**

```

Test on ShanghaiTech dataset:
```
cd bitrap
python tools/test.py --config_file configs/st.yml CKPT_DIR **DIR_TO_CKPT**
```


##### PKL Files
 BiTrap pkl files can be found [here](https://drive.google.com/drive/folders/1ELYuty5kg-J14jrDH66Gv9rhn58O1t9I).
 
 * Download the ```output_bitrap``` folder which contains the pkl file folders for Avenue and ShanghiTech dataset.
 * Naming convention: ```in_3_out_3_K_1``` means input trajectory and output trajectory is set to 3. And K=1 means using Bitrap as unimodal.
 

 
 ## Step 4: AUC Caluation 
 Refer to the main.py for inference. Would need to change the file locations 'train_file', 'test_file', 'pkl_file' in config.py. Note that if running a single input and output, would need to change the input_seq and pred_seq in config.py. However if you want to run multiple experiments at once can look at run_quick function located in main.py as reference. 
 ## Citation 
If you found repo useful, feel free to cite.
```
@INPROCEEDINGS{9660004,
  author={Kanu-Asiegbu, Asiegbu Miracle and Vasudevan, Ram and Du, Xiaoxiao},
  booktitle={2021 IEEE Symposium Series on Computational Intelligence (SSCI)}, 
  title={Leveraging Trajectory Prediction for Pedestrian Video Anomaly Detection}, 
  year={2021},
  volume={},
  number={},
  pages={01-08},
  doi={10.1109/SSCI50451.2021.9660004}}
```
