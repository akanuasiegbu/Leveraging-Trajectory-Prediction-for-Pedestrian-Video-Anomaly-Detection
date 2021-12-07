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


 You can also use docker with 'docker/Dockerfile'
 
 ## BiTrap Data
 BiTrap pkl files can be found [here](https://drive.google.com/drive/folders/1ELYuty5kg-J14jrDH66Gv9rhn58O1t9I?usp=sharing).
 Download pkl file folders for Avenue and ShanghiTech dataset and create a folder called output_bitrap and put both folders inside. 
 Note the name in_3_out_3_K_1 means input trajectory and output trajectory is set to 3. And K=1 means using Bitrap as unimodal.
 
 ## Training
 Users can train their LSTM models on Avenue and ShanghaiTech by using function lstm_train in models.py
 
 For training BiTrap models refer forked repo [here](https://github.com/akanuasiegbu/bidireaction-trajectory-prediction).
 
 Trained models for BiTrap can be found [here](https://drive.google.com/drive/folders/1942GF9FIzoqTVOHyW2Qo86s3R1OOSnsg?usp=sharing) 
 
 
 ## Inference 
 
 
 
 
 ## Citation 
