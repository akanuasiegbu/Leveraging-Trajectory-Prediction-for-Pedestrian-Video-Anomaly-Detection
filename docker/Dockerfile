FROM tensorflow/tensorflow:2.4.0rc3-gpu

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
RUN apt-get update &&\
		apt-get install -y wget openssh-server nano tree

RUN systemctl enable ssh
#RUN systemctl start ssh

#apt-get -y install curl g++ nano libglib2.0-0 tree libsm6

WORKDIR /tmp

RUN     pip3 install \
		scipy==1.4.1 \
		matplotlib==3.3.1 \
		Pillow==7.2.0 \
		scikit_learn==0.23.2\
		opencv-python==4.4.0.42\
		jupyter \
		jupyterthemes==0.20.0 \
		hyperas==0.4.1 \
		pandas==1.1.2\
		seaborn==0.11.0\
		tensorflow_addons==0.11.2 \
		tensorflow_datasets\
		wandb==0.10.12\
		more_itertools==8.8.0





# opencv-python dependencies
RUN apt-get install -y libsm6 libxext6 libxrender-dev libgl1-mesa-glx

RUN rm -rf /var/lib/apt/lists/* /tmp/* /root/.cache/*


RUN mkdir /home/akanu
# This allows me to set Python Path 
ENV PYTHONPATH "/mnt/roahm/users/akanu/projects/Projects_to_make_public/Leveraging-Trajectory-Prediction-for-Pedestrian-Video-Anomaly-Detection/custom_functions:/home/akanu"
