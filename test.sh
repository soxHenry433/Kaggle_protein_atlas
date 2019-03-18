#!/bin/bash


wget -O model1ResNet.h5 https://www.dropbox.com/s/ipu96pzqsg36jru/model1ResNet.h5?dl=1
wget -O myModeBestInceptionl.h5 https://www.dropbox.com/s/29x833k0f4530jl/myModeBestInceptionl.h5?dl=1
wget -O myModelInceptionClassW.h5 https://www.dropbox.com/s/7evd39ve495srqi/myModelInceptionClassW.h5?dl=1
wget -O myModelInceptionClassW10.h5 https://www.dropbox.com/s/kyt9tjqxyx45go9/myModelInceptionClassW10.h5?dl=1
wget -O myModelResNet3_18.h5 https://www.dropbox.com/s/daje80galql2ftg/myModelResNet3_18.h5?dl=1
wget -O ResNet50Class.h5 https://www.dropbox.com/s/budhpqs4bl0kql6/ResNet50Class.h5?dl=1
wget -O myModelResClassW01_186.h5 https://www.dropbox.com/s/5mp9diffli68adg/myModelResClassW01_186.h5?dl=1

CSV_PATH=$1 #"/mnt/e/ML_dataset/final/train.csv"
FILE_PATH=$2 #'/mnt/e/ML_dataset/final/Train'
MODEL_PATH=$(pwd)"/"
TrainScore="False"
SAVE_FILE=$3 #"submission.csv"

python3 set_thred.py $CSV_PATH $FILE_PATH $MODEL_PATH $TrainScore $SAVE_FILE

