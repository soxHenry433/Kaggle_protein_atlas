#!/bin/bash



Train_script="train_myModeBestInception.py
train_ResNet50Class.py
train_myModelResNet3_18.py
train_myModelInceptionClassW.py
train_myModelInceptionClassW10.py
train_myModelResClassW01_186.py
train_model1ResNet.py"
CSV_PATH=$1 #"/mnt/e/ML_dataset/final/sample_submission.csv"
FILE_PATH=$2 #'/mnt/e/ML_dataset/final/Test/'
MODEL_PATH=$(pwd)"/"
TrainScore="True"
SAVE_FILE=$4 #"submission.csv"

for i in $Train_script;do
    echo $i
    python3 $i $FILE_PATH $CSV_PATH
done

python3 set_thred.py $CSV_PATH $FILE_PATH $MODEL_PATH $TrainScore $SAVE_FILE


