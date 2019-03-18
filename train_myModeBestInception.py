import os, sys, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from imgaug import augmenters as iaa
from tqdm import tqdm
from itertools import product

import warnings
warnings.filterwarnings("ignore")

INPUT_SHAPE = (299,299,4)
BATCH_SIZE = 10 


path_to_train = sys.argv[1]#'/mnt/e/ML_dataset/final/Train'
PATH_to_TRAINCSV=sys.argv[2]#"/mnt/e/ML_dataset/final/train.csv"
save_name = sys.argv[0].replace("train_","")
SAVE_NAME = save_name.replace("py","h5")

data = pd.read_csv(PATH_to_TRAINCSV)

train_dataset_info = []
for name, labels in zip(data['Id'], data['Target'].str.split(' ')):
    train_dataset_info.append({
        'path':os.path.join(path_to_train, name),
        'labels':np.array([int(label) for label in labels])})
train_dataset_info = np.array(train_dataset_info)


from sklearn.model_selection import train_test_split
train_ids, test_ids, train_targets, test_target = train_test_split(
    data['Id'], data['Target'], test_size=0.2, random_state=42)


class data_generator:
    
    def create_train(dataset_info, batch_size, shape, augument=True):
        #assert shape[2] == 3
        while True:
            random_indexes = np.random.choice(len(dataset_info), batch_size)
            batch_images = np.empty((batch_size, shape[0], shape[1], shape[2]))
            batch_labels = np.zeros((batch_size, 28))
            for i, idx in enumerate(random_indexes):
                image = data_generator.load_image(
                    dataset_info[idx]['path'], shape)   
                if augument:
                    image = data_generator.augment(image)
                batch_images[i] = image
                batch_labels[i][dataset_info[idx]['labels']] = 1
            yield batch_images, batch_labels
            
    
    def load_image(path, shape):
        R = np.array(Image.open(path+'_red.png'))
        G = np.array(Image.open(path+'_green.png'))
        B = np.array(Image.open(path+'_blue.png'))
        Y = np.array(Image.open(path+'_yellow.png'))

        #image = np.stack((
        #    R/2 + Y/2, 
        #    G/2 + Y/2, 
        #    B),-1)
        image = np.stack((R,G,B,Y),-1)
        image = cv2.resize(image, (shape[0], shape[1]))
        image = np.divide(image, 255)
        return image  
                
            
    def augment(image):
        augment_img = iaa.Sequential([
            iaa.OneOf([
                iaa.Affine(rotate=0),
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
            ])], random_order=True)
        
        image_aug = augment_img.augment_image(image)
        return image_aug
    
    
    



from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.models import Model
from keras.applications import InceptionResNetV2,resnet50
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LambdaCallback
from keras.callbacks import Callback
from keras import metrics
from keras.optimizers import Adam 
from keras import backend as K
import tensorflow as tf
import keras

def create_model(input_shape, n_out):
    '''
    pretrain_model = resnet50.ResNet50(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=(299, 299, 3),
        pooling='avg',
        classes=None)
    '''
    pretrain_model = InceptionResNetV2(
        include_top=False, 
        weights='imagenet',
        input_shape = (299, 299, 3))    
    
    input_tensor = Input(shape=input_shape)
    x = BatchNormalization()(input_tensor)
    x = Conv2D(3, kernel_size=(1,1), activation='relu',input_shape=input_shape)(x)    
    x = pretrain_model(x)
    #x = Conv2D(128, kernel_size=(1,1), activation='relu')(x)
    #x = Flatten()(x)
    x = Dropout(0.25)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.25)(x)
    output = Dense(n_out, activation='sigmoid')(x)
    model = Model(input_tensor, output)
    
    return model

def f1(y_true, y_pred):
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


def show_history(history):
    fig, ax = plt.subplots(1, 3, figsize=(15,5))
    ax[0].set_title('loss')
    ax[0].plot(history.epoch, history.history["loss"], label="Train loss")
    ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax[1].set_title('f1')
    ax[1].plot(history.epoch, history.history["f1"], label="Train f1")
    ax[1].plot(history.epoch, history.history["val_f1"], label="Validation f1")
    ax[2].set_title('acc')
    ax[2].plot(history.epoch, history.history["acc"], label="Train acc")
    ax[2].plot(history.epoch, history.history["val_acc"], label="Validation acc")
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    plt.savefig('History.png')
    

keras.backend.clear_session()

model = create_model(
    input_shape=INPUT_SHAPE, 
    n_out=28)

model.summary()    
    
train_generator = data_generator.create_train(
    train_dataset_info[train_ids.index], BATCH_SIZE, INPUT_SHAPE, augument=True)
validation_generator = data_generator.create_train(
    train_dataset_info[test_ids.index], 256, INPUT_SHAPE, augument=False)

model.layers[3].trainable = True

def GetClassWeight(data):
    pre_Y = data['Target'].str.split().tolist()
    Y = np.zeros((len(pre_Y),28))
    for cc,target in enumerate(pre_Y):
        target = [int(i) for i in target]
        Y[cc,target] = 1
    class_num = np.sum(Y,0)
    class_portion = class_num/np.sum(class_num)
    return -1*np.log2(class_portion)

ClassWeight = GetClassWeight(data)
print(ClassWeight)


pre_Y = data['Target'].str.split().tolist()
Y = np.zeros((len(pre_Y),28))
for cc,target in enumerate(pre_Y):
    target = [int(i) for i in target]
    Y[cc,target] = 1
'''
def calculating_class_weights(y_true):
    from sklearn.utils.class_weight import compute_class_weight
    number_dim = np.shape(y_true)[1]
    weights = np.empty([number_dim, 2])
    for i in range(number_dim):
        weights[i] = compute_class_weight('balanced', [0.,1.], y_true[:, i])
    return weights
ClassWeight = calculating_class_weights(Y)

'''
def get_weighted_loss(weights):
    def weighted_loss(y_true, y_pred):
        return K.mean(weights*K.binary_crossentropy(y_true, y_pred), axis=-1)
    return weighted_loss

loss_fun = get_weighted_loss(ClassWeight)



model.compile(
    loss = loss_fun,#'binary_crossentropy',  
    optimizer=Adam(1e-4),
    metrics=['acc', f1])


check  = ModelCheckpoint(SAVE_NAME,
    monitor='val_f1',
    save_best_only=True,
    mode = 'max')
#"/mnt/e/ML_dataset/final/myModel.h5"
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    validation_data=next(validation_generator),
    epochs=500, 
    verbose=1,
    callbacks=[check],
    )

show_history(history)


#model.save('model1.h5')


