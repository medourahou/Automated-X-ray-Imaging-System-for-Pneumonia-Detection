#!/usr/bin/env python
# coding: utf-8

# <h2>1. Import libraries</h2>

# In[2]:


import sys
import os
import argparse

import random

import time
import datetime

from collections import Counter

import numpy as np
import pandas as pd

from keras import models
from keras.models import Model
from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, GlobalAveragePooling1D, GlobalAveragePooling2D, Flatten, BatchNormalization, Dense
import keras
from keras.utils import np_utils
#from sklearn.utils import class_weight

from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf


import cv2


import shutil
from tqdm import tqdm

import inspect
import gc

import re

from PIL import Image


from keras.applications.inception_v3 import InceptionV3


#for data augmentation
from keras.preprocessing.image import ImageDataGenerator

from keras.constraints import maxnorm


from keras import optimizers
from keras.optimizers import Adam, SGD , RMSprop


from keras import backend as K
K.set_image_data_format('channels_first')

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau

from keras.wrappers.scikit_learn import KerasClassifier


from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report




from IPython.display import display

import seaborn as sns

from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# <h2>2. Read Data and Data augmentation</h2>

# In[6]:



#count the number of images in the dirictory
def count_files(directory):
    return sum([len(files) for r, d, files in os.walk(directory)])



#paths to data
train_data_dir = r"data/train"
val_data_dir = r"data/val"
test_data_dir = r"data/test"


#scale images
rescale = 1./255

#some model parameters 
target_size = (150, 150)
batch_size = 163
class_mode = "categorical"


#=============================== data augmentation========================
gen_train = ImageDataGenerator(
    rescale=rescale,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)


train_augmented = gen_train.flow_from_directory(
    train_data_dir,
    target_size=target_size,
    class_mode=class_mode,
    batch_size=batch_size,
    shuffle=True)


gen_val= ImageDataGenerator(rescale=rescale)

validation_augmented = gen_val.flow_from_directory(
    val_data_dir,
    target_size=target_size,
    class_mode=class_mode,
    batch_size=count_files(val_data_dir),
    shuffle = False)


gen_test = ImageDataGenerator(rescale=rescale)

test_augmented = gen_test.flow_from_directory(
    test_data_dir,
    target_size=target_size,
    class_mode=class_mode,
    batch_size=count_files(test_data_dir),
    shuffle = False)


# In[7]:





# In[8]:





# In[14]:


#=========================== callbacks=================


early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=100,
    verbose=1,
    restore_best_weights=True)


# In[10]:


class_weights = compute_class_weight(
                                        class_weight = "balanced",
                                        classes = np.unique(train_augmented.classes),
                                        y = train_augmented.classes                                                   
                                    )
class_weight = dict(zip(np.unique(train_augmented.classes), class_weights))
class_weight


# <h2>3. The CNN model</h2>

# In[15]:




model = Sequential()
model.add(Conv2D(16, (3, 3), activation='relu', padding="same", input_shape=(3,150,150)))
model.add(Conv2D(16, (3, 3), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=(3,150,150)))
model.add(Conv2D(32, (3, 3), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(96, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same"))
model.add(Conv2D(96, (3, 3), padding="valid", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same"))
model.add(Conv2D(128, (3, 3), padding="valid", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(64, activation='relu'))
#model.add(Dropout(0.4))
model.add(Dense(2 , activation='softmax'))


model.compile(optimizers.Adam(lr = 0.001), loss='categorical_crossentropy', metrics=['accuracy'])


steps_per_epoch=len(test_augmented)
validation_steps=len(validation_augmented)


history = model.fit_generator(
    train_augmented,
    steps_per_epoch = steps_per_epoch,
    epochs=100,
    verbose=2,
    callbacks=[early_stopping],
    validation_data=validation_augmented,
    validation_steps=validation_steps, 
    class_weight=class_weight)


# In[17]:


#history.history['accuracy']

result  = model.evaluate_generator(test_augmented, steps=len(test_augmented), verbose=1)


# <h2>4. VGG16</h2>

# In[ ]:


# Load and configure model InceptionV3 for fine-tuning with new class labels
inputs = keras.Input(shape=(3, 224, 224))
input_shape = (3, 224, 224)   
base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
x = base_model(inputs, training=False)

out = base_model.output


results = Dense(2, activation='softmax')(x) 
    
    
TFmodel = Model(inputs= inputs, outputs=results)
    
    
for layer in base_model.layers:
        layer.trainable = False
        
        
    
TFmodel.compile(optimizers.Adam(lr = 0.001), loss='categorical_crossentropy', metrics=['accuracy'])


steps_per_epoch=len(test_augmented)
validation_steps=len(validation_augmented)


history = TFmodel.fit_generator(
    train_augmented,
    steps_per_epoch = steps_per_epoch,
    epochs=100,
    verbose=1,
    callbacks=[early_stopping],
    validation_data=validation_augmented,
    validation_steps=validation_steps,
    class_weight=class_weight)
        
 

