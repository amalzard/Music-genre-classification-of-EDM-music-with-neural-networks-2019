import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tensorflow
import matplotlib.pyplot as plt
from tf_dataprep import returnClassNames

from keras.models import  Sequential
from keras.layers import  *
from keras.optimizers import  *

def createModel(class_names):
	numberOfClasses = len(class_names)
	model = Sequential()

	model.add(InputLayer(input_shape=[128,128,1]))
	model.add(Conv2D(filters=32, kernel_size=5,strides=1,padding='same', activation='relu'))
	model.add(MaxPool2D(pool_size=5, padding='same'))

	model.add(Conv2D(filters=50, kernel_size=5,strides=1,padding='same', activation='relu'))
	model.add(MaxPool2D(pool_size=5, padding='same'))

	model.add(Conv2D(filters=80, kernel_size=5,strides=1,padding='same', activation='relu'))
	model.add(MaxPool2D(pool_size=5, padding='same'))

	model.add(Dropout(0.25))


	model.add(Flatten())
	model.add(Dense(256, activation=tf.nn.relu))
	model.add(Dense(512, activation=tf.nn.relu))
	model.add(Dropout(rate='0.5'))
	model.add(Dense(numberOfClasses, activation='softmax'))



	model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
	
	model.load_weights('fullModel5.h5')

	print("Weights Loaded!")

	return model



