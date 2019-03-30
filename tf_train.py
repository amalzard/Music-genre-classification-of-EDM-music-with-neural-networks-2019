import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tensorflow
import matplotlib.pyplot as plt
from tensorflow import keras
from collections import defaultdict
from tf_dataprep import genre_label, train_data_gen, test_data_gen, predict_data_gen, returnClassNames
from processPedictionFiles import *
from slicePredictionFiles import splitPredictionSpectrogram, slicePredictionSpectrograms


currentPath = os.path.dirname(os.path.realpath(__file__))
train_data = currentPath+'/spectrogramSlices/train'
test_data = currentPath+'/spectrogramSlices/test'
predict_data = currentPath+'/spectrogramSlices/predict'

class_names = []

from keras.models import  Sequential
from keras.layers import  *
from keras.optimizers import  *

training_images = train_data_gen()
tr_img_data = np.array([i[0] for i in training_images]).reshape(-1,128,128,1)
tr_lbl_data = np.array([i[1] for i in training_images])
print(len(class_names))
testing_images = test_data_gen()
tst_img_data = np.array([i[0] for i in testing_images]).reshape(-1,128,128,1)
tst_lbl_data = np.array([i[1] for i in testing_images])

predicting_images = predict_data_gen()
pred_img_data = np.array([i for i in predicting_images]).reshape(-1,128,128,1)
class_names = returnClassNames()
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

model.load_weights('first_try.h5')

print("Weights Loaded!")

model.fit(x= tr_img_data, y= tr_lbl_data, epochs=10)



model.save_weights('first_try.h5')

print("Weights Saved!")

predictionsOver = 0.0
genre_count = {}



test_loss, test_acc = model.evaluate(tst_img_data, tst_lbl_data)

print('Test accuracy:', test_acc)




