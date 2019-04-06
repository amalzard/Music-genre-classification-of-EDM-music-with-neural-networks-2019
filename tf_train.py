import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tensorflow
import matplotlib.pyplot as plt
from tensorflow import keras
from collections import defaultdict
from tf_dataprep import *
from processPedictionFiles import *
from slicePredictionFiles import splitPredictionSpectrogram, slicePredictionSpectrograms
from tf_model import *


from keras.models import  Sequential
from keras.layers import  *
from keras.optimizers import  *


currentPath = os.path.dirname(os.path.realpath(__file__))
train_data = currentPath+'/spectrogramSlices/train'
test_data = currentPath+'/spectrogramSlices/test'
predict_data = currentPath+'/spectrogramSlices/predict'

def trainModel(class_names):

    #training_images = train_data_gen()
    training_images = train_data_gen()
    #testing_images = test_data_gen()
    tr_img_data = np.array([i[0] for i in training_images]).reshape(-1,128,128,1)
    tr_lbl_data = np.array([i[1] for i in training_images])
    #print(training_images)
    #print(len(class_names))
    #testing_images = test_data_gen()
    #tst_img_data = np.array([i[0] for i in testing_images]).reshape(-1,128,128,1)
    #tst_lbl_data = np.array([i[1] for i in testing_images])
    #print(tst_lbl_data)

    #predicting_images = predict_data_gen()
    #pred_img_data = np.array([i for i in predicting_images]).reshape(-1,128,128,1)
    class_names = returnClassNames()
    numberOfClasses = len(class_names)
    
    #print(numberOfClasses)
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
    
    #model.load_weights('fullModel4.h5')

    #print("Weights Loaded!")

    #model = createModel(class_names)

    model.fit(x= tr_img_data, y= tr_lbl_data, epochs=10)



    model.save('fullModel5.h5')

    print("Model Saved!")




    #test_loss, test_acc = model.evaluate(tst_img_data, tst_lbl_data)

    #print('Test accuracy:', test_acc)




