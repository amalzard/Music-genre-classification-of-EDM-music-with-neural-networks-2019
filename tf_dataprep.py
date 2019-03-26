import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tensorflow
import matplotlib.pyplot as plt
from tensorflow import keras
from collections import defaultdict


currentPath = os.path.dirname(os.path.realpath(__file__))
train_data = currentPath+'/spectrogramSlices/train'
test_data = currentPath+'/spectrogramSlices/test'
predict_data = currentPath+'/spectrogramSlices/predict'

class_names = []

def genre_label(img):
	label = img.split('_')[0]
	if label in class_names:
		genre_label = np.array([class_names.index(label)])
	else:
		class_names.append(label)
		genre_label = np.array([class_names.index(label)])
	
	return genre_label

def train_data_gen():
    train_images = []
    for i in tqdm(os.listdir(train_data)):
        path = os.path.join(train_data,i)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        train_images.append([np.array(img), genre_label(i)])
    shuffle(train_images)
    print("Training images:", len(train_images))
    return train_images

def test_data_gen():
    test_images = []
    for i in tqdm(os.listdir(test_data)):
        path = os.path.join(test_data, i)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        test_images.append([np.array(img), genre_label(i)])
    print("Testing images: ", len(test_images))
    return test_images

def predict_data_gen():
    predict_images = []
    for i in tqdm(os.listdir(predict_data)):
        path = os.path.join(predict_data, i)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        predict_images.append([np.array(img)])
    print("Pediction images: ", len(predict_images))
    return predict_images


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
fig = plt.figure(figsize=(2,300))

for cnt, data in enumerate(predicting_images):
    #y = fig.add_subplot(20,20, cnt+1)
    img = data[0]
    data = img.reshape(1,128,128,1)
    model_out = model.predict([data])
    #print(model_out)

    i = np.argmax(model_out)

    if np.argmax(model_out) == i:
        str_label = class_names[i]



    print("{} {:2.0f}%".format(str_label, 100*np.max(model_out)))
    if 100*np.max(model_out) >= 95:
    	predictionsOver += 1.0
    	if str_label in genre_count:
    		genre_count[str_label] += 1
    	else:
    		genre_count[str_label] = 1


    


print(genre_count)
maximum = max(genre_count.values())
percentOfMax = maximum / predictionsOver
percent = percentOfMax * 100
print("{} {:3.0f}%".format(str_label, percent))

test_loss, test_acc = model.evaluate(tst_img_data, tst_lbl_data)

print('Test accuracy:', test_acc)




