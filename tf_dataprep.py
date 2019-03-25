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
#class_count = 0
#class_names = ['Drum & Bass', 'Glitch', 'Hardcore', 'Hardstyle', 'House', 'Electro House', 'Dubstep']

def genre_label(img):
	label = img.split('_')[0]
	if label in class_names:
		genre_label = np.array([class_names.index(label)])
	else:
		class_names.append(label)
		#class_count += 1
		genre_label = np.array([class_names.index(label)])
	
	return genre_label

def train_data_gen():
    train_images = []
    for i in tqdm(os.listdir(train_data)):
        path = os.path.join(train_data,i)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        #img = cv2.resize(img, (64,64))
        train_images.append([np.array(img), genre_label(i)])
    shuffle(train_images)
    print("Training images:", len(train_images))
    return train_images

#def train_label_gen():
	#train_labels = []
	#for i in tqdm(os.listdir(train_data)):
		#train_labels.append(one_hot_label(i))
	#return train_labels

def test_data_gen():
    test_images = []
    for i in tqdm(os.listdir(test_data)):
        path = os.path.join(test_data, i)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        #img = cv2.resize(img, (64, 64))
        test_images.append([np.array(img), genre_label(i)])
    print("Testing images: ", len(test_images))
    return test_images

def predict_data_gen():
    predict_images = []
    for i in tqdm(os.listdir(predict_data)):
        path = os.path.join(predict_data, i)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        #img = cv2.resize(img, (64, 64))
        predict_images.append([np.array(img)])
    print("Pediction images: ", len(predict_images))
    return predict_images

def plot_image(i, predictions_array, img):
  predictions_array, img = predictions_array[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  plt.xlabel("{} {:2.0f}%".format(class_names[predicted_label],
                                100*np.max(predictions_array)))

def plot_value_array(i, predictions_array):
  predictions_array = predictions_array[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')


#def test_label_gen():
	#test_labels = []
	#for i in tqdm(os.listdir(test_data)):
		#test_labels.append(one_hot_label(i))
	#return test_labels

from keras.models import  Sequential
from keras.layers import  *
from keras.optimizers import  *

training_images = train_data_gen()
#training_images = np.array(training_images)
tr_img_data = np.array([i[0] for i in training_images]).reshape(-1,128,128,1)
tr_lbl_data = np.array([i[1] for i in training_images])
print(len(class_names))
#print(training_images.shape)
#training_labels = train_label_gen()
#training_labels = np.array(training_labels)
#print(training_labels)
testing_images = test_data_gen()
#testing_images = np.array(testing_images)
tst_img_data = np.array([i[0] for i in testing_images]).reshape(-1,128,128,1)
tst_lbl_data = np.array([i[1] for i in testing_images])

predicting_images = predict_data_gen()
pred_img_data = np.array([i for i in predicting_images]).reshape(-1,128,128,1)
#testing_labels = test_label_gen()
#testing_labels = np.array(testing_labels)

#plt.figure(figsize=(10,10))
#for i in range(25):
    #plt.subplot(5,5,i+1)
    #plt.xticks([])
    #plt.yticks([])
    #plt.grid(False)
    #plt.imshow(tr_img_data[i], cmap=plt.cm.binary)
    #plt.xlabel(class_names[tr_lbl_data[i]])
#plt.show()
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



    #y.imshow(img, cmap= 'gray')
    #name = str_label + str(cnt) + ".jpg"
    #cv2.imwrite(name,img)
    #plt.title(str_label)
    #print(str_label)
    print("{} {:2.0f}%".format(str_label, 100*np.max(model_out)))
    if 100*np.max(model_out) >= 95:
    	predictionsOver += 1.0
    	if str_label in genre_count:
    		genre_count[str_label] += 1
    	else:
    		genre_count[str_label] = 1


    


    #y.axes.get_xaxis().set_visible(False)
    #y.axes.get_yaxis().set_visible(False)
#plt.show()
# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
# num_rows = 5
# num_cols = 3
# num_images = num_rows*num_cols
# plt.figure(figsize=(2*2*num_cols, 2*num_rows))

# for i in range(num_images):
# 	predicting = np.array(predicting_images)
# 	img = predicting[i]
# 	data = img.reshape(1,128,128,1)
# 	model_out = model.predict([data])
#  	plt.subplot(num_rows, 2*num_cols, 2*i+1)
#  	plot_image(i, model_out, img)
#  	plt.subplot(num_rows, 2*num_cols, 2*i+2)
#  	#plot_value_array(i, model_out)
# plt.show()
print(genre_count)
maximum = max(genre_count.values())
#print(predictionsOver)
#print(maximum)
percentOfMax = maximum / predictionsOver
percent = percentOfMax * 100
print("{} {:3.0f}%".format(str_label, percent))

test_loss, test_acc = model.evaluate(tst_img_data, tst_lbl_data)

print('Test accuracy:', test_acc)




