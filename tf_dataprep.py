import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tensorflow


currentPath = os.path.dirname(os.path.realpath(__file__))
train_data = currentPath+'/spectrogramSlices/train'
test_data = currentPath+'/spectrogramSlices/test'
predict_data = currentPath+'/spectrogramSlices/predict'

class_names = []
genre_count = {}

def genre_label(img):
	label = img.split('_')[0]
	if label in class_names:
		genre_label = np.array([class_names.index(label)])
        genre_count[genre_label] += 1
	else:
		class_names.append(label)
		genre_label = np.array([class_names.index(label)])
        genre_count[genre_label] = 1
	
	return genre_label

def train_data_gen():
    train_images = []
    for i in tqdm(os.listdir(train_data)):
        path = os.path.join(train_data,i)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        train_images.append([np.array(img), genre_label(i)])
    shuffle(train_images)
    print("Training images:", len(train_images))
    print(class_names)
    return train_images

def test_data_gen():
    test_images = []
    for i in tqdm(os.listdir(test_data)):
        path = os.path.join(test_data, i)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        test_images.append([np.array(img), genre_label(i)])
    print("Testing images: ", len(test_images))
    print(class_names)
    return test_images

def predict_data_gen():
    predict_images = []
    for i in tqdm(os.listdir(predict_data)):
        path = os.path.join(predict_data, i)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        predict_images.append([np.array(img)])
    print("Pediction images: ", len(predict_images))
    return predict_images

def returnClassNames():
  return class_names




