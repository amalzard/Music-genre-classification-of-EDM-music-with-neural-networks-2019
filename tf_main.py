import cv2
import numpy as np
import os
import sys
from random import shuffle
from tqdm import tqdm
import tensorflow as tensorflow
import matplotlib.pyplot as plt

from tf_dataprep import genre_label, train_data_gen, test_data_gen, predict_data_gen, returnClassNames
from processPedictionFiles import createPredictionSpectrogram, convertPredictionFlacToMp3, convertPredictionOogToMp3, renamePredictionMp3Files, convertPredictionMp3ToSpectrogram
from slicePredictionFiles import splitPredictionSpectrogram, slicePredictionSpectrograms
from mp3ToSpectrogram import convertMp3ToSpectrogram
from splitSpectrograms import sliceSpectrograms
from tf_predict import predictGenre
from tf_model import createModel
from tf_train import *

from keras.models import  Sequential
from keras.layers import  *
from keras.optimizers import  *

currentPath = os.path.dirname(os.path.realpath(__file__))
train_data = currentPath+'/spectrogramSlices/train'			#Stores the spectrogram slices for training data
test_data = currentPath+'/spectrogramSlices/test'			#Stores the spectrogram slices for testing data
predict_data = currentPath+'/spectrogramSlices/predict'		#Stores the spectrogram slices of audio files to be predicted


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("mode", help="Trains or tests the CNN", nargs='+', choices=["train","test","convert","slice","predict", "prep"])
args = parser.parse_args()

if "convert" in args.mode:		#Converts the audiofiles to spectrograms
	convertMp3ToSpectrogram()
	sys.exit()

if "slice" in args.mode:		#Slices the spectrogram files for the training and testing datasets
	sliceSpectrograms()
	sys.exit()

if "predict" in args.mode:		#Runs the prediction model
	track_list = convertPredictionMp3ToSpectrogram()
	slicePredictionSpectrograms()
	#print(track_list)
	training_images = train_data_gen()
	class_names = returnClassNames()
	numberOfClasses = len(class_names)
	model = createModel(class_names)

	#print("Weights Loaded!")
	print("------------------------------")
	predictGenre(model, class_names, track_list)
	print("------------------------------")
	sys.exit()

if "prep" in args.mode:		#Prepares the audio files for model prediction
	track_list = convertPredictionMp3ToSpectrogram()
	slicePredictionSpectrograms()
	#print(track_list)
	#training_images = train_data_gen()
	#class_names = returnClassNames()
	#numberOfClasses = len(class_names)
	#model = createModel(class_names)

	#print("Weights Loaded!")
	print("------------------------------")
	#predictGenre(model, class_names, track_list)
	print("Image Slices Created!")
	print("------------------------------")
	sys.exit()

if "train" in args.mode:		#Trains the network model
	
	#predicting_images = predict_data_gen()
	class_names = returnClassNames()
	numberOfClasses = len(class_names)
	#print(class_names)
	trainModel(class_names)
	sys.exit()
