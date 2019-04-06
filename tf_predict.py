import cv2
import numpy as np
import os
import sys
reload(sys)
from random import shuffle
from tqdm import tqdm
import tensorflow as tensorflow
import matplotlib.pyplot as plt
from tensorflow import keras
from collections import defaultdict
from tf_dataprep import genre_label, train_data_gen, test_data_gen, predict_data_gen, returnClassNames
from tf_model import createModel
sys.setdefaultencoding('utf-8')

currentPath = os.path.dirname(os.path.realpath(__file__))

def predictGenre(model, class_names, track_list):
	numberPredictTracks = len(track_list)
	print("Prediction Results")
	print("(Percent values represent the percentage of 2.8 second chunks that the network has at least 80% confidence is of a genre)")
	print("")
	for x in range(numberPredictTracks):
		predictionsOver = 0.0
		genre_count = {}
		predicting_images = predict_data_gen(track_list, x)
		for cnt, data in enumerate(predicting_images):
			img = data[0]
			data = img.reshape(1,128,128,1)
			model_out = model.predict([data])
			i = np.argmax(model_out)
			str_label = class_names[i]
			if 100*np.max(model_out) >= 80:
				predictionsOver += 1.0
				if str_label in genre_count:
					genre_count[str_label] += 1
				else:
					genre_count[str_label] = 1

		#print(genre_count)
		maximum = max(genre_count.values())
		maxGenre = list(genre_count.keys())[list(genre_count.values()).index(maximum)]
		percentOfMax = maximum / predictionsOver
		percent = percentOfMax * 100
		currentTrack = track_list.get(x)
		currentTrackname = currentTrack.split('.')[0]
		print("{}: {} {:3.0f}%".format(currentTrackname, maxGenre, percent))