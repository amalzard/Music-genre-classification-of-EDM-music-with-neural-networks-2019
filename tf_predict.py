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
from tf_model import createModel

currentPath = os.path.dirname(os.path.realpath(__file__))

def predictGenre(model, class_names):
	
	genre_count = {}
	predictionsOver = 0.0
	predicting_images = predict_data_gen()
	for cnt, data in enumerate(predicting_images):
		img = data[0]
		data = img.reshape(1,128,128,1)
		model_out = model.predict([data])
		i = np.argmax(model_out)
		str_label = class_names[i]
		print("{} {:2.0f}%".format(str_label, 100*np.max(model_out)))
		if 100*np.max(model_out) >= 50:
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