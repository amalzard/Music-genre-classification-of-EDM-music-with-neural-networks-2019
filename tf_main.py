import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tensorflow
import matplotlib.pyplot as plt

from tf_dataprep.py import one_hot_label, train_data_with_label, test_data_with_label

currentPath = os.path.dirname(os.path.realpath(__file__))
train_data = currentPath+'/spectrogramSlices/train'
test_data = currentPath+'/spectrogramSlices/test'

class_names = ['Drum & Bass', 'Glitch', 'Hardcore', 'Hardstyle', 'House', 'No Genre']

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("mode", help="Trains or tests the CNN", nargs='+', choices=["train","test","convert","slice","predict", "createdataset"])
args = parser.parse_args()

if createdataset in args.mode:
	training_images = train_data_with_label()
	testing_images = test_data_with_label()



