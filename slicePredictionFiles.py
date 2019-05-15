import os
import sys
from PIL import Image


currentPath = os.path.dirname(os.path.realpath(__file__))
predictionPath=currentPath+"/filesToPredict/"
destPath=currentPath+"/spectrogramSlices/predict/"


def splitPredictionSpectrogram(filename, fileIndex):
	#print(filename)
	img = Image.open(predictionPath + filename)
	width, height = img.size
	nbSamples = int(width / 128)
	for i in range(nbSamples):
		print "Creating slice: ", (i+1), "/", nbSamples, "for", filename
		startPixel = i*128
		imgTmp = img.crop((startPixel, 1, startPixel + 128, 128 + 1))
		imgTmp.save(destPath+"file_{}_{}.png".format(fileIndex, i))
	os.remove(predictionPath + filename)
	return filename




def slicePredictionSpectrograms():
	fileIndex = 0
	track_list = {}
	spectrogramFiles = os.listdir(predictionPath)
	pngFiles = [file for file in spectrogramFiles if file.endswith(".png")]
	nbFiles = len(pngFiles)
	if len(spectrogramFiles) > 0:
		for index, filename in enumerate(pngFiles):
			track_list[fileIndex] = splitPredictionSpectrogram(filename, fileIndex)
			fileIndex = fileIndex + 1
			#os.remove(predictionPath+filename)
		print("Slices Created!")
	else:
		print("No spectrograms to slice")
	return track_list