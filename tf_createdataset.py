from keras.preprocessing.image import ImageDataGenerator
import os

currentPath = os.path.dirname(os.path.realpath(__file__))
trainPath=currentPath+"/spectrogramSlices/train/"
testPath=currentPath+"/spectrogramSlices/test/"
dataPath=currentPath+"/Data/"

trainDatagen = ImageDataGenerator()

trainGenerator = trainDatagen.flow_from_directory(
	trainPath,
	target_size=(128, 128),
	color_mode="grayscale",
	class_mode='categorical',
	save_to_dir=dataPath,
	save_prefix='')
next(trainGenerator)