# USAGE
# python test_recognizer.py --model fer2013/checkpoints/epoch_75.hdf5

# import the necessary packages
from config import emotion_config as config
from network.preprocessing import ImageToArrayPreprocessor
from network.io import HDF5DatasetGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str,
	help="path to model checkpoint to load")
args = vars(ap.parse_args())

# initialize the testing data generator and image preprocessor
testAug = ImageDataGenerator(rescale=1 / 255.0)
iap = ImageToArrayPreprocessor()

# initialize the testing dataset generator
testGen = HDF5DatasetGenerator(config.TEST_HDF5, config.BATCH_SIZE,
	aug=testAug, preprocessors=[iap], classes=config.NUM_CLASSES)

# load the model from disk
print("[INFO] loading {}...".format(args["model"]))
model = load_model(args["model"])

# evaluate the network
(loss, acc) = model.evaluate_generator(
	testGen.generator(),
	steps=testGen.numImages // config.BATCH_SIZE,
	max_queue_size=10)
print("[INFO] accuracy: {:.2f}".format(acc * 100))

# close the testing database
testGen.close()