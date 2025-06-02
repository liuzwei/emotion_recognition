# USAGE
# python train_recognizer.py --checkpoints fer2013/checkpoints
# python train_recognizer.py --checkpoints fer2013/checkpoints --model fer2013/checkpoints/epoch_20.hdf5 \
#	--start-epoch 20

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from config import emotion_config as config
from network.preprocessing import ImageToArrayPreprocessor
from network.callbacks import EpochCheckpoint
from network.callbacks import TrainingMonitor
from network.io import HDF5DatasetGenerator
from network.nn.conv import EmotionVGGNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import argparse
import os
import tensorflow as tf

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True,
	help="path to output checkpoint directory")
ap.add_argument("-m", "--model", type=str,
	help="path to *specific* model checkpoint to load")
ap.add_argument("-s", "--start-epoch", type=int, default=0,
	help="epoch to restart training at")
args = vars(ap.parse_args())

# construct the training and testing image generators for data
# augmentation, then initialize the image preprocessor
trainAug = ImageDataGenerator(rotation_range=10, zoom_range=0.1,
	horizontal_flip=True, rescale=1 / 255.0, fill_mode="nearest")
valAug = ImageDataGenerator(rescale=1 / 255.0)
iap = ImageToArrayPreprocessor()

# initialize the training and validation dataset generators
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, config.BATCH_SIZE,
	aug=trainAug, preprocessors=[iap], classes=config.NUM_CLASSES)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, config.BATCH_SIZE,
	aug=valAug, preprocessors=[iap], classes=config.NUM_CLASSES)

# Enable eager execution
tf.config.run_functions_eagerly(True)

# if there is no specific model checkpoint supplied, then initialize
# the network and compile the model
if args["model"] is None:
	print("[INFO] compiling model...")
	model = EmotionVGGNet.build(width=48, height=48, depth=1,
		classes=config.NUM_CLASSES)
	opt = Adam(learning_rate=1e-3)
	# opt = SGD(learning_rate=1e-2, momentum=0.9)

	model.compile(loss="categorical_crossentropy", optimizer=opt,
		metrics=["accuracy"])

# otherwise, load the checkpoint from disk
else:
	print("[INFO] loading {}...".format(args["model"]))
    
	model = load_model(args["model"])

    # update the learning rate using tf.keras API
    
	current_lr = model.optimizer.learning_rate.numpy()
	print("[INFO] old learning rate: {}".format(current_lr))
    # Set new learning rate

    # 更换成Adam优化器
	model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=1e-4),
		metrics=["accuracy"])
	# model.optimizer.learning_rate.assign(1e-4)
    
	print("[INFO] new learning rate: {}".format(
        model.optimizer.learning_rate.numpy()))

# construct the set of callbacks
figPath = os.path.sep.join([config.OUTPUT_PATH,
	"vggnet_emotion.png"])
jsonPath = os.path.sep.join([config.OUTPUT_PATH,
	"vggnet_emotion.json"])
callbacks = [
	EpochCheckpoint(args["checkpoints"], every=5,
		startAt=args["start_epoch"]),
	TrainingMonitor(figPath, jsonPath=jsonPath,
		startAt=args["start_epoch"])]
try:
	# train the network
	model.fit(
		trainGen.generator(),
		steps_per_epoch=trainGen.numImages // config.BATCH_SIZE,
		validation_data=valGen.generator(),
		validation_steps=valGen.numImages // config.BATCH_SIZE,
		epochs=20,
		callbacks=callbacks, 
		verbose=1)
finally:
	# close the databases
	trainGen.close()
	valGen.close()