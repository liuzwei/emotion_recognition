# import the necessary packages
from os import path

# define the base path to the emotion dataset
# 获取绝对路径
BASE_PATH = path.join(path.dirname(__file__), '..', 'datasets')
BASE_PATH = path.abspath(BASE_PATH)

# use the base path to define the path to the input emotions file
INPUT_PATH = path.sep.join([BASE_PATH, "fer2013/fer2013.csv"])

# define the number of classes (set to 6 if you are ignoring the
# "disgust" class)
# NUM_CLASSES = 7
NUM_CLASSES = 6

# define the path to the output training, validation, and testing
# HDF5 files
TRAIN_HDF5 = path.sep.join([BASE_PATH, "fer2013/hdf5/train.hdf5"])
VAL_HDF5 = path.sep.join([BASE_PATH, "fer2013/hdf5/val.hdf5"])
TEST_HDF5 = path.sep.join([BASE_PATH, "fer2013/hdf5/test.hdf5"])

# define the batch size
BATCH_SIZE = 128

# define the path to where output logs will be stored
OUTPUT_PATH = path.join(path.dirname(__file__), '..', 'output')
OUTPUT_PATH = path.abspath(OUTPUT_PATH)