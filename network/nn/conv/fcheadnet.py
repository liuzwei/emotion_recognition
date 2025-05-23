# import the necessary packages
import tensorflow as tf

class FCHeadNet:
	@staticmethod
	def build(baseModel, classes, D):
		# initialize the head model that will be placed on top of
		# the base, then add a FC layer
		headModel = baseModel.output
		headModel = tf.tensorflow.keras.layers.Flatten(name="flatten")(headModel)
		headModel = tf.tensorflow.keras.layers.Dense(D, activation="relu")(headModel)
		headModel = tf.tensorflow.keras.layers.Dropout(0.5)(headModel)

		# add a softmax layer
		headModel = tf.tensorflow.keras.layers.Dense(classes, activation="softmax")(headModel)

		# return the model
		return headModel