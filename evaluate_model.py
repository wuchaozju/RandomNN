import tensorflow as tf
import numpy as np

def print_model(model_file,layer):
	model = tf.keras.models.load_model(model_file)
	layer_weight = model.get_layer(layer).get_weights()
	print(layer_weight)

print_model("trained_final.h5", "softmax_0")
print_model("trained_7.h5", "softmax_7")
