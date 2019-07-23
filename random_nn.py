from tensorflow.keras.layers import Input, Flatten, Dense, MaxPooling1D, Conv1D, Add
#from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
from data_loader import get_some_data
from network_generator import network
import numpy as np


#Map DAG to a NN
'''
https://keras.io/getting-started/functional-api-guide/
https://www.kaggle.com/achukka/conv-layers-with-batch-normalization-in-keras
https://www.kaggle.com/kentaroyoshioka47/cnn-with-batchnormalization-in-keras-94
'''

inputs = Input(shape=(28, 28))

def create_tensor(inputs):
	tensor = Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')(inputs)
	tensor = MaxPooling1D(pool_size=2)(tensor)

	return tensor



def get_input_nodes(h, node):
	result = []
	for edge in list(h.edges):
		if edge[1] is node:
			result.append(edge[0])
	return result

def iter_node(h, node):
	# If it's an input nodes, return the tensor
	input_nodes = get_input_nodes(h, node)
	if len(input_nodes) == 0:
		return create_tensor(inputs)
	else:
		# Otherwise, find its input nodes and aggregate them
		return Add()([iter_node(h, input_node) for input_node in input_nodes])

def network_tensor(h):
	for edge in list(h.edges):
		if edge[0] > edge[1]:
			h.remove_edge(edge[0], edge[1])

	#input_nodes = list(h.nodes)
	output_nodes = list(h.nodes)

	for edge in list(h.edges):
		#if edge[1] in input_nodes:
		#	input_nodes.remove(edge[1])
		if edge[0] in output_nodes:
			output_nodes.remove(edge[0])

	final_node = len(list(h.nodes))
	for output_node in output_nodes:
		h.add_edge(output_node, final_node)

	output = iter_node(h, final_node)
	
	return output


'''
x_1 = Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')(inputs)
x_1 = MaxPooling1D(pool_size=2)(x_1)

x_2 = Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')(inputs)
x_2 = MaxPooling1D(pool_size=2)(x_2)

x = concatenate([x_1, x_2])
'''

h = network(8, 3)
x = network_tensor(h)
x = Flatten()(x)
predictions = Dense(10, activation='softmax')(x)

# This creates a model that includes
# the Input layer and three Dense layers

model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

(x, y) = get_some_data(5000, -1)
x_train = np.reshape(x, (-1, 28, 28)) / 255.0
y_train = np.array(y)

# Conv1D(64, 3, activation='relu')

model.fit(x_train, y_train, epochs=4)  # starts training

(x_, y_) = get_some_data(50, 9)
x_test = np.reshape(x_, (-1, 28, 28)) / 255.0
y_test = np.array(y_)
model.evaluate(x_test, y_test)
