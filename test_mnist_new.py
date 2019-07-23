from __future__ import division
import tensorflow as tf
import numpy as np
from data_loader import get_some_data


#(x_train, y_train),(x_test, y_test) = mnist.load_data()
#x_train, x_test = x_train / 255.0, x_test / 255.0
#print(x_train)

def train_model_init():
    model = tf.keras.models.Sequential([
          tf.keras.layers.Flatten(input_shape=(28, 28), name="flatten"),
          tf.keras.layers.Dense(512, activation=tf.nn.relu, name="dense"),
          #tf.keras.layers.Dropout(0.2, name="dropout"),
          tf.keras.layers.Dense(10, activation=tf.nn.softmax, name="softmax")
          ])

    model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    (x, y) = get_some_data(20, -1)
    x_train = np.reshape(x, (-1, 28, 28)) / 255.0
    y_train = np.array(y)

    model.fit(x_train, y_train, epochs=1)
    #model.evaluate(x_test, y_test)
    model.save("trained_init.h5")

def train_model_distributed(model_file):
    #model = tf.keras.models.load_model("trained_init.h5")
    # Train 10 models for 10 numbers
    for i in range(0, 10):
        model = tf.keras.models.load_model(model_file)
        (x, y) = get_some_data(500, i)
        x_train = np.reshape(x, (-1, 28, 28)) / 255.0
        y_train = np.array(y)
        model.fit(x_train, y_train, epochs=2)
        model.save("trained_new_" + str(i) + ".h5")
            

def test_model():
    # Model for number '0'
    model = tf.keras.models.load_model("trained_final_new.h5")
    #model = tf.keras.models.load_model("trained_9.h5")

    # Test for same number
    (x, y) = get_some_data(500, 0)
    x_test = np.reshape(x, (-1, 28, 28)) / 255.0
    y_test = np.array(y)
    model.evaluate(x_test, y_test)

    
    # Test for a different number 
    (x, y) = get_some_data(500, 1)
    x_test = np.reshape(x, (-1, 28, 28)) / 255.0
    y_test = np.array(y)
    model.evaluate(x_test, y_test)
    
    # Test for another different number 
    (x, y) = get_some_data(500, 6)
    x_test = np.reshape(x, (-1, 28, 28)) / 255.0
    y_test = np.array(y)
    model.evaluate(x_test, y_test)
    

def avg_models(layers):
    #models = []
    layer_weights = {}
    
    for i in range(0, 10):
        model = tf.keras.models.load_model("trained_new_" + str(i) + ".h5")

        for layer in layers:
            layer_weight = model.get_layer(layer).get_weights()
            
            if layer in layer_weights:
                temp_w = [x * i for x in layer_weights[layer]]

                temp_w = [temp_w[k] + layer_weight[k] for k in range(len(temp_w))] 
                
                layer_weights[layer] = [x / (i + 1) for x in temp_w]
            else:
                layer_weights[layer] = layer_weight
        
            
        #models.append(model)

        #dense_1 = my_model.get_layer('dense_1')
        #w_dense1 = dense_1.get_weights()
        #dense3.set_weights(w_dense1)

    final_model = tf.keras.models.load_model("trained_new_0.h5")
    
    
    for layer in layers:
        layer_ = final_model.get_layer(layer)
        layer_.set_weights(layer_weights[layer])
    

    '''
    fix_layer = final_model.get_layer("dense_0")
    fix_layer.trainable = False
    '''
    '''
    final_model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
    

    (x, y) = get_some_data(20000, -1)
    x_train = np.reshape(x, (-1, 28, 28))
    y_train = np.array(y)
    final_model.fit(x_train, y_train, epochs=10)
    '''
        
    final_model.save("trained_final_new.h5")


train_model_init()
train_model_distributed("trained_init.h5")
avg_models(["dense", "softmax"])
test_model()

'''
my_model = tf.keras.models.load_model("trained.h5")
dense_1 = my_model.get_layer('dense_1')
w_dense1 = dense_1.get_weights()
#dense3.set_weights(w_dense1)
print(w_dense1)
'''

'''
model_json = my_model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
'''

#my_model.load_weights("trained.h5", by_name=True)

#my_model.evaluate(x_test, y_test)

# CIFAR: http://www.cs.toronto.edu/~kriz/cifar.html