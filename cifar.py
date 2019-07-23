'''
https://keras.io/examples/cifar10_cnn/
https://keras.io/examples/cifar10_resnet/
'''

from __future__ import print_function
import keras
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
#from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Add
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model
import os

#from network_generator import network


batch_size = 32
num_classes = 100
epochs = 100
data_augmentation = True
num_predictions = 20 #?
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar100_trained_model.h5'

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

inputs = Input(shape=x_train.shape[1:])

def create_tensor(inputs):
    tensor = Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:])(inputs)
    tensor = Activation('relu')(tensor)
    tensor = MaxPooling2D(pool_size=(2, 2))(tensor)
    tensor = Dropout(0.25)(tensor)
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
        #   input_nodes.remove(edge[1])
        if edge[0] in output_nodes:
            output_nodes.remove(edge[0])

    final_node = len(list(h.nodes))
    for output_node in output_nodes:
        h.add_edge(output_node, final_node)

    output = iter_node(h, final_node)
    
    return output

h = network(20, 3)
n_tensor = network_tensor(h)
n_tensor = Flatten()(n_tensor)
n_tensor = Dense(512)(n_tensor)
n_tensor = Activation('relu')(n_tensor)
n_tensor = Dropout(0.5)(n_tensor)
predictions = Dense(num_classes, activation='softmax')(n_tensor)

model = Model(inputs=inputs, outputs=predictions)
'''
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
'''

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        workers=4, steps_per_epoch=100)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])