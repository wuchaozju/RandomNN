#python-mnist
from mnist import MNIST
import numpy as np

mndata = MNIST('./data')

#images, labels = mndata.load_training()
images, labels = mndata.load_testing()

def get_some_data(count=60000, s_label=-1):
    s_images = []
    s_labels = []

    i = 0
    for label in labels:
        if s_label == -1 or label == s_label:
            s_images.append(images[i])
            s_labels.append(labels[i])
            i += 1

        if i >= count:
            break

    return s_images, s_labels

#print(len(images))
#print(images[0])

#i, l = get_some_data(50, 3)

#print(i[0])
#print(l)
#print(len(l))

#l_array = np.array(l)
#print(l_array)

'''
i_array = np.reshape(i, (-1, 28, 28))
print(i_array)
print(i_array.shape)
'''
