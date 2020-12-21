import keras 
import numpy as np 
import pandas as pd

import tensorflowjs as tfjs


data = pd.read_csv('/Volumes/TOSHIBA EXT 2/ML/MnistClassifire/mnist_train.csv', header = 0)
#arr = np.array(data.iloc[0])
arr = np.array(data.iloc[0:,0])
inputs = np.array(data.iloc[:,1:785], dtype = float)

'''for i in range(len(inputs)):
	for j in range(len(inputs[0])):
		inputs[i][j] = inputs[i][j] / 255.0'''

inputs /= 255

inputs = np.resize(inputs, (59999, 28, 28, 1))

outputs = np.zeros(shape=(59999, 10))
for i in range(len(arr)):
	output = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
	ele = arr[i]
	output[ele] = 1
	outputs[i] = output



model = keras.models.Sequential()

# number of convolutional filters
n_filters = 64

# convolution filter size
# i.e. we will use a n_conv x n_conv filter
n_conv = 3

# pooling window size
# i.e. we will use a n_pool x n_pool pooling window
n_pool = 2


#1st conv layer
model.add(keras.layers.Convolution2D(
		n_filters,
		kernel_size=(n_conv, n_conv),
		# we have a 28x28 single channel (grayscale) image
        # so the input shape should be (28, 28, 1)
        input_shape=(28, 28, 1)

	))
model.add(keras.layers.Activation('relu'))

#2nd conv layer
model.add(keras.layers.Convolution2D(n_filters, kernel_size=(n_conv, n_conv)))
model.add(keras.layers.Activation('relu'))

# then we apply pooling to summarize the features
# extracted thus far
model.add(keras.layers.convolutional.MaxPooling2D(pool_size=(n_pool, n_pool)))

# flatten the data for the 1D layers
model.add(keras.layers.Flatten())

#dense layers
#layer1 
model.add(keras.layers.Dense(256))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dropout(0.5))

#output layer
# the softmax output layer gives us a probablity for each class
model.add(keras.layers.Dense(10))
model.add(keras.layers.Activation('softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(inputs, outputs, epochs = 10, shuffle = True)

model.save('ConvMnistModel.h5')

#model = keras.models.load_model('ConvMnistModel.h5')


#tfjs.converters.save_keras_model(model, '/Users/makarandsubhashlahane/Desktop/Projects/JavaScript/Tensorflow.jsProjects/MnistClassifireTesting/tfjsMnistCNNModel')











