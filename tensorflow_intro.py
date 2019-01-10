import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


print(train_images.shape)
print(len(train_labels))

# show the image of an image
plt.interactive(False)
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show(block=True)

# pre-process data
train_images = train_images / 255.0
test_images = test_images / 255.0

# display first 25 images to ensure data is pre-processed correctly
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
    plt.show(block = True)

# model the data
model = keras.Sequential([
    # This changes the data from a 2D array to a 1D array
    keras.layers.Flatten(input_shape=(28, 28)),
    # creates layer of 128 neurons
    keras.layers.Dense(128, activation=tf.nn.relu),
    # used for percent chance that the image is classified as each of the 10 categories
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# compile the model
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# fit the data
model.fit(train_images, train_labels, epochs=5)

# Test accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# Make Predictions
predictions = model.predict(test_images)
print(predictions[0])

