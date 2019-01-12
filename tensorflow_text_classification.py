import tensorflow as tf
import keras
import numpy as np

imbd = keras.datasets.imbd

(train_data, train_labels), (test_data, test_labels) = imbd.load_data(num_words=10000)
