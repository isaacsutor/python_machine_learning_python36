from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
sample_image = mnist.train.images[0].reshape([28, 28])

plt.gray()
plt.imshow(sample_image)




