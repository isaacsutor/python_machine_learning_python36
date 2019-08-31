import tensorflow as tf

# creating nodes
node = tf.Variable(tf.zeros([2, 2]))

# run session
with tf.compat.v1.Session() as sess:
    # initialize all global variables
    sess.run(tf.compat.v1.global_variables_initializer())

    # evaluating node
    print("Tensor value before addition:\n", sess.run(node))

    # elementwise addition to tensor
    node = node.assign(node + tf.ones([2, 2]))

    # evaluate node again
    print("Tensor value after addition:\n", sess.run(node))