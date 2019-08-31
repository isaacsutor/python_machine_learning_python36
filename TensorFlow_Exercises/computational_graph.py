import tensorflow as tf


node1 = tf.constant(3, dtype=tf.int32)
node2 = tf.constant(5, dtype=tf.int32)
node3 = tf.add(node1, node2)

sess = tf.compat.v1.Session()


print("Sum of node1 and node2 is:", sess.run(node3))


sess.close()

# alternative that does not need to explicitly close the session
# with tf.Session() as sess:
    # print("Sum of node1 and node2 is:",sess.run(node3))