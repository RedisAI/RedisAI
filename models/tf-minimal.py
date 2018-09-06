import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants

with tf.Session() as sess:

    a = tf.placeholder(tf.float32, name='a')
    b = tf.placeholder(tf.float32, name='b')

    c = a * b

    c = tf.identity(c, name='c')

    # res = sess.run(c, feed_dict = {a: 3.0, b: 4.0})

    frozen_graph = convert_variables_to_constants(sess, sess.graph_def, ['c'])
    tf.train.write_graph(frozen_graph, './', 'graph.pb', as_text=False)

