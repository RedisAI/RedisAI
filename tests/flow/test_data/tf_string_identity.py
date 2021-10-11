import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants

with tf.compat.v1.Session() as sess:

    a = tf.compat.v1.placeholder(tf.string, name='input_str')
    c = tf.identity(a, name='c')

    frozen_graph = convert_variables_to_constants(sess, sess.graph_def, ['c'])
    tf.compat.v1.train.write_graph(frozen_graph, './', 'identity_string.pb', as_text=False)
