import tensorflow as tf
import numpy as np
from tensorflow.python.framework.graph_util import convert_variables_to_constants

with tf.Session() as sess:
    a = tf.Variable(tf.convert_to_tensor(5, dtype=tf.uint8), name='a')
    b = tf.Variable(tf.convert_to_tensor(6, dtype=tf.uint8), name='b')
 
    c = tf.multiply(a, b, name="c")

    sess.run(tf.initialize_all_variables())

    print(a.eval()) # 5
    print(b.eval()) # 6
    print(c.eval()) # 30                              

    frozen_graph = convert_variables_to_constants(sess, sess.graph_def, ["c"])
    tf.train.write_graph(frozen_graph, './', 'graph.pb', as_text=False)

