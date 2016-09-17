import tensorflow as tf
import numpy as np

with tf.Session() as sess:
    a = tf.Variable(tf.convert_to_tensor(5, dtype=tf.uint8), name='a')
    b = tf.Variable(tf.convert_to_tensor(6, dtype=tf.uint8), name='b')
 
    a = tf.Variable(5, name='a')
    b = tf.Variable(6, name='b')
    c = tf.mul(a, b, name="c")

    sess.run(tf.initialize_all_variables())

    print(a.eval()) # 5
    print(b.eval()) # 6
    print(c.eval()) # 30                              

    tf.train.write_graph(sess.graph_def, './', 'graph.pb', as_text=False)

