from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants

x = tf.placeholder(tf.float32, name='input')
y_ = tf.placeholder(tf.float32, name='target')

W = tf.Variable(5., name='W')
b = tf.Variable(3., name='b')

y = x * W + b
y = tf.identity(y, name='output')

loss = tf.reduce_mean(tf.square(y - y_))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(loss, name='train')

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

# Creating a tf.train.Saver adds operations to the graph to save and
# restore variables from checkpoints.
# saver_def = tf.train.Saver().as_saver_def()

#print('Operation to initialize variables:       ', init.name)
#print('Tensor to feed as input data:            ', x.name)
#print('Tensor to feed as training targets:      ', y_.name)
#print('Tensor to fetch as prediction:           ', y.name)
#print('Operation to train one step:             ', train_op.name)
#print('Tensor to be fed for checkpoint filename:', saver_def.filename_tensor_name)
#print('Operation to save a checkpoint:          ', saver_def.save_tensor_name)
#print('Operation to restore a checkpoint:       ', saver_def.restore_op_name)
#print('Tensor to read value of W                ', W.value().name)
#print('Tensor to read value of b                ', b.value().name)

frozen_graph = convert_variables_to_constants(sess, sess.graph_def, ['output'])
tf.train.write_graph(frozen_graph, './', 'graph2.pb', as_text=False)

##with open('graph2.pb', 'w') as f:
##  f.write(tf.get_default_graph().as_graph_def().SerializeToString())

#with tf.Session() as sess:
#
#    x = tf.placeholder(tf.float32, name='input')
#    y_ = tf.placeholder(tf.float32, name='target')
#
#    W = tf.Variable(5., name='W')
#    b = tf.Variable(3., name='b')
#
#    y = x * W + b
#    y = tf.identity(y, name='output')
#
#    loss = tf.reduce_mean(tf.square(y - y_))
#    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
#    train_op = optimizer.minimize(loss, name='train')
#
#    init = tf.global_variables_initializer()
#    init.run()
#
#    # Creating a tf.train.Saver adds operations to the graph to save and
#    # restore variables from checkpoints.
#    # saver_def = tf.train.Saver().as_saver_def()
#
#    #print('Operation to initialize variables:       ', init.name)
#    #print('Tensor to feed as input data:            ', x.name)
#    #print('Tensor to feed as training targets:      ', y_.name)
#    #print('Tensor to fetch as prediction:           ', y.name)
#    #print('Operation to train one step:             ', train_op.name)
#    #print('Tensor to be fed for checkpoint filename:', saver_def.filename_tensor_name)
#    #print('Operation to save a checkpoint:          ', saver_def.save_tensor_name)
#    #print('Operation to restore a checkpoint:       ', saver_def.restore_op_name)
#    #print('Tensor to read value of W                ', W.value().name)
#    #print('Tensor to read value of b                ', b.value().name)
#
#    frozen_graph = convert_variables_to_constants(sess, sess.graph_def, ['output'])
#    tf.train.write_graph(frozen_graph, './', 'graph2.pb', as_text=False)
#
#    ##with open('graph2.pb', 'w') as f:
#    ##  f.write(tf.get_default_graph().as_graph_def().SerializeToString())
#
