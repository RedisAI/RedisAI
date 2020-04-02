import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import numpy as np

# From https://github.com/leimao/Frozen_Graph_TensorFlow

tf.random.set_seed(seed=0)

model = keras.Sequential(layers=[
                             keras.layers.InputLayer(input_shape=(28, 28), name="input"),
                             keras.layers.Flatten(input_shape=(28, 28), name="flatten"),
                             keras.layers.Dense(128, activation="relu", name="dense"),
                             keras.layers.Dense(10, activation="softmax", name="output")
                         ], name="FCN")

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

full_model = tf.function(lambda x: model(x))
full_model = full_model.get_concrete_function(
                 tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()

tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir=".",
                  name="graph_v2.pb",
                  as_text=False)

