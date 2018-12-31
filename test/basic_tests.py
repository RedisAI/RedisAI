from RLTest import Env

from multiprocessing import Pool
import redis

import numpy as np
from skimage.io import imread
from skimage.transform import resize
import random
import time
import json
import os


'''
python -m RLTest --test basic_tests.py --module ../src/redisdl.so
'''


def check_cuda():
    return os.system('which nvcc')


def run_test_multiproc(env, n_procs, fn, args=()):
    pool = Pool(processes=n_procs)
    for _ in range(n_procs):
        pool.apply_async(fn, args = (env.getConnectionArgs(), *args))
    pool.close()
    pool.join()


def example_multiproc_fn(connArgs):
    con = redis.Redis(**connArgs)
    con.set('x', 1)


def test_example_multiproc():
    env = Env(testDescription="Basic multiprocessing test")
    run_test_multiproc(env, 10, example_multiproc_fn)
    con = env.getConnection()
    env.assertEqual(con.get('x'), '1')


def test_set_tensor():
    env = Env(testDescription="Set tensor")
    con = env.getConnection()
    con.execute_command('DL.SET', 'TENSOR', 'x', 'FLOAT', 1, 2, 'VALUES', 2, 3)
    tensor = con.execute_command('DL.GET', 'TENSOR', 'x', 'VALUES')
    values = tensor[-1]
    env.assertEqual(
        values, ['2', '3']
    )


def set_tensor(connArgs):
    con = redis.Redis(**connArgs)
    con.execute_command('DL.SET', 'TENSOR', 'x', 'FLOAT', 1, 2, 'VALUES', 2, 3)


def test_set_tensor_multiproc():
    env = Env(testDescription="Set tensor multiprocessing")
    run_test_multiproc(env, 10, set_tensor)
    con = env.getConnection()
    tensor = con.execute_command('DL.GET', 'TENSOR', 'x', 'VALUES')
    values = tensor[-1]
    env.assertEqual(
        values, ['2', '3']
    )


def load_mobilenet_test_data():

    examples_path = os.path.join(os.path.dirname(__file__), '..', 'examples')
    labels_filename = os.path.join(examples_path, 'js/imagenet_class_index.json')
    image_filename = os.path.join(examples_path, 'img/panda.jpg')
    graph_filename = os.path.join(examples_path, 'models/mobilenet_v2_1.4_224_frozen.pb')

    with open(graph_filename, 'rb') as f:
        graph_pb = f.read()

    with open(labels_filename, 'rb') as f:
        labels = json.load(f)

    img_height, img_width = 224, 224

    img = imread(image_filename)
    img = resize(img, (img_height, img_width), mode='constant', anti_aliasing=True)
    img = img.astype(np.float32)

    return graph_pb, labels, img


def test_run_mobilenet():
    input_var = 'input'
    output_var = 'MobilenetV2/Predictions/Reshape_1'

    env = Env(testDescription="Test mobilenet")
    con = env.getConnection(decode_responses=False)

    graph_pb, labels, img = load_mobilenet_test_data()

    con.execute_command('DL.SET', 'GRAPH', 'mobilenet', 'TF', graph_pb)

    con.execute_command('DL.SET', 'TENSOR', 'input',
                        'FLOAT', 4, 1, img.shape[1], img.shape[0], img.shape[2],
                        'BLOB', img.tobytes())

    con.execute_command('DL.RUN', 'GRAPH', 'mobilenet', 1,
                        'input', input_var,
                        'output', output_var)

    dtype, _, shape, _, data = con.execute_command('DL.GET', 'TENSOR', 'output', 'BLOB')

    dtype_map = {b'FLOAT': np.float32}
    tensor = np.frombuffer(data, dtype=dtype_map[dtype]).reshape(shape)
    label_id = np.argmax(tensor) - 1

    _, label = labels[str(label_id)]

    env.assertEqual(
        label, 'giant_panda'
    )


def run_mobilenet(connArgs, img, input_var, output_var):
    time.sleep(0.5 * random.randint(0, 10))

    con = redis.Redis(**connArgs)

    con.execute_command('DL.SET', 'TENSOR', 'input',
                        'FLOAT', 4, 1, img.shape[1], img.shape[0], img.shape[2],
                        'BLOB', img.tobytes())

    con.execute_command('DL.RUN', 'GRAPH', 'mobilenet', 1,
                        'input', input_var,
                        'output', output_var)

    con.execute_command('DEL','input')


def test_run_mobilenet_multiproc():
    input_var = 'input'
    output_var = 'MobilenetV2/Predictions/Reshape_1'

    graph_pb, labels, img = load_mobilenet_test_data()

    env = Env(testDescription="Test mobilenet multiprocessing")
    con = env.getConnection(decode_responses=False)

    con.execute_command('DL.SET', 'GRAPH', 'mobilenet', 'TF', graph_pb)

    run_test_multiproc(env, 30, run_mobilenet, (img, input_var, output_var))

    dtype, _, shape, _, data = con.execute_command('DL.GET', 'TENSOR', 'output', 'BLOB')

    dtype_map = {b'FLOAT': np.float32}
    tensor = np.frombuffer(data, dtype=dtype_map[dtype]).reshape(shape)
    label_id = np.argmax(tensor) - 1

    _, label = labels[str(label_id)]

    env.assertEqual(
        label, 'giant_panda'
    )
