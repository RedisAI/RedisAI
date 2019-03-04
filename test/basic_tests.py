from RLTest import Env

from multiprocessing import Pool, Process
import redis

import numpy as np
from skimage.io import imread
from skimage.transform import resize
import random
import time
import json
import os


'''
python -m RLTest --test basic_tests.py --module ../src/redisai.so
python3 -m RLTest --test test/basic_tests.py --module build/redisai.so --enterprise-lib-path /Users/lantiga/Projects/RedisAI/RedisAI/deps/install/lib --oss-redis-path deps/redis/src/redis-server -s --use-aof
'''


def check_cuda():
    return os.system('which nvcc')


def run_test_multiproc(env, n_procs, fn, args=tuple()):
    procs = []

    def tmpfn():
        e = env.getConnection()
        fn(e, *args)
        return 1

    for _ in range(n_procs):
        p = Process(target=tmpfn)
        p.start()
        procs.append(p)

    [p.join() for p in procs]


def example_multiproc_fn(env):
    env.execute_command('set', 'x', 1)


def test_example_multiproc(env):
    run_test_multiproc(env, 10, lambda x: x.execute_command('set', 'x', 1))
    r = env.cmd('get', 'x')
    env.assertEqual(r, b'1')


def test_set_tensor(env):
    con = env
    con.execute_command('AI.TENSORSET', 'x', 'FLOAT', 2, 'VALUES', 2, 3)
    tensor = con.execute_command('AI.TENSORGET', 'x', 'VALUES')
    values = tensor[-1]
    env.assertEqual(values, [b'2', b'3'])
    con.execute_command('AI.TENSORSET', 'x', 'INT32', 2, 'VALUES', 2, 3)
    tensor = con.execute_command('AI.TENSORGET', 'x', 'VALUES')
    values = tensor[-1]
    env.assertEqual(values, [2, 3])

    try:
        env.execute_command('AI.TENSORSET', 1)
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        env.execute_command('AI.TENSORSET', 'y', 'FLOAT')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        env.execute_command('AI.TENSORSET', 'y', 'FLOAT', '2')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        env.execute_command('AI.TENSORSET', 'y', 'FLOAT', 2, 'VALUES')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        env.execute_command('AI.TENSORSET', 'y', 'FLOAT', 2, 'VALUES', 1)
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        env.execute_command('AI.TENSORSET', 'y', 'FLOAT', 2, 'VALUES', '1')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    for _ in con.reloadingIterator():
        env.assertExists('x')


def test_run_tf_model(env):
    examples_path = os.path.join(os.path.dirname(__file__), '..', 'examples')
    model_filename = os.path.join(examples_path, 'models/graph.pb')

    with open(model_filename, 'rb') as f:
        model_pb = f.read()

    con = env
    ret = con.execute_command('AI.MODELSET', 'm', 'TF', 'CPU',
                              'INPUTS', 'a', 'b', 'OUTPUTS', 'mul', model_pb)
    con.assertEqual(ret, b'OK')

    try:
        env.execute_command('AI.MODELSET', 'm_1', 'TF',
                            'INPUTS', 'a', 'b', 'OUTPUTS', 'mul', model_pb)
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        env.execute_command('AI.MODELSET', 'm_2', 'PORCH', 'CPU',
                            'INPUTS', 'a', 'b', 'OUTPUTS', 'mul', model_pb)
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        env.execute_command('AI.MODELSET', 'm_3', 'TORCH', 'CPU',
                            'INPUTS', 'a', 'b', 'OUTPUTS', 'mul', model_pb)
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        env.execute_command('AI.MODELSET', 'm_4', 'TF',
                            'INPUTS', 'a', 'b', 'OUTPUTS', 'mul', model_pb)
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        env.execute_command('AI.MODELSET', 'm_5', 'TF', 'CPU',
                            'INPUTS', 'a', 'b', 'c', 'OUTPUTS', 'mul', model_pb)
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        env.execute_command('AI.MODELSET', 'm_6', 'TF', 'CPU',
                            'INPUTS', 'a', 'b', 'OUTPUTS', 'mult', model_pb)
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        env.execute_command('AI.MODELSET', 'm_7', 'TF', 'CPU', model_pb)
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        env.execute_command('AI.MODELSET', 'm_8', 'TF', 'CPU',
                            'INPUTS', 'a', 'b', 'OUTPUTS', 'mul')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)
 
    con.execute_command('AI.TENSORSET', 'a', 'FLOAT', 2, 'VALUES', 2, 3)
    con.execute_command('AI.TENSORSET', 'b', 'FLOAT', 2, 'VALUES', 2, 3)

    con.execute_command('AI.MODELRUN', 'm', 'INPUTS', 'a', 'b', 'OUTPUTS', 'c')

    tensor = con.execute_command('AI.TENSORGET', 'c', 'VALUES')
    values = tensor[-1]
    con.assertEqual(values, [b'4', b'9'])

    for _ in con.reloadingIterator():
        env.assertExists('m')
        env.assertExists('a')
        env.assertExists('b')
        env.assertExists('c')


def test_run_torch_model(env):
    examples_path = os.path.join(os.path.dirname(__file__), '..', 'examples')
    model_filename = os.path.join(examples_path, 'models/pt-minimal.pt')

    with open(model_filename, 'rb') as f:
        model_pb = f.read()

    con = env
    ret = con.execute_command('AI.MODELSET', 'm', 'TORCH', 'CPU', model_pb)
    con.assertEqual(ret, b'OK')

    try:
        env.execute_command('AI.MODELSET', 'm_1', 'TORCH', model_pb)
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        env.execute_command('AI.MODELSET', 'm_2', model_pb)
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    con.execute_command('AI.TENSORSET', 'a', 'FLOAT', 2, 'VALUES', 2, 3)
    con.execute_command('AI.TENSORSET', 'b', 'FLOAT', 2, 'VALUES', 2, 3)

    try:
        con.execute_command('AI.MODELRUN', 'm_1', 'INPUTS', 'a', 'b', 'OUTPUTS')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.MODELRUN', 'm_2', 'INPUTS', 'a', 'b', 'c')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.MODELRUN', 'm_3', 'a', 'b', 'c')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.MODELRUN', 'm_1', 'OUTPUTS', 'c')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.MODELRUN', 'm_1', 'INPUTS', 'OUTPUTS')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    con.execute_command('AI.MODELRUN', 'm', 'INPUTS', 'a', 'b', 'OUTPUTS', 'c')

    tensor = con.execute_command('AI.TENSORGET', 'c', 'VALUES')
    values = tensor[-1]
    con.assertEqual(values, [b'4', b'6'])

    for _ in con.reloadingIterator():
        env.assertExists('m')
        env.assertExists('a')
        env.assertExists('b')
        env.assertExists('c')


def test_set_tensor_multiproc(env):
    run_test_multiproc(env, 10,
        lambda con: con.execute_command('AI.TENSORSET', 'x', 'FLOAT', 2, 'VALUES', 2, 3))

    con = env
    tensor = con.execute_command('AI.TENSORGET', 'x', 'VALUES')
    values = tensor[-1]
    env.assertEqual(values, [b'2', b'3'])


def load_mobilenet_test_data():
    examples_path = os.path.join(os.path.dirname(__file__), '..', 'examples')
    labels_filename = os.path.join(examples_path, 'js/imagenet_class_index.json')
    image_filename = os.path.join(examples_path, 'img/panda.jpg')
    model_filename = os.path.join(examples_path, 'models/mobilenet_v2_1.4_224_frozen.pb')

    with open(model_filename, 'rb') as f:
        model_pb = f.read()

    with open(labels_filename, 'rb') as f:
        labels = json.load(f)

    img_height, img_width = 224, 224

    img = imread(image_filename)
    img = resize(img, (img_height, img_width), mode='constant', anti_aliasing=True)
    img = img.astype(np.float32)

    return model_pb, labels, img


def test_run_mobilenet(env):
    input_var = 'input'
    output_var = 'MobilenetV2/Predictions/Reshape_1'
    con = env

    model_pb, labels, img = load_mobilenet_test_data()

    con.execute_command('AI.MODELSET', 'mobilenet', 'TF', 'CPU',
                        'INPUTS', input_var, 'OUTPUTS', output_var, model_pb)

    con.execute_command('AI.TENSORSET', 'input',
                        'FLOAT', 1, img.shape[1], img.shape[0], img.shape[2],
                        'BLOB', img.tobytes())

    con.execute_command('AI.MODELRUN', 'mobilenet',
                        'INPUTS', 'input', 'OUTPUTS', 'output')

    dtype, shape, data = con.execute_command('AI.TENSORGET', 'output', 'BLOB')

    dtype_map = {b'FLOAT': np.float32}
    tensor = np.frombuffer(data, dtype=dtype_map[dtype]).reshape(shape)
    label_id = np.argmax(tensor) - 1

    _, label = labels[str(label_id)]

    env.assertEqual(label, 'giant_panda')


def run_mobilenet(con, img, input_var, output_var):
    time.sleep(0.5 * random.randint(0, 10))
    con.execute_command('AI.TENSORSET', 'input',
                        'FLOAT', 1, img.shape[1], img.shape[0], img.shape[2],
                        'BLOB', img.tobytes())

    con.execute_command('AI.MODELRUN', 'mobilenet',
                        'INPUTS', 'input', 'OUTPUTS', 'output')

    con.execute_command('DEL','input')


def test_run_mobilenet_multiproc(env):
    input_var = 'input'
    output_var = 'MobilenetV2/Predictions/Reshape_1'

    model_pb, labels, img = load_mobilenet_test_data()
    con = env
    con.execute_command('AI.MODELSET', 'mobilenet', 'TF', 'CPU',
                        'INPUTS', input_var, 'OUTPUTS', output_var, model_pb)

    run_test_multiproc(env, 30, run_mobilenet, (img, input_var, output_var))

    dtype, shape, data = con.execute_command('AI.TENSORGET', 'output', 'BLOB')

    dtype_map = {b'FLOAT': np.float32}
    tensor = np.frombuffer(data, dtype=dtype_map[dtype]).reshape(shape)
    label_id = np.argmax(tensor) - 1

    _, label = labels[str(label_id)]

    env.assertEqual(
        label, 'giant_panda'
    )


def test_set_incorrect_script(env):
    try:
        env.execute_command('AI.SCRIPTSET', 'ket', 'CPU', 'return 1')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        env.execute_command('AI.SCRIPTSET', 'nope')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        env.execute_command('AI.SCRIPTSET', 'more', 'CPU')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)


def test_set_correct_script(env):
    examples_path = os.path.join(os.path.dirname(__file__), '..', 'examples')
    script_filename = os.path.join(examples_path, 'models/script.txt')

    with open(script_filename, 'rb') as f:
        script = f.read()

    env.execute_command('AI.SCRIPTSET', 'ket', 'CPU', script)

    for _ in env.reloadingIterator():
        env.assertExists('ket')


def test_run_script(env):
    examples_path = os.path.join(os.path.dirname(__file__), '..', 'examples')
    script_filename = os.path.join(examples_path, 'models/script.txt')

    with open(script_filename, 'rb') as f:
        script = f.read()

    env.execute_command('AI.SCRIPTSET', 'ket', 'CPU', script)

    env.execute_command('AI.TENSORSET', 'a', 'FLOAT', 2, 'VALUES', 2, 3)
    env.execute_command('AI.TENSORSET', 'b', 'FLOAT', 2, 'VALUES', 2, 3)

    try:
        env.execute_command('AI.SCRIPTRUN', 'ket', 'bar', 'INPUTS', 'b', 'OUTPUTS', 'c')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        env.execute_command('AI.SCRIPTRUN', 'ket', 'INPUTS', 'a', 'b', 'OUTPUTS', 'c')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        env.execute_command('AI.SCRIPTRUN', 'ket', 'bar', 'INPUTS', 'b', 'OUTPUTS')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        env.execute_command('AI.SCRIPTRUN', 'ket', 'bar', 'INPUTS', 'OUTPUTS')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    env.execute_command('AI.SCRIPTRUN', 'ket', 'bar', 'INPUTS', 'a', 'b', 'OUTPUTS', 'c')

    tensor = env.execute_command('AI.TENSORGET', 'c', 'VALUES')
    values = tensor[-1]
    env.assertEqual(values, [b'4', b'6'])

    for _ in env.reloadingIterator():
        env.assertExists('ket')
        env.assertExists('a')
        env.assertExists('b')
        env.assertExists('c')
