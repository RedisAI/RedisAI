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


def test_del_tf_model(env):
    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    model_filename = os.path.join(test_data_path, 'graph.pb')

    with open(model_filename, 'rb') as f:
        model_pb = f.read()

    con = env
    ret = con.execute_command('AI.MODELSET', 'm', 'TF', 'CPU',
                              'INPUTS', 'a', 'b', 'OUTPUTS', 'mul', model_pb)
    con.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.MODELDEL', 'm')
    con.assertEqual(ret, b'OK')


def test_run_tf_model(env):
    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    model_filename = os.path.join(test_data_path, 'graph.pb')
    wrong_model_filename = os.path.join(test_data_path, 'pt-minimal.pt')

    with open(model_filename, 'rb') as f:
        model_pb = f.read()

    with open(wrong_model_filename, 'rb') as f:
        wrong_model_pb = f.read()

    con = env
    ret = con.execute_command('AI.MODELSET', 'm', 'TF', 'CPU',
                              'INPUTS', 'a', 'b', 'OUTPUTS', 'mul', model_pb)
    con.assertEqual(ret, b'OK')

    try:
        ret = con.execute_command('AI.MODELSET', 'm', 'TF', 'CPU',
                                  'INPUTS', 'a', 'b', 'OUTPUTS', 'mul', wrong_model_pb)
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

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

    try:
        env.execute_command('AI.MODELSET', 'm_8', 'TF', 'CPU',
                            'INPUTS', 'a_', 'b', 'OUTPUTS', 'mul')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        env.execute_command('AI.MODELSET', 'm_8', 'TF', 'CPU',
                            'INPUTS', 'a', 'b', 'OUTPUTS', 'mul_')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        env.execute_command('AI.MODELRUN', 'm', 'INPUTS', 'a', 'b')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        env.execute_command('AI.MODELRUN', 'm', 'OUTPUTS', 'c')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    con.execute_command('AI.TENSORSET', 'a', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    con.execute_command('AI.TENSORSET', 'b', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)

    con.execute_command('AI.MODELRUN', 'm', 'INPUTS', 'a', 'b', 'OUTPUTS', 'c')

    tensor = con.execute_command('AI.TENSORGET', 'c', 'VALUES')
    values = tensor[-1]
    con.assertEqual(values, [b'4', b'9', b'4', b'9'])

    for _ in con.reloadingIterator():
        env.assertExists('m')
        env.assertExists('a')
        env.assertExists('b')
        env.assertExists('c')


def test_run_torch_model(env):
    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    model_filename = os.path.join(test_data_path, 'pt-minimal.pt')
    wrong_model_filename = os.path.join(test_data_path, 'graph.pb')

    with open(model_filename, 'rb') as f:
        model_pb = f.read()

    with open(wrong_model_filename, 'rb') as f:
        wrong_model_pb = f.read()

    con = env
    ret = con.execute_command('AI.MODELSET', 'm', 'TORCH', 'CPU', model_pb)
    con.assertEqual(ret, b'OK')

    try:
        con.execute_command('AI.MODELSET', 'm', 'TORCH', 'CPU', wrong_model_pb)
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

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

    con.execute_command('AI.TENSORSET', 'a', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    con.execute_command('AI.TENSORSET', 'b', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)

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
        con.execute_command('AI.MODELRUN', 'm', 'OUTPUTS', 'c')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.MODELRUN', 'm', 'INPUTS', 'a', 'b')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.MODELRUN', 'm_1', 'INPUTS', 'OUTPUTS')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.MODELRUN', 'm_1', 'INPUTS', 'a', 'b', 'OUTPUTS', 'c', 'd')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    con.execute_command('AI.MODELRUN', 'm', 'INPUTS', 'a', 'b', 'OUTPUTS', 'c')

    tensor = con.execute_command('AI.TENSORGET', 'c', 'VALUES')
    values = tensor[-1]
    con.assertEqual(values, [b'4', b'6', b'4', b'6'])

    for _ in con.reloadingIterator():
        env.assertExists('m')
        env.assertExists('a')
        env.assertExists('b')
        env.assertExists('c')


def test_run_onnx_model(env):
    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    model_filename = os.path.join(test_data_path, 'mnist.onnx')
    wrong_model_filename = os.path.join(test_data_path, 'graph.pb')
    sample_filename = os.path.join(test_data_path, 'one.raw')

    with open(model_filename, 'rb') as f:
        model_pb = f.read()

    with open(wrong_model_filename, 'rb') as f:
        wrong_model_pb = f.read()

    with open(sample_filename, 'rb') as f:
        sample_raw = f.read()

    con = env
    ret = con.execute_command('AI.MODELSET', 'm', 'ONNX', 'CPU', model_pb)
    con.assertEqual(ret, b'OK')

    try:
        con.execute_command('AI.MODELSET', 'm', 'ONNX', 'CPU', wrong_model_pb)
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        env.execute_command('AI.MODELSET', 'm_1', 'ONNX', model_pb)
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        env.execute_command('AI.MODELSET', 'm_2', model_pb)
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    con.execute_command('AI.TENSORSET', 'a', 'FLOAT', 1, 1, 28, 28, 'BLOB', sample_raw)

    try:
        con.execute_command('AI.MODELRUN', 'm_1', 'INPUTS', 'a', 'OUTPUTS')
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
        con.execute_command('AI.MODELRUN', 'm', 'OUTPUTS', 'c')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.MODELRUN', 'm', 'INPUTS', 'a', 'b')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.MODELRUN', 'm_1', 'INPUTS', 'OUTPUTS')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.MODELRUN', 'm_1', 'INPUTS', 'a', 'OUTPUTS', 'b')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    con.execute_command('AI.MODELRUN', 'm', 'INPUTS', 'a', 'OUTPUTS', 'b')

    tensor = con.execute_command('AI.TENSORGET', 'b', 'VALUES')
    values = tensor[-1]
    argmax = max(range(len(values)), key=lambda i: values[i])

    env.assertEqual(argmax, 1)

    for _ in con.reloadingIterator():
        env.assertExists('m')
        env.assertExists('a')
        env.assertExists('b')


def test_run_onnxml_model(env):
    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    linear_model_filename = os.path.join(test_data_path, 'linear_iris.onnx')
    logreg_model_filename = os.path.join(test_data_path, 'logreg_iris.onnx')

    with open(linear_model_filename, 'rb') as f:
        linear_model = f.read()

    with open(logreg_model_filename, 'rb') as f:
        logreg_model = f.read()

    con = env
    ret = con.execute_command('AI.MODELSET', 'linear', 'ONNX', 'CPU', linear_model)
    con.assertEqual(ret, b'OK')

    con = env
    ret = con.execute_command('AI.MODELSET', 'logreg', 'ONNX', 'CPU', logreg_model)
    con.assertEqual(ret, b'OK')

    con.execute_command('AI.TENSORSET', 'features', 'FLOAT', 1, 4, 'VALUES', 5.1, 3.5, 1.4, 0.2)

    con.execute_command('AI.MODELRUN', 'linear', 'INPUTS', 'features', 'OUTPUTS', 'linear_out')
    con.execute_command('AI.MODELRUN', 'logreg', 'INPUTS', 'features', 'OUTPUTS', 'logreg_out', 'logreg_probs')

    linear_out = con.execute_command('AI.TENSORGET', 'linear_out', 'VALUES')
    logreg_out = con.execute_command('AI.TENSORGET', 'logreg_out', 'VALUES')

    env.assertEqual(float(linear_out[2][0]), -0.090524077415466309)
    env.assertEqual(logreg_out[2][0], 0)

    for _ in con.reloadingIterator():
        env.assertExists('linear')
        env.assertExists('logreg')


def test_set_tensor_multiproc(env):
    run_test_multiproc(env, 10,
        lambda con: con.execute_command('AI.TENSORSET', 'x', 'FLOAT', 2, 'VALUES', 2, 3))

    con = env
    tensor = con.execute_command('AI.TENSORGET', 'x', 'VALUES')
    values = tensor[-1]
    env.assertEqual(values, [b'2', b'3'])


def load_mobilenet_test_data():
    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    labels_filename = os.path.join(test_data_path, 'imagenet_class_index.json')
    image_filename = os.path.join(test_data_path, 'panda.jpg')
    model_filename = os.path.join(test_data_path, 'mobilenet_v2_1.4_224_frozen.pb')

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

    con.execute_command('DEL', 'input')


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
    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    script_filename = os.path.join(test_data_path, 'script.txt')

    with open(script_filename, 'rb') as f:
        script = f.read()

    env.execute_command('AI.SCRIPTSET', 'ket', 'CPU', script)

    for _ in env.reloadingIterator():
        env.assertExists('ket')


def test_del_script(env):
    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    script_filename = os.path.join(test_data_path, 'script.txt')

    with open(script_filename, 'rb') as f:
        script = f.read()

    ret = env.execute_command('AI.SCRIPTSET', 'ket', 'CPU', script)
    env.assertEqual(ret, b'OK')

    ret = env.execute_command('AI.SCRIPTDEL', 'ket')
    env.assertEqual(ret, b'OK')


def test_run_script(env):
    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    script_filename = os.path.join(test_data_path, 'script.txt')

    with open(script_filename, 'rb') as f:
        script = f.read()

    env.execute_command('AI.SCRIPTSET', 'ket', 'CPU', script)

    env.execute_command('AI.TENSORSET', 'a', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.execute_command('AI.TENSORSET', 'b', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)

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
    env.assertEqual(values, [b'4', b'6', b'4', b'6'])

    for _ in env.reloadingIterator():
        env.assertExists('ket')
        env.assertExists('a')
        env.assertExists('b')
        env.assertExists('c')
