from multiprocessing import Pool, Process
import redis

import numpy as np
from skimage.io import imread
from skimage.transform import resize
import random
import time
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../opt/readies"))
import paella

TEST_TF = os.environ.get("TEST_TF") != "0" and os.environ.get("WITH_TF") != "0"
TEST_TFLITE = os.environ.get("TEST_TFLITE") != "0" and os.environ.get("WITH_TFLITE") != "0"
TEST_PT = os.environ.get("TEST_PT") != "0" and os.environ.get("WITH_PT") != "0"
TEST_ONNX = os.environ.get("TEST_ONNX") != "0" and os.environ.get("WITH_ORT") != "0"


'''
python -m RLTest --test basic_tests.py --module path/to/redisai.so
'''

DEVICE = os.environ.get('DEVICE', 'CPU').upper()
print(f"Running tests on {DEVICE}\n")


def check_cuda():
    return os.system('which nvcc')


def run_test_multiproc(env, n_procs, fn, args=tuple()):
    procs = []

    def tmpfn():
        con = env.getConnection()
        fn(con, *args)
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
    con = env.getConnection()
    con.execute_command('AI.TENSORSET', 'x', 'FLOAT', 2, 'VALUES', 2, 3)
    tensor = con.execute_command('AI.TENSORGET', 'x', 'VALUES')
    values = tensor[-1]
    env.assertEqual(values, [b'2', b'3'])
    con.execute_command('AI.TENSORSET', 'x', 'INT32', 2, 'VALUES', 2, 3)
    tensor = con.execute_command('AI.TENSORGET', 'x', 'VALUES')
    values = tensor[-1]
    env.assertEqual(values, [2, 3])

    try:
        con.execute_command('AI.TENSORSET', 1)
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.TENSORSET', 'y', 'FLOAT')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.TENSORSET', 'y', 'FLOAT', '2')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.TENSORSET', 'y', 'FLOAT', 2, 'VALUES')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.TENSORSET', 'y', 'FLOAT', 2, 'VALUES', 1)
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.TENSORSET', 'y', 'FLOAT', 2, 'VALUES', '1')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    time.sleep(0.1)

    for _ in env.reloadingIterator():
        env.assertExists('x')


def test_del_tf_model(env):
    if not TEST_PT:
        return

    con = env.getConnection()

    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    model_filename = os.path.join(test_data_path, 'graph.pb')

    with open(model_filename, 'rb') as f:
        model_pb = f.read()

    ret = con.execute_command('AI.MODELSET', 'm', 'TF', DEVICE,
                              'INPUTS', 'a', 'b', 'OUTPUTS', 'mul', model_pb)
    env.assertEqual(ret, b'OK')

    con.execute_command('AI.MODELDEL', 'm')
    env.assertFalse(env.execute_command('EXISTS', 'm'))


def test_run_tf_model(env):
    if not TEST_PT:
        return

    con = env.getConnection()

    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    model_filename = os.path.join(test_data_path, 'graph.pb')
    wrong_model_filename = os.path.join(test_data_path, 'pt-minimal.pt')

    with open(model_filename, 'rb') as f:
        model_pb = f.read()

    with open(wrong_model_filename, 'rb') as f:
        wrong_model_pb = f.read()
    ret = con.execute_command('AI.MODELSET', 'm', 'TF', DEVICE,
                              'INPUTS', 'a', 'b', 'OUTPUTS', 'mul', model_pb)
    env.assertEqual(ret, b'OK')

    try:
        ret = con.execute_command('AI.MODELSET', 'm', 'TF', DEVICE,
                                  'INPUTS', 'a', 'b', 'OUTPUTS', 'mul', wrong_model_pb)
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.MODELSET', 'm_1', 'TF',
                            'INPUTS', 'a', 'b', 'OUTPUTS', 'mul', model_pb)
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.MODELSET', 'm_2', 'PORCH', DEVICE,
                            'INPUTS', 'a', 'b', 'OUTPUTS', 'mul', model_pb)
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.MODELSET', 'm_3', 'TORCH', DEVICE,
                            'INPUTS', 'a', 'b', 'OUTPUTS', 'mul', model_pb)
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.MODELSET', 'm_4', 'TF',
                            'INPUTS', 'a', 'b', 'OUTPUTS', 'mul', model_pb)
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.MODELSET', 'm_5', 'TF', DEVICE,
                            'INPUTS', 'a', 'b', 'c', 'OUTPUTS', 'mul', model_pb)
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.MODELSET', 'm_6', 'TF', DEVICE,
                            'INPUTS', 'a', 'b', 'OUTPUTS', 'mult', model_pb)
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.MODELSET', 'm_7', 'TF', DEVICE, model_pb)
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.MODELSET', 'm_8', 'TF', DEVICE,
                            'INPUTS', 'a', 'b', 'OUTPUTS', 'mul')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.MODELSET', 'm_8', 'TF', DEVICE,
                            'INPUTS', 'a_', 'b', 'OUTPUTS', 'mul')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.MODELSET', 'm_8', 'TF', DEVICE,
                            'INPUTS', 'a', 'b', 'OUTPUTS', 'mul_')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.MODELRUN', 'm', 'INPUTS', 'a', 'b')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.MODELRUN', 'm', 'OUTPUTS', 'c')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    con.execute_command('AI.TENSORSET', 'a', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    con.execute_command('AI.TENSORSET', 'b', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)

    con.execute_command('AI.MODELRUN', 'm', 'INPUTS', 'a', 'b', 'OUTPUTS', 'c')

    tensor = con.execute_command('AI.TENSORGET', 'c', 'VALUES')
    values = tensor[-1]
    env.assertEqual(values, [b'4', b'9', b'4', b'9'])

    if env.useSlaves:
        con2 = env.getSlaveConnection()
        time.sleep(0.1)
        tensor2 = con2.execute_command('AI.TENSORGET', 'c', 'VALUES')
        env.assertEqual(tensor2, tensor)

    for _ in env.reloadingIterator():
        env.assertExists('m')
        env.assertExists('a')
        env.assertExists('b')
        env.assertExists('c')


def test_run_torch_model(env):
    if not TEST_PT:
        return

    con = env.getConnection()

    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    model_filename = os.path.join(test_data_path, 'pt-minimal.pt')
    wrong_model_filename = os.path.join(test_data_path, 'graph.pb')

    with open(model_filename, 'rb') as f:
        model_pb = f.read()

    with open(wrong_model_filename, 'rb') as f:
        wrong_model_pb = f.read()

    ret = con.execute_command('AI.MODELSET', 'm', 'TORCH', DEVICE, model_pb)
    con.assertEqual(ret, b'OK')

    try:
        con.execute_command('AI.MODELSET', 'm', 'TORCH', DEVICE, wrong_model_pb)
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.MODELSET', 'm_1', 'TORCH', model_pb)
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.MODELSET', 'm_2', model_pb)
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    env.execute_command('AI.TENSORSET', 'a', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.execute_command('AI.TENSORSET', 'b', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)

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
    env.assertEqual(values, [b'4', b'6', b'4', b'6'])

    if env.useSlaves:
        con2 = env.getSlaveConnection()
        time.sleep(0.1)
        tensor2 = con2.execute_command('AI.TENSORGET', 'c', 'VALUES')
        env.assertEqual(tensor2, tensor)

    for _ in env.reloadingIterator():
        env.assertExists('m')
        env.assertExists('a')
        env.assertExists('b')
        env.assertExists('c')


def test_run_onnx_model(env):
    if not TEST_ONNX:
        return

    con = env.getConnection()

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

    ret = con.execute_command('AI.MODELSET', 'm', 'ONNX', DEVICE, model_pb)
    con.assertEqual(ret, b'OK')

    try:
        con.execute_command('AI.MODELSET', 'm', 'ONNX', DEVICE, wrong_model_pb)
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.MODELSET', 'm_1', 'ONNX', model_pb)
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.MODELSET', 'm_2', model_pb)
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

    if env.useSlaves:
        con2 = env.getSlaveConnection()
        time.sleep(0.1)
        tensor2 = con2.execute_command('AI.TENSORGET', 'b', 'VALUES')
        env.assertEqual(tensor2, tensor)

    for _ in env.reloadingIterator():
        env.assertExists('m')
        env.assertExists('a')
        env.assertExists('b')


def test_run_onnxml_model(env):
    if not TEST_ONNX:
        return

    con = env.getConnection()

    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    linear_model_filename = os.path.join(test_data_path, 'linear_iris.onnx')
    logreg_model_filename = os.path.join(test_data_path, 'logreg_iris.onnx')

    with open(linear_model_filename, 'rb') as f:
        linear_model = f.read()

    with open(logreg_model_filename, 'rb') as f:
        logreg_model = f.read()

    ret = con.execute_command('AI.MODELSET', 'linear', 'ONNX', DEVICE, linear_model)
    con.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.MODELSET', 'logreg', 'ONNX', DEVICE, logreg_model)
    con.assertEqual(ret, b'OK')

    con.execute_command('AI.TENSORSET', 'features', 'FLOAT', 1, 4, 'VALUES', 5.1, 3.5, 1.4, 0.2)

    con.execute_command('AI.MODELRUN', 'linear', 'INPUTS', 'features', 'OUTPUTS', 'linear_out')
    con.execute_command('AI.MODELRUN', 'logreg', 'INPUTS', 'features', 'OUTPUTS', 'logreg_out', 'logreg_probs')

    linear_out = con.execute_command('AI.TENSORGET', 'linear_out', 'VALUES')
    logreg_out = con.execute_command('AI.TENSORGET', 'logreg_out', 'VALUES')

    env.assertEqual(float(linear_out[2][0]), -0.090524077415466309)
    env.assertEqual(logreg_out[2][0], 0)

    if env.useSlaves:
        con2 = env.getSlaveConnection()
        time.sleep(0.1)
        linear_out2 = con2.execute_command('AI.TENSORGET', 'linear_out', 'VALUES')
        logreg_out2 = con2.execute_command('AI.TENSORGET', 'logreg_out', 'VALUES')
        env.assertEqual(linear_out, linear_out2)
        env.assertEqual(logreg_out, logreg_out2)

    for _ in env.reloadingIterator():
        env.assertExists('linear')
        env.assertExists('logreg')


def test_run_tflite_model(env):
    if not TEST_TFLITE:
        return

    con = env.getConnection()

    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    model_filename = os.path.join(test_data_path, 'mnist_model_quant.tflite')
    wrong_model_filename = os.path.join(test_data_path, 'graph.pb')
    sample_filename = os.path.join(test_data_path, 'one.raw')

    with open(model_filename, 'rb') as f:
        model_pb = f.read()

    with open(model_filename, 'rb') as f:
        model_pb2 = f.read()

    with open(wrong_model_filename, 'rb') as f:
        wrong_model_pb = f.read()

    with open(sample_filename, 'rb') as f:
        sample_raw = f.read()

    ret = con.execute_command('AI.MODELSET', 'm', 'TFLITE', 'CPU', model_pb)
    env.assertEqual(ret, b'OK')

    # try:
    #     con.execute_command('AI.MODELSET', 'm_1', 'TFLITE', 'CPU', wrong_model_pb)
    # except Exception as e:
    #     exception = e
    # env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.MODELSET', 'm_1', 'TFLITE', model_pb)
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    ret = con.execute_command('AI.MODELSET', 'm_2', 'TFLITE', 'CPU', model_pb2)

    try:
        con.execute_command('AI.MODELSET', 'm_2', model_pb)
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    con.execute_command('AI.TENSORSET', 'a', 'FLOAT', 1, 1, 28, 28, 'BLOB', sample_raw)

    try:
        con.execute_command('AI.MODELRUN', 'm_2', 'INPUTS', 'a', 'OUTPUTS')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.MODELRUN', 'm_2', 'INPUTS', 'a', 'b', 'c')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.MODELRUN', 'm_2', 'a', 'b', 'c')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.MODELRUN', 'm_2', 'OUTPUTS', 'c')
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
        con.execute_command('AI.MODELRUN', 'm', 'INPUTS', 'OUTPUTS')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.MODELRUN', 'm', 'INPUTS', 'a', 'OUTPUTS', 'b')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    con.execute_command('AI.MODELRUN', 'm', 'INPUTS', 'a', 'OUTPUTS', 'b', 'c')

    tensor = con.execute_command('AI.TENSORGET', 'b', 'VALUES')
    value = tensor[-1][0]

    env.assertEqual(value, 1)

    for _ in env.reloadingIterator():
        env.assertExists('m')
        env.assertExists('a')
        env.assertExists('b')
        env.assertExists('c')


def test_set_tensor_multiproc(env):
    run_test_multiproc(env, 10,
        lambda env: env.execute_command('AI.TENSORSET', 'x', 'FLOAT', 2, 'VALUES', 2, 3))

    con = env.getConnection()

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

    with open(labels_filename, 'r') as f:
        labels = json.load(f)

    img_height, img_width = 224, 224

    img = imread(image_filename)
    img = resize(img, (img_height, img_width), mode='constant', anti_aliasing=True)
    img = img.astype(np.float32)

    return model_pb, labels, img


def test_run_mobilenet(env):
    if not TEST_TF:
        return

    con = env.getConnection()

    input_var = 'input'
    output_var = 'MobilenetV2/Predictions/Reshape_1'

    model_pb, labels, img = load_mobilenet_test_data()

    con.execute_command('AI.MODELSET', 'mobilenet', 'TF', DEVICE,
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

    # env.execute_command('DEL', 'input')


def test_run_mobilenet_multiproc(env):
    if not TEST_TF:
        return

    con = env.getConnection()

    input_var = 'input'
    output_var = 'MobilenetV2/Predictions/Reshape_1'

    model_pb, labels, img = load_mobilenet_test_data()
    con.execute_command('AI.MODELSET', 'mobilenet', 'TF', DEVICE,
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

    #@@@ possible workaround for side-effect test failure
    # env.restartAndReload()


def test_set_incorrect_script(env):
    if not TEST_PT:
        return

    con = env.getConnection()

    try:
        con.execute_command('AI.SCRIPTSET', 'ket', DEVICE, 'return 1')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.SCRIPTSET', 'nope')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.SCRIPTSET', 'more', DEVICE)
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)


def test_set_correct_script(env):
    if not TEST_PT:
        return

    con = env.getConnection()

    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    script_filename = os.path.join(test_data_path, 'script.txt')

    with open(script_filename, 'rb') as f:
        script = f.read()

    con.execute_command('AI.SCRIPTSET', 'ket', DEVICE, script)

    for _ in env.reloadingIterator():
        env.assertExists('ket')


def test_del_script(env):
    if not TEST_PT:
        return

    con = env.getConnection()

    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    script_filename = os.path.join(test_data_path, 'script.txt')

    with open(script_filename, 'rb') as f:
        script = f.read()

    ret = con.execute_command('AI.SCRIPTSET', 'ket', DEVICE, script)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.SCRIPTDEL', 'ket')
    env.assertFalse(con.execute_command('EXISTS', 'ket'))


def test_run_script(env):
    if not TEST_PT:
        return

    con = env.getConnection()

    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    script_filename = os.path.join(test_data_path, 'script.txt')

    with open(script_filename, 'rb') as f:
        script = f.read()

    con.execute_command('AI.SCRIPTSET', 'ket', DEVICE, script)

    con.execute_command('AI.TENSORSET', 'a', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    con.execute_command('AI.TENSORSET', 'b', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)

    try:
        con.execute_command('AI.SCRIPTRUN', 'ket', 'bar', 'INPUTS', 'b', 'OUTPUTS', 'c')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.SCRIPTRUN', 'ket', 'INPUTS', 'a', 'b', 'OUTPUTS', 'c')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.SCRIPTRUN', 'ket', 'bar', 'INPUTS', 'b', 'OUTPUTS')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.SCRIPTRUN', 'ket', 'bar', 'INPUTS', 'OUTPUTS')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    con.execute_command('AI.SCRIPTRUN', 'ket', 'bar', 'INPUTS', 'a', 'b', 'OUTPUTS', 'c')

    tensor = con.execute_command('AI.TENSORGET', 'c', 'VALUES')
    values = tensor[-1]
    env.assertEqual(values, [b'4', b'6', b'4', b'6'])

    time.sleep(0.1)

    if env.useSlaves:
        con2 = env.getSlaveConnection()
        time.sleep(0.1)
        tensor2 = con2.execute_command('AI.TENSORGET', 'c', 'VALUES')
        env.assertEqual(tensor2, tensor)

    for _ in env.reloadingIterator():
        env.assertExists('ket')
        env.assertExists('a')
        env.assertExists('b')
        env.assertExists('c')
