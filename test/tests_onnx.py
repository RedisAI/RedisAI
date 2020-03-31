import sys

import redis
from includes import *

'''
python -m RLTest --test tests_onnx.py --module path/to/redisai.so
'''


def test_onnx_modelrun_mnist(env):
    if not TEST_ONNX:
        env.debugPrint("skipping {} since TEST_ONNX=0".format(sys._getframe().f_code.co_name), force=True)
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
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    ret = con.execute_command('AI.MODELGET', 'm')
    env.assertEqual(len(ret), 6)
    env.assertEqual(ret[-1], b'')

    ret = con.execute_command('AI.MODELSET', 'm', 'ONNX', DEVICE, 'TAG', 'asdf', model_pb)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    ret = con.execute_command('AI.MODELGET', 'm')
    env.assertEqual(len(ret), 6)
    env.assertEqual(ret[-1], b'asdf')
 
    # TODO: enable me
    # env.assertEqual(ret[0], b'ONNX')
    # env.assertEqual(ret[1], b'CPU')

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

    ensureSlaveSynced(con, env)

    tensor = con.execute_command('AI.TENSORGET', 'b', 'VALUES')
    values = tensor[-1]
    argmax = max(range(len(values)), key=lambda i: values[i])

    env.assertEqual(argmax, 1)

    if env.useSlaves:
        con2 = env.getSlaveConnection()
        tensor2 = con2.execute_command('AI.TENSORGET', 'b', 'VALUES')
        env.assertEqual(tensor2, tensor)


def test_onnx_modelrun_mnist_autobatch(env):
    if not TEST_PT:
        return

    con = env.getConnection()

    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    model_filename = os.path.join(test_data_path, 'mnist_batched.onnx')
    sample_filename = os.path.join(test_data_path, 'one.raw')

    with open(model_filename, 'rb') as f:
        model_pb = f.read()

    with open(sample_filename, 'rb') as f:
        sample_raw = f.read()

    ret = con.execute_command('AI.MODELSET', 'm', 'ONNX', 'CPU',
                              'BATCHSIZE', 2, 'MINBATCHSIZE', 2, model_pb)
    env.assertEqual(ret, b'OK')

    con.execute_command('AI.TENSORSET', 'a', 'FLOAT', 1, 1, 28, 28, 'BLOB', sample_raw)
    con.execute_command('AI.TENSORSET', 'c', 'FLOAT', 1, 1, 28, 28, 'BLOB', sample_raw)

    ensureSlaveSynced(con, env)

    def run():
        con = env.getConnection()
        con.execute_command('AI.MODELRUN', 'm', 'INPUTS', 'c', 'OUTPUTS', 'd')

    t = threading.Thread(target=run)
    t.start()

    con.execute_command('AI.MODELRUN', 'm', 'INPUTS', 'a', 'OUTPUTS', 'b')

    ensureSlaveSynced(con, env)

    import time
    time.sleep(1)

    tensor = con.execute_command('AI.TENSORGET', 'b', 'VALUES')
    values = tensor[-1]
    argmax = max(range(len(values)), key=lambda i: values[i])

    env.assertEqual(argmax, 1)

    tensor = con.execute_command('AI.TENSORGET', 'd', 'VALUES')
    values = tensor[-1]
    argmax = max(range(len(values)), key=lambda i: values[i])

    env.assertEqual(argmax, 1)


def test_onnx_modelrun_iris(env):
    if not TEST_ONNX:
        env.debugPrint("skipping {} since TEST_ONNX=0".format(sys._getframe().f_code.co_name), force=True)
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
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.MODELSET', 'logreg', 'ONNX', DEVICE, logreg_model)
    env.assertEqual(ret, b'OK')

    con.execute_command('AI.TENSORSET', 'features', 'FLOAT', 1, 4, 'VALUES', 5.1, 3.5, 1.4, 0.2)

    ensureSlaveSynced(con, env)

    con.execute_command('AI.MODELRUN', 'linear', 'INPUTS', 'features', 'OUTPUTS', 'linear_out')
    con.execute_command('AI.MODELRUN', 'logreg', 'INPUTS', 'features', 'OUTPUTS', 'logreg_out', 'logreg_probs')

    ensureSlaveSynced(con, env)

    linear_out = con.execute_command('AI.TENSORGET', 'linear_out', 'VALUES')
    logreg_out = con.execute_command('AI.TENSORGET', 'logreg_out', 'VALUES')

    env.assertEqual(float(linear_out[2][0]), -0.090524077415466309)
    env.assertEqual(logreg_out[2][0], 0)

    if env.useSlaves:
        con2 = env.getSlaveConnection()
        linear_out2 = con2.execute_command('AI.TENSORGET', 'linear_out', 'VALUES')
        logreg_out2 = con2.execute_command('AI.TENSORGET', 'logreg_out', 'VALUES')
        env.assertEqual(linear_out, linear_out2)
        env.assertEqual(logreg_out, logreg_out2)


def test_onnx_modelinfo(env):
    if not TEST_ONNX:
        env.debugPrint("skipping {} since TEST_ONNX=0".format(sys._getframe().f_code.co_name), force=True)
        return

    con = env.getConnection()
    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    linear_model_filename = os.path.join(test_data_path, 'linear_iris.onnx')

    with open(linear_model_filename, 'rb') as f:
        linear_model = f.read()

    ret = con.execute_command('AI.MODELSET', 'linear', 'ONNX', DEVICE, linear_model)
    env.assertEqual(ret, b'OK')

    model_serialized_master = con.execute_command('AI.MODELGET', 'linear')
    con.execute_command('AI.TENSORSET', 'features', 'FLOAT', 1, 4, 'VALUES', 5.1, 3.5, 1.4, 0.2)

    ensureSlaveSynced(con, env)

    if env.useSlaves:
        con2 = env.getSlaveConnection()
        model_serialized_slave = con2.execute_command('AI.MODELGET', 'linear')
        env.assertEqual(len(model_serialized_master), len(model_serialized_slave))
    previous_duration = 0
    for call in range(1, 10):
        res = con.execute_command('AI.MODELRUN', 'linear', 'INPUTS', 'features', 'OUTPUTS', 'linear_out')
        env.assertEqual(res, b'OK')
        ensureSlaveSynced(con, env)

        info = con.execute_command('AI.INFO', 'linear')
        info_dict_0 = info_to_dict(info)

        env.assertEqual(info_dict_0['KEY'], 'linear')
        env.assertEqual(info_dict_0['TYPE'], 'MODEL')
        env.assertEqual(info_dict_0['BACKEND'], 'ONNX')
        env.assertEqual(info_dict_0['DEVICE'], DEVICE)
        env.assertTrue(info_dict_0['DURATION'] > previous_duration)
        env.assertEqual(info_dict_0['SAMPLES'], call)
        env.assertEqual(info_dict_0['CALLS'], call)
        env.assertEqual(info_dict_0['ERRORS'], 0)

        previous_duration = info_dict_0['DURATION']

    res = con.execute_command('AI.INFO', 'linear', 'RESETSTAT')
    env.assertEqual(res, b'OK')

    info = con.execute_command('AI.INFO', 'linear')
    info_dict_0 = info_to_dict(info)
    env.assertEqual(info_dict_0['DURATION'], 0)
    env.assertEqual(info_dict_0['SAMPLES'], 0)
    env.assertEqual(info_dict_0['CALLS'], 0)
    env.assertEqual(info_dict_0['ERRORS'], 0)


def test_onnx_modelrun_disconnect(env):
    if not TEST_ONNX:
        env.debugPrint("skipping {} since TEST_ONNX=0".format(sys._getframe().f_code.co_name), force=True)
        return

    con = env.getConnection()
    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    linear_model_filename = os.path.join(test_data_path, 'linear_iris.onnx')

    with open(linear_model_filename, 'rb') as f:
        linear_model = f.read()

    ret = con.execute_command('AI.MODELSET', 'linear', 'ONNX', DEVICE, linear_model)
    env.assertEqual(ret, b'OK')

    model_serialized_master = con.execute_command('AI.MODELGET', 'linear')
    con.execute_command('AI.TENSORSET', 'features', 'FLOAT', 1, 4, 'VALUES', 5.1, 3.5, 1.4, 0.2)

    ensureSlaveSynced(con, env)

    if env.useSlaves:
        con2 = env.getSlaveConnection()
        model_serialized_slave = con2.execute_command('AI.MODELGET', 'linear')
        env.assertEqual(len(model_serialized_master), len(model_serialized_slave))

    ret = send_and_disconnect(('AI.MODELRUN', 'linear', 'INPUTS', 'features', 'OUTPUTS', 'linear_out'), con)
    env.assertEqual(ret, None)
