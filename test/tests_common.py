import redis

from includes import *

'''
python -m RLTest --test tests_common.py --module path/to/redisai.so
'''


def example_multiproc_fn(env):
    env.execute_command('set', 'x', 1)


def test_example_multiproc(env):
    run_test_multiproc(env, 10, lambda x: x.execute_command('set', 'x', 1))
    r = env.cmd('get', 'x')
    env.assertEqual(r, b'1')


def test_set_tensor(env):
    con = env.getConnection()
    con.execute_command('AI.TENSORSET', 'x', 'FLOAT', 2, 'VALUES', 2, 3)

    ensureSlaveSynced(con, env)

    tensor = con.execute_command('AI.TENSORGET', 'x', 'VALUES')
    values = tensor[-1]
    env.assertEqual(values, [b'2', b'3'])

    con.execute_command('AI.TENSORSET', 'x', 'INT32', 2, 'VALUES', 2, 3)

    ensureSlaveSynced(con, env)

    tensor = con.execute_command('AI.TENSORGET', 'x', 'VALUES')
    values = tensor[-1]
    env.assertEqual(values, [2, 3])

    # ERR unsupported data format
    try:
        con.execute_command('AI.TENSORSET', 'z', 'INT32', 2, 'unsupported', 2, 3)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual(exception.__str__(), "invalid argument found in tensor shape")

    # ERR invalid value
    try:
        con.execute_command('AI.TENSORSET', 'z', 'FLOAT', 2, 'VALUES', 2, 'A')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual(exception.__str__(), "invalid value")

    # ERR invalid value
    try:
        con.execute_command('AI.TENSORSET', 'z', 'INT32', 2, 'VALUES', 2, 'A')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual(exception.__str__(), "invalid value")

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

    ensureSlaveSynced(con, env)


def test_get_tensor(env):
    con = env.getConnection()
    con.execute_command('AI.TENSORSET', 't_FLOAT', 'FLOAT', 2, 'VALUES', 2, 3)
    con.execute_command('AI.TENSORSET', 't_INT8', 'INT8', 2, 'VALUES', 1, 1)
    con.execute_command('AI.TENSORSET', 't_INT16', 'INT8', 2, 'VALUES', 1, 1)
    con.execute_command('AI.TENSORSET', 't_INT32', 'INT8', 2, 'VALUES', 1, 1)
    con.execute_command('AI.TENSORSET', 't_INT64', 'INT8', 2, 'VALUES', 1, 1)

    tensor = con.execute_command('AI.TENSORGET', 't_FLOAT', 'BLOB')
    values = tensor[-1]

    tensor = con.execute_command('AI.TENSORGET', 't_INT8', 'VALUES')
    values = tensor[-1]
    env.assertEqual(values, [1, 1])

    tensor = con.execute_command('AI.TENSORGET', 't_INT16', 'VALUES')
    values = tensor[-1]
    env.assertEqual(values, [1, 1])

    tensor = con.execute_command('AI.TENSORGET', 't_INT32', 'VALUES')
    values = tensor[-1]
    env.assertEqual(values, [1, 1])

    tensor = con.execute_command('AI.TENSORGET', 't_INT64', 'VALUES')
    values = tensor[-1]
    env.assertEqual(values, [1, 1])

    tensor = con.execute_command('AI.TENSORGET', 't_INT32', 'META')
    values = tensor[-1]
    env.assertEqual(values, [2])

    # ERR unsupported data format
    try:
        con.execute_command('AI.TENSORGET', 't_FLOAT', 'unsupported')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual(exception.__str__(), "unsupported data format")


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
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    ret = con.execute_command('AI.MODELGET', 'm')
    env.assertEqual(len(ret), 3)
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


def test_set_tensor_multiproc(env):
    run_test_multiproc(env, 10,
                       lambda env: env.execute_command('AI.TENSORSET', 'x', 'FLOAT', 2, 'VALUES', 2, 3))
    con = env.getConnection()
    ensureSlaveSynced(con, env)
    tensor = con.execute_command('AI.TENSORGET', 'x', 'VALUES')
    values = tensor[-1]
    env.assertEqual(values, [b'2', b'3'])
