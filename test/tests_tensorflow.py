import redis
from functools import wraps
import multiprocessing as mp

from includes import *

'''
python -m RLTest --test tests_tensorflow.py --module path/to/redisai.so
'''


def skip_if_no_TF(f):
    @wraps(f)
    def wrapper(env, *args, **kwargs):
        if not TEST_TF:
            env.debugPrint("skipping {} since TEST_TF=0".format(
                sys._getframe().f_code.co_name), force=True)
            return
        return f(env, *args, **kwargs)
    return wrapper


@skip_if_no_TF
def test_run_mobilenet(env):
    con = env.getConnection()

    input_var = 'input'
    output_var = 'MobilenetV2/Predictions/Reshape_1'

    model_pb, labels, img = load_mobilenet_test_data()

    con.execute_command('AI.MODELSET', 'mobilenet', 'TF', DEVICE,
                        'INPUTS', input_var, 'OUTPUTS', output_var, 'BLOB', model_pb)

    ensureSlaveSynced(con, env)

    mobilenet_model_serialized = con.execute_command(
        'AI.MODELGET', 'mobilenet', 'META')

    ensureSlaveSynced(con, env)
    if env.useSlaves:
        con2 = env.getSlaveConnection()
        slave_mobilenet_model_serialized = con2.execute_command(
            'AI.MODELGET', 'mobilenet', 'META')
        env.assertEqual(len(mobilenet_model_serialized),
                        len(slave_mobilenet_model_serialized))

    con.execute_command('AI.TENSORSET', 'input',
                        'FLOAT', 1, img.shape[1], img.shape[0], img.shape[2],
                        'BLOB', img.tobytes())

    ensureSlaveSynced(con, env)
    input_tensor_meta = con.execute_command('AI.TENSORGET', 'input', 'META')
    env.assertEqual(
        [b'dtype', b'FLOAT', b'shape', [1, img.shape[1], img.shape[0], img.shape[2]]], input_tensor_meta)

    ensureSlaveSynced(con, env)
    if env.useSlaves:
        con2 = env.getSlaveConnection()
        slave_tensor_meta = con2.execute_command(
            'AI.TENSORGET', 'input', 'META')
        env.assertEqual(input_tensor_meta, slave_tensor_meta)

    con.execute_command('AI.MODELRUN', 'mobilenet',
                        'INPUTS', 'input', 'OUTPUTS', 'output')

    ensureSlaveSynced(con, env)

    _, dtype, _, shape, _, data = con.execute_command('AI.TENSORGET', 'output', 'META', 'BLOB')

    dtype_map = {b'FLOAT': np.float32}
    tensor = np.frombuffer(data, dtype=dtype_map[dtype]).reshape(shape)
    label_id = np.argmax(tensor) - 1

    _, label = labels[str(label_id)]

    env.assertEqual(label, 'giant_panda')

    if env.useSlaves:
        con2 = env.getSlaveConnection()
        _, slave_dtype, _, slave_shape, _, slave_data = con2.execute_command(
            'AI.TENSORGET', 'output', 'META', 'BLOB')
        env.assertEqual(dtype, slave_dtype)
        env.assertEqual(shape, slave_shape)
        env.assertEqual(data, slave_data)


@skip_if_no_TF
def test_run_mobilenet_multiproc(env):
    if VALGRIND:
        env.debugPrint("skipping {} since VALGRIND=1".format(
            sys._getframe().f_code.co_name), force=True)
        return

    con = env.getConnection()

    input_var = 'input'
    output_var = 'MobilenetV2/Predictions/Reshape_1'

    model_pb, labels, img = load_mobilenet_test_data()
    con.execute_command('AI.MODELSET', 'mobilenet', 'TF', DEVICE,
                        'INPUTS', input_var, 'OUTPUTS', output_var, 'BLOB', model_pb)
    ensureSlaveSynced(con, env)

    run_test_multiproc(env, 30, run_mobilenet, (img, input_var, output_var))

    ensureSlaveSynced(con, env)

    _, dtype, _, shape, _, data = con.execute_command('AI.TENSORGET', 'output', 'META', 'BLOB')

    dtype_map = {b'FLOAT': np.float32}
    tensor = np.frombuffer(data, dtype=dtype_map[dtype]).reshape(shape)
    label_id = np.argmax(tensor) - 1

    _, label = labels[str(label_id)]

    env.assertEqual(
        label, 'giant_panda'
    )

    if env.useSlaves:
        con2 = env.getSlaveConnection()
        _, slave_dtype, _, slave_shape, _, slave_data = con2.execute_command(
            'AI.TENSORGET', 'output', 'META', 'BLOB')
        env.assertEqual(dtype, slave_dtype)
        env.assertEqual(shape, slave_shape)
        env.assertEqual(data, slave_data)


@skip_if_no_TF
def test_del_tf_model(env):
    con = env.getConnection()

    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    model_filename = os.path.join(test_data_path, 'graph.pb')

    with open(model_filename, 'rb') as f:
        model_pb = f.read()

    ret = con.execute_command('AI.MODELSET', 'm', 'TF', DEVICE,
                              'INPUTS', 'a', 'b', 'OUTPUTS', 'mul', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    con.execute_command('AI.MODELDEL', 'm')
    env.assertFalse(con.execute_command('EXISTS', 'm'))

    ensureSlaveSynced(con, env)
    if env.useSlaves:
        con2 = env.getSlaveConnection()
        env.assertFalse(con2.execute_command('EXISTS', 'm'))

    # ERR no model at key
    try:
        con.execute_command('AI.MODELDEL', 'm')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("model key is empty", exception.__str__())

    # ERR wrong type
    try:
        con.execute_command('SET', 'NOT_MODEL', 'BAR')
        con.execute_command('AI.MODELDEL', 'NOT_MODEL')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual(
            "WRONGTYPE Operation against a key holding the wrong kind of value", exception.__str__())


@skip_if_no_TF
def test_run_tf_model(env):
    con = env.getConnection()

    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    model_filename = os.path.join(test_data_path, 'graph.pb')

    with open(model_filename, 'rb') as f:
        model_pb = f.read()

    ret = con.execute_command('AI.MODELSET', 'm', 'TF', DEVICE,
                              'INPUTS', 'a', 'b', 'OUTPUTS', 'mul', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    ret = con.execute_command('AI.MODELGET', 'm', 'META')
    env.assertEqual(len(ret), 14)
    env.assertEqual(ret[5], b'')
    env.assertEqual(ret[11][0], b'a')
    env.assertEqual(ret[11][1], b'b')
    env.assertEqual(ret[13][0], b'mul')

    ret = con.execute_command('AI.MODELSET', 'm', 'TF', DEVICE, 'TAG', 'version:1',
                              'INPUTS', 'a', 'b', 'OUTPUTS', 'mul', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    ret = con.execute_command('AI.MODELGET', 'm', 'META')
    env.assertEqual(len(ret), 14)
    # TODO: enable me. CI is having issues on GPU asserts of TF and CPU
    if DEVICE == "CPU":
        env.assertEqual(ret[1], b'TF')
        env.assertEqual(ret[3], b'CPU')
    env.assertEqual(ret[5], b'version:1')
    env.assertEqual(ret[11][0], b'a')
    env.assertEqual(ret[11][1], b'b')
    env.assertEqual(ret[13][0], b'mul')

    con.execute_command('AI.TENSORSET', 'a', 'FLOAT',
                        2, 2, 'VALUES', 2, 3, 2, 3)
    con.execute_command('AI.TENSORSET', 'b', 'FLOAT',
                        2, 2, 'VALUES', 2, 3, 2, 3)

    ensureSlaveSynced(con, env)

    con.execute_command('AI.MODELRUN', 'm', 'INPUTS', 'a', 'b', 'OUTPUTS', 'c')

    ensureSlaveSynced(con, env)

    values = con.execute_command('AI.TENSORGET', 'c', 'VALUES')
    env.assertEqual(values, [b'4', b'9', b'4', b'9'])

    if env.useSlaves:
        con2 = env.getSlaveConnection()
        values2 = con2.execute_command('AI.TENSORGET', 'c', 'VALUES')
        env.assertEqual(values2, values)

    for _ in env.reloadingIterator():
        env.assertExists('m')
        env.assertExists('a')
        env.assertExists('b')
        env.assertExists('c')

    con.execute_command('AI.MODELDEL', 'm')
    ensureSlaveSynced(con, env)

    env.assertFalse(con.execute_command('EXISTS', 'm'))

    ensureSlaveSynced(con, env)
    if env.useSlaves:
        con2 = env.getSlaveConnection()
        env.assertFalse(con2.execute_command('EXISTS', 'm'))


@skip_if_no_TF
def test_run_tf2_model(env):
    con = env.getConnection()

    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    model_filename = os.path.join(test_data_path, 'graph_v2.pb')

    with open(model_filename, 'rb') as f:
        model_pb = f.read()

    ret = con.execute_command('AI.MODELSET', 'm', 'TF', DEVICE,
                              'INPUTS', 'x', 'OUTPUTS', 'Identity', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    ret = con.execute_command('AI.MODELGET', 'm', 'META')
    env.assertEqual(len(ret), 14)
    env.assertEqual(ret[5], b'')
    env.assertEqual(ret[11][0], b'x')
    env.assertEqual(ret[13][0], b'Identity')

    ret = con.execute_command('AI.MODELSET', 'm', 'TF', DEVICE, 'TAG', 'asdf',
                              'INPUTS', 'x', 'OUTPUTS', 'Identity', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    ret = con.execute_command('AI.MODELGET', 'm', 'META')
    env.assertEqual(len(ret), 14)
    env.assertEqual(ret[5], b'asdf')
    env.assertEqual(ret[11][0], b'x')
    env.assertEqual(ret[13][0], b'Identity')

    zero_values = [0] * (28 * 28)

    con.execute_command('AI.TENSORSET', 'x', 'FLOAT',
                        1, 1, 28, 28, 'VALUES', *zero_values)

    ensureSlaveSynced(con, env)

    con.execute_command('AI.MODELRUN', 'm', 'INPUTS', 'x', 'OUTPUTS', 'y')

    ensureSlaveSynced(con, env)

    values = con.execute_command('AI.TENSORGET', 'y', 'VALUES')
    for value in values:
        env.assertAlmostEqual(float(value), 0.1, 1E-4)

    if env.useSlaves:
        con2 = env.getSlaveConnection()
        values2 = con2.execute_command('AI.TENSORGET', 'y', 'VALUES')
        env.assertEqual(values2, values)

    for _ in env.reloadingIterator():
        env.assertExists('m')
        env.assertExists('x')
        env.assertExists('y')

    con.execute_command('AI.MODELDEL', 'm')
    ensureSlaveSynced(con, env)

    env.assertFalse(con.execute_command('EXISTS', 'm'))

    ensureSlaveSynced(con, env)
    if env.useSlaves:
        con2 = env.getSlaveConnection()
        env.assertFalse(con2.execute_command('EXISTS', 'm'))


@skip_if_no_TF
def test_run_tf_model_errors(env):
    con = env.getConnection()

    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    model_filename = os.path.join(test_data_path, 'graph.pb')
    wrong_model_filename = os.path.join(test_data_path, 'pt-minimal.pt')

    with open(model_filename, 'rb') as f:
        model_pb = f.read()

    with open(wrong_model_filename, 'rb') as f:
        wrong_model_pb = f.read()

    ret = con.execute_command('AI.MODELSET', 'm', 'TF', DEVICE,
                              'INPUTS', 'a', 'b', 'OUTPUTS', 'mul', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    try:
        con.execute_command('AI.MODELGET')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual(
            "wrong number of arguments for 'AI.MODELGET' command", exception.__str__())

    # ERR WRONGTYPE
    con.execute_command('SET', 'NOT_MODEL', 'BAR')
    try:
        con.execute_command('AI.MODELGET', 'NOT_MODEL')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual(
            "WRONGTYPE Operation against a key holding the wrong kind of value", exception.__str__())
    # cleanup
    con.execute_command('DEL', 'NOT_MODEL')

    # ERR model key is empty
    con.execute_command('DEL', 'DONT_EXIST')
    try:
        con.execute_command('AI.MODELGET', 'DONT_EXIST')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("model key is empty", exception.__str__())

    try:
        ret = con.execute_command('AI.MODELSET', 'm', 'TF', DEVICE,
                                  'INPUTS', 'a', 'b', 'OUTPUTS', 'mul', 'BLOB', wrong_model_pb)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("Invalid GraphDef", exception.__str__())

    try:
        con.execute_command('AI.MODELSET', 'm_1', 'TF',
                            'INPUTS', 'a', 'b', 'OUTPUTS', 'mul', 'BLOB', model_pb)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("Invalid DEVICE", exception.__str__())

    try:
        con.execute_command('AI.MODELSET', 'm_2', 'PORCH', DEVICE,
                            'INPUTS', 'a', 'b', 'OUTPUTS', 'mul', 'BLOB', model_pb)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("unsupported backend", exception.__str__())

    try:
        con.execute_command('AI.MODELSET', 'm_3', 'TORCH', DEVICE,
                            'INPUTS', 'a', 'b', 'OUTPUTS', 'mul', 'BLOB', model_pb)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.MODELSET', 'm_4', 'TF',
                            'INPUTS', 'a', 'b', 'OUTPUTS', 'mul', 'BLOB', model_pb)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("Invalid DEVICE", exception.__str__())

    try:
        con.execute_command('AI.MODELSET', 'm_5', 'TF', DEVICE,
                            'INPUTS', 'a', 'b', 'c', 'OUTPUTS', 'mul', 'BLOB', model_pb)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("WRONGTYPE Operation against a key holding the wrong kind of value", exception.__str__())

    try:
        con.execute_command('AI.MODELSET', 'm_6', 'TF', DEVICE,
                            'INPUTS', 'a', 'b', 'OUTPUTS', 'mult', 'BLOB', model_pb)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("Output node named \"mult\" not found in TF graph", exception.__str__())

    try:
        con.execute_command('AI.MODELSET', 'm_7', 'TF', DEVICE, 'BLOB', model_pb)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("Insufficient arguments, INPUTS and OUTPUTS not specified", exception.__str__())

    try:
        con.execute_command('AI.MODELSET', 'm_8', 'TF', DEVICE,
                            'INPUTS', 'a', 'b', 'OUTPUTS', 'mul')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("Insufficient arguments, missing model BLOB", exception.__str__())

    try:
        con.execute_command('AI.MODELSET', 'm_8', 'TF', DEVICE,
                            'INPUTS', 'a_', 'b', 'OUTPUTS', 'mul')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("Insufficient arguments, missing model BLOB", exception.__str__())

    try:
        con.execute_command('AI.MODELSET', 'm_8', 'TF', DEVICE,
                            'INPUTS', 'a', 'b', 'OUTPUTS', 'mul_')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("Insufficient arguments, missing model BLOB", exception.__str__())

    try:
        con.execute_command('AI.MODELSET', 'm_8', 'TF', DEVICE,
                            'INPUTS', 'a', 'b', 'OUTPUTS')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("Insufficient arguments, missing model BLOB",exception.__str__())

    try:
        con.execute_command('AI.MODELRUN', 'm', 'INPUTS', 'a', 'b')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("tensor key is empty", exception.__str__())

    try:
        con.execute_command('AI.MODELRUN', 'm', 'OUTPUTS', 'c')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("INPUTS not specified", exception.__str__())


@skip_if_no_TF
def test_run_tf_model_autobatch(env):
    if not TEST_PT:
        return

    con = env.getConnection()

    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    model_filename = os.path.join(test_data_path, 'graph.pb')

    with open(model_filename, 'rb') as f:
        model_pb = f.read()

    ret = con.execute_command('AI.MODELSET', 'm', 'TF', 'CPU',
                              'BATCHSIZE', 4, 'MINBATCHSIZE', 3,
                              'INPUTS', 'a', 'b', 'OUTPUTS', 'mul', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    con.execute_command('AI.TENSORSET', 'a', 'FLOAT',
                        2, 2, 'VALUES', 2, 3, 2, 3)
    con.execute_command('AI.TENSORSET', 'b', 'FLOAT',
                        2, 2, 'VALUES', 2, 3, 2, 3)

    con.execute_command('AI.TENSORSET', 'd', 'FLOAT',
                        2, 2, 'VALUES', 2, 3, 2, 3)
    con.execute_command('AI.TENSORSET', 'e', 'FLOAT',
                        2, 2, 'VALUES', 2, 3, 2, 3)

    ensureSlaveSynced(con, env)

    def run():
        con = env.getConnection()
        con.execute_command('AI.MODELRUN', 'm', 'INPUTS',
                            'd', 'e', 'OUTPUTS', 'f')
        ensureSlaveSynced(con, env)

    t = threading.Thread(target=run)
    t.start()

    con.execute_command('AI.MODELRUN', 'm', 'INPUTS', 'a', 'b', 'OUTPUTS', 'c')

    ensureSlaveSynced(con, env)

    values = con.execute_command('AI.TENSORGET', 'c', 'VALUES')
    env.assertEqual(values, [b'4', b'9', b'4', b'9'])

    values = con.execute_command('AI.TENSORGET', 'f', 'VALUES')
    env.assertEqual(values, [b'4', b'9', b'4', b'9'])


@skip_if_no_TF
def test_tensorflow_modelinfo(env):
    con = env.getConnection()
    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    model_filename = os.path.join(test_data_path, 'graph.pb')

    with open(model_filename, 'rb') as f:
        model_pb = f.read()

    ret = con.execute_command('AI.MODELSET', 'm', 'TF', DEVICE,
                              'INPUTS', 'a', 'b', 'OUTPUTS', 'mul', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')
    info = con.execute_command('AI.INFO', 'm')  # Getting initial info before modelrun
    info_dict0 = info_to_dict(info)
    expected = {'key': 'm', 'type': 'MODEL', 'backend': 'TF', 'device': DEVICE,
                'tag': '', 'duration': 0, 'samples': 0, 'calls': 0, 'errors': 0}
    env.assertEqual(info_dict0, expected)

    # second modelset; a corner case
    ret = con.execute_command('AI.MODELSET', 'm', 'TF', DEVICE,
                              'INPUTS', 'a', 'b', 'OUTPUTS', 'mul', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')
    info = con.execute_command('AI.INFO', 'm')  # this will fail
    info_dict1 = info_to_dict(info)
    env.assertEqual(info_dict1, info_dict0)

    ret = con.execute_command(
        'AI.TENSORSET', 'a', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command(
        'AI.TENSORSET', 'b', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    previous_duration = 0
    for call in range(1, 10):
        ret = con.execute_command(
            'AI.MODELRUN', 'm', 'INPUTS', 'a', 'b', 'OUTPUTS', 'c')
        env.assertEqual(ret, b'OK')
        ensureSlaveSynced(con, env)

        info = con.execute_command('AI.INFO', 'm')
        info_dict_0 = info_to_dict(info)

        env.assertEqual(info_dict_0['key'], 'm')
        env.assertEqual(info_dict_0['type'], 'MODEL')
        env.assertEqual(info_dict_0['backend'], 'TF')
        env.assertEqual(info_dict_0['device'], DEVICE)
        env.assertTrue(info_dict_0['duration'] > previous_duration)
        env.assertEqual(info_dict_0['samples'], 2 * call)
        env.assertEqual(info_dict_0['calls'], call)
        env.assertEqual(info_dict_0['errors'], 0)

        previous_duration = info_dict_0['duration']

    res = con.execute_command('AI.INFO', 'm', 'RESETSTAT')
    env.assertEqual(res, b'OK')
    info = con.execute_command('AI.INFO', 'm')
    info_dict_0 = info_to_dict(info)
    env.assertEqual(info_dict_0['duration'], 0)
    env.assertEqual(info_dict_0['samples'], 0)
    env.assertEqual(info_dict_0['calls'], 0)
    env.assertEqual(info_dict_0['errors'], 0)


@skip_if_no_TF
def test_tensorflow_modelrun_disconnect(env):
    red = env.getConnection()

    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    model_filename = os.path.join(test_data_path, 'graph.pb')

    with open(model_filename, 'rb') as f:
        model_pb = f.read()

    ret = red.execute_command('AI.MODELSET', 'm', 'TF', DEVICE,
                              'INPUTS', 'a', 'b', 'OUTPUTS', 'mul', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ret = red.execute_command(
        'AI.TENSORSET', 'a', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')

    ret = red.execute_command(
        'AI.TENSORSET', 'b', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(red, env)

    ret = send_and_disconnect(
        ('AI.MODELRUN', 'm', 'INPUTS', 'a', 'b', 'OUTPUTS', 'c'), red)
    env.assertEqual(ret, None)


@skip_if_no_TF
def test_tensorflow_modelrun_with_batch_and_minbatch(env):
    con = env.getConnection()
    batch_size = 2
    minbatch_size = 2
    model_name = 'model'
    another_model_name = 'another_model'
    inputvar = 'input'
    outputvar = 'MobilenetV2/Predictions/Reshape_1'

    model_pb, labels, img = load_mobilenet_test_data()

    con.execute_command('AI.MODELSET', model_name, 'TF', DEVICE,
                        'BATCHSIZE', batch_size, 'MINBATCHSIZE', minbatch_size,
                        'INPUTS', inputvar,
                        'OUTPUTS', outputvar,
                        'BLOB', model_pb)
    con.execute_command('AI.TENSORSET', 'input',
                        'FLOAT', 1, img.shape[1], img.shape[0], img.shape[2],
                        'BLOB', img.tobytes())

    def run(name=model_name, output_name='output'):
        con = env.getConnection()
        con.execute_command('AI.MODELRUN', name,
                            'INPUTS', 'input', 'OUTPUTS', output_name)

    # Running thrice since minbatchsize = 2
    p1 = mp.Process(target=run)
    p1.start()
    p2 = mp.Process(target=run)
    p2.start()
    p3 = mp.Process(target=run)
    p3.start()

    time.sleep(3)

    con.execute_command('AI.MODELSET', another_model_name, 'TF', DEVICE,
                        'BATCHSIZE', batch_size, 'MINBATCHSIZE', minbatch_size,
                        'INPUTS', inputvar,
                        'OUTPUTS', outputvar,
                        'BLOB', model_pb)

    p1b = mp.Process(target=run, args=(another_model_name, 'final1'))
    p1b.start()

    run(another_model_name, 'final2')

    _, dtype, _, shape, _, data = con.execute_command('AI.TENSORGET', 'final1', 'META', 'BLOB')
    dtype_map = {b'FLOAT': np.float32}
    tensor = np.frombuffer(data, dtype=dtype_map[dtype]).reshape(shape)
    label_id = np.argmax(tensor) - 1

    _, label = labels[str(label_id)]

    env.assertEqual(label, 'giant_panda')

    p3.terminate()


@skip_if_no_TF
def test_tensorflow_modelrun_financialNet(env):
    con = env.getConnection()

    model_pb, creditcard_transactions, creditcard_referencedata = load_creditcardfraud_data(env)

    tensor_number = 1
    for transaction_tensor in creditcard_transactions[:MAX_TRANSACTIONS]:
        ret = con.execute_command('AI.TENSORSET', 'transactionTensor:{0}'.format(tensor_number),
                                  'FLOAT', 1, 30,
                                  'BLOB', transaction_tensor.tobytes())
        env.assertEqual(ret, b'OK')
        tensor_number = tensor_number + 1

    tensor_number = 1
    for reference_tensor in creditcard_referencedata[:MAX_TRANSACTIONS]:
        ret = con.execute_command('AI.TENSORSET', 'referenceTensor:{0}'.format(tensor_number),
                                  'FLOAT', 1, 256,
                                  'BLOB', reference_tensor.tobytes())
        env.assertEqual(ret, b'OK')
        tensor_number = tensor_number + 1

    ret = con.execute_command('AI.MODELSET', 'financialNet', 'TF', DEVICE,
                              'INPUTS', 'transaction', 'reference', 'OUTPUTS', 'output', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    for tensor_number in range(1, MAX_TRANSACTIONS):
        for repetition in range(0, 10):
            ret = con.execute_command('AI.MODELRUN', 'financialNet', 'INPUTS',
                                      'transactionTensor:{}'.format(tensor_number),
                                      'referenceTensor:{}'.format(tensor_number), 'OUTPUTS',
                                      'classificationTensor:{}_{}'.format(tensor_number, repetition))
            env.assertEqual(ret, b'OK')


def test_tensorflow_modelrun_financialNet_multiproc(env):
    con = env.getConnection()

    model_pb, creditcard_transactions, creditcard_referencedata = load_creditcardfraud_data(env)

    tensor_number = 1
    for transaction_tensor in creditcard_transactions[:MAX_TRANSACTIONS]:
        ret = con.execute_command('AI.TENSORSET', 'transactionTensor:{0}'.format(tensor_number),
                                  'FLOAT', 1, 30,
                                  'BLOB', transaction_tensor.tobytes())
        env.assertEqual(ret, b'OK')
        tensor_number = tensor_number + 1

    tensor_number = 1
    for reference_tensor in creditcard_referencedata[:MAX_TRANSACTIONS]:
        ret = con.execute_command('AI.TENSORSET', 'referenceTensor:{0}'.format(tensor_number),
                                  'FLOAT', 1, 256,
                                  'BLOB', reference_tensor.tobytes())
        env.assertEqual(ret, b'OK')
        tensor_number = tensor_number + 1

    ret = con.execute_command('AI.MODELSET', 'financialNet', 'TF', DEVICE,
                              'INPUTS', 'transaction', 'reference', 'OUTPUTS', 'output', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    def functor_financialNet(env, key_max, repetitions):
        for tensor_number in range(1, key_max):
            for repetition in range(1, repetitions):
                ret = env.execute_command('AI.MODELRUN', 'financialNet', 'INPUTS',
                                        'transactionTensor:{}'.format(tensor_number),
                                        'referenceTensor:{}'.format(tensor_number), 'OUTPUTS',
                                        'classificationTensor:{}_{}'.format(tensor_number, repetition))

    t = time.time()
    run_test_multiproc(env, 10,
                       lambda env: functor_financialNet(env,MAX_TRANSACTIONS,100) )
    elapsed_time = time.time() - t
    total_ops = len(transaction_tensor)*100
    avg_ops_sec = total_ops/elapsed_time
    # env.debugPrint("AI.MODELRUN elapsed time(sec) {:6.2f}\tTotal ops  {:10.2f}\tAvg. ops/sec {:10.2f}".format(elapsed_time, total_ops, avg_ops_sec), True)
