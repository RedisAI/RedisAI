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

    model_pb, input_var, output_var, labels, img = load_mobilenet_v2_test_data()

    con.execute_command('AI.MODELSET', 'mobilenet{1}', 'TF', DEVICE,
                        'INPUTS', input_var, 'OUTPUTS', output_var, 'BLOB', model_pb)

    ensureSlaveSynced(con, env)

    mobilenet_model_serialized = con.execute_command(
        'AI.MODELGET', 'mobilenet{1}', 'META')

    ensureSlaveSynced(con, env)
    if env.useSlaves:
        con2 = env.getSlaveConnection()
        slave_mobilenet_model_serialized = con2.execute_command(
            'AI.MODELGET', 'mobilenet{1}', 'META')
        env.assertEqual(len(mobilenet_model_serialized),
                        len(slave_mobilenet_model_serialized))

    con.execute_command('AI.TENSORSET', 'input{1}',
                        'FLOAT', 1, img.shape[1], img.shape[0], img.shape[2],
                        'BLOB', img.tobytes())

    ensureSlaveSynced(con, env)
    input_tensor_meta = con.execute_command('AI.TENSORGET', 'input{1}', 'META')
    env.assertEqual(
        [b'dtype', b'FLOAT', b'shape', [1, img.shape[1], img.shape[0], img.shape[2]]], input_tensor_meta)

    ensureSlaveSynced(con, env)
    if env.useSlaves:
        con2 = env.getSlaveConnection()
        slave_tensor_meta = con2.execute_command(
            'AI.TENSORGET', 'input{1}', 'META')
        env.assertEqual(input_tensor_meta, slave_tensor_meta)

    con.execute_command('AI.MODELRUN', 'mobilenet{1}',
                        'INPUTS', 'input{1}', 'OUTPUTS', 'output{1}')

    ensureSlaveSynced(con, env)

    _, dtype, _, shape, _, data = con.execute_command('AI.TENSORGET', 'output{1}', 'META', 'BLOB')

    dtype_map = {b'FLOAT': np.float32}
    tensor = np.frombuffer(data, dtype=dtype_map[dtype]).reshape(shape)
    label_id = np.argmax(tensor) - 1

    _, label = labels[str(label_id)]

    env.assertEqual(label, 'giant_panda')

    if env.useSlaves:
        con2 = env.getSlaveConnection()
        _, slave_dtype, _, slave_shape, _, slave_data = con2.execute_command(
            'AI.TENSORGET', 'output{1}', 'META', 'BLOB')
        env.assertEqual(dtype, slave_dtype)
        env.assertEqual(shape, slave_shape)
        env.assertEqual(data, slave_data)


@skip_if_no_TF
def test_run_mobilenet_multiproc(env):

    con = env.getConnection()

    model_pb, input_var, output_var, labels, img = load_mobilenet_v2_test_data()
    con.execute_command('AI.MODELSET', 'mobilenet{1}', 'TF', DEVICE,
                        'INPUTS', input_var, 'OUTPUTS', output_var, 'BLOB', model_pb)
    ensureSlaveSynced(con, env)

    run_test_multiproc(env, 30, run_mobilenet, (img, input_var, output_var))

    ensureSlaveSynced(con, env)

    _, dtype, _, shape, _, data = con.execute_command('AI.TENSORGET', 'output{1}', 'META', 'BLOB')

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
            'AI.TENSORGET', 'output{1}', 'META', 'BLOB')
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

    ret = con.execute_command('AI.MODELSET', 'm{1}', 'TF', DEVICE,
                              'INPUTS', 'a', 'b', 'OUTPUTS', 'mul', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    con.execute_command('AI.MODELDEL', 'm{1}')
    env.assertFalse(con.execute_command('EXISTS', 'm{1}'))

    ensureSlaveSynced(con, env)
    if env.useSlaves:
        con2 = env.getSlaveConnection()
        env.assertFalse(con2.execute_command('EXISTS', 'm{1}'))

    # ERR no model at key
    try:
        con.execute_command('AI.MODELDEL', 'm{1}')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("model key is empty", exception.__str__())

    # ERR wrong type
    try:
        con.execute_command('SET', 'NOT_MODEL{1}', 'BAR')
        con.execute_command('AI.MODELDEL', 'NOT_MODEL{1}')
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

    ret = con.execute_command('AI.MODELSET', 'm{1}', 'TF', DEVICE,
                              'INPUTS', 'a', 'b', 'OUTPUTS', 'mul', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    ret = con.execute_command('AI.MODELGET', 'm{1}', 'META')
    env.assertEqual(len(ret), 14)
    env.assertEqual(ret[5], b'')
    env.assertEqual(ret[11][0], b'a')
    env.assertEqual(ret[11][1], b'b')
    env.assertEqual(ret[13][0], b'mul')

    ret = con.execute_command('AI.MODELSET', 'm{1}', 'TF', DEVICE, 'TAG', 'version:1',
                              'INPUTS', 'a', 'b', 'OUTPUTS', 'mul', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    ret = con.execute_command('AI.MODELGET', 'm{1}', 'META')
    env.assertEqual(len(ret), 14)
    # TODO: enable me. CI is having issues on GPU asserts of TF and CPU
    if DEVICE == "CPU":
        env.assertEqual(ret[1], b'TF')
        env.assertEqual(ret[3], b'CPU')
    env.assertEqual(ret[5], b'version:1')
    env.assertEqual(ret[11][0], b'a')
    env.assertEqual(ret[11][1], b'b')
    env.assertEqual(ret[13][0], b'mul')

    con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT',
                        2, 2, 'VALUES', 2, 3, 2, 3)
    con.execute_command('AI.TENSORSET', 'b{1}', 'FLOAT',
                        2, 2, 'VALUES', 2, 3, 2, 3)

    ensureSlaveSynced(con, env)

    con.execute_command('AI.MODELRUN', 'm{1}', 'INPUTS', 'a{1}', 'b{1}', 'OUTPUTS', 'c{1}')

    ensureSlaveSynced(con, env)

    values = con.execute_command('AI.TENSORGET', 'c{1}', 'VALUES')
    env.assertEqual(values, [b'4', b'9', b'4', b'9'])

    if env.useSlaves:
        con2 = env.getSlaveConnection()
        values2 = con2.execute_command('AI.TENSORGET', 'c{1}', 'VALUES')
        env.assertEqual(values2, values)

    for _ in env.reloadingIterator():
        env.assertExists('m{1}')
        env.assertExists('a{1}')
        env.assertExists('b{1}')
        env.assertExists('c{1}')

    con.execute_command('AI.MODELDEL', 'm{1}')
    ensureSlaveSynced(con, env)

    env.assertFalse(con.execute_command('EXISTS', 'm{1}'))

    ensureSlaveSynced(con, env)
    if env.useSlaves:
        con2 = env.getSlaveConnection()
        env.assertFalse(con2.execute_command('EXISTS', 'm{1}'))


@skip_if_no_TF
def test_run_tf2_model(env):
    con = env.getConnection()

    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    model_filename = os.path.join(test_data_path, 'graph_v2.pb')

    with open(model_filename, 'rb') as f:
        model_pb = f.read()

    ret = con.execute_command('AI.MODELSET', 'm{1}', 'TF', DEVICE,
                              'INPUTS', 'x', 'OUTPUTS', 'Identity', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    ret = con.execute_command('AI.MODELGET', 'm{1}', 'META')
    env.assertEqual(len(ret), 14)
    env.assertEqual(ret[5], b'')
    env.assertEqual(ret[11][0], b'x')
    env.assertEqual(ret[13][0], b'Identity')

    ret = con.execute_command('AI.MODELSET', 'm{1}', 'TF', DEVICE, 'TAG', 'asdf',
                              'INPUTS', 'x', 'OUTPUTS', 'Identity', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    ret = con.execute_command('AI.MODELGET', 'm{1}', 'META')
    env.assertEqual(len(ret), 14)
    env.assertEqual(ret[5], b'asdf')
    env.assertEqual(ret[11][0], b'x')
    env.assertEqual(ret[13][0], b'Identity')

    zero_values = [0] * (28 * 28)

    con.execute_command('AI.TENSORSET', 'x{1}', 'FLOAT',
                        1, 1, 28, 28, 'VALUES', *zero_values)

    ensureSlaveSynced(con, env)

    con.execute_command('AI.MODELRUN', 'm{1}', 'INPUTS', 'x{1}', 'OUTPUTS', 'y{1}')

    ensureSlaveSynced(con, env)

    values = con.execute_command('AI.TENSORGET', 'y{1}', 'VALUES')
    for value in values:
        env.assertAlmostEqual(float(value), 0.1, 1E-4)

    if env.useSlaves:
        con2 = env.getSlaveConnection()
        values2 = con2.execute_command('AI.TENSORGET', 'y{1}', 'VALUES')
        env.assertEqual(values2, values)

    for _ in env.reloadingIterator():
        env.assertExists('m{1}')
        env.assertExists('x{1}')
        env.assertExists('y{1}')

    con.execute_command('AI.MODELDEL', 'm{1}')
    ensureSlaveSynced(con, env)

    env.assertFalse(con.execute_command('EXISTS', 'm{1}'))

    ensureSlaveSynced(con, env)
    if env.useSlaves:
        con2 = env.getSlaveConnection()
        env.assertFalse(con2.execute_command('EXISTS', 'm{1}'))


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

    ret = con.execute_command('AI.MODELSET', 'm{1}', 'TF', DEVICE,
                              'INPUTS', 'a', 'b', 'OUTPUTS', 'mul', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    try:
        con.execute_command('AI.MODELGET')
        env.assertFalse(True)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual(
            "wrong number of arguments for 'AI.MODELGET' command", exception.__str__())

    # ERR WRONGTYPE
    con.execute_command('SET', 'NOT_MODEL{1}', 'BAR')
    try:
        con.execute_command('AI.MODELGET', 'NOT_MODEL{1}')
        env.assertFalse(True)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual(
            "WRONGTYPE Operation against a key holding the wrong kind of value", exception.__str__())
    # cleanup
    con.execute_command('DEL', 'NOT_MODEL{1}')

    # ERR model key is empty
    con.execute_command('DEL', 'DONT_EXIST{1}')
    try:
        con.execute_command('AI.MODELGET', 'DONT_EXIST{1}')
        env.assertFalse(True)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("model key is empty", exception.__str__())

    try:
        ret = con.execute_command('AI.MODELSET', 'm{1}', 'TF', DEVICE,
                                  'INPUTS', 'a', 'b', 'OUTPUTS', 'mul', 'BLOB', wrong_model_pb)
        env.assertFalse(True)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("Invalid GraphDef", exception.__str__())

    try:
        con.execute_command('AI.MODELSET', 'm_1{1}', 'TF',
                            'INPUTS', 'a', 'b', 'OUTPUTS', 'mul', 'BLOB', model_pb)
        env.assertFalse(True)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("Invalid DEVICE", exception.__str__())

    try:
        con.execute_command('AI.MODELSET', 'm_2{1}', 'PORCH', DEVICE,
                            'INPUTS', 'a', 'b', 'OUTPUTS', 'mul', 'BLOB', model_pb)
        env.assertFalse(True)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("unsupported backend", exception.__str__())

    try:
        con.execute_command('AI.MODELSET', 'm_3{1}', 'TORCH', DEVICE,
                            'INPUTS', 'a', 'b', 'OUTPUTS', 'mul', 'BLOB', model_pb)
        env.assertFalse(True)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.MODELSET', 'm_4{1}', 'TF',
                            'INPUTS', 'a', 'b', 'OUTPUTS', 'mul', 'BLOB', model_pb)
        env.assertFalse(True)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("Invalid DEVICE", exception.__str__())

    try:
        con.execute_command('AI.MODELSET', 'm_5{1}', 'TF', DEVICE,
                            'INPUTS', 'a', 'b', 'bad_input', 'OUTPUTS', 'mul', 'BLOB', model_pb)
        env.assertFalse(True)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("Input node named \"bad_input\" not found in TF graph.", exception.__str__())

    try:
        con.execute_command('AI.MODELSET', 'm_6{1}', 'TF', DEVICE,
                            'INPUTS', 'a', 'b', 'OUTPUTS', 'mult', 'BLOB', model_pb)
        env.assertFalse(True)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("Output node named \"mult\" not found in TF graph", exception.__str__())

    try:
        con.execute_command('AI.MODELSET', 'm_7{1}', 'TF', DEVICE, 'BLOB', model_pb)
        env.assertFalse(True)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("Insufficient arguments, INPUTS and OUTPUTS not specified for TF model", exception.__str__())

    try:
        con.execute_command('AI.MODELSET', 'm_8{1}', 'TF', DEVICE,
                            'INPUTS', 'a', 'b', 'OUTPUTS', 'mul')
        env.assertFalse(True)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("Insufficient arguments, missing model BLOB", exception.__str__())

    try:
        con.execute_command('AI.MODELSET', 'm_8{1}', 'TF', DEVICE,
                            'INPUTS', 'a_', 'b', 'OUTPUTS', 'mul')
        env.assertFalse(True)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("Insufficient arguments, missing model BLOB", exception.__str__())

    try:
        con.execute_command('AI.MODELSET', 'm_8{1}', 'TF', DEVICE,
                            'INPUTS', 'a{1}', 'b{1}', 'OUTPUTS', 'mul_')
        env.assertFalse(True)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("Insufficient arguments, missing model BLOB", exception.__str__())

    try:
        con.execute_command('AI.MODELSET', 'm_8{1}', 'TF', DEVICE,
                            'INPUTS', 'a', 'b', 'OUTPUTS')
        env.assertFalse(True)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("Insufficient arguments, missing model BLOB",exception.__str__())

    try:
        con.execute_command('AI.MODELRUN', 'm{1}', 'INPUTS', 'a{1}', 'b{1}', 'OUTPUTS')
        env.assertFalse(True)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("Number of keys given as OUTPUTS here does not match model definition", exception.__str__())

    try:
        con.execute_command('AI.MODELRUN', 'm{1}', 'OUTPUTS', 'c{1}', 'd{1}', 'e{1}')
        env.assertFalse(True)
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

    ret = con.execute_command('AI.MODELSET', 'm{1}', 'TF', 'CPU',
                              'BATCHSIZE', 4, 'MINBATCHSIZE', 3,
                              'INPUTS', 'a', 'b', 'OUTPUTS', 'mul', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT',
                        2, 2, 'VALUES', 2, 3, 2, 3)
    con.execute_command('AI.TENSORSET', 'b{1}', 'FLOAT',
                        2, 2, 'VALUES', 2, 3, 2, 3)

    con.execute_command('AI.TENSORSET', 'd{1}', 'FLOAT',
                        2, 2, 'VALUES', 2, 3, 2, 3)
    con.execute_command('AI.TENSORSET', 'e{1}', 'FLOAT',
                        2, 2, 'VALUES', 2, 3, 2, 3)

    ensureSlaveSynced(con, env)

    def run():
        con = env.getConnection()
        con.execute_command('AI.MODELRUN', 'm{1}', 'INPUTS',
                            'd{1}', 'e{1}', 'OUTPUTS', 'f{1}')
        ensureSlaveSynced(con, env)

    t = threading.Thread(target=run)
    t.start()

    con.execute_command('AI.MODELRUN', 'm{1}', 'INPUTS', 'a{1}', 'b{1}', 'OUTPUTS', 'c{1}')
    t.join()

    ensureSlaveSynced(con, env)

    values = con.execute_command('AI.TENSORGET', 'c{1}', 'VALUES')
    env.assertEqual(values, [b'4', b'9', b'4', b'9'])

    values = con.execute_command('AI.TENSORGET', 'f{1}', 'VALUES')
    env.assertEqual(values, [b'4', b'9', b'4', b'9'])


@skip_if_no_TF
def test_tensorflow_modelinfo(env):
    con = env.getConnection()
    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    model_filename = os.path.join(test_data_path, 'graph.pb')

    with open(model_filename, 'rb') as f:
        model_pb = f.read()

    ret = con.execute_command('AI.MODELSET', 'm{1}', 'TF', DEVICE,
                              'INPUTS', 'a', 'b', 'OUTPUTS', 'mul', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')
    info = con.execute_command('AI.INFO', 'm{1}')  # Getting initial info before modelrun
    info_dict0 = info_to_dict(info)
    expected = {'key': 'm{1}', 'type': 'MODEL', 'backend': 'TF', 'device': DEVICE,
                'tag': '', 'duration': 0, 'samples': 0, 'calls': 0, 'errors': 0}
    env.assertEqual(info_dict0, expected)

    # second modelset; a corner case
    ret = con.execute_command('AI.MODELSET', 'm{1}', 'TF', DEVICE,
                              'INPUTS', 'a', 'b', 'OUTPUTS', 'mul', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')
    info = con.execute_command('AI.INFO', 'm{1}')  # this will fail
    info_dict1 = info_to_dict(info)
    env.assertEqual(info_dict1, info_dict0)

    ret = con.execute_command(
        'AI.TENSORSET', 'a{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command(
        'AI.TENSORSET', 'b{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    previous_duration = 0
    for call in range(1, 10):
        ret = con.execute_command(
            'AI.MODELRUN', 'm{1}', 'INPUTS', 'a{1}', 'b{1}', 'OUTPUTS', 'c{1}')
        env.assertEqual(ret, b'OK')
        ensureSlaveSynced(con, env)

        info = con.execute_command('AI.INFO', 'm{1}')
        info_dict_0 = info_to_dict(info)

        env.assertEqual(info_dict_0['key'], 'm{1}')
        env.assertEqual(info_dict_0['type'], 'MODEL')
        env.assertEqual(info_dict_0['backend'], 'TF')
        env.assertEqual(info_dict_0['device'], DEVICE)
        env.assertTrue(info_dict_0['duration'] > previous_duration)
        env.assertEqual(info_dict_0['samples'], 2 * call)
        env.assertEqual(info_dict_0['calls'], call)
        env.assertEqual(info_dict_0['errors'], 0)

        previous_duration = info_dict_0['duration']

    res = con.execute_command('AI.INFO', 'm{1}', 'RESETSTAT')
    env.assertEqual(res, b'OK')
    info = con.execute_command('AI.INFO', 'm{1}')
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

    ret = red.execute_command('AI.MODELSET', 'm{1}', 'TF', DEVICE,
                              'INPUTS', 'a', 'b', 'OUTPUTS', 'mul', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ret = red.execute_command(
        'AI.TENSORSET', 'a{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')

    ret = red.execute_command(
        'AI.TENSORSET', 'b{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(red, env)

    ret = send_and_disconnect(
        ('AI.MODELRUN', 'm{1}', 'INPUTS', 'a{1}', 'b{1}', 'OUTPUTS', 'c{1}'), red)
    env.assertEqual(ret, None)


@skip_if_no_TF
def test_tensorflow_modelrun_with_batch_and_minbatch(env):

    con = env.getConnection()
    batch_size = 2
    minbatch_size = 2
    model_name = 'model{1}'
    another_model_name = 'another_model{1}'
    model_pb, input_var, output_var, labels, img = load_mobilenet_v2_test_data()

    con.execute_command('AI.MODELSET', model_name, 'TF', DEVICE,
                        'BATCHSIZE', batch_size, 'MINBATCHSIZE', minbatch_size,
                        'INPUTS', input_var,
                        'OUTPUTS', output_var,
                        'BLOB', model_pb)
    con.execute_command('AI.TENSORSET', 'input{1}',
                        'FLOAT', 1, img.shape[1], img.shape[0], img.shape[2],
                        'BLOB', img.tobytes())

    def run(name=model_name, output_name='output{1}'):
        con = env.getConnection()
        con.execute_command('AI.MODELRUN', name,
                            'INPUTS', 'input{1}', 'OUTPUTS', output_name)
    

    # Running thrice since minbatchsize = 2
    # The third process will hang until termintation or until a new process will execute the model with the same properties.
    processes = []
    for i in range(3):
        p = mp.Process(target=run)
        p.start()
        processes.append(p)

    time.sleep(3)

    con.execute_command('AI.MODELSET', another_model_name, 'TF', DEVICE,
                        'BATCHSIZE', batch_size, 'MINBATCHSIZE', minbatch_size,
                        'INPUTS', input_var,
                        'OUTPUTS', output_var,
                        'BLOB', model_pb)

    p1b = mp.Process(target=run, args=(another_model_name, 'final1{1}'))
    p1b.start()
    run(another_model_name, 'final2{1}')

    p1b.join()

    _, dtype, _, shape, _, data = con.execute_command('AI.TENSORGET', 'final1{1}', 'META', 'BLOB')
    dtype_map = {b'FLOAT': np.float32}
    tensor = np.frombuffer(data, dtype=dtype_map[dtype]).reshape(shape)
    label_id = np.argmax(tensor) - 1

    _, label = labels[str(label_id)]

    env.assertEqual(label, 'giant_panda')

    for p in processes:
        p.terminate()


# @skip_if_no_TF
# def test_tensorflow_modelrun_with_timeout(env):
#     con = env.getConnection()
#     batch_size = 2
#     minbatch_size = 2
#     timeout = 1
#     model_name = 'model{1}'
#     model_pb, input_var, output_var, labels, img = load_mobilenet_v2_test_data()

#     con.execute_command('AI.MODELSET', model_name, 'TF', DEVICE,
#                         'BATCHSIZE', batch_size, 'MINBATCHSIZE', minbatch_size,
#                         'INPUTS', input_var,
#                         'OUTPUTS', output_var,
#                         'BLOB', model_pb)
#     con.execute_command('AI.TENSORSET', 'input{1}',
#                         'FLOAT', 1, img.shape[1], img.shape[0], img.shape[2],
#                         'BLOB', img.tobytes())

#     res = con.execute_command('AI.MODELRUN', model_name,
#                         'TIMEOUT', timeout,
#                         'INPUTS', 'input{1}', 'OUTPUTS', 'output{1}')

#     env.assertEqual(b'TIMEDOUT', res)


@skip_if_no_TF
def test_tensorflow_modelrun_with_batch_minbatch_and_timeout(env):
    con = env.getConnection()
    batch_size = 2
    minbatch_size = 2
    minbatch_timeout = 1000
    model_name = 'model{1}'
    model_pb, input_var, output_var, labels, img = load_mobilenet_v2_test_data()

    con.execute_command('AI.MODELSET', model_name, 'TF', DEVICE,
                        'BATCHSIZE', batch_size, 'MINBATCHSIZE', minbatch_size,
                        'MINBATCHTIMEOUT', minbatch_timeout,
                        'INPUTS', input_var,
                        'OUTPUTS', output_var,
                        'BLOB', model_pb)
    con.execute_command('AI.TENSORSET', 'input{1}',
                        'FLOAT', 1, img.shape[1], img.shape[0], img.shape[2],
                        'BLOB', img.tobytes())

    con.execute_command('AI.MODELRUN', model_name,
                        'INPUTS', 'input{1}', 'OUTPUTS', 'output{1}')

    _, dtype, _, shape, _, data = con.execute_command('AI.TENSORGET', 'output{1}', 'META', 'BLOB')
    dtype_map = {b'FLOAT': np.float32}
    tensor = np.frombuffer(data, dtype=dtype_map[dtype]).reshape(shape)
    label_id = np.argmax(tensor) - 1

    _, label = labels[str(label_id)]

    env.assertEqual(label, 'giant_panda')


@skip_if_no_TF
def test_tensorflow_modelrun_financialNet(env):
    con = env.getConnection()

    model_pb, creditcard_transactions, creditcard_referencedata = load_creditcardfraud_data(env)

    tensor_number = 1
    for transaction_tensor in creditcard_transactions[:MAX_TRANSACTIONS]:
        ret = con.execute_command('AI.TENSORSET', 'transactionTensor{{1}}:{0}'.format(tensor_number),
                                  'FLOAT', 1, 30,
                                  'BLOB', transaction_tensor.tobytes())
        env.assertEqual(ret, b'OK')
        tensor_number = tensor_number + 1

    tensor_number = 1
    for reference_tensor in creditcard_referencedata[:MAX_TRANSACTIONS]:
        ret = con.execute_command('AI.TENSORSET', 'referenceTensor{{1}}:{0}'.format(tensor_number),
                                  'FLOAT', 1, 256,
                                  'BLOB', reference_tensor.tobytes())
        env.assertEqual(ret, b'OK')
        tensor_number = tensor_number + 1

    ret = con.execute_command('AI.MODELSET', 'financialNet{1}', 'TF', DEVICE,
                              'INPUTS', 'transaction', 'reference', 'OUTPUTS', 'output', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    for tensor_number in range(1, MAX_TRANSACTIONS):
        for repetition in range(0, 10):
            ret = con.execute_command('AI.MODELRUN', 'financialNet{1}', 'INPUTS',
                                      'transactionTensor{{1}}:{}'.format(tensor_number),
                                      'referenceTensor{{1}}:{}'.format(tensor_number), 'OUTPUTS',
                                      'classificationTensor{{1}}:{}_{}'.format(tensor_number, repetition))
            env.assertEqual(ret, b'OK')


def test_tensorflow_modelrun_financialNet_multiproc(env):
    con = env.getConnection()

    model_pb, creditcard_transactions, creditcard_referencedata = load_creditcardfraud_data(env)

    tensor_number = 1
    for transaction_tensor in creditcard_transactions[:MAX_TRANSACTIONS]:
        ret = con.execute_command('AI.TENSORSET', 'transactionTensor{{1}}:{0}'.format(tensor_number),
                                  'FLOAT', 1, 30,
                                  'BLOB', transaction_tensor.tobytes())
        env.assertEqual(ret, b'OK')
        tensor_number = tensor_number + 1

    tensor_number = 1
    for reference_tensor in creditcard_referencedata[:MAX_TRANSACTIONS]:
        ret = con.execute_command('AI.TENSORSET', 'referenceTensor{{1}}:{0}'.format(tensor_number),
                                  'FLOAT', 1, 256,
                                  'BLOB', reference_tensor.tobytes())
        env.assertEqual(ret, b'OK')
        tensor_number = tensor_number + 1

    ret = con.execute_command('AI.MODELSET', 'financialNet{1}', 'TF', DEVICE,
                              'INPUTS', 'transaction', 'reference', 'OUTPUTS', 'output', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    def functor_financialNet(env, key_max, repetitions):
        for tensor_number in range(1, key_max):
            for repetition in range(1, repetitions):
                ret = env.execute_command('AI.MODELRUN', 'financialNet{1}', 'INPUTS',
                                          'transactionTensor{{1}}:{}'.format(tensor_number),
                                          'referenceTensor{{1}}:{}'.format(tensor_number), 'OUTPUTS',
                                          'classificationTensor{{1}}:{}_{}'.format(tensor_number, repetition))

    t = time.time()
    run_test_multiproc(env, 10,
                       lambda env: functor_financialNet(env, MAX_TRANSACTIONS, 100) )
    elapsed_time = time.time() - t
    total_ops = len(transaction_tensor)*100
    avg_ops_sec = total_ops/elapsed_time
    # env.debugPrint("AI.MODELRUN elapsed time(sec) {:6.2f}\tTotal ops  {:10.2f}\tAvg. ops/sec {:10.2f}".format(elapsed_time, total_ops, avg_ops_sec), True)


def test_tensorflow_modelrun_scriptrun_resnet(env):
    if (not TEST_TF or not TEST_PT):
        return
    con = env.getConnection()
    model_name = 'imagenet_model{1}'
    script_name = 'imagenet_script{1}'
    inputvar = 'images'
    outputvar = 'output'


    model_pb, script, labels, img = load_resnet_test_data()

    ret = con.execute_command('AI.MODELSET', model_name, 'TF', DEVICE,
                              'INPUTS', inputvar,
                              'OUTPUTS', outputvar,
                              'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.SCRIPTSET', script_name, DEVICE, 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    image_key = 'image1{1}'
    temp_key1 = 'temp_key1{1}'
    temp_key2 = 'temp_key2{1}'
    output_key = 'output{1}'

    ret = con.execute_command('AI.TENSORSET', image_key,
                              'UINT8', img.shape[1], img.shape[0], 3,
                              'BLOB', img.tobytes())
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.SCRIPTRUN',  script_name,
                              'pre_process_3ch', 'INPUTS', image_key, 'OUTPUTS', temp_key1 )
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.MODELRUN', model_name,
                              'INPUTS', temp_key1, 'OUTPUTS', temp_key2 )
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.SCRIPTRUN',  script_name,
                              'post_process', 'INPUTS', temp_key2, 'OUTPUTS', output_key )
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    ret = con.execute_command('AI.TENSORGET', output_key, 'VALUES' )
    # tf model has 100 classes [0,999]
    env.assertEqual(ret[0]>=0 and ret[0]<1001, True)

@skip_if_no_TF
def test_tf_info(env):
    con = env.getConnection()

    ret = con.execute_command('AI.INFO')
    env.assertEqual(6, len(ret))

    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    model_filename = os.path.join(test_data_path, 'graph.pb')

    with open(model_filename, 'rb') as f:
        model_pb = f.read()

    con.execute_command('AI.MODELSET', 'm{1}', 'TF', DEVICE,
                              'INPUTS', 'a', 'b', 'OUTPUTS', 'mul', 'BLOB', model_pb)
    
    ret = con.execute_command('AI.INFO')
    env.assertEqual(8, len(ret))
    env.assertEqual(b'TensorFlow version', ret[6])
