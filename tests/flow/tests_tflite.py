import redis

from includes import *

'''
python -m RLTest --test tests_tflite.py --module path/to/redisai.so
'''


def test_run_tflite_model(env):
    if not TEST_TFLITE:
        env.debugPrint("skipping {} since TEST_TFLITE=0".format(sys._getframe().f_code.co_name), force=True)
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

    ret = con.execute_command('AI.MODELSET', 'm{1}', 'TFLITE', 'CPU', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.MODELGET', 'm{1}', 'META')
    env.assertEqual(len(ret), 14)
    env.assertEqual(ret[5], b'')

    ret = con.execute_command('AI.MODELSET', 'm{1}', 'TFLITE', 'CPU', 'TAG', 'asdf', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.MODELGET', 'm{1}', 'META')
    env.assertEqual(len(ret), 14)
    env.assertEqual(ret[5], b'asdf')

    ret = con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 1, 1, 28, 28, 'BLOB', sample_raw)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    ret = con.execute_command('AI.MODELGET', 'm{1}', 'META')
    env.assertEqual(len(ret), 14)
    # TODO: enable me. CI is having issues on GPU asserts of TFLITE and CPU
    if DEVICE == "CPU":
        env.assertEqual(ret[1], b'TFLITE')
        env.assertEqual(ret[3], b'CPU')

    con.execute_command('AI.MODELRUN', 'm{1}', 'INPUTS', 'a{1}', 'OUTPUTS', 'b{1}', 'c{1}')

    ensureSlaveSynced(con, env)

    values = con.execute_command('AI.TENSORGET', 'b{1}', 'VALUES')

    env.assertEqual(values[0], 1)


def test_run_tflite_model_errors(env):
    if not TEST_TFLITE:
        env.debugPrint("skipping {} since TEST_TFLITE=0".format(sys._getframe().f_code.co_name), force=True)
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

    ret = con.execute_command('AI.MODELSET', 'm_2{1}', 'TFLITE', 'CPU', 'BLOB', model_pb2)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.MODELSET', 'm{1}', 'TFLITE', 'CPU', 'TAG', 'asdf', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 1, 1, 28, 28, 'BLOB', sample_raw)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    try:
        con.execute_command('AI.MODELSET', 'm_1{1}', 'TFLITE', model_pb)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("Invalid DEVICE", exception.__str__())

    try:
        con.execute_command('AI.MODELSET', 'm_2{1}', 'BLOB', model_pb)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("unsupported backend", exception.__str__())

    try:
        con.execute_command('AI.MODELRUN', 'm_2{1}', 'INPUTS', 'EMPTY_INPUT{1}', 'OUTPUTS', 'EMPTY_OUTPUT{1}')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("Number of keys given as OUTPUTS here does not match model definition", exception.__str__())

    try:
        con.execute_command('AI.MODELRUN', 'm_2{1}')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("wrong number of arguments for 'AI.MODELRUN' command", exception.__str__())

    try:
        con.execute_command('AI.MODELRUN', 'EMPTY', 'INPUTS')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("wrong number of arguments for 'AI.MODELRUN' command", exception.__str__())

    try:
        con.execute_command('AI.MODELRUN', 'm_2{1}', 'INPUTS', 'a{1}', 'b{1}', 'c{1}')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("Number of keys given as INPUTS here does not match model definition", exception.__str__())

    try:
        con.execute_command('AI.MODELRUN', 'm_2{1}', 'a{1}', 'b{1}', 'OUTPUTS', 'c{1}')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("INPUTS not specified", exception.__str__())

    try:
        con.execute_command('AI.MODELRUN', 'm{1}', 'OUTPUTS', 'c{1}', 'd{1}', 'e{1}')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("INPUTS not specified", exception.__str__())

    try:
        con.execute_command('AI.MODELRUN', 'm{1}', 'INPUTS', 'a{1}', 'b{1}', 'OUTPUTS')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("Number of keys given as INPUTS here does not match model definition", exception.__str__())

    try:
        con.execute_command('AI.MODELRUN', 'm{1}', 'INPUTS', 'OUTPUTS', 'c{1}', 'd{1}')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("Number of keys given as INPUTS here does not match model definition", exception.__str__())

    try:
        con.execute_command('AI.MODELRUN', 'm{1}', 'INPUTS', 'a{1}', 'OUTPUTS', 'c{1}', 'd{1}')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("Number of keys given as OUTPUTS here does not match model definition", exception.__str__())

    try:
        con.execute_command('AI.MODELRUN', 'm{1}', 'INPUTS', 'a{1}', 'OUTPUTS', 'b{1}')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("Number of keys given as OUTPUTS here does not match model definition", exception.__str__())


# TODO: Autobatch is tricky with TFLITE because TFLITE expects a fixed batch
#       size. At least we should constrain MINBATCHSIZE according to the
#       hard-coded dims in the tflite model.
def test_run_tflite_model_autobatch(env):
    if not TEST_TFLITE:
        return

    con = env.getConnection()

    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    model_filename = os.path.join(test_data_path, 'mnist_model_quant.tflite')
    sample_filename = os.path.join(test_data_path, 'one.raw')

    with open(model_filename, 'rb') as f:
        model_pb = f.read()

    with open(sample_filename, 'rb') as f:
        sample_raw = f.read()

    try:
        ret = con.execute_command('AI.MODELSET', 'm{1}', 'TFLITE', 'CPU',
                                  'BATCHSIZE', 2, 'MINBATCHSIZE', 2, 'BLOB', model_pb)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("Auto-batching not supported by the TFLITE backend", exception.__str__())

    # env.assertEqual(ret, b'OK')

    # con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 1, 1, 28, 28, 'BLOB', sample_raw)
    # con.execute_command('AI.TENSORSET', 'c{1}', 'FLOAT', 1, 1, 28, 28, 'BLOB', sample_raw)

    # def run():
    #     con = env.getConnection()
    #     con.execute_command('AI.MODELRUN', 'm{1}', 'INPUTS', 'c{1}', 'OUTPUTS', 'd', 'd2')

    # t = threading.Thread(target=run)
    # t.start()

    # con.execute_command('AI.MODELRUN', 'm{1}', 'INPUTS', 'a{1}', 'OUTPUTS', 'b{1}', 'b2')

    # values = con.execute_command('AI.TENSORGET', 'b{1}', 'VALUES')

    # env.assertEqual(values[0], 1)

    # values = con.execute_command('AI.TENSORGET', 'd', 'VALUES')

    # env.assertEqual(values[0], 1)


def test_tflite_modelinfo(env):
    if not TEST_TFLITE:
        env.debugPrint("skipping {} since TEST_TFLITE=0".format(sys._getframe().f_code.co_name), force=True)
        return

    if DEVICE == "GPU":
        env.debugPrint("skipping {} since it's hanging CI".format(sys._getframe().f_code.co_name), force=True)
        return

    con = env.getConnection()
    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    model_filename = os.path.join(test_data_path, 'mnist_model_quant.tflite')
    sample_filename = os.path.join(test_data_path, 'one.raw')

    with open(model_filename, 'rb') as f:
        model_pb = f.read()

    with open(sample_filename, 'rb') as f:
        sample_raw = f.read()

    ret = con.execute_command('AI.MODELSET', 'mnist{1}', 'TFLITE', 'CPU', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 1, 1, 28, 28, 'BLOB', sample_raw)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    previous_duration = 0
    for call in range(1, 10):
        ret = con.execute_command('AI.MODELRUN', 'mnist{1}', 'INPUTS', 'a{1}', 'OUTPUTS', 'b{1}', 'c{1}')
        env.assertEqual(ret, b'OK')
        ensureSlaveSynced(con, env)

        info = con.execute_command('AI.INFO', 'mnist{1}')
        info_dict_0 = info_to_dict(info)

        env.assertEqual(info_dict_0['key'], 'mnist{1}')
        env.assertEqual(info_dict_0['type'], 'MODEL')
        env.assertEqual(info_dict_0['backend'], 'TFLITE')
        env.assertEqual(info_dict_0['device'], DEVICE)
        env.assertTrue(info_dict_0['duration'] > previous_duration)
        env.assertEqual(info_dict_0['samples'], call)
        env.assertEqual(info_dict_0['calls'], call)
        env.assertEqual(info_dict_0['errors'], 0)

        previous_duration = info_dict_0['duration']

    res = con.execute_command('AI.INFO', 'mnist{1}', 'RESETSTAT')
    env.assertEqual(res, b'OK')
    info = con.execute_command('AI.INFO', 'mnist{1}')
    info_dict_0 = info_to_dict(info)
    env.assertEqual(info_dict_0['duration'], 0)
    env.assertEqual(info_dict_0['samples'], 0)
    env.assertEqual(info_dict_0['calls'], 0)
    env.assertEqual(info_dict_0['errors'], 0)


def test_tflite_modelrun_disconnect(env):
    if not TEST_TFLITE:
        env.debugPrint("skipping {} since TEST_TFLITE=0".format(sys._getframe().f_code.co_name), force=True)
        return

    red = env.getConnection()
    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    model_filename = os.path.join(test_data_path, 'mnist_model_quant.tflite')
    sample_filename = os.path.join(test_data_path, 'one.raw')

    with open(model_filename, 'rb') as f:
        model_pb = f.read()

    with open(sample_filename, 'rb') as f:
        sample_raw = f.read()

    ret = red.execute_command('AI.MODELSET', 'mnist{1}', 'TFLITE', 'CPU', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ret = red.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 1, 1, 28, 28, 'BLOB', sample_raw)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(red, env)

    ret = send_and_disconnect(('AI.MODELRUN', 'mnist{1}', 'INPUTS', 'a{1}', 'OUTPUTS', 'b{1}', 'c{1}'), red)
    env.assertEqual(ret, None)


def test_tflite_model_rdb_save_load(env):
    env.skipOnCluster()
    if env.useAof or not TEST_TFLITE:
        env.debugPrint("skipping {}".format(sys._getframe().f_code.co_name), force=True)
        return

    con = env.getConnection()
    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    model_filename = os.path.join(test_data_path, 'mnist_model_quant.tflite')

    with open(model_filename, 'rb') as f:
        model_pb = f.read()

    ret = con.execute_command('AI.MODELSET', 'mnist{1}', 'TFLITE', 'CPU', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    model_serialized_memory = con.execute_command('AI.MODELGET', 'mnist{1}', 'BLOB')

    ensureSlaveSynced(con, env)
    ret = con.execute_command('SAVE')
    env.assertEqual(ret, True)

    env.stop()
    env.start()
    con = env.getConnection()
    model_serialized_after_rdbload = con.execute_command('AI.MODELGET', 'mnist{1}', 'BLOB')
    env.assertEqual(len(model_serialized_memory), len(model_serialized_after_rdbload))
    env.assertEqual(len(model_pb), len(model_serialized_after_rdbload))
    # Assert in memory model binary is equal to loaded model binary
    env.assertTrue(model_serialized_memory == model_serialized_after_rdbload)
    # Assert input model binary is equal to loaded model binary
    env.assertTrue(model_pb == model_serialized_after_rdbload)

def test_tflite_info(env):
    if not TEST_TFLITE:
        env.debugPrint("skipping {}".format(sys._getframe().f_code.co_name), force=True)
        return
    con = env.getConnection()

    ret = con.execute_command('AI.INFO')
    env.assertEqual(6, len(ret))

    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    model_filename = os.path.join(test_data_path, 'mnist_model_quant.tflite')

    with open(model_filename, 'rb') as f:
        model_pb = f.read()

    ret = con.execute_command('AI.MODELSET', 'mnist{1}', 'TFLITE', 'CPU', 'BLOB', model_pb)

    ret = con.execute_command('AI.INFO')
    env.assertEqual(8, len(ret))
    env.assertEqual(b'TFLite version', ret[6])
