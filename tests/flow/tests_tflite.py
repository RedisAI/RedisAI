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
    model_pb = load_file_content('mnist_model_quant.tflite')
    sample_raw = load_file_content('one.raw')

    ret = con.execute_command('AI.MODELSTORE', 'm{1}', 'TFLITE', DEVICE, 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.MODELGET', 'm{1}', 'META')
    env.assertEqual(len(ret), 16)
    env.assertEqual(ret[5], b'')

    ret = con.execute_command('AI.MODELSTORE', 'm{1}', 'TFLITE', DEVICE, 'TAG', 'asdf', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.MODELGET', 'm{1}', 'META')
    env.assertEqual(len(ret), 16)
    env.assertEqual(ret[5], b'asdf')

    ret = con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 1, 1, 28, 28, 'BLOB', sample_raw)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    ret = con.execute_command('AI.MODELGET', 'm{1}', 'META')
    env.assertEqual(len(ret), 16)
    # TODO: enable me. CI is having issues on GPU asserts of TFLITE and CPU
    if DEVICE == "CPU":
        env.assertEqual(ret[1], b'TFLITE')
        env.assertEqual(ret[3], bDEVICE)

    con.execute_command('AI.MODELEXECUTE', 'm{1}', 'INPUTS', 1, 'a{1}', 'OUTPUTS', 2, 'b{1}', 'c{1}')
    values = con.execute_command('AI.TENSORGET', 'b{1}', 'VALUES')
    env.assertEqual(values[0], 1)


def test_run_tflite_model_errors(env):
    if not TEST_TFLITE:
        env.debugPrint("skipping {} since TEST_TFLITE=0".format(sys._getframe().f_code.co_name), force=True)
        return

    con = env.getConnection()

    model_pb = load_file_content('mnist_model_quant.tflite')
    sample_raw = load_file_content('one.raw')
    wrong_model_pb = load_file_content('graph.pb')

    ret = con.execute_command('AI.MODELSTORE', 'm_2{1}', 'TFLITE', DEVICE, 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    check_error_message(env, con, "Failed to load model from buffer",
                        'AI.MODELSTORE', 'm{1}', 'TFLITE', DEVICE, 'TAG', 'asdf', 'BLOB', wrong_model_pb)

    # TODO: Autobatch is tricky with TFLITE because TFLITE expects a fixed batch
    #       size. At least we should constrain MINBATCHSIZE according to the
    #       hard-coded dims in the tflite model.
    check_error_message(env, con, "Auto-batching not supported by the TFLITE backend",
                        'AI.MODELSTORE', 'm{1}', 'TFLITE', DEVICE,
                        'BATCHSIZE', 2, 'MINBATCHSIZE', 2, 'BLOB', model_pb)

    ret = con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 1, 1, 28, 28, 'BLOB', sample_raw)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    check_error_message(env, con, "Number of keys given as OUTPUTS here does not match model definition",
                        'AI.MODELEXECUTE', 'm_2{1}', 'INPUTS', 1, 'EMPTY_INPUT{1}', 'OUTPUTS', 1, 'EMPTY_OUTPUT{1}')

    check_error_message(env, con, "Number of keys given as INPUTS here does not match model definition",
                        'AI.MODELEXECUTE', 'm_2{1}', 'INPUTS', 3, 'a{1}', 'b{1}', 'c{1}', 'OUTPUTS', 1, 'd{1}')


def test_tflite_modelinfo(env):
    if not TEST_TFLITE:
        env.debugPrint("skipping {} since TEST_TFLITE=0".format(sys._getframe().f_code.co_name), force=True)
        return

    if DEVICE == "GPU":
        env.debugPrint("skipping {} since it's hanging CI".format(sys._getframe().f_code.co_name), force=True)
        return

    con = env.getConnection()
    model_pb = load_file_content('mnist_model_quant.tflite')
    sample_raw = load_file_content('one.raw')

    ret = con.execute_command('AI.MODELSTORE', 'mnist{1}', 'TFLITE', DEVICE, 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 1, 1, 28, 28, 'BLOB', sample_raw)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    previous_duration = 0
    for call in range(1, 10):
        ret = con.execute_command('AI.MODELEXECUTE', 'mnist{1}', 'INPUTS', 1, 'a{1}', 'OUTPUTS', 2, 'b{1}', 'c{1}')
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
    model_pb = load_file_content('mnist_model_quant.tflite')
    sample_raw = load_file_content('one.raw')

    ret = red.execute_command('AI.MODELSTORE', 'mnist{1}', 'TFLITE', DEVICE, 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ret = red.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 1, 1, 28, 28, 'BLOB', sample_raw)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(red, env)

    ret = send_and_disconnect(('AI.MODELEXECUTE', 'mnist{1}', 'INPUTS', 1, 'a{1}', 'OUTPUTS', 2, 'b{1}', 'c{1}'), red)
    env.assertEqual(ret, None)


def test_tflite_model_rdb_save_load(env):
    env.skipOnCluster()
    if env.useAof or not TEST_TFLITE:
        env.debugPrint("skipping {}".format(sys._getframe().f_code.co_name), force=True)
        return

    con = env.getConnection()
    model_pb = load_file_content('mnist_model_quant.tflite')

    ret = con.execute_command('AI.MODELSTORE', 'mnist{1}', 'TFLITE', DEVICE, 'BLOB', model_pb)
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

    model_pb = load_file_content('mnist_model_quant.tflite')

    con.execute_command('AI.MODELSTORE', 'mnist{1}', 'TFLITE', DEVICE, 'BLOB', model_pb)

    ret = con.execute_command('AI.INFO')
    env.assertEqual(8, len(ret))
    env.assertEqual(b'TFLite version', ret[6])
