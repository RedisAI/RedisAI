# Copyright Redis Ltd. 2018 - present
# Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
# the Server Side Public License v1 (SSPLv1).

import numpy as np

from includes import *

'''
python -m RLTest --test tests_tflite.py --module path/to/redisai.so
'''


def test_run_tflite_model(env):
    if not TEST_TFLITE:
        env.debugPrint("skipping {} since TEST_TFLITE=0".format(sys._getframe().f_code.co_name), force=True)
        return

    con = get_connection(env, '{1}')
    model_pb = load_file_content('mnist_model_quant.tflite')
    sample_raw = load_file_content('one.raw')

    ret = con.execute_command('AI.MODELSTORE', 'm{1}', 'TFLITE', 'CPU', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.MODELGET', 'm{1}', 'META')
    env.assertEqual(len(ret), 16)
    env.assertEqual(ret[5], b'')

    ret = con.execute_command('AI.MODELSTORE', 'm{1}', 'TFLITE', 'CPU', 'TAG', 'asdf', 'BLOB', model_pb)
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
        env.assertEqual(ret[3], b'CPU')

    con.execute_command('AI.MODELEXECUTE', 'm{1}', 'INPUTS', 1, 'a{1}', 'OUTPUTS', 2, 'b{1}', 'c{1}')
    values = con.execute_command('AI.TENSORGET', 'b{1}', 'VALUES')
    env.assertEqual(values[0], 1)


def test_run_tflite_model_autobatch(env):
    if not TEST_TFLITE:
        env.debugPrint("skipping {} since TEST_TFLITE=0".format(sys._getframe().f_code.co_name), force=True)
        return

    con = get_connection(env, '{1}')
    model_pb = load_file_content('lite-model_imagenet_mobilenet_v3_small_100_224_classification_5_default_1.tflite')
    _, _, _, img = load_resnet_test_data()
    img = img.astype(np.float32) / 255

    ret = con.execute_command('AI.MODELSTORE', 'm{1}', 'TFLITE', 'CPU',
                              'BATCHSIZE', 4, 'MINBATCHSIZE', 2,
                              'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.MODELGET', 'm{1}', 'META')
    env.assertEqual(len(ret), 16)
    if DEVICE == "CPU":
        env.assertEqual(ret[1], b'TFLITE')
        env.assertEqual(ret[3], b'CPU')

    ret = con.execute_command('AI.TENSORSET', 'a{1}',
                              'FLOAT', 1, img.shape[1], img.shape[0], 3,
                              'BLOB', img.tobytes())
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.TENSORSET', 'b{1}',
                              'FLOAT', 1, img.shape[1], img.shape[0], 3,
                              'BLOB', img.tobytes())
    env.assertEqual(ret, b'OK')

    def run():
        con = get_connection(env, '{1}')
        con.execute_command('AI.MODELEXECUTE', 'm{1}', 'INPUTS', 1,
                            'b{1}', 'OUTPUTS', 1, 'd{1}')
        ensureSlaveSynced(con, env)

    t = threading.Thread(target=run)
    t.start()

    con.execute_command('AI.MODELEXECUTE', 'm{1}', 'INPUTS', 1, 'a{1}', 'OUTPUTS', 1, 'c{1}')
    t.join()

    ensureSlaveSynced(con, env)

    c_values = np.array(con.execute_command('AI.TENSORGET', 'c{1}', 'VALUES'), dtype=np.float32)
    c_idx = np.argmax(c_values)
    d_values = np.array(con.execute_command('AI.TENSORGET', 'd{1}', 'VALUES'), dtype=np.float32)
    d_idx = np.argmax(d_values)
    env.assertEqual(c_idx, d_idx)
    env.assertFalse(np.isnan(c_values[c_idx]))
    env.assertFalse(np.isinf(c_values[c_idx]))



def test_run_tflite_errors(env):
    if not TEST_TFLITE:
        env.debugPrint("skipping {} since TEST_TFLITE=0".format(sys._getframe().f_code.co_name), force=True)
        return

    con = get_connection(env, '{1}')

    model_pb = load_file_content('mnist_model_quant.tflite')
    sample_raw = load_file_content('one.raw')
    wrong_model_pb = load_file_content('graph.pb')

    ret = con.execute_command('AI.MODELSTORE', 'm_2{1}', 'TFLITE', 'CPU', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    check_error_message(env, con, "Failed to load model from buffer",
                        'AI.MODELSTORE', 'm{1}', 'TFLITE', 'CPU', 'TAG', 'asdf', 'BLOB', wrong_model_pb)

    ret = con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 1, 1, 28, 28, 'BLOB', sample_raw)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    check_error_message(env, con, "Number of keys given as OUTPUTS here does not match model definition",
                        'AI.MODELEXECUTE', 'm_2{1}', 'INPUTS', 1, 'EMPTY_INPUT{1}', 'OUTPUTS', 1, 'EMPTY_OUTPUT{1}')

    check_error_message(env, con, "Number of keys given as INPUTS here does not match model definition",
                        'AI.MODELEXECUTE', 'm_2{1}', 'INPUTS', 3, 'a{1}', 'b{1}', 'c{1}', 'OUTPUTS', 1, 'd{1}')

    model_pb = load_file_content('lite-model_imagenet_mobilenet_v3_small_100_224_classification_5_default_1.tflite')
    _, _, _, img = load_resnet_test_data()

    ret = con.execute_command('AI.MODELSTORE', 'image_net{1}', 'TFLITE', 'CPU', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')
    ret = con.execute_command('AI.TENSORSET', 'dog{1}', 'UINT8', 1, img.shape[1], img.shape[0], 3,
                              'BLOB', img.tobytes())
    env.assertEqual(ret, b'OK')

    # The model expects FLOAT input, but UINT8 tensor is given.
    check_error_message(env, con, "Input tensor type doesn't match the type expected by the model definition",
                        'AI.MODELEXECUTE', 'image_net{1}', 'INPUTS', 1, 'dog{1}', 'OUTPUTS', 1, 'output{1}')


def test_tflite_modelinfo(env):
    if not TEST_TFLITE:
        env.debugPrint("skipping {} since TEST_TFLITE=0".format(sys._getframe().f_code.co_name), force=True)
        return

    if DEVICE == "GPU":
        env.debugPrint("skipping {} since it's hanging CI".format(sys._getframe().f_code.co_name), force=True)
        return

    con = get_connection(env, '{1}')
    model_pb = load_file_content('mnist_model_quant.tflite')
    sample_raw = load_file_content('one.raw')

    ret = con.execute_command('AI.MODELSTORE', 'mnist{1}', 'TFLITE', 'CPU', 'BLOB', model_pb)
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

    con = get_connection(env, '{1}')
    model_pb = load_file_content('mnist_model_quant.tflite')
    sample_raw = load_file_content('one.raw')

    ret = con.execute_command('AI.MODELSTORE', 'mnist{1}', 'TFLITE', 'CPU', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 1, 1, 28, 28, 'BLOB', sample_raw)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    ret = send_and_disconnect(('AI.MODELEXECUTE', 'mnist{1}', 'INPUTS', 1, 'a{1}', 'OUTPUTS', 2, 'b{1}', 'c{1}'), con)
    env.assertEqual(ret, None)


def test_tflite_model_rdb_save_load(env):
    env.skipOnCluster()
    if env.useAof or not TEST_TFLITE:
        env.debugPrint("skipping {}".format(sys._getframe().f_code.co_name), force=True)
        return

    con = get_connection(env, '{1}')
    model_pb = load_file_content('mnist_model_quant.tflite')

    ret = con.execute_command('AI.MODELSTORE', 'mnist{1}', 'TFLITE', 'CPU', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    model_serialized_memory = con.execute_command('AI.MODELGET', 'mnist{1}', 'BLOB')

    ensureSlaveSynced(con, env)
    ret = con.execute_command('SAVE')
    env.assertEqual(ret, True)

    env.stop()
    env.start()
    con = get_connection(env, '{1}')
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
    con = get_connection(env, '{1}')

    backends_info = get_info_section(con, 'backends_info')
    env.assertFalse('ai_TensorFlowLite_version' in backends_info)

    model_pb = load_file_content('mnist_model_quant.tflite')
    con.execute_command('AI.MODELSTORE', 'mnist{1}', 'TFLITE', 'CPU', 'BLOB', model_pb)

    backends_info = get_info_section(con, 'backends_info')
    env.assertTrue('ai_TensorFlowLite_version' in backends_info)
