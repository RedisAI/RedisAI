import redis

from includes import *

'''
python -m RLTest --test tests_deprecated_commands.py --module path/to/redisai.so
'''

def test_modelset_errors(env):
    if not TEST_PT:
        env.debugPrint("skipping {} since TEST_PT=0".format(sys._getframe().f_code.co_name), force=True)
        return

    con = env.getConnection()
    model_pb = load_from_file('pt-minimal.pt')

    # test validity of backend and device args.
    check_error_message(env, con, "wrong number of arguments for 'AI.MODELSET' command",
                        'AI.MODELSET', 'm{1}', 'BLOB')
    check_error_message(env, con, "unsupported backend",
                        'AI.MODELSET', 'm{1}', 'PORCH', DEVICE, 'BLOB', model_pb)
    check_error_message(env, con, "Invalid DEVICE",
                        'AI.MODELSET', 'm{1}', 'TORCH', 'BLOB', model_pb)

    # test validity of batching args.
    check_error_message(env, con, "Invalid argument for BATCHSIZE",
                        'AI.MODELSET', 'm{1}', 'TORCH', DEVICE, 'BATCHSIZE', 'bad_size', 'BLOB', model_pb)
    check_error_message(env, con, "MINBATCHSIZE specified without BATCHSIZE",
                        'AI.MODELSET', 'm{1}', 'TORCH', DEVICE, 'MINBATCHSIZE', '2', 'BLOB', model_pb)
    check_error_message(env, con, "Invalid argument for MINBATCHSIZE",
                        'AI.MODELSET', 'm{1}', 'TORCH', DEVICE, 'BATCHSIZE', 2, 'MINBATCHSIZE', 'bad_size', 'BLOB', model_pb)
    check_error_message(env, con, "MINBATCHTIMEOUT specified without MINBATCHSIZE",
                        'AI.MODELSET', 'm{1}', 'TORCH', DEVICE, 'BATCHSIZE', 2, 'MINBATCHTIMEOUT', 1000,
                        'BLOB', model_pb)
    check_error_message(env, con, "Invalid argument for MINBATCHTIMEOUT",
                        'AI.MODELSET', 'm{1}', 'TORCH', DEVICE, 'BATCHSIZE', 2, 'MINBATCHSIZE', 2,
                        'MINBATCHTIMEOUT', 'bad_timeout', 'BLOB', model_pb)

    # test validity of BLOB args.
    check_error_message(env, con, "Insufficient arguments, missing model BLOB",
                        'AI.MODELSET', 'm{1}', 'TORCH', DEVICE, 'BATCHSIZE', 2)
    check_error_message(env, con, "Insufficient arguments, missing model BLOB",
                        'AI.MODELSET', 'm{1}', 'TORCH', DEVICE, 'BATCHSIZE', 2, 'NOT_BLOB')
    check_error_message(env, con, "Insufficient arguments, missing model BLOB",
                        'AI.MODELSET', 'm{1}', 'TORCH', DEVICE, 'BATCHSIZE', 2, 'BLOB')

    # test INPUTS and OUTPUTS args for TF backend
    model_pb = load_from_file('graph.pb')
    check_error_message(env, con, "Insufficient arguments, INPUTS and OUTPUTS not specified",
                        'AI.MODELSET', 'm_1{1}', 'TF', DEVICE, 'BLOB', model_pb)
    check_error_message(env, con, "INPUTS not specified",
                        'AI.MODELSET', 'm_1{1}', 'TF', DEVICE, 'not_inputs', 'BLOB', model_pb)
    check_error_message(env, con, "OUTPUTS not specified",
                        'AI.MODELSET', 'm_1{1}', 'TF', DEVICE, 'INPUTS', 'a', 'b')
    check_error_message(env, con, "Insufficient arguments, missing model BLOB",
                        'AI.MODELSET', 'm_1{1}', 'TF', DEVICE, 'INPUTS', 'a', 'b', 'OUTPUTS', 'mul')


def test_modelrun_errors(env):
    if not TEST_TF:
        env.debugPrint("Skipping test since TF is not available", force=True)
        return
    con = env.getConnection()

    model_pb = load_from_file('graph.pb')
    ret = con.execute_command('AI.MODELSET', 'm{1}', 'TF', DEVICE,
                              'INPUTS', 'a', 'b', 'OUTPUTS', 'mul', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    # Expect at least 6 arguments, the second one is the model's key.
    check_error_message(env, con, "wrong number of arguments for 'AI.MODELRUN' command",
                        'AI.MODELRUN', 'm{1}', 'INPUTS', 'a{1}', 'OUTPUTS')
    check_error_message(env, con, "model key is empty",
                        'AI.MODELRUN', 'not_exiting_model{1}', 'INPUTS','a{1}', 'b{1}', 'OUTPUTS', 'c{1}')

    # Only INPUTS or TIMEOUT must be the fourth argument. Expected 2 inputs and 1 outputs.
    check_error_message(env, con, "INPUTS not specified",
                        'AI.MODELRUN', 'm{1}', 'NOT_INPUTS', 'a{1}', 'b{1}', 'OUTPUTS', 'c{1}')
    check_error_message(env, con, "Number of keys given as INPUTS here does not match model definition",
                        'AI.MODELRUN', 'm{1}', 'INPUTS', 'a{1}', 'OUTPUTS', 'c{1}')
    check_error_message(env, con, "Number of keys given as OUTPUTS here does not match model definition",
                        'AI.MODELRUN', 'm{1}', 'INPUTS', 'a{1}', 'b{1}', 'OUTPUTS', 'c{1}', 'd{1}')

    # TIMEOUT is allowed before INPUTS, check for its validity.
    check_error_message(env, con, "Invalid value for TIMEOUT",
                        'AI.MODELRUN', 'm{1}', 'TIMEOUT', -43, 'INPUTS', 'a{1}', 'b{1}', 'OUTPUTS', 'c{1}')
    check_error_message(env, con, "INPUTS not specified",
                        'AI.MODELRUN', 'm{1}', 'TIMEOUT', 1000, 'NOT_INPUTS', 'a{1}', 'b{1}', 'OUTPUTS', 'c{1}')


def test_modelset_modelrun_tf(env):
    if not TEST_TF:
        env.debugPrint("Skipping test since TF is not available", force=True)
        return
    con = env.getConnection()

    model_pb = load_from_file('graph.pb')
    ret = con.execute_command('AI.MODELSET', 'm{1}', 'TF', DEVICE, 'TAG', 'version:1',
                              'INPUTS', 'a', 'b', 'OUTPUTS', 'mul', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.MODELGET', 'm{1}', 'META')
    env.assertEqual(len(ret), 14)
    if DEVICE == "CPU":
        env.assertEqual(ret[1], b'TF')
        env.assertEqual(ret[3], b'CPU')
    env.assertEqual(ret[5], b'version:1')
    env.assertEqual(ret[11][0], b'a')
    env.assertEqual(ret[11][1], b'b')
    env.assertEqual(ret[13][0], b'mul')

    con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    con.execute_command('AI.TENSORSET', 'b{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)

    con.execute_command('AI.MODELRUN', 'm{1}', 'INPUTS', 'a{1}', 'b{1}', 'OUTPUTS', 'c{1}')
    values = con.execute_command('AI.TENSORGET', 'c{1}', 'VALUES')
    env.assertEqual(values, [b'4', b'9', b'4', b'9'])


def test_modelset_modelrun_tflite(env):
    if not TEST_TFLITE:
        env.debugPrint("skipping {} since TEST_TFLITE=0".format(sys._getframe().f_code.co_name), force=True)
        return

    con = env.getConnection()
    model_pb = load_from_file('mnist_model_quant.tflite')
    sample_raw = load_from_file('one.raw')

    ret = con.execute_command('AI.MODELSET', 'm{1}', 'TFLITE', 'CPU', 'TAG', 'asdf', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.MODELGET', 'm{1}', 'META')
    env.assertEqual(len(ret), 14)
    env.assertEqual(ret[5], b'asdf')

    ret = con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 1, 1, 28, 28, 'BLOB', sample_raw)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.MODELGET', 'm{1}', 'META')
    env.assertEqual(len(ret), 14)
    if DEVICE == "CPU":
        env.assertEqual(ret[1], b'TFLITE')
        env.assertEqual(ret[3], b'CPU')

    con.execute_command('AI.MODELRUN', 'm{1}', 'INPUTS', 'a{1}', 'OUTPUTS', 'b{1}', 'c{1}')
    values = con.execute_command('AI.TENSORGET', 'b{1}', 'VALUES')
    env.assertEqual(values[0], 1)


def test_modelset_modelrun_pytorch(env):
    if not TEST_PT:
        env.debugPrint("skipping {} since TEST_PT=0".format(sys._getframe().f_code.co_name), force=True)
        return

    con = env.getConnection()
    model_pb = load_from_file('pt-minimal.pt')

    ret = con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')
    ret = con.execute_command('AI.TENSORSET', 'b{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.MODELSET', 'm{1}', 'TORCH', DEVICE, 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.MODELGET', 'm{1}', 'META')
    env.assertEqual(len(ret), 14)
    if DEVICE == "CPU":
        env.assertEqual(ret[1], b'TORCH')
        env.assertEqual(ret[3], b'CPU')
    env.assertEqual(ret[5], b'')
    env.assertEqual(ret[7], 0)
    env.assertEqual(ret[9], 0)
    # assert there are no inputs or outputs
    env.assertEqual(len(ret[11]), 2)
    env.assertEqual(len(ret[13]), 1)

    con.execute_command('AI.MODELRUN', 'm{1}', 'INPUTS', 'a{1}', 'b{1}', 'OUTPUTS', 'c{1}')

    values = con.execute_command('AI.TENSORGET', 'c{1}', 'VALUES')
    env.assertEqual(values, [b'4', b'6', b'4', b'6'])


def test_modelset_modelrun_onnx(env):
    if not TEST_ONNX:
        env.debugPrint("skipping {} since TEST_ONNX=0".format(sys._getframe().f_code.co_name), force=True)
        return

    con = env.getConnection()
    model_pb = load_from_file('mnist.onnx')
    sample_raw = load_from_file('one.raw')

    ret = con.execute_command('AI.MODELSET', 'm{1}', 'ONNX', DEVICE, 'TAG', 'version:2', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.MODELGET', 'm{1}', 'META')
    env.assertEqual(len(ret), 14)
    if DEVICE == "CPU":
        env.assertEqual(ret[1], b'ONNX')
        env.assertEqual(ret[3], b'CPU')
    env.assertEqual(ret[5], b'version:2')
    env.assertEqual(len(ret[11]), 1)
    env.assertEqual(len(ret[13]), 1)

    con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 1, 1, 28, 28, 'BLOB', sample_raw)

    con.execute_command('AI.MODELRUN', 'm{1}', 'INPUTS', 'a{1}', 'OUTPUTS', 'b{1}')

    values = con.execute_command('AI.TENSORGET', 'b{1}', 'VALUES')
    argmax = max(range(len(values)), key=lambda i: values[i])
    env.assertEqual(argmax, 1)
