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
    model_pb = load_file_content('pt-minimal.pt')

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
    model_pb = load_file_content('graph.pb')
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

    model_pb = load_file_content('graph.pb')
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

    model_pb = load_file_content('graph.pb')
    ret = con.execute_command('AI.MODELSET', 'm{1}', 'TF', DEVICE, 'TAG', 'version:1',
                              'INPUTS', 'a', 'b', 'OUTPUTS', 'mul', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.MODELGET', 'm{1}', 'META')
    env.assertEqual(len(ret), 16)
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
    model_pb = load_file_content('mnist_model_quant.tflite')
    sample_raw = load_file_content('one.raw')

    ret = con.execute_command('AI.MODELSET', 'm{1}', 'TFLITE', 'CPU', 'TAG', 'asdf', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.MODELGET', 'm{1}', 'META')
    env.assertEqual(len(ret), 16)
    env.assertEqual(ret[5], b'asdf')

    ret = con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 1, 1, 28, 28, 'BLOB', sample_raw)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.MODELGET', 'm{1}', 'META')
    env.assertEqual(len(ret), 16)
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
    model_pb = load_file_content('pt-minimal.pt')

    ret = con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')
    ret = con.execute_command('AI.TENSORSET', 'b{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.MODELSET', 'm{1}', 'TORCH', DEVICE, 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.MODELGET', 'm{1}', 'META')
    env.assertEqual(len(ret), 16)
    if DEVICE == "CPU":
        env.assertEqual(ret[1], b'TORCH')
        env.assertEqual(ret[3], b'CPU')
    env.assertEqual(ret[5], b'')
    env.assertEqual(ret[7], 0)
    env.assertEqual(ret[9], 0)
    env.assertEqual(ret[15], 0)
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
    model_pb = load_file_content('mnist.onnx')
    sample_raw = load_file_content('one.raw')

    ret = con.execute_command('AI.MODELSET', 'm{1}', 'ONNX', DEVICE, 'TAG', 'version:2', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.MODELGET', 'm{1}', 'META')
    env.assertEqual(len(ret), 16)
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


def test_pytorch_scriptset(env):
    if not TEST_PT:
        env.debugPrint("skipping {} since TEST_PT=0".format(sys._getframe().f_code.co_name), force=True)
        return

    con = env.getConnection()

    check_error(env, con, 'AI.SCRIPTSET', 'ket{1}', DEVICE, 'SOURCE', 'return 1')

    check_error(env, con, 'AI.SCRIPTSET', 'nope')

    check_error(env, con, 'AI.SCRIPTSET', 'nope', 'SOURCE')

    check_error(env, con, 'AI.SCRIPTSET', 'more', DEVICE)

    script = load_file_content('script.txt')

    ret = con.execute_command('AI.SCRIPTSET', 'ket{1}', DEVICE, 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    ret = con.execute_command('AI.SCRIPTSET', 'ket{1}', DEVICE, 'TAG', 'asdf', 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

def test_pytorch_scriptrun(env):
    if not TEST_PT:
        env.debugPrint("skipping {} since TEST_PT=0".format(sys._getframe().f_code.co_name), force=True)
        return

    con = env.getConnection()
    script = load_file_content('script.txt')

    ret = con.execute_command('AI.SCRIPTSET', 'myscript{1}', DEVICE, 'TAG', 'version1', 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')
    ret = con.execute_command('AI.TENSORSET', 'b{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    for _ in range( 0,100):

        ret = con.execute_command('AI.SCRIPTRUN', 'myscript{1}', 'bar', 'INPUTS', 'a{1}', 'b{1}', 'OUTPUTS', 'c{1}')
        env.assertEqual(ret, b'OK')


    ensureSlaveSynced(con, env)

    info = con.execute_command('AI.INFO', 'myscript{1}')
    info_dict_0 = info_to_dict(info)

    env.assertEqual(info_dict_0['key'], 'myscript{1}')
    env.assertEqual(info_dict_0['type'], 'SCRIPT')
    env.assertEqual(info_dict_0['backend'], 'TORCH')
    env.assertEqual(info_dict_0['tag'], 'version1')
    env.assertTrue(info_dict_0['duration'] > 0)
    env.assertEqual(info_dict_0['samples'], -1)
    env.assertEqual(info_dict_0['calls'], 100)
    env.assertEqual(info_dict_0['errors'], 0)

    values = con.execute_command('AI.TENSORGET', 'c{1}', 'VALUES')
    env.assertEqual(values, [b'4', b'6', b'4', b'6'])

    ensureSlaveSynced(con, env)

    if env.useSlaves:
        con2 = env.getSlaveConnection()
        values2 = con2.execute_command('AI.TENSORGET', 'c{1}', 'VALUES')
        env.assertEqual(values2, values)


def test_pytorch_scriptrun_variadic(env):
    if not TEST_PT:
        env.debugPrint("skipping {} since TEST_PT=0".format(sys._getframe().f_code.co_name), force=True)
        return

    con = env.getConnection()

    script = load_file_content('script.txt')

    ret = con.execute_command('AI.SCRIPTSET', 'myscript{$}', DEVICE, 'TAG', 'version1', 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.TENSORSET', 'a{$}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')
    ret = con.execute_command('AI.TENSORSET', 'b1{$}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')
    ret = con.execute_command('AI.TENSORSET', 'b2{$}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    for _ in range( 0,100):
        ret = con.execute_command('AI.SCRIPTRUN', 'myscript{$}', 'bar_variadic', 'INPUTS', 'a{$}', '$', 'b1{$}', 'b2{$}', 'OUTPUTS', 'c{$}')
        env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    info = con.execute_command('AI.INFO', 'myscript{$}')
    info_dict_0 = info_to_dict(info)

    env.assertEqual(info_dict_0['key'], 'myscript{$}')
    env.assertEqual(info_dict_0['type'], 'SCRIPT')
    env.assertEqual(info_dict_0['backend'], 'TORCH')
    env.assertEqual(info_dict_0['tag'], 'version1')
    env.assertTrue(info_dict_0['duration'] > 0)
    env.assertEqual(info_dict_0['samples'], -1)
    env.assertEqual(info_dict_0['calls'], 100)
    env.assertEqual(info_dict_0['errors'], 0)

    values = con.execute_command('AI.TENSORGET', 'c{$}', 'VALUES')
    env.assertEqual(values, [b'4', b'6', b'4', b'6'])

    ensureSlaveSynced(con, env)

    if env.useSlaves:
        con2 = env.getSlaveConnection()
        values2 = con2.execute_command('AI.TENSORGET', 'c{$}', 'VALUES')
        env.assertEqual(values2, values)


def test_pytorch_scriptrun_errors(env):
    if not TEST_PT:
        env.debugPrint("skipping {} since TEST_PT=0".format(sys._getframe().f_code.co_name), force=True)
        return

    con = env.getConnection()
    script = load_file_content('script.txt')

    ret = con.execute_command('AI.SCRIPTSET', 'ket{1}', DEVICE, 'TAG', 'asdf', 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')
    ret = con.execute_command('AI.TENSORSET', 'b{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    con.execute_command('DEL', 'EMPTY{1}')
    # ERR no script at key from SCRIPTGET
    check_error_message(env, con, "script key is empty", 'AI.SCRIPTGET', 'EMPTY{1}')

    con.execute_command('SET', 'NOT_SCRIPT{1}', 'BAR')
    # ERR wrong type from SCRIPTGET
    check_error_message(env, con, "WRONGTYPE Operation against a key holding the wrong kind of value", 'AI.SCRIPTGET', 'NOT_SCRIPT{1}')

    con.execute_command('DEL', 'EMPTY{1}')
    # ERR no script at key from SCRIPTRUN
    check_error_message(env, con, "script key is empty", 'AI.SCRIPTRUN', 'EMPTY{1}', 'bar', 'INPUTS', 'b{1}', 'OUTPUTS', 'c{1}')

    con.execute_command('SET', 'NOT_SCRIPT{1}', 'BAR')
    # ERR wrong type from SCRIPTRUN
    check_error_message(env, con, "WRONGTYPE Operation against a key holding the wrong kind of value", 'AI.SCRIPTRUN', 'NOT_SCRIPT{1}', 'bar', 'INPUTS', 'b{1}', 'OUTPUTS', 'c{1}')

    con.execute_command('DEL', 'EMPTY{1}')
    # ERR Input key is empty
    check_error_message(env, con, "tensor key is empty or in a different shard", 'AI.SCRIPTRUN', 'ket{1}', 'bar', 'INPUTS', 'EMPTY{1}', 'b{1}', 'OUTPUTS', 'c{1}')

    con.execute_command('SET', 'NOT_TENSOR{1}', 'BAR')
    # ERR Input key not tensor
    check_error_message(env, con, "WRONGTYPE Operation against a key holding the wrong kind of value", 'AI.SCRIPTRUN', 'ket{1}', 'bar', 'INPUTS', 'NOT_TENSOR{1}', 'b{1}', 'OUTPUTS', 'c{1}')

    check_error(env, con, 'AI.SCRIPTRUN', 'ket{1}', 'bar', 'INPUTS', 'b{1}', 'OUTPUTS', 'c{1}')

    check_error(env, con, 'AI.SCRIPTRUN', 'ket{1}', 'INPUTS', 'a{1}', 'b{1}', 'OUTPUTS', 'c{1}')

    check_error(env, con, 'AI.SCRIPTRUN', 'ket{1}', 'bar', 'INPUTS', 'b{1}', 'OUTPUTS')

    check_error(env, con, 'AI.SCRIPTRUN', 'ket{1}', 'bar', 'INPUTS', 'OUTPUTS')


def test_pytorch_scriptrun_variadic_errors(env):
    if not TEST_PT:
        env.debugPrint("skipping {} since TEST_PT=0".format(sys._getframe().f_code.co_name), force=True)
        return

    con = env.getConnection()

    script = load_file_content('script.txt')

    ret = con.execute_command('AI.SCRIPTSET', 'ket{$}', DEVICE, 'TAG', 'asdf', 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.TENSORSET', 'a{$}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')
    ret = con.execute_command('AI.TENSORSET', 'b{$}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    con.execute_command('DEL', 'EMPTY{$}')
    # ERR Variadic input key is empty
    check_error_message(env, con, "tensor key is empty or in a different shard", 'AI.SCRIPTRUN', 'ket{$}', 'bar_variadic', 'INPUTS', 'a{$}', '$', 'EMPTY{$}', 'b{$}', 'OUTPUTS', 'c{$}')
 
    con.execute_command('SET', 'NOT_TENSOR{$}', 'BAR')
    # ERR Variadic input key not tensor
    check_error_message(env, con, "WRONGTYPE Operation against a key holding the wrong kind of value", 'AI.SCRIPTRUN', 'ket{$}', 'bar_variadic', 'INPUTS', 'a{$}', '$' , 'NOT_TENSOR{$}', 'b{$}', 'OUTPUTS', 'c{$}')

    check_error(env, con, 'AI.SCRIPTRUN', 'ket{$}', 'bar_variadic', 'INPUTS', 'b{$}', '${$}', 'OUTPUTS', 'c{$}')

    check_error(env, con, 'AI.SCRIPTRUN', 'ket{$}', 'bar_variadic', 'INPUTS', 'b{$}', '$', 'OUTPUTS')

    check_error(env, con, 'AI.SCRIPTRUN', 'ket{$}', 'bar_variadic', 'INPUTS', '$', 'OUTPUTS')

    check_error_message(env, con, "Already encountered a variable size list of tensors", 'AI.SCRIPTRUN', 'ket{$}', 'bar_variadic', 'INPUTS', '$', 'a{$}', '$', 'b{$}' 'OUTPUTS')


def test_dagrun_common_errors(env):
    con = env.getConnection()

    model_pb = load_file_content('graph.pb')
    ret = con.execute_command('AI.MODELSET', 'm{1}', 'TF', DEVICE,
                              'INPUTS', 'a', 'b', 'OUTPUTS', 'mul', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')
    script = load_file_content('script.txt')
    ret = con.execute_command('AI.SCRIPTSET', 'script{1}', DEVICE, 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    # ERR bad syntax
    check_error_message(env, con, "Invalid DAG command. Unexpected argument:  BAD_ARG",
                        "AI.DAGRUN PERSIST 1 a{1} BAD_ARG")

    # ERR unsupported command within DAG
    check_error_message(env, con, "Unsupported command within DAG",
                        "AI.DAGRUN |> AI.NOCOMMAND a{1} FLOAT 1 2 VALUES 5 10")

    # ERR wrong number of arguments for 'AI.DAGRUN' command
    check_error_message(env, con, "wrong number of arguments for 'AI.DAGRUN' command", "AI.DAGRUN ")

    # ERR DAG with no ops
    command = "AI.TENSORSET volatile_tensor{1} FLOAT 2 2 VALUES 5 10 5 10"
    ret = con.execute_command(command)
    env.assertEqual(ret, b'OK')
    check_error_message(env, con, "DAG is empty", "AI.DAGRUN LOAD 1 volatile_tensor{1}")

    # ERR new commands in deprecated AI.DAGRUN
    check_error_message(env, con, "AI.MODELEXECUTE cannot be used in a deprecated AI.DAGRUN command",
                        "AI.DAGRUN LOAD 1 volatile_tensor{1} "
                        "|> AI.MODELEXECUTE m{1} INPUTS 2 volatile_tensor{1} volatile_tensor{1} OUTPUTS 1 output_tensor{1}")

    check_error_message(env, con, "AI.SCRIPTEXECUTE cannot be used in a deprecated AI.DAGRUN command",
                        "AI.DAGRUN LOAD 1 volatile_tensor{1} "
                        "|> AI.SCRIPTEXECUTE script{1} bar INPUTS 2 volatile_tensor{1} volatile_tensor{1} OUTPUTS 1 output_tensor{1}")

    # ERR wrong number of arguments for 'AI.DAGEXECUTE_RO' command
    check_error_message(env, con, "wrong number of arguments for 'AI.DAGRUN_RO' command", "AI.DAGRUN_RO ")

    # ERR persist in not allowed in AI.DAGEXECUTE_RO
    check_error_message(env, con, "PERSIST cannot be specified in a read-only DAG",
                        "AI.DAGRUN_RO PERSIST 1 tensor1{1} |> AI.TENSORSET tensor1{1} FLOAT 1 2 VALUES 5 10")


def test_dagrun_modelrun_multidevice_resnet_ensemble_alias(env):
    if (not TEST_TF or not TEST_PT):
        return
    con = env.getConnection()

    model_name_0 = 'imagenet_model1:{1}'
    model_name_1 = 'imagenet_model2:{1}'
    script_name_0 = 'imagenet_script1:{1}'
    script_name_1 = 'imagenet_script2:{1}'
    inputvar = 'images'
    outputvar = 'output'
    image_key = 'image:{1}'
    temp_key1 = 'temp_key1:{1}'
    temp_key2_0 = 'temp_key2_0'
    temp_key2_1 = 'temp_key2_1'
    class_key_0 = 'output0:{1}'
    class_key_1 = 'output1:{1}'

    model_pb, script, labels, img = load_resnet_test_data()

    device_0 = 'CPU:1'
    device_1 = DEVICE

    ret = con.execute_command('AI.MODELSET', model_name_0, 'TF', device_0,
                              'INPUTS', inputvar,
                              'OUTPUTS', outputvar,
                              'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    ret = con.execute_command('AI.MODELSET', model_name_1, 'TF', device_1,
                              'INPUTS', inputvar,
                              'OUTPUTS', outputvar,
                              'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    ret = con.execute_command('AI.SCRIPTSET', script_name_0, device_0, 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    ret = con.execute_command('AI.SCRIPTSET', script_name_1, device_1, 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    # Cannot persist class_key_1
    check_error_message(env, con, "PERSIST key cannot be found in DAG",
                        'AI.DAGRUN',
                        'PERSIST', '2', class_key_0, class_key_1, '|>',
                        'AI.TENSORSET', image_key, 'UINT8', img.shape[1], img.shape[0], 3, 'BLOB', img.tobytes(),
                        '|>',
                        'AI.SCRIPTRUN',  script_name_0, 'pre_process_3ch',
                        'INPUTS', image_key,
                        'OUTPUTS', temp_key1,
                        '|>',
                        'AI.MODELRUN', model_name_0,
                        'INPUTS', temp_key1,
                        'OUTPUTS', temp_key2_0,
                        '|>',
                        'AI.MODELRUN', model_name_1,
                        'INPUTS', temp_key1,
                        'OUTPUTS', temp_key2_1,
                        '|>',
                        'AI.SCRIPTRUN', script_name_1, 'ensemble',
                        'INPUTS', temp_key2_0, temp_key2_1,
                        'OUTPUTS', temp_key1,
                        '|>',
                        'AI.SCRIPTRUN', script_name_0, 'post_process',
                        'INPUTS', temp_key1,
                        'OUTPUTS', class_key_0)


    try:
        ret = con.execute_command(
            'AI.DAGRUN',
            'PERSIST', '1', class_key_0, '|>',
            'AI.TENSORSET', image_key, 'UINT8', img.shape[1], img.shape[0], 3, 'BLOB', img.tobytes(),
            '|>',
            'AI.SCRIPTRUN',  script_name_0, 'pre_process_3ch',
            'INPUTS', image_key,
            'OUTPUTS', temp_key1,
            '|>',
            'AI.MODELRUN', model_name_0,
            'INPUTS', temp_key1,
            'OUTPUTS', temp_key2_0,
            '|>',
            'AI.MODELRUN', model_name_1,
            'INPUTS', temp_key1,
            'OUTPUTS', temp_key2_1,
            '|>',
            'AI.SCRIPTRUN', script_name_0, 'ensemble',
            'INPUTS', temp_key2_0,
            'OUTPUTS', temp_key1,
            '|>',
            'AI.SCRIPTRUN', script_name_0, 'post_process',
            'INPUTS', temp_key1,
            'OUTPUTS', class_key_0,
        )
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertTrue(exception.__str__().startswith("expected 2 inputs, but got only 1"))

    ret = con.execute_command(
        'AI.DAGRUN',
        'PERSIST', '1', class_key_0,
        '|>',
        'AI.TENSORSET', image_key, 'UINT8', img.shape[1], img.shape[0], 3, 'BLOB', img.tobytes(),
        '|>',
        'AI.SCRIPTRUN',  script_name_0, 'pre_process_3ch',
        'INPUTS', image_key,
        'OUTPUTS', temp_key1,
        '|>',
        'AI.MODELRUN', model_name_0,
        'INPUTS', temp_key1,
        'OUTPUTS', temp_key2_0,
        '|>',
        'AI.MODELRUN', model_name_1,
        'INPUTS', temp_key1,
        'OUTPUTS', temp_key2_1,
        '|>',
        'AI.SCRIPTRUN', script_name_0, 'ensemble',
        'INPUTS', temp_key2_0, temp_key2_1,
        'OUTPUTS', temp_key1,
        '|>',
        'AI.SCRIPTRUN', script_name_0, 'post_process',
        'INPUTS', temp_key1,
        'OUTPUTS', class_key_0,
    )
    env.assertEqual([b'OK', b'OK', b'OK', b'OK', b'OK', b'OK'], ret)

    ensureSlaveSynced(con, env)

    ret = con.execute_command('AI.TENSORGET', class_key_0, 'VALUES' )
    # tf model has 100 classes [0,999]
    env.assertEqual(ret[0]>=0 and ret[0]<1001, True)
