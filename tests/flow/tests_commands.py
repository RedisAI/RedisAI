import redis

from includes import *

'''
python -m RLTest --test tests_commands.py --module path/to/redisai.so
'''

'''
Here we test that we return the right error messages when AI.MODELSTORE command is invalid (for all backends).
Additional tests for storing a model for TF backend (in which case we must specify INPUTS and OUTPUTS in the command)
can be found in tests_tensorflow.py
'''
def test_modelstore_errors(env):
    if not TEST_PT:
        env.debugPrint("skipping {} since TEST_PT=0".format(sys._getframe().f_code.co_name), force=True)
        return

    con = env.getConnection()
    model_pb = load_file_content('pt-minimal.pt')

    # Check that the basic arguments are valid (model's key, device, backend, blob)
    check_error_message(env, con, "wrong number of arguments for 'AI.MODELSTORE' command",
                        'AI.MODELSTORE', 'm{1}', 'TORCH', 'BLOB', model_pb)
    check_error_message(env, con, "unsupported backend",
                        'AI.MODELSTORE', 'm{1}', 'PORCH', DEVICE, 'BLOB', model_pb)
    check_error_message(env, con, "Invalid DEVICE",
                        'AI.MODELSTORE', 'm{1}', 'TORCH', 'bad_device', 'BLOB', model_pb)

    # Check for validity of batching arguments.
    check_error_message(env, con, "Invalid argument for BATCHSIZE",
                        'AI.MODELSTORE', 'm{1}', 'TORCH', DEVICE, 'BATCHSIZE', 'bad_size', 'BLOB', model_pb)
    check_error_message(env, con, "MINBATCHSIZE specified without BATCHSIZE",
                        'AI.MODELSTORE', 'm{1}', 'TORCH', DEVICE, 'MINBATCHSIZE', '2', 'BLOB', model_pb)
    check_error_message(env, con, "Invalid argument for MINBATCHSIZE",
                        'AI.MODELSTORE', 'm{1}', 'TORCH', DEVICE, 'BATCHSIZE', 2, 'MINBATCHSIZE', 'bad_size', 'BLOB', model_pb)
    check_error_message(env, con, "MINBATCHTIMEOUT specified without MINBATCHSIZE",
                        'AI.MODELSTORE', 'm{1}', 'TORCH', DEVICE, 'BATCHSIZE', 2, 'MINBATCHTIMEOUT', 1000,
                        'BLOB', model_pb)
    check_error_message(env, con, "Invalid argument for MINBATCHTIMEOUT",
                        'AI.MODELSTORE', 'm{1}', 'TORCH', DEVICE, 'BATCHSIZE', 2, 'MINBATCHSIZE', 2,
                        'MINBATCHTIMEOUT', 'bad_timeout', 'BLOB', model_pb)

    # INPUTS and OUTPUTS args are relevant only for TF.
    check_error_message(env, con, "INPUTS argument should not be specified for this backend",
                        'AI.MODELSTORE', 'm{1}', 'TORCH', DEVICE, 'INPUTS', 2, 'a', 'b', 'OUTPUTS', 1, 'c', 'BLOB', model_pb)

    # Check for existence and validity of blob
    check_error_message(env, con, "Insufficient arguments, missing model BLOB",
                        'AI.MODELSTORE', 'm{1}', 'TORCH', DEVICE, 'BATCHSIZE', 2)
    check_error_message(env, con, "Invalid argument, expected BLOB",
                        'AI.MODELSTORE', 'm{1}', 'TORCH', DEVICE, 'BATCHSIZE', 2, 'NOT_BLOB')
    check_error_message(env, con, "Insufficient arguments, missing model BLOB",
                        'AI.MODELSTORE', 'm{1}', 'TORCH', DEVICE, 'BATCHSIZE', 2, 'BLOB')


def test_modelget_errors(env):
    if not TEST_TF:
        env.debugPrint("Skipping test since TF is not available", force=True)
        return

    con = env.getConnection()
    # ERR WRONGTYPE
    con.execute_command('SET', 'NOT_MODEL{1}', 'BAR')
    check_error_message(env, con, "WRONGTYPE Operation against a key holding the wrong kind of value",
                        'AI.MODELGET', 'NOT_MODEL{1}')
    # cleanup
    con.execute_command('DEL', 'NOT_MODEL{1}')

    # ERR model key is empty
    con.execute_command('DEL', 'DONT_EXIST{1}')
    check_error_message(env, con, "model key is empty",
                        'AI.MODELGET', 'DONT_EXIST{1}')


def test_modelexecute_errors(env):
    if not TEST_TF:
        env.debugPrint("Skipping test since TF is not available", force=True)
        return
    con = env.getConnection()

    model_pb = load_file_content('graph.pb')
    ret = con.execute_command('AI.MODELSTORE', 'm{1}', 'TF', DEVICE,
                              'INPUTS', 2, 'a', 'b', 'OUTPUTS', 1, 'mul', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')
    ensureSlaveSynced(con, env)

    # Expect at least 8 arguments, the second one is the model's key.
    check_error_message(env, con, "wrong number of arguments for 'AI.MODELEXECUTE' command",
                        'AI.MODELEXECUTE', 'm{1}', 'INPUTS', 2, 'a{1}', 'b{1}', 'OUTPUTS')
    check_error_message(env, con, "model key is empty",
                        'AI.MODELEXECUTE', 'not_exiting_model{1}', 'INPUTS', 2, 'a{1}', 'b{1}', 'OUTPUTS', 1, 'c{1}')

    # INPUTS must be the fourth argument (timeout is allowed only at the end). Expected 2 inputs.
    check_error_message(env, con, "INPUTS not specified",
                        'AI.MODELEXECUTE', 'm{1}', 'timeout', 1000, 'INPUTS', 2, 'a{1}', 'b{1}', 'OUTPUTS', 1, 'c{1}')
    check_error_message(env, con, "Invalid argument for input_count",
                        'AI.MODELEXECUTE', 'm{1}', 'INPUTS', 'not_a_number', 'a{1}', 'b{1}', 'OUTPUTS', 1, 'c{1}')
    check_error_message(env, con, "Input count must be a positive integer",
                        'AI.MODELEXECUTE', 'm{1}', 'INPUTS', 0, 'a{1}', 'b{1}', 'OUTPUTS', 1, 'c{1}')
    check_error_message(env, con, "Number of keys given as INPUTS here does not match model definition",
                        'AI.MODELEXECUTE', 'm{1}', 'INPUTS', 1, 'a{1}', 'OUTPUTS', 1, 'c{1}')

    # Similar checks for OUTPUTS (expect only 1 output).
    check_error_message(env, con, "OUTPUTS not specified",
                        'AI.MODELEXECUTE', 'm{1}', 'INPUTS', 2, 'a{1}', 'b{1}', 'NOT_OUTPUTS', 1, 'c{1}')
    check_error_message(env, con, "Invalid argument for output_count",
                        'AI.MODELEXECUTE', 'm{1}', 'INPUTS', 2, 'a{1}', 'b{1}', 'OUTPUTS', 'not_a_number', 'c{1}')
    check_error_message(env, con, "Output count must be a positive integer",
                        'AI.MODELEXECUTE', 'm{1}', 'INPUTS', 2, 'a{1}', 'b{1}', 'OUTPUTS', 0, 'c{1}')
    check_error_message(env, con, "Number of keys given as OUTPUTS here does not match model definition",
                       'AI.MODELEXECUTE', 'm{1}', 'INPUTS', 2, 'a{1}', 'b{1}', 'OUTPUTS', 2, 'c{1}', 'd{1}')
    check_error_message(env, con, "number of output keys to AI.MODELEXECUTE command does not match the number of "
                                  "given arguments",
                        'AI.MODELEXECUTE', 'm{1}', 'INPUTS', 2, 'a{1}', 'b{1}', 'OUTPUTS', 1)

    # Only TIMEOUT is allowed after the output keys (optional), and nothing else after.
    check_error_message(env, con, "Invalid argument: bad_arg",
                        'AI.MODELEXECUTE', 'm{1}', 'INPUTS', 2, 'a{1}', 'b{1}', 'OUTPUTS', 1, 'c{1}', 'bad_arg')
    check_error_message(env, con, "No value provided for TIMEOUT",
                        'AI.MODELEXECUTE', 'm{1}', 'INPUTS', 2, 'a{1}', 'b{1}', 'OUTPUTS', 1, 'c{1}', 'TIMEOUT')
    check_error_message(env, con, "Invalid value for TIMEOUT",
                        'AI.MODELEXECUTE', 'm{1}', 'INPUTS', 2, 'a{1}', 'b{1}', 'OUTPUTS', 1, 'c{1}', 'TIMEOUT', -43)
    check_error_message(env, con, "Invalid argument: bad_arg",
                        'AI.MODELEXECUTE', 'm{1}', 'INPUTS', 2, 'a{1}', 'b{1}', 'OUTPUTS', 1, 'c{1}', 'TIMEOUT', 1000, 'bad_arg')

    con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    con.execute_command('AI.TENSORSET', 'b', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)

    # The following 2 commands should raise an error on cluster mode (keys are not on the same shard)
    if env.isCluster():
        check_error_message(env, con, "CROSSSLOT Keys in request don't hash to the same slot",
                            'AI.MODELEXECUTE', 'm{1}', 'INPUTS', 2, 'a{1}', 'b', 'OUTPUTS', 1, 'c{1}')
        check_error_message(env, con, "CROSSSLOT Keys in request don't hash to the same slot",
                            'AI.MODELEXECUTE', 'm{1}', 'INPUTS', 2, 'a{1}', 'b{1}', 'OUTPUTS', 1, 'c')


def test_keys_syntax(env):
    if not TEST_PT:
        env.debugPrint("skipping {} since TEST_PT=0".format(sys._getframe().f_code.co_name), force=True)
        return
    # the KEYS keyword must appears in every AI.SCRIPTEXECUTE command, an may appear in AI.DAGEXECUTE(_RO) command.

    con = env.getConnection()
    script = load_file_content('script.txt')
    ret = con.execute_command('AI.SCRIPTSET', 'script{1}', DEVICE, 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    # ERR wrong number of arguments for KEYS
    check_error_message(env, con, "Missing arguments after KEYS keyword",
                        "AI.SCRIPTEXECUTE script{1} bar KEYS 1")

    # ERR invalid or negative number of arguments for KEYS
    check_error_message(env, con, "Invalid or negative value found in number of KEYS",
                        "AI.SCRIPTEXECUTE script{1} bar KEYS not_number key{1}")

    con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)

    # ERR number of KEYS does not match the number of given arguments.
    check_error_message(env, con, "Number of pre declared KEYS to be used in the command does not match the number "
                                  "of given arguments",
                        "AI.SCRIPTEXECUTE script{1} bar KEYS 2 key{1}")

    # ERR KEYS missing in AI.SCRIPTEXECUTE
    check_error_message(env, con, "KEYS scope must be provided first for AI.SCRIPTEXECUTE command",
                        "AI.SCRIPTEXECUTE script{1} bar INPUTS 2 a{1} a{1}")

    # ERR KEYS section in an inner AI.SCRIPTEXEUTE command within a DAG is not allowed.
    check_error_message(env, con, "Already encountered KEYS scope in current command",
                        "AI.DAGEXECUTE KEYS 1 a{1} |> AI.SCRIPTEXECUTE script{1} bar KEYS 1 a{1}")


# Todo: this test should change once the script store command is implemented.
def test_scriptstore(env):
    con = env.getConnection()
    script = load_file_content('script.txt')
    ret = con.execute_command('AI.SCRIPTSTORE', 'ket{1}', DEVICE, 'SOURCE', script)
    env.assertEqual(ret, b'OK')
