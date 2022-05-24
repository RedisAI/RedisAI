from includes import *
from tests_llapi import with_test_module

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

    con = get_connection(env, '{1}')
    model_pb = load_file_content('pt-minimal.pt')

    # Check the validity of backend name and number of arguments
    check_error_message(env, con, "wrong number of arguments for 'AI.MODELSTORE' command",
                        'AI.MODELSTORE', 'm{1}', 'TORCH', 'BLOB', model_pb)
    check_error_message(env, con, "unsupported backend",
                        'AI.MODELSTORE', 'm{1}', 'PORCH', DEVICE, 'BLOB', model_pb)

    # Check for valid device argument (should only be "CPU:<n>" or "GPU:<n>, where <n> is a number
    # (contains digits only, up to 5).
    check_error_message(env, con, "Invalid DEVICE",
                        'AI.MODELSTORE', 'm{1}', 'TORCH', 'bad_device', 'BLOB', model_pb)
    check_error_message(env, con, "Invalid DEVICE",
                        'AI.MODELSTORE', 'm{1}', 'TORCH', 'CPU::', 'BLOB', model_pb)
    check_error_message(env, con, "Invalid DEVICE",
                        'AI.MODELSTORE', 'm{1}', 'TORCH', 'CPU:1.8', 'BLOB', model_pb)
    check_error_message(env, con, "Invalid DEVICE",
                        'AI.MODELSTORE', 'm{1}', 'TORCH', 'CPU:123456', 'BLOB', model_pb)

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


def test_modelget(env):
    if not TEST_TF:
        env.debugPrint("Skipping test since TF is not available", force=True)
        return

    con = get_connection(env, '{1}')
    # ERR WRONGTYPE
    con.execute_command('SET', 'NOT_MODEL{1}', 'BAR')
    check_error_message(env, con, "WRONGTYPE Operation against a key holding the wrong kind of value",
                        'AI.MODELGET', 'NOT_MODEL{1}')
    # cleanup
    con.execute_command('DEL', 'NOT_MODEL{1}')

    # ERR model key is empty
    con.execute_command('DEL', 'DONT_EXIST{1}')
    check_error_message(env, con, "model key is empty", 'AI.MODELGET', 'DONT_EXIST{1}')

    # The default behaviour on success is return META+BLOB
    model_pb = load_file_content('graph.pb')
    con.execute_command('AI.MODELSTORE', 'm{1}', 'TF', DEVICE, 'INPUTS', 2, 'a', 'b', 'OUTPUTS', 1, 'mul',
                              'BLOB', model_pb)
    _, backend, _, device, _, tag, _, batchsize, _, minbatchsize, _, inputs, _, outputs, _, minbatchtimeout, _, blob = \
            con.execute_command('AI.MODELGET', 'm{1}')
    env.assertEqual([backend, device, tag, batchsize, minbatchsize, minbatchtimeout, inputs, outputs],
                         [b"TF", bytes(DEVICE, "utf8"), b"", 0, 0, 0, [b"a", b"b"], [b"mul"]])
    env.assertEqual(blob, model_pb)


def test_modelexecute_errors(env):
    if not TEST_TF:
        env.debugPrint("Skipping test since TF is not available", force=True)
        return
    con = get_connection(env, '{1}')

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
    con.execute_command('AI.TENSORSET', 'b{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)

    # The following 2 commands should raise an error on cluster mode (keys are not on the same shard)
    if env.isCluster():
        check_error_message(env, con, "Keys in request don't hash to the same slot",
                            'AI.MODELEXECUTE', 'm{1}', 'INPUTS', 2, 'a{1}', 'b', 'OUTPUTS', 1, 'c{1}', error_type=redis.exceptions.ClusterCrossSlotError)
        check_error_message(env, con, "Keys in request don't hash to the same slot",
                            'AI.MODELEXECUTE', 'm{1}', 'INPUTS', 2, 'a{1}', 'b{1}', 'OUTPUTS', 1, 'c', error_type=redis.exceptions.ClusterCrossSlotError)


def test_keys_syntax(env):
    if not TEST_PT:
        env.debugPrint("skipping {} since TEST_PT=0".format(sys._getframe().f_code.co_name), force=True)
        return
    # the KEYS keyword must appears in every AI.SCRIPTEXECUTE command, an may appear in AI.DAGEXECUTE(_RO) command.

    con = get_connection(env, '{1}')
    script = load_file_content('script.txt')
    ret = con.execute_command('AI.SCRIPTSTORE', 'script{1}', DEVICE, 'ENTRY_POINTS', 2, 'bar', 'bar_variadic', 'SOURCE', script)
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

    # ERR KEYS or INPUTS missing in AI.SCRIPTEXECUTE
    check_error_message(env, con, "KEYS or INPUTS scope must be provided first for AI.SCRIPTEXECUTE command",
                        "AI.SCRIPTEXECUTE script{1} bar OUTPUTS 2 a{1} a{1}")


def test_scriptstore(env):
    if not TEST_PT:
        env.debugPrint("skipping {} since TEST_PT=0".format(sys._getframe().f_code.co_name), force=True)
        return

    con = get_connection(env, '{1}')
    script = load_file_content('script.txt')

    ret = con.execute_command('AI.SCRIPTSTORE', 'ket{1}', DEVICE, 'ENTRY_POINTS', 2, 'bar', 'bar_variadic', 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    ret = con.execute_command('AI.SCRIPTSTORE', 'ket{1}', DEVICE, 'TAG', 'asdf', 'ENTRY_POINTS', 2, 'bar', 'bar_variadic', 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

def test_scriptstore_errors(env):
    if not TEST_PT:
        env.debugPrint("skipping {} since TEST_PT=0".format(sys._getframe().f_code.co_name), force=True)
        return

    con = get_connection(env, '{1}')
    script = load_file_content('script.txt')
    old_script = load_file_content('old_script.txt')
    bad_script = load_file_content('script_bad.txt')

    check_error_message(env, con, "wrong number of arguments for 'AI.SCRIPTSTORE' command", 'AI.SCRIPTSTORE', 'ket{1}', DEVICE, 'SOURCE', 'return 1')

    check_error_message(env, con, "wrong number of arguments for 'AI.SCRIPTSTORE' command", 'AI.SCRIPTSTORE', 'nope{1}')

    check_error_message(env, con, "wrong number of arguments for 'AI.SCRIPTSTORE' command", 'AI.SCRIPTSTORE', 'nope{1}', 'SOURCE')

    check_error_message(env, con, "wrong number of arguments for 'AI.SCRIPTSTORE' command", 'AI.SCRIPTSTORE', 'more{1}', DEVICE)

    check_error_message(env, con, "Insufficient arguments, missing script entry points", 'AI.SCRIPTSTORE', 'ket{1}', DEVICE, 'NO_ENTRY_POINTS', 2, 'bar', 'bar_variadic', 'SOURCE', script)

    check_error_message(env, con, "Non numeric entry points number provided to AI.SCRIPTSTORE command", 'AI.SCRIPTSTORE', 'ket{1}', DEVICE, 'ENTRY_POINTS', 'ENTRY_POINTS', 'bar', 'bar_variadic', 'SOURCE', script)

    check_error_message(env, con, "Function bar1 does not exist in the given script.", 'AI.SCRIPTSTORE', 'ket{1}', DEVICE, 'ENTRY_POINTS', 2, 'bar', 'bar1', 'SOURCE', script)

    check_error_message(env, con, "Insufficient arguments, missing script SOURCE", 'AI.SCRIPTSTORE', 'ket{1}', DEVICE, 'ENTRY_POINTS', 2, 'bar', 'bar_variadic', 'SOURCE')

    check_error_message(env, con, "Wrong number of inputs in function bar. Expected 3 but was 2", 'AI.SCRIPTSTORE', 'ket{1}', DEVICE, 'ENTRY_POINTS', 2, 'bar', 'bar_variadic', 'SOURCE', old_script)

    check_error_message(env, con, "Wrong inputs type in function bar. Expected signature similar to: def bar(tensors: List[Tensor], keys: List[str], args: List[str])", 'AI.SCRIPTSTORE', 'ket{1}', DEVICE, 'ENTRY_POINTS', 2, 'bar', 'bar_variadic', 'SOURCE', bad_script)

    check_error_message(env, con, "Insufficient arguments, missing script entry points", 'AI.SCRIPTSTORE', 'ket{1}', DEVICE, 'ENTRY_POINTS', 5 , 'bar', 'bar_variadic', 'SOURCE', script)


def test_pytrorch_scriptget_errors(env):
    if not TEST_PT:
        env.debugPrint("skipping {} since TEST_PT=0".format(sys._getframe().f_code.co_name), force=True)
        return

    con = get_connection(env, '{1}')

    script = load_file_content('script.txt')

    ret = con.execute_command('AI.SCRIPTSTORE', 'ket{1}', DEVICE, 'ENTRY_POINTS', 2, 'bar', 'bar_variadic', 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')
    ret = con.execute_command('AI.TENSORSET', 'b{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    # ERR no script at key from SCRIPTGET
    check_error_message(env, con, "script key is empty", 'AI.SCRIPTGET', 'EMPTY{1}')

    con.execute_command('SET', 'NOT_SCRIPT{1}', 'BAR')
    # ERR wrong type from SCRIPTGET
    check_error_message(env, con, "WRONGTYPE Operation against a key holding the wrong kind of value", 'AI.SCRIPTGET', 'NOT_SCRIPT{1}')


def test_pytorch_scriptexecute_errors(env):
    if not TEST_PT:
        env.debugPrint("skipping {} since TEST_PT=0".format(sys._getframe().f_code.co_name), force=True)
        return

    con = get_connection(env, '{1}')

    script = load_file_content('script.txt')

    ret = con.execute_command('AI.SCRIPTSTORE', 'ket{1}', DEVICE, 'ENTRY_POINTS', 2, 'bar', 'bar_variadic', 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')
    ret = con.execute_command('AI.TENSORSET', 'b{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    con.execute_command('DEL', 'EMPTY{1}')
    # ERR no script at key from SCRIPTEXECUTE
    check_error_message(env, con, "script key is empty", 'AI.SCRIPTEXECUTE', 'EMPTY{1}', 'bar', 'KEYS', 1 , '{1}', 'INPUTS', 1, 'b{1}', 'OUTPUTS', 1, 'c{1}')

    con.execute_command('SET', 'NOT_SCRIPT{1}', 'BAR')
    # ERR wrong type from SCRIPTEXECUTE
    check_error_message(env, con, "WRONGTYPE Operation against a key holding the wrong kind of value", 'AI.SCRIPTEXECUTE', 'NOT_SCRIPT{1}', 'bar', 'KEYS', 1 , '{1}', 'INPUTS', 1, 'b{1}', 'OUTPUTS', 1, 'c{1}')

    con.execute_command('DEL', 'EMPTY{1}')
    # ERR Input key is empty
    check_error_message(env, con, "tensor key is empty or in a different shard", 'AI.SCRIPTEXECUTE', 'ket{1}', 'bar', 'KEYS', 1 , '{1}', 'INPUTS', 2, 'EMPTY{1}', 'b{1}', 'OUTPUTS', 1, 'c{1}')

    con.execute_command('SET', 'NOT_TENSOR{1}', 'BAR')
    # ERR Input key not tensor
    check_error_message(env, con, "WRONGTYPE Operation against a key holding the wrong kind of value", 'AI.SCRIPTEXECUTE', 'ket{1}', 'bar', 'KEYS', 1 , '{1}', 'INPUTS', 2, 'NOT_TENSOR{1}', 'b{1}', 'OUTPUTS', 1, 'c{1}')

    check_error(env, con, 'AI.SCRIPTEXECUTE', 'ket{1}', 'bar', 'KEYS', 1 , '{1}', 'INPUTS', 1, 'b{1}', 'OUTPUTS', 1, 'c{1}')

    check_error(env, con, 'AI.SCRIPTEXECUTE', 'ket{1}', 'KEYS', 1 , '{1}', 'INPUTS', 2, 'a{1}', 'b{1}', 'OUTPUTS', 1, 'c{1}')

    check_error(env, con, 'AI.SCRIPTEXECUTE', 'ket{1}', 'bar', 'KEYS', 1 , '{1}', 'INPUTS', 1, 'b{1}', 'OUTPUTS')

    check_error(env, con, 'AI.SCRIPTEXECUTE', 'ket{1}', 'bar', 'KEYS', 1 , '{1}', 'INPUTS', 'OUTPUTS')

    check_error_message(env, con, "Invalid arguments provided to AI.SCRIPTEXECUTE",
                        'AI.SCRIPTEXECUTE', 'ket{1}', 'bar', 'KEYS', 1, '{1}', 'ARGS')

    check_error_message(env, con, "Invalid argument for inputs count in AI.SCRIPTEXECUTE",
                        'AI.SCRIPTEXECUTE', 'ket{1}', 'bar', 'INPUTS', 'OUTPUTS')

    check_error_message(env, con, "Invalid value for TIMEOUT",
                        'AI.SCRIPTEXECUTE', 'ket{1}', 'bar', 'INPUTS', 2, 'a{1}', 'b{1}', 'OUTPUTS', 1, 'c{1}', 'TIMEOUT', 'TIMEOUT')

    check_error_message(env, con, "No value provided for TIMEOUT in AI.SCRIPTEXECUTE",
                        'AI.SCRIPTEXECUTE', 'ket{1}', 'bar', 'INPUTS', 2, 'a{1}', 'b{1}', 'OUTPUTS', 1, 'c{1}', 'TIMEOUT')

    if env.isCluster():
        # cross shard
        check_error_message(env, con, "Keys in request don't hash to the same slot", 'AI.SCRIPTEXECUTE', 'ket{1}', 'bar', 'KEYS', 1 , '{2}', 'INPUTS', 2, 'a{1}', 'b{1}', 'OUTPUTS', 1, 'c{1}', error_type=redis.exceptions.ClusterCrossSlotError)

        # key doesn't exist
        check_error_message(env, con, "Keys in request don't hash to the same slot", 'AI.SCRIPTEXECUTE', 'ket{1}', 'bar', 'KEYS', 1 , '{1}', 'INPUTS', 2, 'a{1}', 'b{2}', 'OUTPUTS', 1, 'c{1}', error_type=redis.exceptions.ClusterCrossSlotError)


def test_pytorch_scriptexecute_variadic_errors(env):
    if not TEST_PT:
        env.debugPrint("skipping {} since TEST_PT=0".format(sys._getframe().f_code.co_name), force=True)
        return

    con = get_connection(env, '{1}')

    script = load_file_content('script.txt')

    ret = con.execute_command('AI.SCRIPTSTORE', 'ket{$}', DEVICE, 'ENTRY_POINTS', 2, 'bar', 'bar_variadic', 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.TENSORSET', 'a{$}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')
    ret = con.execute_command('AI.TENSORSET', 'b{$}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    con.execute_command('DEL', 'EMPTY{$}')
    # ERR Variadic input key is empty
    check_error_message(env, con, "tensor key is empty or in a different shard", 'AI.SCRIPTEXECUTE', 'ket{$}', 'bar_variadic', 'KEYS', 1 , '{$}', 'INPUTS', 3, 'a{$}', 'EMPTY{$}', 'b{$}', 'OUTPUTS', 1, 'c{$}')

    con.execute_command('SET', 'NOT_TENSOR{$}', 'BAR')
    # ERR Variadic input key not tensor
    check_error_message(env, con, "WRONGTYPE Operation against a key holding the wrong kind of value", 'AI.SCRIPTEXECUTE', 'ket{$}', 'bar_variadic', 'KEYS', 1 , '{$}', 'INPUTS', 3, 'a{$}', 'NOT_TENSOR{$}', 'b{$}', 'OUTPUTS', 1, 'c{$}')

    check_error(env, con, 'AI.SCRIPTEXECUTE', 'ket{$}', 'bar_variadic', 'KEYS', 1 , '{$}', 'INPUTS', 2, 'b{$}', '${$}', 'OUTPUTS', 1, 'c{$}')

    check_error(env, con, 'AI.SCRIPTEXECUTE', 'ket{$}', 'bar_variadic', 'KEYS', 1 , '{$}', 'INPUTS', 1, 'b{$}', 'OUTPUTS')

    check_error(env, con, 'AI.SCRIPTEXECUTE', 'ket{$}', 'bar_variadic', 'KEYS', 1 , '{$}', 'INPUTS', 'OUTPUTS')


def create_run_update_models_parallel(con, client_id,  # these are mandatory
                                      env, model_pb, pipes, iterations_num, multiple_keys):
    my_pipe = pipes[client_id]
    total_success_num = 0
    model_key_name = str(client_id)+'_m{1}' if multiple_keys else 'm{1}'
    ret = con.execute_command('AI.MODELSTORE', model_key_name, 'TF', DEVICE,
                              'INPUTS', 2, 'a', 'b', 'OUTPUTS', 1, 'mul', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    # Sanity test to verify that AI.INFO command can run safely in parallel with write and execution commands, which
    # read and write (atomically) to same counters of a RunStats entry, and change the global RunStats dict.
    for i in range(iterations_num):
        try:
            con.execute_command('AI.MODELDEL', model_key_name)
            ret = con.execute_command('AI.MODELSTORE', model_key_name, 'TF', DEVICE,
                                  'INPUTS', 2, 'a', 'b', 'OUTPUTS', 1, 'mul', 'BLOB', model_pb)
            env.assertEqual(ret, b'OK')
            env.assertEqual(con.execute_command('AI.MODELEXECUTE', model_key_name,
                                            'INPUTS', 2, 'a{1}', 'b{1}', 'OUTPUTS', 1, 'c{1}'), b'OK')
            total_success_num += 1
            info = info_to_dict(con.execute_command('AI.INFO', model_key_name))
            env.assertEqual(info['key'], model_key_name)
            env.assertEqual(info['device'], DEVICE)
            if multiple_keys:
                # Calls number can be verified only in a multiple keys scenario, since when parallel clients use
                # the same key, one might delete the key before another try to run the model
                # (hence the try/catch structure of the test, which is relevant only for a single key scenario)
                env.assertEqual(info['calls'], 1)
                con.execute_command('AI.INFO', model_key_name, 'RESETSTAT')
                info = info_to_dict(con.execute_command('AI.INFO', model_key_name))
                env.assertEqual(info['calls'], 0)
        except Exception as e:
            env.assertEqual(type(e), redis.exceptions.ResponseError)
            env.assertTrue("model key is empty" == str(e) or "cannot find run info for key" == str(e))
    my_pipe.send(total_success_num)


def run_model_execute_from_llapi_parallel(con, client_id,  # these are mandatory
                                 env, model_pb, pipes, iterations_num):

    my_pipe = pipes[client_id]
    total_success_num = 0
    model_key_name = str(client_id)+'_m{1}'

    # Use a different device than the one that is run from command API, to test also multi-device scenario.
    ret = con.execute_command('AI.MODELSTORE', model_key_name, 'TF', 'CPU:1',
                              'INPUTS', 2, 'a', 'b', 'OUTPUTS', 1, 'mul', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')
    for i in range(1, iterations_num+1):
        # In every call, this commands runs the model twice - once it returns with an error, and the other returns OK.
        env.assertEqual(con.execute_command("RAI_llapi.modelRun", model_key_name), b'Async run success')
        total_success_num += 1
        info = info_to_dict(con.execute_command('AI.INFO', model_key_name))
        env.assertEqual(info['calls'], 2*i)
        env.assertEqual(info['errors'], i)
    my_pipe.send(total_success_num)


def test_ai_info_multiproc_multi_keys(env):
    if not TEST_TF:
        env.debugPrint("Skipping test since TF is not available", force=True)
        return
    con = get_connection(env, '{1}')

    # Load model protobuf and store its input tensors.
    model_pb = load_file_content('graph.pb')
    con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    con.execute_command('AI.TENSORSET', 'b{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)

    num_parallel_clients = 20
    num_iterations_per_client = 50

    # Create a pipe for every child process, so it can report number of successful runs.
    parent_end_pipes, children_end_pipes = get_parent_children_pipes(num_parallel_clients)

    # Run create_run_delete_models_parallel where every client uses a different model key.
    # In every iteration, clients update the model key - which triggers deletion and insertion of the  model's stats
    # to the global dict.
    run_test_multiproc(env, '{1}', num_parallel_clients, create_run_update_models_parallel,
                       args=(env, model_pb, children_end_pipes, num_iterations_per_client, True))
    # Expect that every child will succeed in every model execution - and report it to the parent thorough the pipe.
    env.assertEqual(sum([p.recv() for p in parent_end_pipes]), num_parallel_clients*num_iterations_per_client)

    # Get the list of models in the system and verify that every model appears once in the global dict.
    models = con.execute_command('AI._MODELSCAN')
    env.assertEqual(len(models), num_parallel_clients)


def test_ai_info_multiproc_single_key(env):
    if not TEST_TF:
        env.debugPrint("Skipping test since TF is not available", force=True)
        return
    con = get_connection(env, '{1}')

    # Load model protobuf and store its input tensors.
    model_pb = load_file_content('graph.pb')
    con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    con.execute_command('AI.TENSORSET', 'b{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)

    num_parallel_clients = 20
    num_iterations_per_client = 50

    # Run create_run_delete_models_parallel, but this time over the same model key.
    # Note that there may be cases where the model is deleted by some client while other clients
    # try to access the model key.
    ret = con.execute_command('AI.MODELSTORE', 'm{1}', 'TF', DEVICE,
                              'INPUTS', 2, 'a', 'b', 'OUTPUTS', 1, 'mul', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')
    parent_end_pipes, children_end_pipes = get_parent_children_pipes(num_parallel_clients)
    run_test_multiproc(env, '{1}', num_parallel_clients, create_run_update_models_parallel,
                       args=(env, model_pb, children_end_pipes, num_iterations_per_client, False))

    # Expect minimal number of success (one per client in average)
    # in running the models (without having it deleted by another client).
    num_success = sum([p.recv() for p in parent_end_pipes])
    env.assertGreaterEqual(num_parallel_clients*num_iterations_per_client, num_success)
    # Valgrind impacts the timings, so number of success may be lower.
    env.assertGreaterEqual(num_success, 1 if VALGRIND or DEVICE == "GPU" else num_parallel_clients)

    # At the end, expect that every client ran the last created model at most once.
    info = info_to_dict(con.execute_command('AI.INFO', 'm{1}'))
    env.assertGreaterEqual(info['calls'], 0)
    env.assertGreaterEqual(num_parallel_clients, info['calls'])


@with_test_module
def test_ai_info_multiproc_with_llapi(env):
    if not TEST_TF:
        env.debugPrint("Skipping test since TF is not available", force=True)
        return
    con = get_connection(env, '{1}')

    # Load model protobuf and store its input tensors.
    model_pb = load_file_content('graph.pb')
    con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    con.execute_command('AI.TENSORSET', 'b{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)

    num_parallel_clients = 20
    num_iterations_per_client = 50

    # This is just a wrapper that triggers the multi-proc test
    def run_model_execute_from_llapi():
        parent_end_pipes_llapi, children_end_pipes_llapi = get_parent_children_pipes(num_parallel_clients)
        run_test_multiproc(env, '{1}', num_parallel_clients, run_model_execute_from_llapi_parallel,
                           args=(env, model_pb, children_end_pipes_llapi, num_iterations_per_client))
        # Expect that every child will succeed in every model execution - and report it to the parent thorough the pipe.
        env.assertEqual(sum([p.recv() for p in parent_end_pipes_llapi]),
                        num_parallel_clients*num_iterations_per_client)

    # Run models both from low-level API and from command.
    t = threading.Thread(target=run_model_execute_from_llapi)
    t.start()

    # Run with single model key.
    ret = con.execute_command('AI.MODELSTORE', 'm{1}', 'TF', DEVICE,
                              'INPUTS', 2, 'a', 'b', 'OUTPUTS', 1, 'mul', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')
    parent_end_pipes, children_end_pipes = get_parent_children_pipes(num_parallel_clients)
    run_test_multiproc(env, '{1}', num_parallel_clients, create_run_update_models_parallel,
                       args=(env, model_pb, children_end_pipes, num_iterations_per_client, False))

    # Expect minimal number of success (one per five clients in average)
    # in running the models (without having it deleted by another client).
    num_success = sum([p.recv() for p in parent_end_pipes])
    env.assertGreaterEqual(num_parallel_clients*num_iterations_per_client, num_success)
    # Valgrind impacts the timings, so number of success may be lower.
    env.assertGreaterEqual(num_success, 1 if VALGRIND or DEVICE == "GPU" else num_parallel_clients/5)

    t.join()
    # Get the list of models in the system and verify that every model appears once in the global dict.
    models = con.execute_command('AI._MODELSCAN')
    # In LLAPI, every client used a distinct model name, while clients that ran via command line used a different name.
    env.assertEqual(len(models), num_parallel_clients + 1)

    # At the end, expect that every client (that didn't use the LLAPI) ran the last created model at most once.
    info = info_to_dict(con.execute_command('AI.INFO', 'm{1}'))
    env.assertGreaterEqual(info['calls'], 0)
    env.assertGreaterEqual(num_parallel_clients, info['calls'])


def test_ai_config(env):
    con = get_connection(env, '{1}')

    # Get the default configs.
    res = con.execute_command('AI.CONFIG', 'GET', 'BACKENDSPATH')
    env.assertEqual(res, None)
    res = con.execute_command('AI.CONFIG', 'GET', 'MODEL_CHUNK_SIZE')
    env.assertEqual(res, 511*1024*1024)

    # Change the default configuration and validate the change.
    print(ROOT)
    res = con.execute_command('AI.CONFIG', 'BACKENDSPATH', ROOT+"/install-cpu/backends")
