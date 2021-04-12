import redis

from includes import *

'''
python -m RLTest --test tests_execute_commands.py --module path/to/redisai.so
'''


def test_modelexecute_errors(env):
    if not TEST_TF:
        env.debugPrint("Skipping test since TF is not available", force=True)
        return
    con = env.getConnection()

    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    model_filename = os.path.join(test_data_path, 'graph.pb')

    with open(model_filename, 'rb') as f:
        model_pb = f.read()
    ret = con.execute_command('AI.MODELSET', 'm{1}', 'TF', DEVICE,
                              'INPUTS', 'a', 'b', 'OUTPUTS', 'mul', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')
    ensureSlaveSynced(con, env)

    try:
        con.execute_command('AI.MODELEXECUTE', 'm{1}', 'INPUTS', 2, 'a{1}', 'b{1}', 'OUTPUTS')
        env.assertFalse(True)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("wrong number of arguments for 'AI.MODELEXECUTE' command", str(exception))

    try:
        con.execute_command('AI.MODELEXECUTE', 'not_exiting_model{1}', 'INPUTS', 2, 'a{1}', 'b{1}', 'OUTPUTS', 1, 'c{1}')
        env.assertFalse(True)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("model key is empty", str(exception))

    # INPUTS must be the fourth argument (timeout is allowed only at the end)
    try:
        con.execute_command('AI.MODELEXECUTE', 'm{1}', 'timeout', 1000, 'INPUTS', 2, 'a{1}', 'b{1}', 'OUTPUTS', 1, 'c{1}')
        env.assertFalse(True)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("INPUTS not specified", str(exception))

    try:
        con.execute_command('AI.MODELEXECUTE', 'm{1}', 'INPUTS', 'not_a_number', 'a{1}', 'b{1}', 'OUTPUTS', 1, 'c{1}')
        env.assertFalse(True)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("Invalid argument for input_count", str(exception))

    try:
        con.execute_command('AI.MODELEXECUTE', 'm{1}', 'INPUTS', 0, 'a{1}', 'b{1}', 'OUTPUTS', 1, 'c{1}')
        env.assertFalse(True)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("Input count must be a positive integer", str(exception))

    try:
        con.execute_command('AI.MODELEXECUTE', 'm{1}', 'INPUTS', 1, 'a{1}', 'OUTPUTS', 1, 'c{1}')
        env.assertFalse(True)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("Number of keys given as INPUTS here does not match model definition", str(exception))

    try:
        con.execute_command('AI.MODELEXECUTE', 'm{1}', 'INPUTS', 2, 'a{1}', 'b{1}', 'NOT_OUTPUTS', 1, 'c{1}')
        env.assertFalse(True)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("OUTPUTS not specified", str(exception))

    try:
        con.execute_command('AI.MODELEXECUTE', 'm{1}', 'INPUTS', 2, 'a{1}', 'b{1}', 'OUTPUTS', 'not_a_number', 'c{1}')
        env.assertFalse(True)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("Invalid argument for output_count", str(exception))

    try:
        con.execute_command('AI.MODELEXECUTE', 'm{1}', 'INPUTS', 2, 'a{1}', 'b{1}', 'OUTPUTS', 0, 'c{1}')
        env.assertFalse(True)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("Input count must be a positive integer", str(exception))

    # This model expects only 1 output
    try:
        con.execute_command('AI.MODELEXECUTE', 'm{1}', 'INPUTS', 2, 'a{1}', 'b{1}', 'OUTPUTS', 2, 'c{1}', 'd{1}')
        env.assertFalse(True)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("Number of keys given as OUTPUTS here does not match model definition", str(exception))

    try:
        con.execute_command('AI.MODELEXECUTE', 'm{1}', 'INPUTS', 2, 'a{1}', 'b{1}', 'OUTPUTS', 1)
        env.assertFalse(True)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("number of output keys to AI.MODELEXECUTE command does not match the number of "
                        "given arguments", str(exception))

    try:
        con.execute_command('AI.MODELEXECUTE', 'm{1}', 'INPUTS', 2, 'a{1}', 'b{1}', 'OUTPUTS', 1, 'c{1}', 'bad_arg')
        env.assertFalse(True)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("Invalid argument: bad_arg", str(exception))

    try:
        con.execute_command('AI.MODELEXECUTE', 'm{1}', 'INPUTS', 2, 'a{1}', 'b{1}', 'OUTPUTS', 1, 'c{1}', 'TIMEOUT')
        env.assertFalse(True)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("No value provided for TIMEOUT", str(exception))

    try:
        con.execute_command('AI.MODELEXECUTE', 'm{1}', 'INPUTS', 2, 'a{1}', 'b{1}', 'OUTPUTS', 1, 'c{1}', 'TIMEOUT', -43)
        env.assertFalse(True)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("Invalid value for TIMEOUT", str(exception))

    try:
        con.execute_command('AI.MODELEXECUTE', 'm{1}', 'INPUTS', 2, 'a{1}', 'b{1}', 'OUTPUTS', 1, 'c{1}', 'TIMEOUT', 1000, 'bad_arg')
        env.assertFalse(True)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("Invalid argument: bad_arg", str(exception))


def test_modelexecute(env):
    if not TEST_TF:
        env.debugPrint("Skipping test since TF is not available", force=True)
        return
    con = env.getConnection()

    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    model_filename = os.path.join(test_data_path, 'graph.pb')

    with open(model_filename, 'rb') as f:
        model_pb = f.read()
    ret = con.execute_command('AI.MODELSET', 'm{1}', 'TF', DEVICE,
                              'INPUTS', 'a', 'b', 'OUTPUTS', 'mul', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')
    ensureSlaveSynced(con, env)
    con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    con.execute_command('AI.TENSORSET', 'b{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    ensureSlaveSynced(con, env)

    # Run the model and check for the expected result.
    con.execute_command('AI.MODELEXECUTE', 'm{1}', 'INPUTS', 2, 'a{1}', 'b{1}', 'OUTPUTS', 1, 'c{1}')
    ensureSlaveSynced(con, env)
    values = con.execute_command('AI.TENSORGET', 'c{1}', 'VALUES')
    env.assertEqual(values, [b'4', b'9', b'4', b'9'])

    if env.useSlaves:
        con2 = env.getSlaveConnection()
        values2 = con2.execute_command('AI.TENSORGET', 'c{1}', 'VALUES')
        env.assertEqual(values2, values)

    # This should an error on cluster (keys are not on the same shard)
    con.execute_command('AI.TENSORSET', 'b', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    try:
        con.execute_command('AI.MODELEXECUTE', 'm{1}', 'INPUTS', 2, 'a{1}', 'b', 'OUTPUTS', 1, 'c{1}')
        if env.isCluster():
            env.assertFalse(True)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("CROSSSLOT Keys in request don't hash to the same slot", str(exception))
