import redis

from includes import *

'''
python -m RLTest --test tests_common.py --module path/to/redisai.so
'''

def test_run_torch_model(env):
    if not TEST_PT:
        return

    con = env.getConnection()

    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    model_filename = os.path.join(test_data_path, 'pt-minimal.pt')
    wrong_model_filename = os.path.join(test_data_path, 'graph.pb')

    with open(model_filename, 'rb') as f:
        model_pb = f.read()

    with open(wrong_model_filename, 'rb') as f:
        wrong_model_pb = f.read()

    ret = con.execute_command('AI.MODELSET', 'm', 'TORCH', DEVICE, model_pb)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    ret = con.execute_command('AI.MODELGET', 'm')
    # TODO: enable me
    # env.assertEqual(ret[0], b'TORCH')
    # env.assertEqual(ret[1], b'CPU')

    try:
        con.execute_command('AI.MODELSET', 'm', 'TORCH', DEVICE, wrong_model_pb)
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.MODELSET', 'm_1', 'TORCH', model_pb)
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.MODELSET', 'm_2', model_pb)
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    env.execute_command('AI.TENSORSET', 'a', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.execute_command('AI.TENSORSET', 'b', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)

    try:
        con.execute_command('AI.MODELRUN', 'm_1', 'INPUTS', 'a', 'b', 'OUTPUTS')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.MODELRUN', 'm_2', 'INPUTS', 'a', 'b', 'c')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.MODELRUN', 'm_3', 'a', 'b', 'c')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.MODELRUN', 'm_1', 'OUTPUTS', 'c')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.MODELRUN', 'm', 'OUTPUTS', 'c')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.MODELRUN', 'm', 'INPUTS', 'a', 'b')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.MODELRUN', 'm_1', 'INPUTS', 'OUTPUTS')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.MODELRUN', 'm_1', 'INPUTS', 'a', 'b', 'OUTPUTS', 'c', 'd')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    con.execute_command('AI.MODELRUN', 'm', 'INPUTS', 'a', 'b', 'OUTPUTS', 'c')

    ensureSlaveSynced(con, env)

    tensor = con.execute_command('AI.TENSORGET', 'c', 'VALUES')
    values = tensor[-1]
    env.assertEqual(values, [b'4', b'6', b'4', b'6'])

    ensureSlaveSynced(con, env)
    if env.useSlaves:
        con2 = env.getSlaveConnection()
        tensor2 = con2.execute_command('AI.TENSORGET', 'c', 'VALUES')
        env.assertEqual(tensor2, tensor)


def test_set_script(env):
    if not TEST_PT:
        return

    con = env.getConnection()

    try:
        con.execute_command('AI.SCRIPTSET', 'ket', DEVICE, 'return 1')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.SCRIPTSET', 'nope')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.SCRIPTSET', 'more', DEVICE)
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    script_filename = os.path.join(test_data_path, 'script.txt')

    with open(script_filename, 'rb') as f:
        script = f.read()

    ret = con.execute_command('AI.SCRIPTSET', 'ket', DEVICE, script)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)


def test_del_script(env):
    if not TEST_PT:
        return

    con = env.getConnection()

    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    script_filename = os.path.join(test_data_path, 'script.txt')

    with open(script_filename, 'rb') as f:
        script = f.read()

    ret = con.execute_command('AI.SCRIPTSET', 'ket', DEVICE, script)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    ret = con.execute_command('AI.SCRIPTDEL', 'ket')
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    env.assertFalse(con.execute_command('EXISTS', 'ket'))

    # ERR no script at key from SCRIPTDEL
    try:
        con.execute_command('DEL', 'EMPTY')
        con.execute_command('AI.SCRIPTDEL', 'EMPTY')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)
    env.assertEqual("no script at key", exception.__str__())

    # ERR wrong type from SCRIPTDEL
    try:
        con.execute_command('SET', 'NOT_SCRIPT', 'BAR')
        con.execute_command('AI.SCRIPTDEL', 'NOT_SCRIPT')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)
    env.assertEqual("WRONGTYPE Operation against a key holding the wrong kind of value", exception.__str__())


def test_run_script(env):
    if not TEST_PT:
        return

    con = env.getConnection()

    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    script_filename = os.path.join(test_data_path, 'script.txt')

    with open(script_filename, 'rb') as f:
        script = f.read()

    ret = con.execute_command('AI.SCRIPTSET', 'ket', DEVICE, script)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.TENSORSET', 'a', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')
    ret = con.execute_command('AI.TENSORSET', 'b', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    # TODO: enable me ( this is hanging CI )
    # ret = con.execute_command('AI.SCRIPTGET', 'ket')
    # TODO: enable me
    # env.assertEqual([b'CPU',script],ret)

    # ERR no script at key from SCRIPTGET
    try:
        con.execute_command('DEL', 'EMPTY')
        con.execute_command('AI.SCRIPTGET', 'EMPTY')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("cannot get script from empty key", exception.__str__())

    # ERR wrong type from SCRIPTGET
    try:
        con.execute_command('SET', 'NOT_SCRIPT', 'BAR')
        con.execute_command('AI.SCRIPTGET', 'NOT_SCRIPT')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("WRONGTYPE Operation against a key holding the wrong kind of value", exception.__str__())

    # ERR no script at key from SCRIPTRUN
    try:
        con.execute_command('DEL', 'EMPTY')
        con.execute_command('AI.SCRIPTRUN', 'EMPTY', 'bar', 'INPUTS', 'b', 'OUTPUTS', 'c')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("script key is empty", exception.__str__())

    # ERR wrong type from SCRIPTRUN
    try:
        con.execute_command('SET', 'NOT_SCRIPT', 'BAR')
        con.execute_command('AI.SCRIPTRUN', 'NOT_SCRIPT', 'bar', 'INPUTS', 'b', 'OUTPUTS', 'c')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("WRONGTYPE Operation against a key holding the wrong kind of value", exception.__str__())

    # ERR Input key is empty
    try:
        con.execute_command('DEL', 'EMPTY')
        con.execute_command('AI.SCRIPTRUN', 'ket', 'bar', 'INPUTS', 'EMPTY', 'b', 'OUTPUTS', 'c')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("Input key is empty", exception.__str__())

    # ERR Input key not tensor
    try:
        con.execute_command('SET', 'NOT_TENSOR', 'BAR')
        con.execute_command('AI.SCRIPTRUN', 'ket', 'bar', 'INPUTS', 'NOT_TENSOR', 'b', 'OUTPUTS', 'c')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("WRONGTYPE Operation against a key holding the wrong kind of value", exception.__str__())

    try:
        con.execute_command('AI.SCRIPTRUN', 'ket', 'bar', 'INPUTS', 'b', 'OUTPUTS', 'c')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.SCRIPTRUN', 'ket', 'INPUTS', 'a', 'b', 'OUTPUTS', 'c')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.SCRIPTRUN', 'ket', 'bar', 'INPUTS', 'b', 'OUTPUTS')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.SCRIPTRUN', 'ket', 'bar', 'INPUTS', 'OUTPUTS')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)

    con.execute_command('AI.SCRIPTRUN', 'ket', 'bar', 'INPUTS', 'a', 'b', 'OUTPUTS', 'c')

    ensureSlaveSynced(con, env)

    info = con.execute_command('AI.INFO', 'ket')
    info_dict_0 = info_to_dict(info)

    env.assertEqual(info_dict_0['KEY'], 'ket')
    env.assertEqual(info_dict_0['TYPE'], 'SCRIPT')
    env.assertEqual(info_dict_0['BACKEND'], 'TORCH')
    env.assertTrue(info_dict_0['DURATION'] > 0)
    env.assertEqual(info_dict_0['SAMPLES'], -1)
    env.assertEqual(info_dict_0['CALLS'], 4)
    env.assertEqual(info_dict_0['ERRORS'], 3)

    con.execute_command('AI.SCRIPTRUN', 'ket', 'bar', 'INPUTS', 'a', 'b', 'OUTPUTS', 'c')

    ensureSlaveSynced(con, env)

    info = con.execute_command('AI.INFO', 'ket')
    info_dict_1 = info_to_dict(info)

    env.assertTrue(info_dict_1['DURATION'] > info_dict_0['DURATION'])
    env.assertEqual(info_dict_1['SAMPLES'], -1)
    env.assertEqual(info_dict_1['CALLS'], 5)
    env.assertEqual(info_dict_1['ERRORS'], 3)

    ret = con.execute_command('AI.INFO', 'ket', 'RESETSTAT')
    env.assertEqual(ret, b'OK')

    con.execute_command('AI.SCRIPTRUN', 'ket', 'bar', 'INPUTS', 'a', 'b', 'OUTPUTS', 'c')

    ensureSlaveSynced(con, env)

    info = con.execute_command('AI.INFO', 'ket')
    info_dict_2 = info_to_dict(info)

    env.assertTrue(info_dict_2['DURATION'] < info_dict_1['DURATION'])
    env.assertEqual(info_dict_2['SAMPLES'], -1)
    env.assertEqual(info_dict_2['CALLS'], 1)
    env.assertEqual(info_dict_2['ERRORS'], 0)

    tensor = con.execute_command('AI.TENSORGET', 'c', 'VALUES')
    values = tensor[-1]
    env.assertEqual(values, [b'4', b'6', b'4', b'6'])

    ensureSlaveSynced(con, env)
    if env.useSlaves:
        con2 = env.getSlaveConnection()
        tensor2 = con2.execute_command('AI.TENSORGET', 'c', 'VALUES')
        env.assertEqual(tensor2, tensor)
