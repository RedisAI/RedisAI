import redis

from includes import *

'''
python -m RLTest --test tests_pytorch.py --module path/to/redisai.so
'''


def test_pytorch_chunked_modelset(env):
    if not TEST_PT:
        env.debugPrint("skipping {} since TEST_PT=0".format(sys._getframe().f_code.co_name), force=True)
        return

    con = env.getConnection()

    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    model_filename = os.path.join(test_data_path, 'pt-minimal.pt')

    with open(model_filename, 'rb') as f:
        model = f.read()

    chunk_size = len(model) // 3

    model_chunks = [model[i:i + chunk_size] for i in range(0, len(model), chunk_size)]

    ret = con.execute_command('AI.MODELSET', 'm1', 'TORCH', DEVICE, 'BLOB', model)
    ret = con.execute_command('AI.MODELSET', 'm2', 'TORCH', DEVICE, 'BLOB', *model_chunks)

    model1 = con.execute_command('AI.MODELGET', 'm1', 'BLOB')
    model2 = con.execute_command('AI.MODELGET', 'm2', 'BLOB')

    env.assertEqual(model1, model2)

    ret = con.execute_command('AI.CONFIG', 'MODEL_CHUNK_SIZE', chunk_size)

    model2 = con.execute_command('AI.MODELGET', 'm2', 'BLOB')
    env.assertEqual(len(model2), len(model_chunks))
    env.assertTrue(all([el1 == el2 for el1, el2 in zip(model2, model_chunks)]))

    model3 = con.execute_command('AI.MODELGET', 'm2', 'META', 'BLOB')[-1]
    env.assertEqual(len(model3), len(model_chunks))
    env.assertTrue(all([el1 == el2 for el1, el2 in zip(model3, model_chunks)]))


def test_pytorch_modelrun(env):
    if not TEST_PT:
        env.debugPrint("skipping {} since TEST_PT=0".format(sys._getframe().f_code.co_name), force=True)
        return

    con = env.getConnection()

    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    model_filename = os.path.join(test_data_path, 'pt-minimal.pt')
    wrong_model_filename = os.path.join(test_data_path, 'graph.pb')

    ret = con.execute_command('AI.TENSORSET', 'a', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.TENSORSET', 'b', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')

    with open(model_filename, 'rb') as f:
        model_pb = f.read()

    with open(wrong_model_filename, 'rb') as f:
        wrong_model_pb = f.read()

    ret = con.execute_command('AI.MODELSET', 'm', 'TORCH', DEVICE, 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    ret = con.execute_command('AI.MODELGET', 'm', 'META')
    ret = con.execute_command('AI.MODELGET', 'm', 'META')
    env.assertEqual(len(ret), 14)
    # TODO: enable me. CI is having issues on GPU asserts of TORCH and CPU
    if DEVICE == "CPU":
        env.assertEqual(ret[1], b'TORCH')
        env.assertEqual(ret[3], b'CPU')
    env.assertEqual(ret[5], b'')
    env.assertEqual(ret[7], 0)
    env.assertEqual(ret[9], 0)
    # assert there are no inputs or outputs
    env.assertEqual(len(ret[11]), 0)
    env.assertEqual(len(ret[13]), 0)

    ret = con.execute_command('AI.MODELSET', 'm', 'TORCH', DEVICE, 'TAG', 'my:tag:v3', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    ret = con.execute_command('AI.MODELGET', 'm', 'META')
    env.assertEqual(len(ret), 14)
    env.assertEqual(ret[5], b'my:tag:v3')
    # TODO: enable me. CI is having issues on GPU asserts of TORCH and CPU
    if DEVICE == "CPU":
        env.assertEqual(ret[1], b'TORCH')
        env.assertEqual(ret[3], b'CPU')

    try:
        con.execute_command('AI.MODELSET', 'm', 'TORCH', DEVICE, 'BLOB', wrong_model_pb)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.MODELSET', 'm_1', 'TORCH', 'BLOB', model_pb)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.MODELSET', 'm_2', 'BLOB', model_pb)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)

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

    values = con.execute_command('AI.TENSORGET', 'c', 'VALUES')
    env.assertEqual(values, [b'4', b'6', b'4', b'6'])

    if env.useSlaves:
        con2 = env.getSlaveConnection()
        values2 = con2.execute_command('AI.TENSORGET', 'c', 'VALUES')
        env.assertEqual(values2, values)


def test_pytorch_modelrun_batchdim_mismatch(env):
    con = env.getConnection()

    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    model_filename = os.path.join(test_data_path, 'batchdim_mismatch.pt')

    with open(model_filename, 'rb') as f:
        model_pb = f.read()

    ret = con.execute_command('AI.MODELSET', 'm', 'TORCH', DEVICE, 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    con.execute_command('AI.TENSORSET', 'a', 'FLOAT', 2, 'VALUES', 1, 1)
    con.execute_command('AI.TENSORSET', 'b', 'FLOAT', 2, 'VALUES', 1, 1)

    con.execute_command('AI.MODELRUN', 'm', 'INPUTS', 'a', 'b', 'OUTPUTS', 'c', 'd')


def test_pytorch_modelrun_autobatch(env):
    if not TEST_PT:
        return

    con = env.getConnection()

    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    model_filename = os.path.join(test_data_path, 'pt-minimal.pt')

    with open(model_filename, 'rb') as f:
        model_pb = f.read()

    ret = con.execute_command('AI.MODELSET', 'm', 'TORCH', 'CPU',
                              'BATCHSIZE', 4, 'MINBATCHSIZE', 3, 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    con.execute_command('AI.TENSORSET', 'a', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    con.execute_command('AI.TENSORSET', 'b', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)

    con.execute_command('AI.TENSORSET', 'd', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    con.execute_command('AI.TENSORSET', 'e', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)

    ensureSlaveSynced(con, env)

    def run():
        con = env.getConnection()
        con.execute_command('AI.MODELRUN', 'm', 'INPUTS', 'd', 'e', 'OUTPUTS', 'f')
        ensureSlaveSynced(con, env)

    t = threading.Thread(target=run)
    t.start()

    con.execute_command('AI.MODELRUN', 'm', 'INPUTS', 'a', 'b', 'OUTPUTS', 'c')

    ensureSlaveSynced(con, env)

    values = con.execute_command('AI.TENSORGET', 'c', 'VALUES')
    env.assertEqual(values, [b'4', b'6', b'4', b'6'])

    values = con.execute_command('AI.TENSORGET', 'f', 'VALUES')
    env.assertEqual(values, [b'4', b'6', b'4', b'6'])


def test_pytorch_modelrun_autobatch_badbatch(env):
    if not TEST_PT:
        return

    con = env.getConnection()

    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    model_filename = os.path.join(test_data_path, 'pt-minimal-bb.pt')

    with open(model_filename, 'rb') as f:
        model_pb = f.read()

    ret = con.execute_command('AI.MODELSET', 'm', 'TORCH', 'CPU',
                              'BATCHSIZE', 4, 'MINBATCHSIZE', 3, 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    con.execute_command('AI.TENSORSET', 'a', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    con.execute_command('AI.TENSORSET', 'b', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)

    con.execute_command('AI.TENSORSET', 'd', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    con.execute_command('AI.TENSORSET', 'e', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)

    ensureSlaveSynced(con, env)

    def run():
        con = env.getConnection()
        try:
            con.execute_command('AI.MODELRUN', 'm', 'INPUTS', 'd', 'e', 'OUTPUTS', 'f1', 'f2')
        except Exception as e:
            exception = e
            env.assertEqual(type(exception), redis.exceptions.ResponseError)
            env.assertEqual("Model did not generate the expected batch size", exception.__str__())

    t = threading.Thread(target=run)
    t.start()

    try:
        con.execute_command('AI.MODELRUN', 'm', 'INPUTS', 'a', 'b', 'OUTPUTS', 'c1', 'c2')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("Model did not generate the expected batch size", exception.__str__())



def test_pytorch_modelinfo(env):
    if not TEST_PT:
        env.debugPrint("skipping {} since TEST_PT=0".format(sys._getframe().f_code.co_name), force=True)
        return

    con = env.getConnection()

    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    model_filename = os.path.join(test_data_path, 'pt-minimal.pt')

    with open(model_filename, 'rb') as f:
        model_pb = f.read()

    ret = con.execute_command('AI.MODELSET', 'm', 'TORCH', DEVICE, 'TAG', 'asdf', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.TENSORSET', 'a', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.TENSORSET', 'b', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    previous_duration = 0
    for call in range(1, 10):
        ret = con.execute_command('AI.MODELRUN', 'm', 'INPUTS', 'a', 'b', 'OUTPUTS', 'c')
        env.assertEqual(ret, b'OK')
        ensureSlaveSynced(con, env)

        info = con.execute_command('AI.INFO', 'm')
        info_dict_0 = info_to_dict(info)

        env.assertEqual(info_dict_0['key'], 'm')
        env.assertEqual(info_dict_0['type'], 'MODEL')
        env.assertEqual(info_dict_0['backend'], 'TORCH')
        env.assertEqual(info_dict_0['device'], DEVICE)
        env.assertEqual(info_dict_0['tag'], 'asdf')
        env.assertTrue(info_dict_0['duration'] > previous_duration)
        env.assertEqual(info_dict_0['samples'], 2 * call)
        env.assertEqual(info_dict_0['calls'], call)
        env.assertEqual(info_dict_0['errors'], 0)

        previous_duration = info_dict_0['duration']

    res = con.execute_command('AI.INFO', 'm', 'RESETSTAT')
    env.assertEqual(res, b'OK')
    info = con.execute_command('AI.INFO', 'm')
    info_dict_0 = info_to_dict(info)
    env.assertEqual(info_dict_0['duration'], 0)
    env.assertEqual(info_dict_0['samples'], 0)
    env.assertEqual(info_dict_0['calls'], 0)
    env.assertEqual(info_dict_0['errors'], 0)


def test_pytorch_scriptset(env):
    if not TEST_PT:
        env.debugPrint("skipping {} since TEST_PT=0".format(sys._getframe().f_code.co_name), force=True)
        return

    con = env.getConnection()

    try:
        con.execute_command('AI.SCRIPTSET', 'ket', DEVICE, 'SOURCE', 'return 1')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.SCRIPTSET', 'nope')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.SCRIPTSET', 'nope', 'SOURCE')
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

    ret = con.execute_command('AI.SCRIPTSET', 'ket', DEVICE, 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    with open(script_filename, 'rb') as f:
        script = f.read()

    ret = con.execute_command('AI.SCRIPTSET', 'ket', DEVICE, 'TAG', 'asdf', 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    # TODO: Check why this COMMAND is hanging CI
    # ret = con.execute_command('AI.SCRIPTGET', 'ket')
    # env.assertEqual([b'CPU',script],ret)
    #
    # if env.useSlaves:
    #     con2 = env.getSlaveConnection()
    #     script_slave = con2.execute_command('AI.SCRIPTGET', 'ket')
    #     env.assertEqual(ret, script_slave)


def test_pytorch_scriptdel(env):
    if not TEST_PT:
        env.debugPrint("skipping {} since TEST_PT=0".format(sys._getframe().f_code.co_name), force=True)
        return

    con = env.getConnection()

    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    script_filename = os.path.join(test_data_path, 'script.txt')

    with open(script_filename, 'rb') as f:
        script = f.read()

    ret = con.execute_command('AI.SCRIPTSET', 'ket', DEVICE, 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    ret = con.execute_command('AI.SCRIPTDEL', 'ket')
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    env.assertFalse(con.execute_command('EXISTS', 'ket'))

    if env.useSlaves:
        con2 = env.getSlaveConnection()
        env.assertFalse(con2.execute_command('EXISTS', 'ket'))

    # ERR no script at key from SCRIPTDEL
    try:
        con.execute_command('DEL', 'EMPTY')
        con.execute_command('AI.SCRIPTDEL', 'EMPTY')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("script key is empty", exception.__str__())

    # ERR wrong type from SCRIPTDEL
    try:
        con.execute_command('SET', 'NOT_SCRIPT', 'BAR')
        con.execute_command('AI.SCRIPTDEL', 'NOT_SCRIPT')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("WRONGTYPE Operation against a key holding the wrong kind of value", exception.__str__())


def test_pytorch_scriptrun(env):
    if not TEST_PT:
        env.debugPrint("skipping {} since TEST_PT=0".format(sys._getframe().f_code.co_name), force=True)
        return

    con = env.getConnection()

    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    script_filename = os.path.join(test_data_path, 'script.txt')

    with open(script_filename, 'rb') as f:
        script = f.read()

    ret = con.execute_command('AI.SCRIPTSET', 'myscript', DEVICE, 'TAG', 'version1', 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.TENSORSET', 'a', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')
    ret = con.execute_command('AI.TENSORSET', 'b', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    for _ in range( 0,100):

        ret = con.execute_command('AI.SCRIPTRUN', 'myscript', 'bar', 'INPUTS', 'a', 'b', 'OUTPUTS', 'c')
        env.assertEqual(ret, b'OK')


    ensureSlaveSynced(con, env)

    info = con.execute_command('AI.INFO', 'myscript')
    info_dict_0 = info_to_dict(info)

    env.assertEqual(info_dict_0['key'], 'myscript')
    env.assertEqual(info_dict_0['type'], 'SCRIPT')
    env.assertEqual(info_dict_0['backend'], 'TORCH')
    env.assertEqual(info_dict_0['tag'], 'version1')
    env.assertTrue(info_dict_0['duration'] > 0)
    env.assertEqual(info_dict_0['samples'], -1)
    env.assertEqual(info_dict_0['calls'], 100)
    env.assertEqual(info_dict_0['errors'], 0)

    values = con.execute_command('AI.TENSORGET', 'c', 'VALUES')
    env.assertEqual(values, [b'4', b'6', b'4', b'6'])

    ensureSlaveSynced(con, env)

    if env.useSlaves:
        con2 = env.getSlaveConnection()
        values2 = con2.execute_command('AI.TENSORGET', 'c', 'VALUES')
        env.assertEqual(values2, values)


def test_pytorch_scriptrun_variadic(env):
    if not TEST_PT:
        env.debugPrint("skipping {} since TEST_PT=0".format(sys._getframe().f_code.co_name), force=True)
        return

    con = env.getConnection()

    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    script_filename = os.path.join(test_data_path, 'script.txt')

    with open(script_filename, 'rb') as f:
        script = f.read()

    ret = con.execute_command('AI.SCRIPTSET', 'myscript', DEVICE, 'TAG', 'version1', 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.TENSORSET', 'a', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')
    ret = con.execute_command('AI.TENSORSET', 'b1', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')
    ret = con.execute_command('AI.TENSORSET', 'b2', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    for _ in range( 0,100):
        ret = con.execute_command('AI.SCRIPTRUN', 'myscript', 'bar_variadic', 'INPUTS', 'a', '$', 'b1', 'b2', 'OUTPUTS', 'c')
        env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    info = con.execute_command('AI.INFO', 'myscript')
    info_dict_0 = info_to_dict(info)

    env.assertEqual(info_dict_0['key'], 'myscript')
    env.assertEqual(info_dict_0['type'], 'SCRIPT')
    env.assertEqual(info_dict_0['backend'], 'TORCH')
    env.assertEqual(info_dict_0['tag'], 'version1')
    env.assertTrue(info_dict_0['duration'] > 0)
    env.assertEqual(info_dict_0['samples'], -1)
    env.assertEqual(info_dict_0['calls'], 100)
    env.assertEqual(info_dict_0['errors'], 0)

    values = con.execute_command('AI.TENSORGET', 'c', 'VALUES')
    env.assertEqual(values, [b'4', b'6', b'4', b'6'])

    ensureSlaveSynced(con, env)

    if env.useSlaves:
        con2 = env.getSlaveConnection()
        values2 = con2.execute_command('AI.TENSORGET', 'c', 'VALUES')
        env.assertEqual(values2, values)


def test_pytorch_scriptrun_errors(env):
    if not TEST_PT:
        env.debugPrint("skipping {} since TEST_PT=0".format(sys._getframe().f_code.co_name), force=True)
        return

    con = env.getConnection()

    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    script_filename = os.path.join(test_data_path, 'script.txt')

    with open(script_filename, 'rb') as f:
        script = f.read()

    ret = con.execute_command('AI.SCRIPTSET', 'ket', DEVICE, 'TAG', 'asdf', 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.TENSORSET', 'a', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')
    ret = con.execute_command('AI.TENSORSET', 'b', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    # ERR no script at key from SCRIPTGET
    try:
        con.execute_command('DEL', 'EMPTY')
        con.execute_command('AI.SCRIPTGET', 'EMPTY')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("script key is empty", exception.__str__())

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
        env.assertEqual("tensor key is empty", exception.__str__())

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


def test_pytorch_scriptrun_variadic_errors(env):
    if not TEST_PT:
        env.debugPrint("skipping {} since TEST_PT=0".format(sys._getframe().f_code.co_name), force=True)
        return

    con = env.getConnection()

    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    script_filename = os.path.join(test_data_path, 'script.txt')

    with open(script_filename, 'rb') as f:
        script = f.read()

    ret = con.execute_command('AI.SCRIPTSET', 'ket', DEVICE, 'TAG', 'asdf', 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.TENSORSET', 'a', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')
    ret = con.execute_command('AI.TENSORSET', 'b', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    # ERR Variadic input key is empty
    try:
        con.execute_command('DEL', 'EMPTY')
        con.execute_command('AI.SCRIPTRUN', 'ket', 'bar_variadic', 'INPUTS', 'a', '$', 'EMPTY', 'b', 'OUTPUTS', 'c')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("tensor key is empty", exception.__str__())

    # ERR Variadic input key not tensor
    try:
        con.execute_command('SET', 'NOT_TENSOR', 'BAR')
        con.execute_command('AI.SCRIPTRUN', 'ket', 'bar_variadic', 'INPUTS', 'a', '$' , 'NOT_TENSOR', 'b', 'OUTPUTS', 'c')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("WRONGTYPE Operation against a key holding the wrong kind of value", exception.__str__())

    try:
        con.execute_command('AI.SCRIPTRUN', 'ket', 'bar_variadic', 'INPUTS', 'b', '$', 'OUTPUTS', 'c')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.SCRIPTRUN', 'ket', 'bar_variadic', 'INPUTS', 'b', '$', 'OUTPUTS')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.SCRIPTRUN', 'ket', 'bar_variadic', 'INPUTS', '$', 'OUTPUTS')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
    
    # "ERR Already encountered a variable size list of tensors"
    try:
        con.execute_command('AI.SCRIPTRUN', 'ket', 'bar_variadic', 'INPUTS', '$', 'a', '$', 'b' 'OUTPUTS')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)


def test_pytorch_scriptinfo(env):
    if not TEST_PT:
        env.debugPrint("skipping {} since TEST_PT=0".format(sys._getframe().f_code.co_name), force=True)
        return

    # env.debugPrint("skipping this test for now", force=True)
    # return

    con = env.getConnection()

    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    script_filename = os.path.join(test_data_path, 'script.txt')

    with open(script_filename, 'rb') as f:
        script = f.read()

    ret = con.execute_command('AI.SCRIPTSET', 'ket_script', DEVICE, 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.TENSORSET', 'a', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')
    ret = con.execute_command('AI.TENSORSET', 'b', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    previous_duration = 0
    for call in range(1, 100):
        ret = con.execute_command('AI.SCRIPTRUN', 'ket_script', 'bar', 'INPUTS', 'a', 'b', 'OUTPUTS', 'c')
        env.assertEqual(ret, b'OK')
        ensureSlaveSynced(con, env)

        info = con.execute_command('AI.INFO', 'ket_script')
        info_dict_0 = info_to_dict(info)

        env.assertEqual(info_dict_0['key'], 'ket_script')
        env.assertEqual(info_dict_0['type'], 'SCRIPT')
        env.assertEqual(info_dict_0['backend'], 'TORCH')
        env.assertEqual(info_dict_0['device'], DEVICE)
        env.assertTrue(info_dict_0['duration'] > previous_duration)
        env.assertEqual(info_dict_0['samples'], -1)
        env.assertEqual(info_dict_0['calls'], call)
        env.assertEqual(info_dict_0['errors'], 0)

        previous_duration = info_dict_0['duration']

    res = con.execute_command('AI.INFO', 'ket_script', 'RESETSTAT')
    env.assertEqual(res, b'OK')
    info = con.execute_command('AI.INFO', 'ket_script')
    info_dict_0 = info_to_dict(info)
    env.assertEqual(info_dict_0['duration'], 0)
    env.assertEqual(info_dict_0['samples'], -1)
    env.assertEqual(info_dict_0['calls'], 0)
    env.assertEqual(info_dict_0['errors'], 0)


def test_pytorch_scriptrun_disconnect(env):
    if not TEST_PT:
        env.debugPrint("skipping {} since TEST_PT=0".format(sys._getframe().f_code.co_name), force=True)
        return

    if DEVICE == "GPU":
        env.debugPrint("skipping {} since it's hanging CI".format(sys._getframe().f_code.co_name), force=True)
        return

    con = env.getConnection()

    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    script_filename = os.path.join(test_data_path, 'script.txt')

    with open(script_filename, 'rb') as f:
        script = f.read()

    ret = con.execute_command('AI.SCRIPTSET', 'ket_script', DEVICE, 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.TENSORSET', 'a', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')
    ret = con.execute_command('AI.TENSORSET', 'b', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    ret = send_and_disconnect(('AI.SCRIPTRUN', 'ket_script', 'bar', 'INPUTS', 'a', 'b', 'OUTPUTS', 'c'), con)
    env.assertEqual(ret, None)


def test_pytorch_modelrun_disconnect(env):
    if not TEST_PT:
        env.debugPrint("skipping {} since TEST_PT=0".format(sys._getframe().f_code.co_name), force=True)
        return

    if DEVICE == "GPU":
        env.debugPrint("skipping {} since it's hanging CI".format(sys._getframe().f_code.co_name), force=True)
        return

    con = env.getConnection()

    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    model_filename = os.path.join(test_data_path, 'pt-minimal.pt')

    with open(model_filename, 'rb') as f:
        model_pb = f.read()

    ret = con.execute_command('AI.MODELSET', 'm', 'TORCH', DEVICE, 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.TENSORSET', 'a', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.TENSORSET', 'b', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    ret = send_and_disconnect(('AI.MODELRUN', 'm', 'INPUTS', 'a', 'b', 'OUTPUTS', 'c'), con)
    env.assertEqual(ret, None)


def test_pytorch_modelscan_scriptscan(env):
    if not TEST_PT:
        env.debugPrint("skipping {} since TEST_PT=0".format(sys._getframe().f_code.co_name), force=True)
        return

    con = env.getConnection()

    # ensure cleaned DB
    # env.flush()

    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    model_filename = os.path.join(test_data_path, 'pt-minimal.pt')

    with open(model_filename, 'rb') as f:
        model_pb = f.read()

    ret = con.execute_command('AI.MODELSET', 'm1', 'TORCH', DEVICE, 'TAG', 'm:v1', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.MODELSET', 'm2', 'TORCH', DEVICE, 'TAG', 'm:v1', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    script_filename = os.path.join(test_data_path, 'script.txt')

    with open(script_filename, 'rb') as f:
        script = f.read()

    ret = con.execute_command('AI.SCRIPTSET', 's1', DEVICE, 'TAG', 's:v1', 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.SCRIPTSET', 's2', DEVICE, 'TAG', 's:v1', 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    ret = con.execute_command('AI._MODELSCAN')
    env.assertEqual(2, len(ret[0]))
    env.assertEqual(2, len(ret[1]))

    ret = con.execute_command('AI._SCRIPTSCAN')

    env.assertEqual(2, len(ret[0]))
    env.assertEqual(2, len(ret[1]))


def test_pytorch_model_rdb_save_load(env):
    env.skipOnCluster()
    if env.useAof or not TEST_PT:
        env.debugPrint("skipping {}".format(sys._getframe().f_code.co_name), force=True)
        return
    if DEVICE == "GPU":
        env.debugPrint("skipping {} since it's hanging CI".format(sys._getframe().f_code.co_name), force=True)
        return

    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    model_filename = os.path.join(test_data_path, 'pt-minimal.pt')

    with open(model_filename, 'rb') as f:
        model_pb = f.read()

    con = env.getConnection()

    ret = con.execute_command('AI.MODELSET', 'm', 'TORCH', DEVICE, 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    model_serialized_memory = con.execute_command('AI.MODELGET', 'm', 'BLOB')

    con.execute_command('AI.TENSORSET', 'a', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    con.execute_command('AI.TENSORSET', 'b', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    con.execute_command('AI.MODELRUN', 'm', 'INPUTS', 'a', 'b', 'OUTPUTS', 'c')
    _, dtype_memory, _, shape_memory, _, data_memory = con.execute_command('AI.TENSORGET', 'c', 'META', 'VALUES')

    ensureSlaveSynced(con, env)
    ret = con.execute_command('SAVE')
    env.assertEqual(ret, True)

    env.stop()
    env.start()
    con = env.getConnection()
    model_serialized_after_rdbload = con.execute_command('AI.MODELGET', 'm', 'BLOB')
    con.execute_command('AI.MODELRUN', 'm', 'INPUTS', 'a', 'b', 'OUTPUTS', 'c')
    _, dtype_after_rdbload, _, shape_after_rdbload, _, data_after_rdbload = con.execute_command('AI.TENSORGET', 'c', 'META', 'VALUES')

    # Assert in memory model metadata is equal to loaded model metadata
    env.assertTrue(model_serialized_memory[1:6] == model_serialized_after_rdbload[1:6])
    # Assert in memory tensor data is equal to loaded tensor data
    env.assertTrue(dtype_memory == dtype_after_rdbload)
    env.assertTrue(shape_memory == shape_after_rdbload)
    env.assertTrue(data_memory == data_after_rdbload)
