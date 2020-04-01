import redis

from includes import *

'''
python -m RLTest --test tests_pytorch.py --module path/to/redisai.so
'''


def test_pytorch_modelrun(env):
    if not TEST_PT:
        env.debugPrint("skipping {} since TEST_PT=0".format(sys._getframe().f_code.co_name), force=True)
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
    env.assertEqual(len(ret), 6)
    env.assertEqual(ret[-1], b'')

    ret = con.execute_command('AI.MODELSET', 'm', 'TORCH', DEVICE, 'TAG', 'asdf', model_pb)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    ret = con.execute_command('AI.MODELGET', 'm')
    env.assertEqual(len(ret), 6)
    env.assertEqual(ret[-1], b'asdf')


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

    if env.useSlaves:
        con2 = env.getSlaveConnection()
        tensor2 = con2.execute_command('AI.TENSORGET', 'c', 'VALUES')
        env.assertEqual(tensor2, tensor)


def test_pytorch_modelrun_autobatch(env):
    if not TEST_PT:
        return

    con = env.getConnection()

    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    model_filename = os.path.join(test_data_path, 'pt-minimal.pt')

    with open(model_filename, 'rb') as f:
        model_pb = f.read()

    ret = con.execute_command('AI.MODELSET', 'm', 'TORCH', 'CPU',
                              'BATCHSIZE', 4, 'MINBATCHSIZE', 3, model_pb)
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

    tensor = con.execute_command('AI.TENSORGET', 'c', 'VALUES')
    values = tensor[-1]
    env.assertEqual(values, [b'4', b'6', b'4', b'6'])

    tensor = con.execute_command('AI.TENSORGET', 'f', 'VALUES')
    values = tensor[-1]
    env.assertEqual(values, [b'4', b'6', b'4', b'6'])


def test_pytorch_modelinfo(env):
    if not TEST_PT:
        env.debugPrint("skipping {} since TEST_PT=0".format(sys._getframe().f_code.co_name), force=True)
        return

    con = env.getConnection()

    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    model_filename = os.path.join(test_data_path, 'pt-minimal.pt')

    with open(model_filename, 'rb') as f:
        model_pb = f.read()

    ret = con.execute_command('AI.MODELSET', 'm', 'TORCH', DEVICE, 'TAG', 'asdf', model_pb)
    env.assertEqual(ret, b'OK')

    ret = env.execute_command('AI.TENSORSET', 'a', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')

    ret = env.execute_command('AI.TENSORSET', 'b', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    previous_duration = 0
    for call in range(1, 10):
        ret = con.execute_command('AI.MODELRUN', 'm', 'INPUTS', 'a', 'b', 'OUTPUTS', 'c')
        env.assertEqual(ret, b'OK')
        ensureSlaveSynced(con, env)

        info = con.execute_command('AI.INFO', 'm')
        info_dict_0 = info_to_dict(info)

        env.assertEqual(info_dict_0['KEY'], 'm')
        env.assertEqual(info_dict_0['TYPE'], 'MODEL')
        env.assertEqual(info_dict_0['BACKEND'], 'TORCH')
        env.assertEqual(info_dict_0['DEVICE'], DEVICE)
        env.assertEqual(info_dict_0['TAG'], 'asdf')
        env.assertTrue(info_dict_0['DURATION'] > previous_duration)
        env.assertEqual(info_dict_0['SAMPLES'], 2 * call)
        env.assertEqual(info_dict_0['CALLS'], call)
        env.assertEqual(info_dict_0['ERRORS'], 0)

        previous_duration = info_dict_0['DURATION']

    res = con.execute_command('AI.INFO', 'm', 'RESETSTAT')
    env.assertEqual(res, b'OK')
    info = con.execute_command('AI.INFO', 'm')
    info_dict_0 = info_to_dict(info)
    env.assertEqual(info_dict_0['DURATION'], 0)
    env.assertEqual(info_dict_0['SAMPLES'], 0)
    env.assertEqual(info_dict_0['CALLS'], 0)
    env.assertEqual(info_dict_0['ERRORS'], 0)


def test_pytorch_scriptset(env):
    if not TEST_PT:
        env.debugPrint("skipping {} since TEST_PT=0".format(sys._getframe().f_code.co_name), force=True)
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

    with open(script_filename, 'rb') as f:
        script = f.read()

    ret = con.execute_command('AI.SCRIPTSET', 'ket', DEVICE, 'TAG', 'asdf', script)
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

    ret = con.execute_command('AI.SCRIPTSET', 'ket', DEVICE, script)
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
        env.assertEqual("no script at key", exception.__str__())

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

    ret = con.execute_command('AI.SCRIPTSET', 'ket', DEVICE, 'TAG', 'asdf', script)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.TENSORSET', 'a', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')
    ret = con.execute_command('AI.TENSORSET', 'b', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    # TODO: Check why this COMMAND is hanging CI
    # master_scriptget_result = con.execute_command('AI.SCRIPTGET', 'ket')
    # env.assertEqual([b'CPU',script],master_scriptget_result)
    #
    # if env.useSlaves:
    #     con2 = env.getSlaveConnection()
    #     slave_scriptget_result = con2.execute_command('AI.SCRIPTGET', 'ket')
    #     env.assertEqual(master_scriptget_result, slave_scriptget_result)

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
    env.assertEqual(info_dict_0['TAG'], 'asdf')
    env.assertTrue(info_dict_0['DURATION'] > 0)
    env.assertEqual(info_dict_0['SAMPLES'], -1)
    env.assertEqual(info_dict_0['CALLS'], 4)
    env.assertEqual(info_dict_0['ERRORS'], 3)

    tensor = con.execute_command('AI.TENSORGET', 'c', 'VALUES')
    values = tensor[-1]
    env.assertEqual(values, [b'4', b'6', b'4', b'6'])

    if env.useSlaves:
        con2 = env.getSlaveConnection()
        tensor2 = con2.execute_command('AI.TENSORGET', 'c', 'VALUES')
        env.assertEqual(tensor2, tensor)


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

    ret = con.execute_command('AI.SCRIPTSET', 'ket_script', DEVICE, script)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.TENSORSET', 'a', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')
    ret = con.execute_command('AI.TENSORSET', 'b', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    previous_duration = 0
    for call in range(1, 10):
        ret = con.execute_command('AI.SCRIPTRUN', 'ket_script', 'bar', 'INPUTS', 'a', 'b', 'OUTPUTS', 'c')
        env.assertEqual(ret, b'OK')
        ensureSlaveSynced(con, env)

        info = con.execute_command('AI.INFO', 'ket_script')
        info_dict_0 = info_to_dict(info)

        env.assertEqual(info_dict_0['KEY'], 'ket_script')
        env.assertEqual(info_dict_0['TYPE'], 'SCRIPT')
        env.assertEqual(info_dict_0['BACKEND'], 'TORCH')
        env.assertEqual(info_dict_0['DEVICE'], DEVICE)
        env.assertTrue(info_dict_0['DURATION'] > previous_duration)
        env.assertEqual(info_dict_0['SAMPLES'], -1)
        env.assertEqual(info_dict_0['CALLS'], call)
        env.assertEqual(info_dict_0['ERRORS'], 0)

        previous_duration = info_dict_0['DURATION']

    res = con.execute_command('AI.INFO', 'ket_script', 'RESETSTAT')
    env.assertEqual(res, b'OK')
    info = con.execute_command('AI.INFO', 'ket_script')
    info_dict_0 = info_to_dict(info)
    env.assertEqual(info_dict_0['DURATION'], 0)
    env.assertEqual(info_dict_0['SAMPLES'], -1)
    env.assertEqual(info_dict_0['CALLS'], 0)
    env.assertEqual(info_dict_0['ERRORS'], 0)


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

    ret = con.execute_command('AI.SCRIPTSET', 'ket_script', DEVICE, script)
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

    ret = con.execute_command('AI.MODELSET', 'm', 'TORCH', DEVICE, model_pb)
    env.assertEqual(ret, b'OK')

    ret = env.execute_command('AI.TENSORSET', 'a', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')

    ret = env.execute_command('AI.TENSORSET', 'b', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    ret = send_and_disconnect(('AI.MODELRUN', 'm', 'INPUTS', 'a', 'b', 'OUTPUTS', 'c'), con)
    env.assertEqual(ret, None)


def test_pytorch_modellist_scriptlist(env):
    if not TEST_PT:
        env.debugPrint("skipping {} since TEST_PT=0".format(sys._getframe().f_code.co_name), force=True)
        return

    con = env.getConnection()

    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    model_filename = os.path.join(test_data_path, 'pt-minimal.pt')

    with open(model_filename, 'rb') as f:
        model_pb = f.read()

    ret = con.execute_command('AI.MODELSET', 'm1', 'TORCH', DEVICE, 'TAG', 'm:v1', model_pb)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.MODELSET', 'm2', 'TORCH', DEVICE, 'TAG', 'm:v1', model_pb)
    env.assertEqual(ret, b'OK')

    script_filename = os.path.join(test_data_path, 'script.txt')

    with open(script_filename, 'rb') as f:
        script = f.read()

    ret = con.execute_command('AI.SCRIPTSET', 's1', DEVICE, 'TAG', 's:v1', script)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.SCRIPTSET', 's2', DEVICE, 'TAG', 's:v1', script)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    ret = con.execute_command('AI._MODELLIST')

    env.assertEqual(ret[0], [b'm1', b'm:v1'])
    env.assertEqual(ret[1], [b'm2', b'm:v1'])

    ret = con.execute_command('AI._SCRIPTLIST')

    env.assertEqual(ret[0], [b's1', b's:v1'])
    env.assertEqual(ret[1], [b's2', b's:v1'])


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

    ret = con.execute_command('AI.MODELSET', 'm', 'TORCH', DEVICE, model_pb)
    env.assertEqual(ret, b'OK')

    model_serialized_memory = con.execute_command('AI.MODELGET', 'm', 'BLOB')

    env.execute_command('AI.TENSORSET', 'a', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.execute_command('AI.TENSORSET', 'b', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    con.execute_command('AI.MODELRUN', 'm', 'INPUTS', 'a', 'b', 'OUTPUTS', 'c')
    dtype_memory, shape_memory, data_memory = con.execute_command('AI.TENSORGET', 'c', 'VALUES')

    ret = con.execute_command('SAVE')
    env.assertEqual(ret, True)

    env.stop()
    env.start()
    con = env.getConnection()
    model_serialized_after_rdbload = con.execute_command('AI.MODELGET', 'm', 'BLOB')
    con.execute_command('AI.MODELRUN', 'm', 'INPUTS', 'a', 'b', 'OUTPUTS', 'c')
    dtype_after_rdbload, shape_after_rdbload, data_after_rdbload = con.execute_command('AI.TENSORGET', 'c', 'VALUES')

    # Assert in memory model metadata is equal to loaded model metadata
    env.assertTrue(model_serialized_memory[1:6] == model_serialized_after_rdbload[1:6])
    # Assert in memory tensor data is equal to loaded tensor data
    env.assertTrue(dtype_memory == dtype_after_rdbload)
    env.assertTrue(shape_memory == shape_after_rdbload)
    env.assertTrue(data_memory == data_after_rdbload)