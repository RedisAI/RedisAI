import redis
import time

from includes import *
from RLTest import Env

'''
python -m RLTest --test tests_pytorch.py --module path/to/redisai.so
'''


def test_pytorch_modelrun(env):
    if not TEST_PT:
        env.debugPrint("skipping {} since TEST_PT=0".format(sys._getframe().f_code.co_name), force=True)
        return

    con = get_connection(env, '{1}')

    model_pb = load_file_content('pt-minimal.pt')
    wrong_model_pb = load_file_content('graph.pb')

    ret = con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.TENSORSET', 'b{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.MODELSTORE', 'm{1}', 'TORCH', DEVICE, 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    ret = con.execute_command('AI.MODELGET', 'm{1}', 'META')
    env.assertEqual(len(ret), 16)
    # TODO: enable me. CI is having issues on GPU asserts of TORCH and CPU
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

    ret = con.execute_command('AI.MODELSTORE', 'm{1}', 'TORCH', DEVICE, 'TAG', 'my:tag:v3', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    ret = con.execute_command('AI.MODELGET', 'm{1}', 'META')
    env.assertEqual(len(ret), 16)
    env.assertEqual(ret[5], b'my:tag:v3')
    # TODO: enable me. CI is having issues on GPU asserts of TORCH and CPU
    if DEVICE == "CPU":
        env.assertEqual(ret[1], b'TORCH')
        env.assertEqual(ret[3], b'CPU')

    check_error(env, con, 'AI.MODELSTORE', 'm{1}', 'TORCH', DEVICE, 'BLOB', wrong_model_pb)

    con.execute_command('AI.MODELEXECUTE', 'm{1}', 'INPUTS', 2, 'a{1}', 'b{1}', 'OUTPUTS', 1, 'c{1}')

    ensureSlaveSynced(con, env)

    values = con.execute_command('AI.TENSORGET', 'c{1}', 'VALUES')
    env.assertEqual(values, [b'4', b'6', b'4', b'6'])

    if env.useSlaves:
        con2 = env.getSlaveConnection()
        values2 = con2.execute_command('AI.TENSORGET', 'c{1}', 'VALUES')
        env.assertEqual(values2, values)


def test_pytorch_modelrun_autobatch(env):
    if not TEST_PT:
        return

    con = get_connection(env, '{1}')

    model_pb = load_file_content('pt-minimal.pt')

    ret = con.execute_command('AI.MODELSTORE', 'm{1}', 'TORCH', 'CPU',
                              'BATCHSIZE', 4, 'MINBATCHSIZE', 2, 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    con.execute_command('AI.TENSORSET', 'b{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)

    con.execute_command('AI.TENSORSET', 'd{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    con.execute_command('AI.TENSORSET', 'e{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)

    ensureSlaveSynced(con, env)

    def run():
        con = get_connection(env, '{1}')
        con.execute_command('AI.MODELEXECUTE', 'm{1}', 'INPUTS', 2, 'd{1}', 'e{1}', 'OUTPUTS', 1, 'f{1}')
        ensureSlaveSynced(con, env)

    t = threading.Thread(target=run)
    t.start()

    con.execute_command('AI.MODELEXECUTE', 'm{1}', 'INPUTS', 2, 'a{1}', 'b{1}', 'OUTPUTS', 1, 'c{1}')
    t.join()

    ensureSlaveSynced(con, env)

    values = con.execute_command('AI.TENSORGET', 'c{1}', 'VALUES')
    env.assertEqual(values, [b'4', b'6', b'4', b'6'])

    values = con.execute_command('AI.TENSORGET', 'f{1}', 'VALUES')
    env.assertEqual(values, [b'4', b'6', b'4', b'6'])


def test_pytorch_modelrun_autobatch_badbatch(env):
    if not TEST_PT:
        return

    con = get_connection(env, '{1}')
    model_pb = load_file_content('pt-minimal-bb.pt')

    ret = con.execute_command('AI.MODELSTORE', 'm{1}', 'TORCH', 'CPU',
                              'BATCHSIZE', 4, 'MINBATCHSIZE', 3, 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    con.execute_command('AI.TENSORSET', 'b{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)

    con.execute_command('AI.TENSORSET', 'd{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    con.execute_command('AI.TENSORSET', 'e{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)

    ensureSlaveSynced(con, env)

    def run():
        con = get_connection(env, '{1}')
        check_error_message(env, con, "Model did not generate the expected batch size",
                            'AI.MODELEXECUTE', 'm{1}', 'INPUTS', 2, 'd{1}', 'e{1}', 'OUTPUTS', 2, 'f1{1}', 'f2{1}')

    t = threading.Thread(target=run)
    t.start()

    check_error_message(env, con, "Model did not generate the expected batch size",
                        'AI.MODELEXECUTE', 'm{1}', 'INPUTS', 2, 'a{1}', 'b{1}', 'OUTPUTS', 2, 'c1{1}', 'c2{1}')
    t.join()


def test_pytorch_modelinfo(env):
    if not TEST_PT:
        env.debugPrint("skipping {} since TEST_PT=0".format(sys._getframe().f_code.co_name), force=True)
        return

    con = get_connection(env, '{1}')

    model_pb = load_file_content('pt-minimal.pt')
    model_key = 'm{1}'
    tensor_a_key = 'a{1}'
    tensor_b_key = 'b{1}'
    tensor_c_key = 'c{1}'

    ret = con.execute_command('AI.MODELSTORE', model_key, 'TORCH', DEVICE, 'TAG', 'asdf', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.TENSORSET', tensor_a_key, 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.TENSORSET', tensor_b_key, 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    previous_duration = 0
    for call in range(1, 100):
        ret = con.execute_command('AI.MODELEXECUTE', model_key, 'INPUTS', 2, tensor_a_key, tensor_b_key, 'OUTPUTS', 1, tensor_c_key)
        env.assertEqual(ret, b'OK')
        ensureSlaveSynced(con, env)

        info = con.execute_command('AI.INFO', model_key)
        info_dict_0 = info_to_dict(info)

        env.assertEqual(info_dict_0['key'], model_key)
        env.assertEqual(info_dict_0['type'], 'MODEL')
        env.assertEqual(info_dict_0['backend'], 'TORCH')
        env.assertEqual(info_dict_0['device'], DEVICE)
        env.assertEqual(info_dict_0['tag'], 'asdf')
        env.assertTrue(info_dict_0['duration'] > previous_duration)
        env.assertEqual(info_dict_0['samples'], 2 * call)
        env.assertEqual(info_dict_0['calls'], call)
        env.assertEqual(info_dict_0['errors'], 0)

        previous_duration = info_dict_0['duration']

    res = con.execute_command('AI.INFO', model_key, 'RESETSTAT')
    env.assertEqual(res, b'OK')
    info = con.execute_command('AI.INFO', model_key)
    info_dict_0 = info_to_dict(info)
    env.assertEqual(info_dict_0['duration'], 0)
    env.assertEqual(info_dict_0['samples'], 0)
    env.assertEqual(info_dict_0['calls'], 0)
    env.assertEqual(info_dict_0['errors'], 0)


def test_pytorch_scriptget(env):
    if not TEST_PT:
        env.debugPrint("skipping {} since TEST_PT=0".format(sys._getframe().f_code.co_name), force=True)
        return

    con = get_connection(env, '{1}')
    con.execute_command('DEL', 'EMPTY{1}')
    # ERR no script at key from SCRIPTGET
    check_error_message(env, con, "script key is empty", 'AI.SCRIPTGET', 'EMPTY{1}')

    con.execute_command('SET', 'NOT_SCRIPT{1}', 'BAR')
    # ERR wrong type from SCRIPTGET
    check_error_message(env, con, "WRONGTYPE Operation against a key holding the wrong kind of value", 'AI.SCRIPTGET', 'NOT_SCRIPT{1}')

    script = load_file_content('script.txt')
    ret = con.execute_command('AI.SCRIPTSTORE', 'my_script{1}', DEVICE, 'TAG', 'my_tag',
                              'ENTRY_POINTS', 2, 'bar', 'bar_variadic', 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    # return meta + source
    _, device, _, tag, _, entry_points, _, source = con.execute_command('AI.SCRIPTGET', 'my_script{1}')
    env.assertEqual([device, tag, entry_points, source], [bytes(DEVICE, "utf8"), b"my_tag", [b'bar', b'bar_variadic'], script])
    # return source only
    source = con.execute_command('AI.SCRIPTGET', 'my_script{1}', 'SOURCE')
    env.assertEqual(source, script)


def test_pytorch_scriptdel(env):
    if not TEST_PT:
        env.debugPrint("skipping {} since TEST_PT=0".format(sys._getframe().f_code.co_name), force=True)
        return

    con = get_connection(env, '{1}')

    script = load_file_content('script.txt')

    ret = con.execute_command('AI.SCRIPTSTORE', 'ket{1}', DEVICE, 'ENTRY_POINTS', 2, 'bar', 'bar_variadic', 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    ret = con.execute_command('AI.SCRIPTDEL', 'ket{1}')
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    env.assertFalse(con.execute_command('EXISTS', 'ket{1}'))

    if env.useSlaves:
        con2 = env.getSlaveConnection()
        env.assertFalse(con2.execute_command('EXISTS', 'ket{1}'))

    con.execute_command('DEL', 'EMPTY{1}')
    # ERR no script at key from SCRIPTDEL
    check_error_message(env, con, "script key is empty", 'AI.SCRIPTDEL', 'EMPTY{1}')

    con.execute_command('SET', 'NOT_SCRIPT{1}', 'BAR')
    # ERR wrong type from SCRIPTDEL
    check_error_message(env, con, "WRONGTYPE Operation against a key holding the wrong kind of value", 'AI.SCRIPTDEL', 'NOT_SCRIPT{1}')

def test_pytorch_scriptexecute(env):
    if not TEST_PT:
        env.debugPrint("skipping {} since TEST_PT=0".format(sys._getframe().f_code.co_name), force=True)
        return

    con = get_connection(env, '{1}')

    script = load_file_content('script.txt')

    ret = con.execute_command('AI.SCRIPTSTORE', 'myscript{1}', DEVICE, 'TAG', 'version1', 'ENTRY_POINTS', 2, 'bar', 'bar_variadic', 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')
    ret = con.execute_command('AI.TENSORSET', 'b{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    for _ in range( 0,100):

        ret = con.execute_command('AI.SCRIPTEXECUTE', 'myscript{1}', 'bar', 'KEYS', 1, '{1}', 'INPUTS', 2, 'a{1}', 'b{1}', 'OUTPUTS', 1, 'c{1}')
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


def test_pytorch_scriptexecute_list_input(env):
    if not TEST_PT:
        env.debugPrint("skipping {} since TEST_PT=0".format(sys._getframe().f_code.co_name), force=True)
        return

    con = get_connection(env, '{1}')

    script = load_file_content('script.txt')

    ret = con.execute_command('AI.SCRIPTSTORE', 'myscript{$}', DEVICE, 'TAG', 'version1', 'ENTRY_POINTS', 2, 'bar', 'bar_variadic', 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.TENSORSET', 'a{$}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')
    ret = con.execute_command('AI.TENSORSET', 'b1{$}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')
    ret = con.execute_command('AI.TENSORSET', 'b2{$}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    for _ in range( 0,100):
        ret = con.execute_command('AI.SCRIPTEXECUTE', 'myscript{$}', 'bar_variadic', 'KEYS', 1, '{$}', 'INPUTS', 3, 'a{$}', 'b1{$}', 'b2{$}', 'OUTPUTS', 1, 'c{$}')
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


def test_pytorch_scriptexecute_with_timeout(env):
    if not TEST_PT:
        env.debugPrint("skipping {} since TEST_PT=0".format(sys._getframe().f_code.co_name), force=True)
        return

    con = get_connection(env, '{$}')
    script = load_file_content('script.txt')
    ret = con.execute_command('AI.SCRIPTSTORE', 'my_script{$}', DEVICE,
                              'ENTRY_POINTS', 2, 'bar', 'long_func', 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    con.execute_command('AI.TENSORSET', 'a{$}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    con.execute_command('AI.TENSORSET', 'b{$}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)

    def run():
        con2 = get_connection(env, '{$}')
        con2.execute_command('AI.SCRIPTEXECUTE', 'my_script{$}', 'long_func', 'KEYS', 1, '{$}')

    t = threading.Thread(target=run)
    t.start()

    # make sure that we have a long operation that RedisAI will run upon sending the following
    # command, to assure that timeout will occur.
    time.sleep(0.5)
    ret = con.execute_command('AI.SCRIPTEXECUTE', 'my_script{$}', 'bar',
                              'INPUTS', 2, 'a{$}', 'b{$}', 'OUTPUTS', 1, 'c{$}', 'TIMEOUT', 1)
    env.assertEqual(ret, b'TIMEDOUT')
    t.join()


def test_pytorch_scriptinfo(env):
    if not TEST_PT:
        env.debugPrint("skipping {} since TEST_PT=0".format(sys._getframe().f_code.co_name), force=True)
        return

    con = get_connection(env, '{1}')

    script = load_file_content('script.txt')

    ret = con.execute_command('AI.SCRIPTSTORE', 'ket_script{1}', DEVICE, 'ENTRY_POINTS', 2, 'bar', 'bar_variadic', 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')
    ret = con.execute_command('AI.TENSORSET', 'b{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    previous_duration = 0
    for call in range(1, 100):
        ret = con.execute_command('AI.SCRIPTEXECUTE', 'ket_script{1}', 'bar', 'KEYS', 1, '{1}', 'INPUTS', 2, 'a{1}', 'b{1}', 'OUTPUTS', 1, 'c{1}')
        env.assertEqual(ret, b'OK')
        ensureSlaveSynced(con, env)

        info = con.execute_command('AI.INFO', 'ket_script{1}')
        info_dict_0 = info_to_dict(info)

        env.assertEqual(info_dict_0['key'], 'ket_script{1}')
        env.assertEqual(info_dict_0['type'], 'SCRIPT')
        env.assertEqual(info_dict_0['backend'], 'TORCH')
        env.assertEqual(info_dict_0['device'], DEVICE)
        env.assertTrue(info_dict_0['duration'] > previous_duration)
        env.assertEqual(info_dict_0['samples'], -1)
        env.assertEqual(info_dict_0['calls'], call)
        env.assertEqual(info_dict_0['errors'], 0)

        previous_duration = info_dict_0['duration']

    res = con.execute_command('AI.INFO', 'ket_script{1}', 'RESETSTAT')
    env.assertEqual(res, b'OK')
    info = con.execute_command('AI.INFO', 'ket_script{1}')
    info_dict_0 = info_to_dict(info)
    env.assertEqual(info_dict_0['duration'], 0)
    env.assertEqual(info_dict_0['samples'], -1)
    env.assertEqual(info_dict_0['calls'], 0)
    env.assertEqual(info_dict_0['errors'], 0)


def test_pytorch_scriptexecute_disconnect(env):
    if not TEST_PT:
        env.debugPrint("skipping {} since TEST_PT=0".format(sys._getframe().f_code.co_name), force=True)
        return

    if DEVICE == "GPU":
        env.debugPrint("skipping {} since it's hanging CI".format(sys._getframe().f_code.co_name), force=True)
        return

    con = get_connection(env, '{1}')

    script = load_file_content('script.txt')

    ret = con.execute_command('AI.SCRIPTSTORE', 'ket_script{1}', DEVICE, 'ENTRY_POINTS', 2, 'bar', 'bar_variadic', 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')
    ret = con.execute_command('AI.TENSORSET', 'b{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    ret = send_and_disconnect(('AI.SCRIPTEXECUTE', 'ket_script{1}', 'bar', 'KEYS', '{1}', 'INPUTS', 2, 'a{1}', 'b{1}', 'OUTPUTS', 1, 'c{1}'), con)
    env.assertEqual(ret, None)


def test_pytorch_modelrun_disconnect(env):
    if not TEST_PT:
        env.debugPrint("skipping {} since TEST_PT=0".format(sys._getframe().f_code.co_name), force=True)
        return

    if DEVICE == "GPU":
        env.debugPrint("skipping {} since it's hanging CI".format(sys._getframe().f_code.co_name), force=True)
        return

    con = get_connection(env, '{1}')

    model_pb = load_file_content('pt-minimal.pt')

    ret = con.execute_command('AI.MODELSTORE', 'm{1}', 'TORCH', DEVICE, 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.TENSORSET', 'b{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    ret = send_and_disconnect(('AI.MODELEXECUTE', 'm{1}', 'INPUTS', 2, 'a{1}', 'b{1}', 'OUTPUTS', 1, 'c{1}'), con)
    env.assertEqual(ret, None)


def test_pytorch_modelscan_scriptscan(env):
    if not TEST_PT:
        env.debugPrint("skipping {} since TEST_PT=0".format(sys._getframe().f_code.co_name), force=True)
        return

    con = get_connection(env, '{1}')

    # ensure cleaned DB
    # env.flush()

    model_pb = load_file_content('pt-minimal.pt')

    ret = con.execute_command('AI.MODELSTORE', 'm1{1}', 'TORCH', DEVICE, 'TAG', 'm:v1', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.MODELSTORE', 'm2{1}', 'TORCH', DEVICE, 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    script = load_file_content('script.txt')

    ret = con.execute_command('AI.SCRIPTSTORE', 's1{1}', DEVICE, 'TAG', 's:v1', 'ENTRY_POINTS', 2, 'bar', 'bar_variadic', 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.SCRIPTSTORE', 's2{1}', DEVICE, 'ENTRY_POINTS', 2, 'bar', 'bar_variadic', 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    ret = con.execute_command('AI._MODELSCAN')
    env.assertEqual(2, len(ret[0]))
    env.assertEqual(2, len(ret[1]))

    ret = con.execute_command('AI._SCRIPTSCAN')

    env.assertEqual(2, len(ret[0]))
    env.assertEqual(2, len(ret[1]))


def test_parallelism():
    env = Env(moduleArgs='INTRA_OP_PARALLELISM 1 INTER_OP_PARALLELISM 1')
    if not TEST_PT:
        env.debugPrint("skipping {} since TEST_PT=0".format(sys._getframe().f_code.co_name), force=True)
        return

    con = get_connection(env, '{1}')

    model_pb = load_file_content('pt-minimal.pt')

    ret = con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    ret = con.execute_command('AI.TENSORSET', 'b{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    ret = con.execute_command('AI.MODELSTORE', 'm{1}', 'TORCH', DEVICE, 'BLOB', model_pb)
    ensureSlaveSynced(con, env)
    con.execute_command('AI.MODELEXECUTE', 'm{1}', 'INPUTS', 2, 'a{1}', 'b{1}', 'OUTPUTS', 1, 'c{1}')
    ensureSlaveSynced(con, env)
    values = con.execute_command('AI.TENSORGET', 'c{1}', 'VALUES')
    env.assertEqual(values, [b'4', b'6', b'4', b'6'])
    load_time_config = get_info_section(con, 'load_time_configs')
    env.assertEqual(load_time_config["ai_inter_op_parallelism"], "1")
    env.assertEqual(load_time_config["ai_intra_op_parallelism"], "1")

    env = Env(moduleArgs='INTRA_OP_PARALLELISM 2 INTER_OP_PARALLELISM 2')
    load_time_config = get_info_section(con, 'load_time_configs')
    env.assertEqual(load_time_config["ai_inter_op_parallelism"], "2")
    env.assertEqual(load_time_config["ai_intra_op_parallelism"], "2")


def test_modelget_for_tuple_output(env):
    if not TEST_PT:
        env.debugPrint("skipping {} since TEST_PT=0".format(sys._getframe().f_code.co_name), force=True)
        return
    con = get_connection(env, '{1}')

    model_pb = load_file_content('pt-minimal-bb.pt')
    ret = con.execute_command('AI.MODELSTORE', 'm{1}', 'TORCH', DEVICE, 'BLOB', model_pb)
    ensureSlaveSynced(con, env)
    env.assertEqual(b'OK', ret)
    ret = con.execute_command('AI.MODELGET', 'm{1}', 'META')
    env.assertEqual(ret[1], b'TORCH')
    env.assertEqual(ret[5], b'')
    env.assertEqual(ret[7], 0)
    env.assertEqual(ret[9], 0)
    env.assertEqual(ret[15], 0)
    env.assertEqual(len(ret[11]), 2)
    env.assertEqual(len(ret[13]), 2)


def test_torch_info(env):
    if not TEST_PT:
        env.debugPrint("skipping {}".format(sys._getframe().f_code.co_name), force=True)
        return
    con = get_connection(env, '{1}')

    backends_info = get_info_section(con, 'backends_info')
    env.assertFalse('ai_Torch_version' in backends_info)

    model_pb = load_file_content('pt-minimal-bb.pt')
    ret = con.execute_command('AI.MODELSTORE', 'm{1}', 'TORCH', DEVICE, 'BLOB', model_pb)

    backends_info = get_info_section(con, 'backends_info')
    env.assertTrue('ai_Torch_version' in backends_info)
