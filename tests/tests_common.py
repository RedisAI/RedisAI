import redis

from includes import *

'''
python -m RLTest --tests tests_common.py --module path/to/redisai.so
'''


def test_common_tensorset(env):
    con = env.getConnection()

    tested_datatypes = ["FLOAT", "DOUBLE", "INT8", "INT16", "INT32", "INT64", "UINT8", "UINT16"]
    for datatype in tested_datatypes:
        ret = con.execute_command('AI.TENSORSET', 'tensor_{0}'.format(datatype), datatype, 2, 'VALUES', 1, 1)
        env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    # AI.TENSORGET in BLOB format and set in a new key
    for datatype in tested_datatypes:
        _, tensor_dtype, _, tensor_dim, _, tensor_blob = con.execute_command('AI.TENSORGET', 'tensor_{0}'.format(datatype),
                                                                             'META', 'BLOB')
        ret = con.execute_command('AI.TENSORSET', 'tensor_blob_{0}'.format(datatype), datatype, 2, 'BLOB', tensor_blob)
        env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    reply_types = ["META", "VALUES", "BLOB"]
    # Confirm that tensor_{0} and tensor_blob_{0} are equal for META VALUES BLOB
    for datatype in tested_datatypes:
        for reply_type in reply_types:
            tensor_1_reply = con.execute_command('AI.TENSORGET', 'tensor_{0}'.format(datatype), reply_type)
            tensor_2_reply = con.execute_command('AI.TENSORGET', 'tensor_blob_{0}'.format(datatype), reply_type)
            env.assertEqual(tensor_1_reply, tensor_2_reply)


def test_common_tensorset_error_replies(env):
    con = env.getConnection()
    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    sample_filename = os.path.join(test_data_path, 'one.raw')

    with open(sample_filename, 'rb') as f:
        sample_raw = f.read()

    ret = con.execute_command('AI.TENSORSET', 'sample_raw_ok', 'FLOAT', 1, 1, 28, 28, 'BLOB', sample_raw)
    env.assertEqual(ret, b'OK')

    # WRONGTYPE Operation against a key holding the wrong kind of value
    try:
        con.execute_command('SET','non-tensor','value')
        con.execute_command('AI.TENSORSET', 'non-tensor', 'INT32', 2, 'unsupported', 2, 3)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual(exception.__str__(), "WRONGTYPE Operation against a key holding the wrong kind of value")

    # ERR invalid data type
    try:
        con.execute_command('AI.TENSORSET', 'z', 'INT128', 2, 'VALUES', 2, 3)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual(exception.__str__(), "invalid data type")

    # ERR invalid or negative value found in tensor shape
    try:
        con.execute_command('AI.TENSORSET', 'z', 'INT32', -1, 'VALUES', 2, 3)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("invalid or negative value found in tensor shape",exception.__str__())

    # ERR invalid argument found in tensor shape
    try:
        con.execute_command('AI.TENSORSET', 'z', 'INT32', 2, 'unsupported', 2, 3)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("invalid or negative value found in tensor shape",exception.__str__())

    # ERR invalid value
    try:
        con.execute_command('AI.TENSORSET', 'z', 'FLOAT', 2, 'VALUES', 2, 'A')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("invalid value",exception.__str__())

    # ERR invalid value
    try:
        con.execute_command('AI.TENSORSET', 'z', 'INT32', 2, 'VALUES', 2, 'A')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual(exception.__str__(), "invalid value")

    try:
        con.execute_command('AI.TENSORSET', 1)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.TENSORSET', 'y', 'FLOAT')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.TENSORSET', 'y', 'FLOAT', '2')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.TENSORSET', 'y', 'FLOAT', 2, 'VALUES')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.TENSORSET', 'y', 'FLOAT', 2, 'VALUES', 1)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.TENSORSET', 'y', 'FLOAT', 2, 'VALUES', '1')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.TENSORSET', 'blob_tensor_moreargs', 'FLOAT', 2, 'BLOB', '\x00', 'extra-argument')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("wrong number of arguments for 'AI.TENSORSET' command", exception.__str__())

    try:
        con.execute_command('AI.TENSORSET', 'blob_tensor_lessargs', 'FLOAT', 2, 'BLOB')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("wrong number of arguments for 'AI.TENSORSET' command", exception.__str__())

    # ERR data length does not match tensor shape and type
    try:
        con.execute_command('AI.TENSORSET', 'sample_raw_wrong_blob_for_dim', 'FLOAT', 1, 1, 28, 280, 'BLOB', sample_raw)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("data length does not match tensor shape and type", exception.__str__())


def test_common_tensorget(env):
    con = env.getConnection()
    tested_datatypes = ["FLOAT", "DOUBLE", "INT8", "INT16", "INT32", "INT64", "UINT8", "UINT16"]
    tested_datatypes_fp = ["FLOAT", "DOUBLE"]
    tested_datatypes_int = ["INT8", "INT16", "INT32", "INT64", "UINT8", "UINT16"]
    for datatype in tested_datatypes:
        ret = con.execute_command('AI.TENSORSET', 'tensor_{0}'.format(datatype), datatype, 2, 'VALUES', 1, 1)
        env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    # AI.TENSORGET in BLOB format and set in a new key
    for datatype in tested_datatypes:
        tensor_blob = con.execute_command('AI.TENSORGET', 'tensor_{0}'.format(datatype), 'BLOB')
        ret = con.execute_command('AI.TENSORSET', 'tensor_blob_{0}'.format(datatype), datatype, 2, 'BLOB', tensor_blob)
        env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    reply_types = ["META", "VALUES", "BLOB"]
    # Confirm that tensor_{0} and tensor_blog_{0} are equal for META VALUES BLOB
    for datatype in tested_datatypes:
        for reply_type in reply_types:
            tensor_1_reply = con.execute_command('AI.TENSORGET', 'tensor_{0}'.format(datatype), reply_type)
            tensor_2_reply = con.execute_command('AI.TENSORGET', 'tensor_blob_{0}'.format(datatype), reply_type)
            env.assertEqual(tensor_1_reply, tensor_2_reply)

    # Confirm that the output is the expected for META
    for datatype in tested_datatypes:
        _, tensor_dtype, _, tensor_dim = con.execute_command('AI.TENSORGET', 'tensor_{0}'.format(datatype), "META")
        env.assertEqual(datatype.encode('utf-8'), tensor_dtype)
        env.assertEqual([2], tensor_dim)

    # Confirm that the output is the expected for VALUES
    for datatype in tested_datatypes:
        _, tensor_dtype, _, tensor_dim, _, tensor_values = con.execute_command('AI.TENSORGET', 'tensor_{0}'.format(datatype),
                                                                               'META', 'VALUES')
        env.assertEqual(datatype.encode('utf-8'), tensor_dtype)
        env.assertEqual([2], tensor_dim)
        if datatype in tested_datatypes_fp:
            env.assertEqual([b'1', b'1'], tensor_values)
        if datatype in tested_datatypes_int:
            env.assertEqual([1, 1], tensor_values)

    # Confirm that the output is the expected for BLOB
    for datatype in tested_datatypes:
        _, tensor_dtype, _, tensor_dim, _, tensor_blob = con.execute_command('AI.TENSORGET', 'tensor_{0}'.format(datatype),
                                                                             'META', 'BLOB')
        env.assertEqual(datatype.encode('utf-8'), tensor_dtype)
        env.assertEqual([2], tensor_dim)


def test_common_tensorget_error_replies(env):
    con = env.getConnection()

    # ERR tensor key is empty
    try:
        con.execute_command('AI.TENSORGET', 'empty', 'unsupported')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("tensor key is empty",exception.__str__())

    # WRONGTYPE Operation against a key holding the wrong kind of value
    try:
        con.execute_command('SET', 'non-tensor', 'value')
        con.execute_command('AI.TENSORGET', 'non-tensor', 'unsupported')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("WRONGTYPE Operation against a key holding the wrong kind of value",exception.__str__())

    # ERR unsupported data format
    ret = con.execute_command('AI.TENSORSET', "T_FLOAT", "FLOAT", 2, 'VALUES', 1, 1)
    env.assertEqual(ret, b'OK')
    try:
        con.execute_command('AI.TENSORGET', 'T_FLOAT', 'unsupported')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("unsupported data format",exception.__str__())


def test_common_tensorset_multiproc(env):
    run_test_multiproc(env, 10,
                       lambda env: env.execute_command('AI.TENSORSET', 'x', 'FLOAT', 2, 'VALUES', 2, 3))

    con = env.getConnection()
    ensureSlaveSynced(con, env)
    values = con.execute_command('AI.TENSORGET', 'x', 'VALUES')
    env.assertEqual(values, [b'2', b'3'])


def test_common_tensorset_multiproc_blob(env):
    con = env.getConnection()
    tested_datatypes = ["FLOAT", "DOUBLE", "INT8", "INT16", "INT32", "INT64", "UINT8", "UINT16"]
    tested_datatypes_map = {}
    for datatype in tested_datatypes:
        ret = con.execute_command('AI.TENSORSET', 'tensor_{0}'.format(datatype), datatype, 1, 256)
        env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    # AI.TENSORGET in BLOB format and set in a new key
    for datatype in tested_datatypes:
        tensor_blob = con.execute_command('AI.TENSORGET', 'tensor_{0}'.format(datatype),
                                                                    'BLOB')
        ret = con.execute_command('AI.TENSORSET', 'tensor_blob_{0}'.format(datatype), datatype, 1, 256, 'BLOB', tensor_blob)
        tested_datatypes_map[datatype] = tensor_blob
        env.assertEqual(ret, b'OK')

    def funcname(env, blob, repetitions, same_key):
        for _ in range(1,same_key):
            for repetion in range(1,repetitions):
                env.execute_command('AI.TENSORSET', 'tensor_{0}'.format(repetitions), 'FLOAT', 1, 256, 'BLOB', blob)
    
    tensor_blob = tested_datatypes_map["FLOAT"]
    t = time.time()
    run_test_multiproc(env, 10,
                       lambda env: funcname(env,tensor_blob,MAX_TRANSACTIONS,10) )
    elapsed_time = time.time() - t
    avg_ops_sec = 100000*10/elapsed_time
    # env.debugPrint("AI.TENSORSET elapsed time(sec) {:6.2f}\tAvg. ops/sec {:10.2f}".format(elapsed_time, avg_ops_sec), True)


def test_tensorset_disconnect(env):
    red = env.getConnection()
    ret = send_and_disconnect(('AI.TENSORSET', 't_FLOAT', 'FLOAT', 2, 'VALUES', 2, 3), red)
    env.assertEqual(ret, None)


def test_tensorget_disconnect(env):
    red = env.getConnection()
    ret = red.execute_command('AI.TENSORSET', 't_FLOAT', 'FLOAT', 2, 'VALUES', 2, 3)
    env.assertEqual(ret, b'OK')
    ret = send_and_disconnect(('AI.TENSORGET', 't_FLOAT', 'META'), red)
    env.assertEqual(ret, None)

def test_info_modules(env):
    red = env.getConnection()
    ret = red.execute_command('INFO','MODULES')
    env.assertEqual( ret['ai_threads_per_queue'], 1 )
    # minimum cpu properties
    env.assertEqual( 'ai_self_used_cpu_sys' in ret, True )
    env.assertEqual( 'ai_self_used_cpu_user' in ret, True )
    env.assertEqual( 'ai_children_used_cpu_sys' in ret, True )
    env.assertEqual( 'ai_children_used_cpu_user' in ret, True )
    env.assertEqual( 'ai_queue_CPU_bthread_#1_used_cpu_total' in ret, True )