import redis

from includes import *

'''
python -m RLTest --test tests_common.py --module path/to/redisai.so
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
        tensor_dtype, tensor_dim, tensor_blob = con.execute_command('AI.TENSORGET', 'tensor_{0}'.format(datatype),
                                                                    'BLOB')
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


def test_common_tensorset_error_replies(env):
    con = env.getConnection()

    # ERR unsupported data format
    try:
        con.execute_command('AI.TENSORSET', 'z', 'INT32', 2, 'unsupported', 2, 3)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual(exception.__str__(), "invalid argument found in tensor shape")

    # ERR invalid value
    try:
        con.execute_command('AI.TENSORSET', 'z', 'FLOAT', 2, 'VALUES', 2, 'A')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual(exception.__str__(), "invalid value")

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
        tensor_dtype, tensor_dim, tensor_blob = con.execute_command('AI.TENSORGET', 'tensor_{0}'.format(datatype),
                                                                    'BLOB')
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
        tensor_dtype, tensor_dim = con.execute_command('AI.TENSORGET', 'tensor_{0}'.format(datatype), "META")
        env.assertEqual(datatype.encode('utf-8'), tensor_dtype)
        env.assertEqual([2], tensor_dim)

    # Confirm that the output is the expected for VALUES
    for datatype in tested_datatypes:
        tensor_dtype, tensor_dim, tensor_values = con.execute_command('AI.TENSORGET', 'tensor_{0}'.format(datatype),
                                                                      "VALUES")
        env.assertEqual(datatype.encode('utf-8'), tensor_dtype)
        env.assertEqual([2], tensor_dim)
        if datatype in tested_datatypes_fp:
            env.assertEqual([b'1', b'1'], tensor_values)
        if datatype in tested_datatypes_int:
            env.assertEqual([1, 1], tensor_values)

    # Confirm that the output is the expected for BLOB
    for datatype in tested_datatypes:
        tensor_dtype, tensor_dim, tensor_blog = con.execute_command('AI.TENSORGET', 'tensor_{0}'.format(datatype),
                                                                    "BLOB")
        env.assertEqual(datatype.encode('utf-8'), tensor_dtype)
        env.assertEqual([2], tensor_dim)


def test_common_tensorget_error_replies(env):
    con = env.getConnection()

    # ERR unsupported data format
    try:
        ret = con.execute_command('AI.TENSORSET', "T_FLOAT", "FLOAT", 2, 'VALUES', 1, 1)
        env.assertEqual(ret, b'OK')
        con.execute_command('AI.TENSORGET', 'T_FLOAT', 'unsupported')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual(exception.__str__(), "unsupported data format")


def test_common_tensorset_multiproc(env):
    run_test_multiproc(env, 10,
                       lambda env: env.execute_command('AI.TENSORSET', 'x', 'FLOAT', 2, 'VALUES', 2, 3))

    con = env.getConnection()
    ensureSlaveSynced(con, env)
    tensor = con.execute_command('AI.TENSORGET', 'x', 'VALUES')
    values = tensor[-1]
    env.assertEqual(values, [b'2', b'3'])


def test_tensorset_disconnect(env):
    red = env.getConnection()
    ret = send_and_disconnect(('AI.TENSORSET', 't_FLOAT', 'FLOAT', 2, 'VALUES', 2, 3), red)
    env.assertEqual(ret, None)


def test_tensorget_disconnect(env):
    red = env.getConnection()
    ret = red.execute_command('AI.TENSORSET', 't_FLOAT', 'FLOAT', 2, 'VALUES', 2, 3)
    env.assertEqual(ret, b'OK')
    ret = send_and_disconnect(('AI.TENSORGET', 't_FLOAT'), red)
    env.assertEqual(ret, None)
